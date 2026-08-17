"""日志底座: setup_logging() 落盘 + 控制台双通道
request_id 经 ContextVar 注入。

核心特性：
    - RotatingFileHandler 落盘 logs/plain_rag.log(10MB x 5, 编码 utf-8)
    - StreamHandler(stderr) 保留终端可见性(print 换 logger 后状态行不消失)
    - logging.Filter 从 request_id_var 注入 request_id, 日志行自动带请求标识
    - OBS_ENABLED 关时只留控制台、不落盘(观测总开关)
    - 幂等: 同一进程只配置一次, 重复调用直接返回

调用点(一期): eval/runner.py main() 与 agent_pipeline main() 入口各调一次。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config
from obs.trace import request_id_var

FILE_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
CONSOLE_LOG_FORMAT = "%(levelname)s: %(message)s"
LOG_FILE_NAME = "plain_rag.log"
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

_configured: bool = False


def make_stdio_encoding_safe() -> None:
    """把 sys.stdout/stderr 编码错误策略设为 replace: GBK 重定向下不可编码字符(如 ¥)替换为 ? 而不崩溃。

    final_report 等 print 到被重定向(GBK/cp936)的 stdout 时, 文本若含 U+00A5 ¥ 等 GBK 缺码字符,
    strict 默认抛 UnicodeEncodeError, 被 final_report 的裸 except 吞成"(最终报告生成失败)"并截断面板。
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


class RequestIdFilter(logging.Filter):
    """把 request_id_var 的当前值注入每条日志记录;
       var 空值时用默认 "-", 不崩。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging(log_dir: str | Path | None = None, force: bool = False) -> None:
    """配置 root logger: 控制台(stderr) + 落盘(logs/plain_rag.log, OBS_ENABLED 开启时)。

    Args:
        log_dir: 日志目录, 默认取 config.OBS_LOG_DIR(测试可显式传入临时目录)。
        force: True 时忽略幂等标记强制重配(测试用)。

    Returns:
        None: 无返回值, 副作用是给 root logger 挂 handler。
    """
    make_stdio_encoding_safe()
    global _configured
    if _configured and not force:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(logging.Formatter(CONSOLE_LOG_FORMAT))
    console_handler.addFilter(RequestIdFilter())
    root.addHandler(console_handler)

    if config.OBS_ENABLED:
        log_path = Path(config.OBS_LOG_DIR if log_dir is None else log_dir) / LOG_FILE_NAME
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
        file_handler.addFilter(RequestIdFilter())
        root.addHandler(file_handler)
