"""文档解析失败清单落盘。

路径为 <项目根>/eval/results/failure_list.json

目标管线不可用或全部失败时, 把文档记入本地失败清单(JSON) + 日志, 供人工处理后重跑。
失败清单不阻塞主流程(调用方捕获后继续), 也不允许落轻量。

清单记录字段: 文件路径 / 路由决策(text/vlm/colpali/skip) / 失败原因 / 各后端尝试结果 / 版本串。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FailureList:
    """本地失败清单: 追加记录并落盘 JSON(每次运行覆盖, 供人工处理后重跑)。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: list[dict[str, Any]] = []

    def record(self, file_path: str, route_decision: str, reason: str,
               backend_attempts: list[str] | None = None,
               version_string: str = "") -> None:
        """追加一条失败记录并立即落盘(进程即使崩溃也不会丢, 供人工处理后重跑)。

        Args:
            file_path: 文档路径。
            route_decision: 路由决策(text/vlm/colpali/skip)。
            reason: 失败原因(含失败分类)。
            backend_attempts: 各后端尝试结果(如 ["docling:quality_fail", "mineru:parse_error"])。
            version_string: 版本串(后端版本, 便于溯源)。
        """
        self._records.append({
            "file_path": str(file_path),
            "route_decision": route_decision,
            "reason": reason,
            "backend_attempts": backend_attempts or [],
            "version_string": version_string,
        })
        logger.warning("失败清单记录: %s (%s) %s", file_path, route_decision, reason)
        self.write()

    def write(self) -> None:
        """把记录落盘为 JSON(覆盖写, 目录不存在则创建)。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"records": self._records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self._records)


# 模块级单例: 每次进程(=一次 run)从空开始, record 即重写完整 JSON(每次运行覆盖)
failure_list = FailureList(Path(__import__("config").FAILURE_LIST_PATH))


def reset_failure_list() -> None:
    """清空失败清单(测试隔离/新 run 开始)。"""
    failure_list._records.clear()
