"""文件持久化基础设施: 跨平台文件锁, 原子写, 换行保序; 与功能项语义解耦。

被 registry 用于 features.yaml 的并发安全读写(加锁读-改-写 + 临时文件替换), 防御多个 harness 进程同时写坏
  features.yaml
这里的"并发"是指"多个 harness 命令进程并发写同一个文件"
harness CLI 每次 uv run python -m harness verify F0X 都是一个独立进程
如果多个进程要同时改写features.yaml这一个文件(多 agent/脚本/CI 并行触发多个 verify 进程), 就需要锁生效, 保证不丢更新
锁只在两个进程几乎同时进入 registry._commit 时起作用

独立成模块便于复用与单独测试, 不依赖 registry 的任何领域逻辑。
"""
import os
import time
from pathlib import Path

from harness.core.errors import HarnessError


class LockTimeoutError(HarnessError):
    """文件锁等待超时, 无法取得写权限。"""


class _FileLock:
    """跨平台文件锁: O_EXCL 创建锁文件, 失败重试, 超时报错, 陈旧锁自动破。

    用 os.open(O_CREAT|O_EXCL) 实现"同一时间只有一个写者"。进程崩溃遗留的
    锁文件按 mtime 年龄判定为陈旧并自动清理, 避免永久卡死后续写命令。
    """

    def __init__(self, lock_path: Path, timeout: float = 5.0, stale_after: float = 10.0) -> None:
        self._lock_path = lock_path
        self._timeout = timeout
        self._stale_after = stale_after
        self._held = False

    def __enter__(self) -> "_FileLock":
        """获取文件锁: O_EXCL 独占创建锁文件, 冲突则重试至超时, 陈旧锁自动破除。

        Returns:
            _FileLock: 持有锁的自身实例, 供 with 语句绑定。
        """
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, str(os.getpid()).encode("ascii"))
                finally:
                    os.close(fd)
                self._held = True
                return self
            except FileExistsError:
                if self._break_if_stale():
                    continue
                if time.monotonic() >= deadline:
                    raise LockTimeoutError(f"文件锁等待超时: {self._lock_path}")
                time.sleep(0.05)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """释放文件锁: 删除锁文件, 幂等(未持有锁时不动)。"""
        if self._held:
            try:
                os.remove(self._lock_path)
            except OSError:
                pass
            self._held = False

    def _break_if_stale(self) -> bool:
        """按锁文件 mtime 年龄判定陈旧并删除, 进程崩溃遗留的锁不会卡死后续写。

        Returns:
            bool: 是否已清理陈旧锁文件(清理后调用方可立即重试获取)。
        """
        try:
            # st_mtime 是墙上时钟(time.time 基准), 不能用 monotonic 与其比较
            age = time.time() - self._lock_path.stat().st_mtime
        except OSError:
            return False
        if age > self._stale_after:
            try:
                os.remove(self._lock_path)
                return True
            except OSError:
                return False
        return False


def atomic_write(text: str, path: Path, newline: str) -> None:
    """临时文件 + os.replace 原子落盘, 防写坏; CRLF 时先统一换行再写。

    Args:
        text: 待写入文本。
        path: 目标路径。
        newline: 探测到的换行风格("\n" 或 "\r\n")。
    """
    if newline == "\r\n":
        text = text.replace("\n", "\r\n")
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp_path, path)


def detect_newline(path: Path) -> str:
    """探测文件换行风格, 写回时保持既有风格避免整文件改行(Windows)。

    Args:
        path: 目标文件。

    Returns:
        str: "\r\n" 或 "\n", 文件不存在时默认 "\n"。
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096)
    except FileNotFoundError:
        return "\n"
    return "\r\n" if b"\r\n" in head else "\n"
