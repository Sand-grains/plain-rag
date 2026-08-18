"""日志/trace 保留策略: 防 logs/ 无限累积
供 finalize_traces 与 scripts/prune_logs.py(定时计划任务入口)共用。

两条规则:
    - trace 文件(logs/traces/**/*.jsonl): 超过 max_traces 时保留最近 max_traces 个(scripts/prune_logs.py 每 3 天触发)
    - logs/ 下全部文件: 超过 max_all_logs 时保留最近 max_all_logs 个(任何时候硬上限)

文件按 mtime 排序, 最近者优先保留; 删除后清理空的天级子目录(logs/traces/YYYY-MM-DD/)。
"""

from __future__ import annotations

import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def prune_logs(logs_dir: str | Path | None = None,
               max_traces: int | None = 100,
               max_all_logs: int | None = 200) -> int:
    """清理 logs/ 下过旧文件, 并返回删除的文件数。

    Args:
        logs_dir: 日志根目录, 默认 config.OBS_LOG_DIR(测试可显式传入临时目录)。
        max_traces: trace 文件保留上限; None 时跳过 trace 规则。
        max_all_logs: logs/ 下全部文件保留上限; None 时跳过总上限规则。

    Returns:
        int: 删除的文件数。
    """
    logs_path = Path(config.OBS_LOG_DIR if logs_dir is None else logs_dir)
    pruned = 0
    trace_dir = logs_path / "traces"
    if max_traces is not None and trace_dir.exists():
        pruned += _prune_oldest(list(trace_dir.rglob("*.jsonl")), max_traces)
        _remove_empty_day_dirs(trace_dir)
    if max_all_logs is not None:
        all_files = [path for path in logs_path.rglob("*") if path.is_file()]
        pruned += _prune_oldest(all_files, max_all_logs)
    if pruned:
        logger.info("prune_logs: 清理 %d 个过旧文件", pruned)
    return pruned


def _prune_oldest(files: list[Path], keep: int) -> int:
    """按 mtime 保留最近 keep 个, 删除更旧者; 返回删除数。"""
    if len(files) <= keep:
        return 0
    ordered = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)
    pruned = 0
    for path in ordered[keep:]:
        try:
            path.unlink()
        except OSError:
            continue
        pruned += 1
    return pruned


def _remove_empty_day_dirs(trace_dir: Path) -> None:
    """清理 trace 根下的空天级子目录(如 logs/traces/2026-08-16/)。"""
    for day_dir in trace_dir.iterdir():
        if day_dir.is_dir() and not any(day_dir.iterdir()):
            try:
                day_dir.rmdir()
            except OSError:
                continue
