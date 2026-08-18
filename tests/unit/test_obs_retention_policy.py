"""unit：obs/retention_policy 保留策略——trace>N 保留最近 / 总文件上限 / 按天分包 / 空目录清理。"""
import os
from datetime import datetime
from pathlib import Path

import pytest

from obs.lifecycle import finalize_traces
from obs.retention_policy import prune_logs
from obs.trace import clear_session_traces, trace_scope


@pytest.fixture(autouse=True)
def _clean_session():
    clear_session_traces()
    yield
    clear_session_traces()


def _touch(path: Path, mtime: float) -> None:
    """写文件并显式设 mtime, 制造新旧次序(避免同秒写入 mtime 相同)。"""
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (mtime, mtime))


class TestPruneLogs:
    def test_trace_over_limit_keeps_most_recent(self, tmp_path):
        day_dir = tmp_path / "traces" / "2026-08-16"
        day_dir.mkdir(parents=True)
        for index in range(3):
            _touch(day_dir / f"trace-{index}.jsonl", 1000 + index)
        pruned = prune_logs(logs_dir=tmp_path, max_traces=2, max_all_logs=None)
        assert pruned == 1
        remaining = sorted(path.name for path in day_dir.glob("*.jsonl"))
        assert remaining == ["trace-1.jsonl", "trace-2.jsonl"]  # 保留最近 2 个

    def test_total_cap_keeps_most_recent(self, tmp_path):
        for index in range(3):
            _touch(tmp_path / f"file-{index}.txt", 1000 + index)
        pruned = prune_logs(logs_dir=tmp_path, max_traces=None, max_all_logs=2)
        assert pruned == 1
        remaining = sorted(path.name for path in tmp_path.iterdir() if path.is_file())
        assert remaining == ["file-1.txt", "file-2.txt"]

    def test_empty_day_dir_removed_after_prune(self, tmp_path):
        old_day = tmp_path / "traces" / "2026-08-16"
        old_day.mkdir(parents=True)
        _touch(old_day / "trace-old.jsonl", 1000)
        new_day = tmp_path / "traces" / "2026-08-17"
        new_day.mkdir(parents=True)
        _touch(new_day / "trace-new.jsonl", 2000)
        prune_logs(logs_dir=tmp_path, max_traces=1, max_all_logs=None)
        assert not old_day.exists()  # 空的天级目录被清理
        assert new_day.exists()

    def test_below_limit_no_prune(self, tmp_path):
        day_dir = tmp_path / "traces" / "2026-08-16"
        day_dir.mkdir(parents=True)
        for index in range(2):
            _touch(day_dir / f"trace-{index}.jsonl", 1000 + index)
        assert prune_logs(logs_dir=tmp_path, max_traces=5, max_all_logs=None) == 0


class TestFinalizePerDay:
    def test_finalize_writes_to_day_subdir(self, tmp_path):
        with trace_scope("Q1", "q"):
            pass
        trace_path = finalize_traces([], [], out_dir=tmp_path)
        assert trace_path is not None
        assert trace_path.parent.name == datetime.now().strftime("%Y-%m-%d")
        assert trace_path.parent.parent.name == "traces"
        assert trace_path.exists()
