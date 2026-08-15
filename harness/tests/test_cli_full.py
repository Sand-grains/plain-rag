"""unit：CLI 完整子命令(F07b), 依赖 F04-F06。

next / status / verify-all / report 经 cli.main() 端到端驱动。
"""
from harness.cli import main
from harness.core.models import FeatureItem, FeatureList
from harness.core.registry import Registry
from harness.core.states import ACTIVE, NOT_STARTED, PASSED, REGRESSED

PASSED_GATE = "uv run pytest harness/tests/test_state_machine.py -q"


def _write_features(tmp_path, items):
    path = tmp_path / "features.yaml"
    registry = Registry(path)
    registry.save(FeatureList(
        schema=1,
        milestone="007-harness-governance",
        items=[FeatureItem(id=item_id, behavior=f"行为-{item_id}",
                           gate=gate, state=state)
               for item_id, state, gate in items],
    ))
    return path


def _failing_gate(tmp_path) -> str:
    failing = tmp_path / "failing_test.py"
    failing.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    return f"uv run pytest {failing} -q"


def _state(path, item_id: str) -> str:
    for item in Registry(path).load().items:
        if item.id == item_id:
            return item.state
    raise AssertionError(item_id)


class TestNext:
    def test_next_prints_candidate(self, tmp_path, capsys):
        path = _write_features(tmp_path, [
            ("F02", PASSED, PASSED_GATE),
            ("F01", NOT_STARTED, PASSED_GATE),
        ])
        assert main(["next"], features_path=path) == 0
        out = capsys.readouterr().out
        assert "F01" in out and "行为-F01" in out

    def test_next_none_when_no_candidate(self, tmp_path, capsys):
        path = _write_features(tmp_path, [("F01", PASSED, PASSED_GATE)])
        assert main(["next"], features_path=path) == 0
        assert "无候选" in capsys.readouterr().out


class TestStatus:
    def test_status_prints_distribution_and_health(self, tmp_path, capsys):
        path = _write_features(tmp_path, [
            ("F01", PASSED, PASSED_GATE),
            ("F02", ACTIVE, PASSED_GATE),
        ])
        assert main(["status"], features_path=path) == 0
        out = capsys.readouterr().out
        assert "passed: 1" in out and "active: 1" in out
        assert "回归率" in out and "验证通过率" in out


class TestVerifyAll:
    def test_regressed_on_broken_gate(self, tmp_path):
        failing = _failing_gate(tmp_path)
        path = _write_features(tmp_path, [
            ("F01", PASSED, PASSED_GATE),
            ("F02", PASSED, failing),
        ])
        assert main(["verify-all"], features_path=path) == 1
        assert _state(path, "F01") == PASSED
        assert _state(path, "F02") == REGRESSED

    def test_all_pass_returns_zero(self, tmp_path):
        path = _write_features(tmp_path, [
            ("F01", PASSED, PASSED_GATE),
            ("F02", PASSED, PASSED_GATE),
        ])
        assert main(["verify-all"], features_path=path) == 0
        assert _state(path, "F01") == PASSED and _state(path, "F02") == PASSED


class TestReport:
    def test_report_preserves_manual_zone(self, tmp_path):
        path = _write_features(tmp_path, [("F01", PASSED, PASSED_GATE)])
        progress = tmp_path / "PROGRESS.md"
        progress.write_text("# 旧\n<!-- manual -->\n人工保护区内容\n", encoding="utf-8")
        assert main(["report"], features_path=path, progress_path=progress) == 0
        content = progress.read_text(encoding="utf-8")
        assert "人工保护区内容" in content
        assert "## 健康度" in content and "F01" in content
