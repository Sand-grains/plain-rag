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


class TestArchive:
    def test_archive_via_cli(self, tmp_path):
        path = _write_features(tmp_path, [
            ("F01", PASSED, PASSED_GATE),
            ("F02", ACTIVE, PASSED_GATE),
        ])
        assert main(["archive"], features_path=path) == 0
        assert {item.id for item in Registry(path).load().items} == {"F02"}
        history = Registry(path).history_files()
        assert len(history) == 1 and history[0]["count"] == 1

    def test_archive_exclude_via_cli(self, tmp_path):
        path = _write_features(tmp_path, [
            ("F01", PASSED, PASSED_GATE),
            ("F02", PASSED, PASSED_GATE),
        ])
        assert main(["archive", "--exclude", "F01"], features_path=path) == 0
        assert {item.id for item in Registry(path).load().items} == {"F01"}

    def test_archive_nothing_to_archive(self, tmp_path, capsys):
        path = _write_features(tmp_path, [("F01", ACTIVE, PASSED_GATE)])
        assert main(["archive"], features_path=path) == 0
        assert "无已 commit" in capsys.readouterr().out


class TestHistory:
    def test_history_lists_archive_files(self, tmp_path, capsys):
        path = _write_features(tmp_path, [("F01", PASSED, PASSED_GATE)])
        main(["archive"], features_path=path)
        assert main(["history"], features_path=path) == 0
        out = capsys.readouterr().out
        assert "archive-" in out and "1 项" in out

    def test_history_empty(self, tmp_path, capsys):
        path = _write_features(tmp_path, [("F01", ACTIVE, PASSED_GATE)])
        assert main(["history"], features_path=path) == 0
        assert "无历史归档" in capsys.readouterr().out


class TestLoadAllRouting:
    def test_status_counts_history_items(self, tmp_path, capsys):
        path = _write_features(tmp_path, [("F01", PASSED, PASSED_GATE)])
        main(["archive"], features_path=path)  # F01 -> history, features.yaml 空
        assert main(["status"], features_path=path) == 0
        out = capsys.readouterr().out
        assert "passed: 1" in out  # 全量统计含历史项

    def test_verify_all_promotes_failed_history_item(self, tmp_path):
        failing = _failing_gate(tmp_path)
        path = _write_features(tmp_path, [
            ("F01", PASSED, PASSED_GATE),
            ("F02", PASSED, failing),
        ])
        main(["archive"], features_path=path)  # 两项都归档
        assert main(["verify-all"], features_path=path) == 1
        states = {item.id: item.state for item in Registry(path).load().items}
        assert states["F02"] == REGRESSED  # 历史项失败 promote 回并转 regressed
        history = Registry(path).history_files()
        assert history[0]["count"] == 1  # F01 通过保持归档

    def test_report_active_table_and_full_health(self, tmp_path):
        path = _write_features(tmp_path, [
            ("F01", PASSED, PASSED_GATE),
            ("F02", ACTIVE, PASSED_GATE),
        ])
        main(["archive"], features_path=path)  # F01 -> history, F02 保留
        progress = tmp_path / "PROGRESS.md"
        assert main(["report"], features_path=path, progress_path=progress) == 0
        content = progress.read_text(encoding="utf-8")
        # 主表只渲染活跃集: F02 在主表, F01 不进主表
        main_table = content.split("## 功能项状态表")[1].split("## 活跃与阻塞项叙述")[0]
        assert "F02" in main_table
        assert "F01" not in main_table
        assert "## 历史归档" in content
        assert "passed: 1" in content  # 健康度按全量统计
