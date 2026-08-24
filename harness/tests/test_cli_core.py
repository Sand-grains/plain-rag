"""unit：CLI 核心子命令(F07a), 不依赖 F04-F06。

start/verify/block/unblock/reactivate/abandon/note 经 cli.main() 端到端驱动,
只 import 公开接口 main/Registry, 覆盖验收主路径与非法转移拒绝。
"""
import pytest

from harness.cli import main
from harness.core.models import FeatureItem, FeatureList
from harness.core.registry import Registry
from harness.core.states import ABANDONED, ACTIVE, BLOCKED, NOT_STARTED, PASSED, REGRESSED

PASSED_GATE = "uv run pytest harness/tests/test_state_machine.py -q"


def _write_features(tmp_path, state=NOT_STARTED, gate=PASSED_GATE,
                    e2e=None):
    path = tmp_path / "features.yaml"
    registry = Registry(path)
    registry.save(FeatureList(
        schema=1,
        milestone="007-harness-governance",
        items=[FeatureItem(id="F01", behavior="测试功能项", gate=gate, e2e=e2e, state=state)],
    ))
    return path


def _state(path) -> str:
    return Registry(path).load().items[0].state


class TestStart:
    def test_start_transitions_to_active(self, tmp_path):
        path = _write_features(tmp_path)
        assert main(["start", "F01"], features_path=path) == 0
        assert _state(path) == ACTIVE

    def test_start_unknown_item_fails(self, tmp_path):
        path = _write_features(tmp_path)
        assert main(["start", "F99"], features_path=path) == 1

    def test_start_illegal_from_active(self, tmp_path):
        path = _write_features(tmp_path, state=ACTIVE)
        assert main(["start", "F01"], features_path=path) == 1


class TestVerify:
    def test_verify_pass_to_passed_with_evidence(self, tmp_path):
        path = _write_features(tmp_path, state=ACTIVE)
        assert main(["verify", "F01"], features_path=path) == 0
        item = Registry(path).load().items[0]
        assert item.state == PASSED
        assert item.last_verify["tests"] > 0

    def test_verify_fail_keeps_active(self, tmp_path):
        failing = tmp_path / "failing_test.py"
        failing.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
        path = _write_features(tmp_path, state=ACTIVE, gate=f"uv run pytest {failing} -q")
        assert main(["verify", "F01"], features_path=path) == 1
        assert _state(path) == ACTIVE

    def test_verify_fail_writes_cardpoint_note(self, tmp_path):
        failing = tmp_path / "failing_test.py"
        failing.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
        path = _write_features(tmp_path, state=ACTIVE, gate=f"uv run pytest {failing} -q")
        main(["verify", "F01", "-m", "门禁未过: 断言不符"], features_path=path)
        assert Registry(path).load().items[0].note == "门禁未过: 断言不符"

    def test_verify_not_started_rejected(self, tmp_path):
        path = _write_features(tmp_path, state=NOT_STARTED)
        assert main(["verify", "F01"], features_path=path) == 1

    def test_verify_blocked_rejected(self, tmp_path):
        path = _write_features(tmp_path, state=BLOCKED)
        assert main(["verify", "F01"], features_path=path) == 1

    def test_verify_e2e_without_e2e_command(self, tmp_path):
        path = _write_features(tmp_path, state=ACTIVE, e2e=None)
        assert main(["verify", "F01", "--e2e"], features_path=path) == 1


class TestBlockUnblock:
    def test_block_requires_note(self, tmp_path):
        path = _write_features(tmp_path, state=ACTIVE)
        with pytest.raises(SystemExit):
            main(["block", "F01"], features_path=path)

    def test_block_then_unblock(self, tmp_path):
        path = _write_features(tmp_path, state=ACTIVE)
        assert main(["block", "F01", "-m", "卡点: 等接口"], features_path=path) == 0
        item = Registry(path).load().items[0]
        assert item.state == BLOCKED and item.note == "卡点: 等接口"
        assert main(["unblock", "F01", "-m", "接口已冻结"], features_path=path) == 0
        assert _state(path) == ACTIVE


class TestReactivateRecovery:
    def test_full_acceptance_path(self, tmp_path):
        # start -> verify(通过) -> passed -> 改坏门禁 -> verify(失败) -> regressed
        # -> 修复 -> reactivate(带note, 用户执行) -> verify -> passed
        path = _write_features(tmp_path, state=ACTIVE, gate=PASSED_GATE)
        assert main(["verify", "F01"], features_path=path) == 0
        assert _state(path) == PASSED

        failing = tmp_path / "failing_test.py"
        failing.write_text("def test_bad():\n    assert False\n", encoding="utf-8")
        registry = Registry(path)
        feature_list = registry.load()
        feature_list.items[0].gate = f"uv run pytest {failing} -q"
        registry.save(feature_list)

        assert main(["verify", "F01"], features_path=path) == 1
        assert _state(path) == REGRESSED

        # 修复门禁 + reactivate(强制带 note)
        registry = Registry(path)
        feature_list = registry.load()
        feature_list.items[0].gate = PASSED_GATE
        registry.save(feature_list)
        assert main(["reactivate", "F01", "-m", "修复了回归"], features_path=path) == 0
        assert _state(path) == ACTIVE

        # reactivate 不判定通过, 必须再 verify
        assert main(["verify", "F01"], features_path=path) == 0
        assert _state(path) == PASSED


class TestAbandonAndNote:
    def test_abandon_from_passed(self, tmp_path):
        path = _write_features(tmp_path, state=PASSED)
        assert main(["abandon", "F01"], features_path=path) == 0
        assert _state(path) == ABANDONED

    def test_note_does_not_touch_state(self, tmp_path):
        path = _write_features(tmp_path, state=ACTIVE)
        assert main(["note", "F01", "卡点说明"], features_path=path) == 0
        item = Registry(path).load().items[0]
        assert item.note == "卡点说明"
        assert item.state == ACTIVE


class TestArchiveHistory:
    def test_archive_via_cli(self, tmp_path):
        path = tmp_path / "features.yaml"
        Registry(path).save(FeatureList(
            schema=1, milestone="007-harness-governance",
            items=[
                FeatureItem(id="F01", behavior="b1", gate=PASSED_GATE, state=PASSED),
                FeatureItem(id="F02", behavior="b2", gate=PASSED_GATE, state=ACTIVE),
            ],
        ))
        assert main(["archive"], features_path=path) == 0
        assert {item.id for item in Registry(path).load().items} == {"F02"}

    def test_archive_exclude_via_cli(self, tmp_path):
        path = tmp_path / "features.yaml"
        Registry(path).save(FeatureList(
            schema=1, milestone="007-harness-governance",
            items=[
                FeatureItem(id="F01", behavior="b1", gate=PASSED_GATE, state=PASSED),
                FeatureItem(id="F02", behavior="b2", gate=PASSED_GATE, state=PASSED),
            ],
        ))
        assert main(["archive", "--exclude", "F01"], features_path=path) == 0
        assert {item.id for item in Registry(path).load().items} == {"F01"}

    def test_history_via_cli(self, tmp_path, capsys):
        path = tmp_path / "features.yaml"
        Registry(path).save(FeatureList(
            schema=1, milestone="007-harness-governance",
            items=[FeatureItem(id="F01", behavior="b1", gate=PASSED_GATE, state=PASSED)],
        ))
        main(["archive"], features_path=path)
        assert main(["history"], features_path=path) == 0
        assert "archive-" in capsys.readouterr().out

    def test_history_empty(self, tmp_path, capsys):
        path = _write_features(tmp_path, state=ACTIVE)
        assert main(["history"], features_path=path) == 0
        assert "无历史归档" in capsys.readouterr().out
