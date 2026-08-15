"""unit：F08 端到端验证接入 (e2e/precheck/e2e_passed)。

registry e2e/precheck/e2e_passed 字段往返 + timeout floor 校验; verifier e2e/precheck
流程(仅退出码判定); cli verify --e2e 端到端(active->passed / passed->regressed /
precheck 退 2 且状态不变)。本文件与 tests/unit/test_eval_runner_modes.py 是 F08 门禁,
覆盖 harness.verifier/registry/cli 至 --cov-fail-under=80。
"""
from pathlib import Path

import pytest

from harness.cli import main
from harness.core.errors import HarnessError
from harness.core.models import FeatureItem, FeatureList
from harness.core.registry import E2E_MIN_TIMEOUT, Registry, ValidationError
from harness.core.states import ACTIVE, NOT_STARTED, PASSED, REGRESSED
from harness.service.verifier import Verifier

E2E_PASS = "uv run python -c 'raise SystemExit(0)'"
E2E_FAIL = "uv run python -c 'raise SystemExit(1)'"
PRECHECK_FAIL = E2E_FAIL
TIMEOUT = E2E_MIN_TIMEOUT
PASSED_GATE = "uv run pytest harness/tests/test_state_machine.py -q"


def _item(item_id="F01", gate=PASSED_GATE, e2e=None, precheck=None,
          state=ACTIVE, timeout=TIMEOUT) -> FeatureItem:
    return FeatureItem(id=item_id, behavior="测试功能项", gate=gate,
                       e2e=e2e, precheck=precheck, state=state, timeout=timeout)


def _write_features(tmp_path, item: FeatureItem) -> "Path":
    path = tmp_path / "features.yaml"
    registry = Registry(path)
    registry.save(FeatureList(schema=1, milestone="007-harness-governance", items=[item]))
    return path


class TestRegistryE2EFields:
    def test_roundtrip_e2e_fields(self, tmp_path):
        yaml_text = (
            "schema: 1\n"
            "milestone: 007-harness-governance\n"
            "items:\n"
            "  - id: F01\n"
            "    behavior: harness-e2e\n"
            "    gate: uv run pytest -q\n"
            "    e2e: uv run python -m eval.runner --precheck\n"
            "    precheck: uv run python -m eval.runner --precheck\n"
            "    e2e_passed: true\n"
            "    state: passed\n"
            "    note: null\n"
            "    timeout: 600\n"
        )
        path = tmp_path / "features.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        registry = Registry(path)
        item = registry.load().items[0]
        assert item.e2e == "uv run python -m eval.runner --precheck"
        assert item.precheck == "uv run python -m eval.runner --precheck"
        assert item.e2e_passed is True
        # 原子写回后字段保留
        registry.apply_note("F01", "n")
        item = registry.load().items[0]
        assert item.e2e == "uv run python -m eval.runner --precheck"
        assert item.precheck == "uv run python -m eval.runner --precheck"
        assert item.e2e_passed is True

    def test_e2e_requires_timeout(self, tmp_path):
        registry = Registry(tmp_path / "features.yaml")
        with pytest.raises(ValidationError):
            registry.save(FeatureList(schema=1, milestone="007", items=[
                _item(e2e=E2E_PASS, timeout=None)]))

    def test_e2e_timeout_below_floor_rejected(self, tmp_path):
        registry = Registry(tmp_path / "features.yaml")
        with pytest.raises(ValidationError):
            registry.save(FeatureList(schema=1, milestone="007", items=[
                _item(e2e="x", timeout=60)]))

    def test_e2e_with_timeout_saved(self, tmp_path):
        path = tmp_path / "features.yaml"
        registry = Registry(path)
        registry.save(FeatureList(schema=1, milestone="007", items=[
            _item(e2e="x", timeout=TIMEOUT)]))
        assert registry.load().items[0].timeout == TIMEOUT


class TestVerifierE2E:
    def test_e2e_runs_exit_code_only(self):
        result = Verifier().verify(_item(e2e=E2E_PASS), e2e=True)
        assert result.passed is True
        assert result.test_count is None  # e2e 无 junitxml
        assert result.mode == "e2e"
        assert result.precheck_passed is True

    def test_e2e_failure(self):
        result = Verifier().verify(_item(e2e=E2E_FAIL), e2e=True)
        assert result.passed is False
        assert result.exit_code == 1

    def test_e2e_empty_rejected(self):
        with pytest.raises(HarnessError):
            Verifier().verify(_item(e2e=None), e2e=True)

    def test_precheck_failure_blocks_e2e(self):
        item = _item(e2e=E2E_PASS, precheck=PRECHECK_FAIL)
        result = Verifier().verify(item, e2e=True)
        assert result.precheck_passed is False
        assert result.passed is False
        assert result.exit_code == 1
        assert result.command == PRECHECK_FAIL  # 跑的是 precheck, 未到 e2e

    def test_precheck_pass_then_e2e(self):
        result = Verifier().verify(_item(e2e=E2E_PASS, precheck=E2E_PASS), e2e=True)
        assert result.precheck_passed is True
        assert result.passed is True

    def test_gate_path_ignores_precheck(self):
        item = _item(e2e=E2E_PASS, precheck=PRECHECK_FAIL)
        result = Verifier().verify(item, e2e=False)
        assert result.mode == "gate"
        assert result.passed is True  # precheck 只影响 e2e 路径
        assert result.test_count and result.test_count > 0

    def test_e2e_evidence_has_mode(self):
        result = Verifier().verify(_item(e2e=E2E_PASS), e2e=True)
        assert result.to_evidence()["mode"] == "e2e"


class TestCliE2E:
    def test_e2e_active_to_passed(self, tmp_path):
        path = _write_features(tmp_path, _item(state=ACTIVE, e2e=E2E_PASS))
        assert main(["verify", "F01", "--e2e"], features_path=path) == 0
        item = Registry(path).load().items[0]
        assert item.state == PASSED
        assert item.e2e_passed is True
        assert item.last_verify["mode"] == "e2e"

    def test_e2e_pass_to_regressed(self, tmp_path):
        path = _write_features(tmp_path, _item(state=PASSED, e2e=E2E_FAIL))
        assert main(["verify", "F01", "--e2e"], features_path=path) == 1
        item = Registry(path).load().items[0]
        assert item.state == REGRESSED
        assert item.e2e_passed is False

    def test_precheck_unmet_exit_two_state_unchanged(self, tmp_path):
        path = _write_features(tmp_path, _item(state=ACTIVE, e2e=E2E_PASS, precheck=PRECHECK_FAIL))
        assert main(["verify", "F01", "--e2e"], features_path=path) == 2
        item = Registry(path).load().items[0]
        assert item.state == ACTIVE  # 不转移状态
        assert item.last_verify is None  # 不落证据
        assert item.e2e_passed is False

    def test_gate_verify_active_to_passed(self, tmp_path):
        path = _write_features(tmp_path, _item(state=ACTIVE, e2e=None))
        assert main(["verify", "F01"], features_path=path) == 0
        item = Registry(path).load().items[0]
        assert item.state == PASSED
        assert item.e2e_passed is False  # gate 通过不清 e2e, 也不置 True


class TestCliStatusAndBranches:
    def test_status_shows_e2e_unverified(self, tmp_path, capsys):
        path = _write_features(tmp_path, _item(state=PASSED, e2e=E2E_PASS))
        assert main(["status"], features_path=path) == 0
        out = capsys.readouterr().out
        assert "端到端未验: F01" in out

    def test_start_and_abandon(self, tmp_path):
        path = _write_features(tmp_path, _item(state=NOT_STARTED))
        assert main(["start", "F01"], features_path=path) == 0
        assert main(["abandon", "F01"], features_path=path) == 0

    def test_block_unblock(self, tmp_path):
        path = _write_features(tmp_path, _item(state=ACTIVE))
        assert main(["block", "F01", "-m", "卡点"], features_path=path) == 0
        assert main(["unblock", "F01", "-m", "恢复"], features_path=path) == 0

    def test_next_candidate(self, tmp_path, capsys):
        path = _write_features(tmp_path, _item(state=NOT_STARTED))
        assert main(["next"], features_path=path) == 0
        assert "F01" in capsys.readouterr().out

    def test_report_writes_progress(self, tmp_path):
        path = _write_features(tmp_path, _item(state=ACTIVE))
        progress = tmp_path / "PROGRESS.md"
        assert main(["report"], features_path=path, progress_path=progress) == 0
        assert "## 健康度" in progress.read_text(encoding="utf-8")
