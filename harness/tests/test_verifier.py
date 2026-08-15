"""unit：验证器(F03)。

退出码 + junitxml 双重判定, 超时/白名单/coverage 门禁/证据记录。fake 命令用
uv run python 覆盖退出码 0/1/2/5; pytest 空收集(退出码 5)视为失败; 真实
gate 验证覆盖 cobertura xml 证据与 --cov-fail-under 门禁。
"""
import pytest

from harness.core.errors import HarnessError
from harness.core.models import FeatureItem
from harness.service.verifier import Verifier


def _item(gate: str, e2e: str | None = None, timeout: float | None = None) -> FeatureItem:
    return FeatureItem(id="FX", behavior="测试项", gate=gate, e2e=e2e, timeout=timeout)


def _fake_verifier(timeout: float = 10.0) -> Verifier:
    # 放宽白名单以允许 python 作为 fake 命令测试退出码判定
    return Verifier(timeout=timeout, gate_prefixes=(("uv", "run", "python"),))


class TestExitCodeDetermination:
    def test_exit_zero_passes(self):
        result = _fake_verifier().verify(_item("uv run python -c 'raise SystemExit(0)'"))
        assert result.passed is True
        assert result.exit_code == 0

    def test_exit_one_fails(self):
        result = _fake_verifier().verify(_item("uv run python -c 'raise SystemExit(1)'"))
        assert result.passed is False
        assert result.exit_code == 1

    def test_exit_two_fails(self):
        result = _fake_verifier().verify(_item("uv run python -c 'raise SystemExit(2)'"))
        assert result.passed is False
        assert result.exit_code == 2

    def test_exit_five_fails(self):
        result = _fake_verifier().verify(_item("uv run python -c 'raise SystemExit(5)'"))
        assert result.passed is False
        assert result.exit_code == 5

    def test_timeout_is_failure(self):
        result = _fake_verifier(timeout=0.5).verify(
            _item("uv run python -c 'import time; time.sleep(5)'"))
        assert result.timed_out is True
        assert result.passed is False
        assert result.exit_code is None


class TestEmptyCollection:
    def test_pytest_no_tests_is_failure(self, tmp_path):
        # 无 test 函数 → pytest 退出码 5, junitxml 0 测试 → 未通过
        empty_test = tmp_path / "empty_test.py"
        empty_test.write_text("x = 1\n", encoding="utf-8")
        result = Verifier().verify(_item(f"uv run pytest {empty_test} -q"))
        assert result.passed is False
        assert result.test_count == 0


class TestGateWhitelist:
    def test_non_whitelisted_gate_rejected(self):
        with pytest.raises(HarnessError):
            Verifier().verify(_item("python -c 'raise SystemExit(0)'"))

    def test_whitelisted_pytest_gate_accepted(self):
        result = Verifier().verify(
            _item("uv run pytest harness/tests/test_state_machine.py -q"))
        assert result.passed is True


class TestCoverageEvidence:
    def test_coverage_parsed_from_scoped_gate(self):
        gate = "uv run pytest harness/tests/test_state_machine.py -q --cov=harness.core.state_machine"
        result = Verifier().verify(_item(gate))
        assert result.passed is True
        assert result.coverage is not None and result.coverage > 0
        assert result.test_count and result.test_count > 0
        assert result.test_names  # 测试名列表记录, 证据可回放

    def test_cov_fail_under_blocks_low_coverage(self):
        # 覆盖率门禁: 统一 --cov-fail-under, 低于阈值 → 退出码非 0 → 未通过
        # 该 gate 的测试不 import harness.core.registry → 覆盖率 0%, 触发门禁失败
        gate = ("uv run pytest harness/tests/test_state_machine.py -q "
                "--cov=harness.core.registry --cov-fail-under=1")
        result = Verifier().verify(_item(gate))
        assert result.passed is False
        assert result.test_count == 8  # 测试本身通过, 拦截发生在覆盖率门禁
        assert "coverage of 1%" in result.output  # 证据可回放: 失败原因留在输出里


class TestE2EVerification:
    def test_e2e_command_runs(self):
        item = _item(gate="uv run pytest harness/tests/test_state_machine.py -q",
                     e2e="uv run python -c 'raise SystemExit(0)'")
        result = Verifier().verify(item, e2e=True)
        assert result.passed is True
        assert result.test_count is None  # e2e 是 eval/benchmark, 无 junitxml
        assert result.mode == "e2e"
        assert result.precheck_passed is True

    def test_e2e_without_command_rejected(self):
        with pytest.raises(HarnessError):
            Verifier().verify(_item("uv run pytest harness/tests/test_state_machine.py -q"), e2e=True)


class TestEvidence:
    def test_to_evidence_compact(self):
        result = _fake_verifier().verify(_item("uv run python -c 'raise SystemExit(0)'"))
        evidence = result.to_evidence()
        assert evidence["exit_code"] == 0
        assert "tests" in evidence and "coverage" in evidence and "duration" in evidence
