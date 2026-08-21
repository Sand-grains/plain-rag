"""unit：决策留痕(F27)。

状态转移命令(start/verify/block/unblock/reactivate/abandon)在转移锁内自动 append 决策到
harness/decisions.md(单源, 追加不重写, user 默认 sca/env HARNESS_USER); verify-all 巡检不写;
verify 完整输出落 logs/verify/verify-<ts>-<ID>.log + last_verify.artifact; note CLI 结构化输入
与自由文本向后兼容。不触真 eval。
"""
import pytest

import config
from harness.cli import main
from harness.core.models import FeatureItem, FeatureList
from harness.core.registry import Registry
from harness.core.states import ACTIVE, BLOCKED, NOT_STARTED, PASSED
from harness.service.verifier import VerificationResult, Verifier

_GATE = "uv run python -c 'raise SystemExit(0)'"


def _write_features(tmp_path, state=NOT_STARTED, gate=_GATE) -> Registry:
    path = tmp_path / "features.yaml"
    registry = Registry(path)
    registry.save(FeatureList(
        schema=1,
        milestone="010-obs-hardening",
        items=[FeatureItem(id="F01", behavior="harness-decision-record", gate=gate, state=state)],
    ))
    return registry


def _decisions(tmp_path) -> str:
    path = tmp_path / "decisions.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _fake_verifier() -> Verifier:
    return Verifier(timeout=10.0, gate_prefixes=(("uv", "run", "python"),))


class TestStateTransitionDecisions:
    def test_start_appends_decision(self, tmp_path):
        registry = _write_features(tmp_path, state=NOT_STARTED)
        registry.apply_state("F01", ACTIVE, note="开始 F27", action="start")
        text = _decisions(tmp_path)
        assert text.startswith("# harness 决策日志")
        assert "start F01" in text
        assert "reason: 开始 F27" in text
        assert "user: sca" in text

    def test_no_action_no_decision(self, tmp_path):
        registry = _write_features(tmp_path, state=NOT_STARTED)
        registry.apply_state("F01", ACTIVE)
        assert _decisions(tmp_path) == ""

    def test_block_reason_from_note(self, tmp_path):
        registry = _write_features(tmp_path, state=ACTIVE)
        registry.apply_state("F01", BLOCKED, note="e2e 前置未过", action="block")
        text = _decisions(tmp_path)
        assert "block F01" in text
        assert "reason: e2e 前置未过" in text

    def test_append_not_rewrite(self, tmp_path):
        registry = _write_features(tmp_path, state=ACTIVE)
        registry.apply_state("F01", BLOCKED, note="首次 block", action="block")
        registry.apply_state("F01", ACTIVE, note="恢复", action="unblock")
        text = _decisions(tmp_path)
        assert text.count("## ") == 2
        assert "block F01" in text and "unblock F01" in text

    def test_user_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HARNESS_USER", "alice")
        registry = _write_features(tmp_path, state=NOT_STARTED)
        registry.apply_state("F01", ACTIVE, action="start")
        assert "user: alice" in _decisions(tmp_path)


class TestVerifyDecision:
    def test_verify_pass_appends_with_artifact(self, tmp_path):
        registry = _write_features(tmp_path, state=ACTIVE)
        registry.record_verification("F01", passed=True,
                                     evidence={"exit_code": 0, "artifact": "verify-123-F01.log"})
        text = _decisions(tmp_path)
        assert "verify F01" in text
        assert "reason: verify 通过" in text
        assert "evidence: verify-123-F01.log" in text

    def test_verify_fail_appends(self, tmp_path):
        registry = _write_features(tmp_path, state=ACTIVE)
        registry.record_verification("F01", passed=False, evidence={"exit_code": 1})
        assert "reason: verify 未通过" in _decisions(tmp_path)

    def test_record_decision_false_skips(self, tmp_path):
        registry = _write_features(tmp_path, state=ACTIVE)
        registry.record_verification("F01", passed=True, evidence={"exit_code": 0}, record_decision=False)
        assert _decisions(tmp_path) == ""

    def test_no_artifact_no_evidence_line(self, tmp_path):
        registry = _write_features(tmp_path, state=ACTIVE)
        registry.record_verification("F01", passed=True, evidence={"exit_code": 0})
        assert "evidence:" not in _decisions(tmp_path)


class TestNoteDecision:
    def test_note_structured_appends_decision_and_note(self, tmp_path):
        registry = _write_features(tmp_path, state=ACTIVE)
        item = registry.apply_note("F01", "暂缓记录", action="note", reason="人工决策: 暂缓", evidence="e.log")
        assert item.note == "暂缓记录"
        text = _decisions(tmp_path)
        assert "note F01" in text
        assert "reason: 人工决策: 暂缓" in text
        assert "evidence: e.log" in text

    def test_note_free_text_backward_compat(self, tmp_path):
        registry = _write_features(tmp_path, state=ACTIVE)
        item = registry.apply_note("F01", "自由文本叙述")
        assert item.note == "自由文本叙述"
        assert _decisions(tmp_path) == ""


class TestVerifyEvidencePersistence:
    def test_gate_writes_artifact(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_LOG_DIR", str(tmp_path / "logs"))
        result = _fake_verifier().verify(FeatureItem(id="F01", behavior="x", gate="uv run python -c 'print(123)'"))
        assert result.artifact is not None
        assert result.artifact.startswith("verify-") and result.artifact.endswith("-F01.log")
        log_path = tmp_path / "logs" / "verify" / result.artifact
        assert log_path.exists()
        assert "123" in log_path.read_text(encoding="utf-8")
        assert result.to_evidence()["artifact"] == result.artifact

    def test_empty_output_no_artifact(self):
        result = VerificationResult(
            item_id="F01", command="x", exit_code=0, timed_out=False, output="",
            test_names=[], test_count=1, coverage=None, duration_seconds=1.0, passed=True,
        )
        assert result.artifact is None


class TestCliDecisions:
    def test_start_via_cli_appends_decision(self, tmp_path):
        path = tmp_path / "features.yaml"
        Registry(path).save(FeatureList(
            schema=1, milestone="010-obs-hardening",
            items=[FeatureItem(id="F01", behavior="x", gate=_GATE, state=NOT_STARTED)],
        ))
        main(["start", "F01", "-m", "开始实现"], features_path=path)
        text = (tmp_path / "decisions.md").read_text(encoding="utf-8")
        assert "start F01" in text
        assert "reason: 开始实现" in text

    def test_note_structured_via_cli(self, tmp_path):
        path = tmp_path / "features.yaml"
        Registry(path).save(FeatureList(
            schema=1, milestone="010-obs-hardening",
            items=[FeatureItem(id="F01", behavior="x", gate=_GATE, state=ACTIVE)],
        ))
        main(["note", "F01", "--action", "block", "--reason", "暂缓", "--evidence", "e.log"], features_path=path)
        text = (tmp_path / "decisions.md").read_text(encoding="utf-8")
        assert "block F01" in text
        assert "reason: 暂缓" in text
        assert "evidence: e.log" in text

    def test_note_free_text_no_decision(self, tmp_path):
        path = tmp_path / "features.yaml"
        Registry(path).save(FeatureList(
            schema=1, milestone="010-obs-hardening",
            items=[FeatureItem(id="F01", behavior="x", gate=_GATE, state=ACTIVE)],
        ))
        main(["note", "F01", "纯叙述"], features_path=path)
        assert Registry(path).load().items[0].note == "纯叙述"
        assert not (tmp_path / "decisions.md").exists()
