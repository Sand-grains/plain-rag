"""unit：验收指标门(F26)。

FeatureItem.metric_gate 字段解析/校验(schema 非法报错)、verifier 阈值 min/max 与 delta drop 判定
(无基线只判阈值)、指标门集成(命令失败/无新 run/阈值/delta 过与不过)、基线持久化(通过更新/
失败不更新)、证据带 metric 摘要。指标值用 synthetic metrics_sink run 记录, 不触真 eval。
"""
import config
import pytest

from harness.core.models import FeatureItem, behavior_hash
from harness.core.registry import Registry, ValidationError
from harness.service.verifier import VerificationResult, Verifier


def _write_features(tmp_path, yaml_text: str) -> Registry:
    path = tmp_path / "features.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return Registry(path)


def _sample_yaml(state, metric_gate_block: str = "", baseline: str | None = None) -> str:
    baseline_line = f"    baseline_run_id: {baseline}\n" if baseline else ""
    return (
        "schema: 1\n"
        "milestone: 010-obs-hardening\n"
        "items:\n"
        "  - id: F01\n"
        "    behavior: harness-metric-gate\n"
        f"    behavior_hash: '{behavior_hash('harness-metric-gate')}'\n"
        "    gate: uv run pytest harness/tests/test_metric_gate.py -q\n"
        "    e2e: null\n"
        f"    state: {state}\n"
        f"{metric_gate_block}"
        f"{baseline_line}"
        "    note: null\n"
        "    timeout: null\n"
    )


_VALID_METRIC_GATE_BLOCK = (
    "    metric_gate:\n"
    "      command: uv run python -m eval.runner --mode retrieval --no-report\n"
    "      thresholds:\n"
    "        hit_at_k: {min: 0.7}\n"
    "      delta:\n"
    "        hit_at_k: {drop: 0.03}\n"
)


def _item(gate="uv run python -c 'raise SystemExit(0)'", metric_gate=None,
          baseline: str | None = None) -> FeatureItem:
    return FeatureItem(id="FX", behavior="测试项", gate=gate, metric_gate=metric_gate, baseline_run_id=baseline)


def _fake_verifier(timeout: float = 10.0) -> Verifier:
    return Verifier(timeout=timeout, gate_prefixes=(("uv", "run", "python"),))


def _run_record(run_id: str, aggregate: dict) -> dict:
    return {"run_id": run_id, "summary": {"aggregate": aggregate}, "per_query": {}}


def _call_sequence(calls: dict, results: list) -> tuple:
    """按调用次数依次返回结果(gate 第 1 次, metric_gate command 第 2 次)。"""
    index = calls["n"]
    calls["n"] += 1
    return results[min(index, len(results) - 1)]


class TestMetricGateSchema:
    def test_valid_metric_gate_parsed(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml("not_started", _VALID_METRIC_GATE_BLOCK))
        item = registry.load().items[0]
        assert item.metric_gate["command"] == "uv run python -m eval.runner --mode retrieval --no-report"
        assert item.metric_gate["thresholds"] == {"hit_at_k": {"min": 0.7}}
        assert item.metric_gate["delta"] == {"hit_at_k": {"drop": 0.03}}

    def test_metric_gate_none_ok(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml("not_started"))
        assert registry.load().items[0].metric_gate is None

    def test_missing_command_rejected(self, tmp_path):
        block = "    metric_gate:\n      thresholds:\n        hit_at_k: {min: 0.7}\n"
        with pytest.raises(ValidationError, match="缺 command"):
            _write_features(tmp_path, _sample_yaml("not_started", block)).load()

    def test_metric_gate_not_dict_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="metric_gate 必须是映射"):
            _write_features(tmp_path, _sample_yaml("not_started", "    metric_gate: nope\n")).load()

    def test_thresholds_not_dict_rejected(self, tmp_path):
        block = "    metric_gate:\n      command: x\n      thresholds: nope\n"
        with pytest.raises(ValidationError, match="thresholds 必须是映射"):
            _write_features(tmp_path, _sample_yaml("not_started", block)).load()

    def test_threshold_missing_min_max_rejected(self, tmp_path):
        block = "    metric_gate:\n      command: x\n      thresholds:\n        hit_at_k: {}\n"
        with pytest.raises(ValidationError, match="缺 .* 任一键"):
            _write_features(tmp_path, _sample_yaml("not_started", block)).load()

    def test_threshold_invalid_key_rejected(self, tmp_path):
        block = "    metric_gate:\n      command: x\n      thresholds:\n        hit_at_k: {min: 0.7, foo: 1}\n"
        with pytest.raises(ValidationError, match="非法键"):
            _write_features(tmp_path, _sample_yaml("not_started", block)).load()

    def test_baseline_run_id_roundtrip(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml("not_started", baseline="R9"))
        assert registry.load().items[0].baseline_run_id == "R9"


class TestEvaluateMetricGate:
    def test_threshold_min_pass_at_boundary(self):
        verifier = Verifier()
        run = _run_record("R2", {"hit_at_k": 0.7})
        checks, passed = verifier._evaluate_metric_gate(run, None, {"command": "x", "thresholds": {"hit_at_k": {"min": 0.7}}})
        assert passed
        assert checks[0]["passed"] is True

    def test_threshold_min_fail_below(self):
        verifier = Verifier()
        run = _run_record("R2", {"hit_at_k": 0.69})
        _, passed = verifier._evaluate_metric_gate(run, None, {"command": "x", "thresholds": {"hit_at_k": {"min": 0.7}}})
        assert not passed

    def test_threshold_max_fail_above(self):
        verifier = Verifier()
        run = _run_record("R2", {"cost.total_tokens": 200})
        _, passed = verifier._evaluate_metric_gate(run, None, {"command": "x", "thresholds": {"cost.total_tokens": {"max": 150}}})
        assert not passed

    def test_threshold_missing_metric_fails(self):
        verifier = Verifier()
        run = _run_record("R2", {})
        checks, passed = verifier._evaluate_metric_gate(run, None, {"command": "x", "thresholds": {"hit_at_k": {"min": 0.7}}})
        assert not passed
        assert checks[0]["reason"] is not None

    def test_delta_pass_within_drop(self):
        verifier = Verifier()
        run = _run_record("R2", {"hit_at_k": 0.82})
        checks, passed = verifier._evaluate_metric_gate(run, {"hit_at_k": 0.85},
                                                    {"command": "x", "delta": {"hit_at_k": {"drop": 0.03}}})
        assert passed
        assert checks[0]["delta"] == 0.03  # 0.85 - 0.82, 边界等于 drop 仍过

    def test_delta_fail_exceeds_drop(self):
        verifier = Verifier()
        run = _run_record("R2", {"hit_at_k": 0.70})
        _, passed = verifier._evaluate_metric_gate(run, {"hit_at_k": 0.85},
                                               {"command": "x", "delta": {"hit_at_k": {"drop": 0.03}}})
        assert not passed

    def test_no_baseline_skips_delta(self):
        verifier = Verifier()
        run = _run_record("R2", {"hit_at_k": 0.82})
        checks, passed = verifier._evaluate_metric_gate(run, None, {"command": "x", "delta": {"hit_at_k": {"drop": 0.03}}})
        assert passed
        assert checks == []  # 无基线只判阈值, delta 段跳过

    def test_delta_missing_baseline_value_fails(self):
        verifier = Verifier()
        run = _run_record("R2", {"hit_at_k": 0.82})
        checks, passed = verifier._evaluate_metric_gate(run, {"other": 0.5},
                                                    {"command": "x", "delta": {"hit_at_k": {"drop": 0.03}}})
        assert not passed
        assert checks[0]["reason"] is not None


class TestMetricGateIntegration:
    def _wire(self, monkeypatch, tmp_path, run, run_id, baseline_agg):
        monkeypatch.setattr(config, "OBS_LOG_DIR", str(tmp_path / "logs"))
        verifier = _fake_verifier()
        monkeypatch.setattr(verifier, "_run", lambda cmd, timeout: (0, False, ""))
        monkeypatch.setattr(verifier, "_latest_run_id", lambda: run_id)
        monkeypatch.setattr(verifier, "_latest_run", lambda: run)
        monkeypatch.setattr(verifier, "_baseline_aggregate", lambda bid: baseline_agg)
        return verifier

    def test_no_metric_gate_declared_skips(self, tmp_path):
        result = _fake_verifier().verify(_item())
        assert result.passed
        assert result.metric_gate is None
        assert result.metric_baseline_run_id is None

    def test_metric_gate_pass_sets_baseline_and_report(self, monkeypatch, tmp_path):
        verifier = self._wire(monkeypatch, tmp_path, _run_record("R2", {"hit_at_k": 0.82}), "R1",
                              {"hit_at_k": 0.85})
        metric_gate = {"command": "x", "thresholds": {"hit_at_k": {"min": 0.7}}, "delta": {"hit_at_k": {"drop": 0.03}}}
        result = verifier.verify(_item(metric_gate=metric_gate))
        assert result.passed
        assert result.metric_baseline_run_id == "R2"
        assert result.metric_gate["passed"] is True
        assert len(result.metric_gate["checks"]) == 2
        report_files = list((tmp_path / "logs" / "verify").glob("metric_gate-*.json"))
        assert len(report_files) == 1

    def test_metric_gate_threshold_fail_no_baseline_update(self, monkeypatch, tmp_path):
        verifier = self._wire(monkeypatch, tmp_path, _run_record("R2", {"hit_at_k": 0.5}), "R1", None)
        metric_gate = {"command": "x", "thresholds": {"hit_at_k": {"min": 0.7}}}
        result = verifier.verify(_item(metric_gate=metric_gate))
        assert not result.passed
        assert result.metric_baseline_run_id is None
        assert result.metric_gate["passed"] is False
        assert result.metric_gate["checks"][0]["value"] == 0.5

    def test_metric_gate_delta_fail(self, monkeypatch, tmp_path):
        verifier = self._wire(monkeypatch, tmp_path, _run_record("R2", {"hit_at_k": 0.70}), "R1",
                              {"hit_at_k": 0.85})
        metric_gate = {"command": "x", "delta": {"hit_at_k": {"drop": 0.03}}}
        result = verifier.verify(_item(metric_gate=metric_gate))
        assert not result.passed
        assert result.metric_baseline_run_id is None

    def test_metric_gate_no_baseline_delta_skipped_passes(self, monkeypatch, tmp_path):
        verifier = self._wire(monkeypatch, tmp_path, _run_record("R2", {"hit_at_k": 0.82}), "R1", None)
        metric_gate = {"command": "x", "delta": {"hit_at_k": {"drop": 0.03}}}
        result = verifier.verify(_item(metric_gate=metric_gate))
        assert result.passed  # 无基线 delta 跳过, 无阈值 → 过
        assert result.metric_gate["checks"] == []

    def test_metric_gate_command_fail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "OBS_LOG_DIR", str(tmp_path / "logs"))
        verifier = _fake_verifier()
        calls = {"n": 0}
        # 第 1 次调用是 gate 命令(必须过), 第 2 次是 metric_gate command(让它失败)
        monkeypatch.setattr(verifier, "_run", lambda cmd, timeout: _call_sequence(calls, [(0, False, ""), (1, False, "boom")]))
        result = verifier.verify(_item(metric_gate={"command": "x"}))
        assert not result.passed
        assert result.metric_gate["reason"] is not None
        assert "boom" in "\n".join(result.metric_gate.get("command_output_tail", []))

    def test_metric_gate_timeout_fail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "OBS_LOG_DIR", str(tmp_path / "logs"))
        verifier = _fake_verifier()
        calls = {"n": 0}
        monkeypatch.setattr(verifier, "_run", lambda cmd, timeout: _call_sequence(calls, [(0, False, ""), (None, True, "")]))
        result = verifier.verify(_item(metric_gate={"command": "x"}))
        assert not result.passed
        assert result.metric_gate["command_timed_out"] is True
        assert result.metric_gate["reason"] is not None

    def test_metric_gate_no_new_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "OBS_LOG_DIR", str(tmp_path / "logs"))
        verifier = _fake_verifier()
        monkeypatch.setattr(verifier, "_run", lambda cmd, timeout: (0, False, ""))
        monkeypatch.setattr(verifier, "_latest_run_id", lambda: "R1")
        monkeypatch.setattr(verifier, "_latest_run", lambda: _run_record("R1", {"hit_at_k": 0.9}))
        result = verifier.verify(_item(metric_gate={"command": "x", "thresholds": {"hit_at_k": {"min": 0.7}}}))
        assert not result.passed
        assert result.metric_gate["reason"] == "metric_gate command 未在 metrics_sink 写入新 run"


class TestRealMetricsSink:
    """真实 metrics_sink 路径: 覆盖 _latest_run/_latest_run_id/_baseline_aggregate 与完整门禁集成。"""

    def _seed_baseline(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_METRICS_DIR", str(tmp_path / "metrics"))
        from obs import metrics_sink
        metrics_sink.record_run(
            run_id="R8", benchmark="b", config={}, corpus_signature="s", test_mode="retrieval",
            num_queries=1, expected_parent_ids_by_query={},
            summary={"aggregate": {"hit_at_k": 0.85}}, per_query={}, attribution={},
        )

    def _command_writes_run(self, tmp_path, monkeypatch):
        from obs import metrics_sink
        calls = {"n": 0}

        def _run(cmd, timeout):
            calls["n"] += 1
            if calls["n"] >= 2:  # 第 1 次调用是 gate 命令(不写), 第 2 次是 metric_gate command(写新 run R9)
                metrics_sink.record_run(
                    run_id="R9", benchmark="b", config={}, corpus_signature="s", test_mode="retrieval",
                    num_queries=1, expected_parent_ids_by_query={},
                    summary={"aggregate": {"hit_at_k": 0.82}}, per_query={}, attribution={},
                )
            return (0, False, "")

        return _run

    def test_real_sink_latest_and_baseline(self, tmp_path, monkeypatch):
        self._seed_baseline(tmp_path, monkeypatch)
        verifier = _fake_verifier()
        assert verifier._latest_run_id() == "R8"
        assert verifier._baseline_aggregate("R8") == {"hit_at_k": 0.85}
        assert verifier._baseline_aggregate("GONE") is None
        assert verifier._baseline_aggregate(None) is None

    def test_metric_gate_with_real_sink(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_LOG_DIR", str(tmp_path / "logs"))
        self._seed_baseline(tmp_path, monkeypatch)
        verifier = _fake_verifier()
        monkeypatch.setattr(verifier, "_run", self._command_writes_run(tmp_path, monkeypatch))
        metric_gate = {"command": "x", "thresholds": {"hit_at_k": {"min": 0.7}}, "delta": {"hit_at_k": {"drop": 0.03}}}
        result = verifier.verify(_item(metric_gate=metric_gate, baseline="R8"))
        assert result.passed
        assert result.metric_baseline_run_id == "R9"  # 新 run 成为基线
        assert result.metric_gate["baseline_run_id"] == "R8"
        delta_check = [c for c in result.metric_gate["checks"] if c["kind"] == "delta"][0]
        assert delta_check["baseline"] == 0.85
        assert delta_check["delta"] == 0.03

    def test_metric_gate_real_sink_no_new_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OBS_LOG_DIR", str(tmp_path / "logs"))
        self._seed_baseline(tmp_path, monkeypatch)
        verifier = _fake_verifier()
        monkeypatch.setattr(verifier, "_run", lambda cmd, timeout: (0, False, ""))  # 不写新 run
        result = verifier.verify(_item(metric_gate={"command": "x"}))
        assert not result.passed
        assert result.metric_gate["reason"] == "metric_gate command 未在 metrics_sink 写入新 run"


class TestBaselinePersistence:
    def test_baseline_updated_on_pass(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml("active"))
        item = registry.record_verification("F01", passed=True, evidence={"tests": 8}, metric_baseline="R9")
        assert item.baseline_run_id == "R9"
        assert registry.load().items[0].baseline_run_id == "R9"

    def test_baseline_kept_on_fail(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml("active", baseline="R8"))
        item = registry.record_verification("F01", passed=False, evidence={"tests": 8}, metric_baseline="R9")
        assert item.baseline_run_id == "R8"  # 失败不更新, 基线保持上次 good run

    def test_baseline_untouched_without_metric_baseline(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml("active", baseline="R8"))
        item = registry.record_verification("F01", passed=True, evidence={"tests": 8}, metric_baseline=None)
        assert item.baseline_run_id == "R8"


class TestEvidence:
    def test_to_evidence_includes_metric_summary(self):
        result = VerificationResult(
            item_id="FX", command="x", exit_code=0, timed_out=False, output="",
            test_names=[], test_count=8, coverage=None, duration_seconds=1.0, passed=True,
            metric_gate={"passed": True, "run_id": "R2", "checks": [{"name": "hit_at_k"}]},
        )
        evidence = result.to_evidence()
        assert evidence["metric"] == {"passed": True, "run_id": "R2", "checks": 1}

    def test_to_evidence_plain_when_no_metric_gate(self):
        result = VerificationResult(
            item_id="FX", command="x", exit_code=0, timed_out=False, output="",
            test_names=[], test_count=8, coverage=None, duration_seconds=1.0, passed=True,
        )
        assert "metric" not in result.to_evidence()
