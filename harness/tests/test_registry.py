"""unit：功能清单数据模型(F02)。

features.yaml 解析/校验/原子改写/渲染/行为哈希回退/CRLF 保序。只 import 公开接口
Registry/FeatureItem/FeatureList/behavior_hash, 不依赖其它 harness 组件实现内部符号。
"""
import pytest

from harness.core.models import FeatureItem, FeatureList, behavior_hash
from harness.core.registry import ItemNotFoundError, Registry, ValidationError
from harness.core.state_machine import IllegalTransitionError
from harness.core.states import ACTIVE, BLOCKED, NOT_STARTED, PASSED, REGRESSED


def _write_features(tmp_path, yaml_text: str) -> Registry:
    path = tmp_path / "features.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return Registry(path)


def _sample_yaml(state=NOT_STARTED, behavior="harness-状态机",
                 behavior_hash_value: str | None = None) -> str:
    # 默认给出与行为匹配的正确哈希, 避免被行为哈希回退逻辑改掉状态
    if behavior_hash_value is None:
        behavior_hash_value = behavior_hash(behavior)
    return (
        "schema: 1\n"
        "milestone: 007-harness-governance\n"
        "items:\n"
        "  - id: F01\n"
        f"    behavior: {behavior}\n"
        f"    behavior_hash: '{behavior_hash_value}'\n"
        "    gate: uv run pytest harness/tests/test_state_machine.py -q\n"
        "    e2e: null\n"
        f"    state: {state}\n"
        "    note: null\n"
    )


class TestParse:
    def test_load_valid_yaml(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml())
        feature_list = registry.load()
        assert feature_list.schema == 1
        assert feature_list.milestone == "007-harness-governance"
        assert feature_list.items[0].id == "F01"
        assert feature_list.items[0].state == NOT_STARTED
        # behavior_hash 缺失时自动按行为文本补齐
        assert feature_list.items[0].behavior_hash == behavior_hash("harness-状态机")

    def test_load_missing_field_rejected(self, tmp_path):
        yaml_text = _sample_yaml().replace("    gate: uv run pytest", "    e2e: null")
        registry = _write_features(tmp_path, yaml_text)
        with pytest.raises(ValidationError):
            registry.load()

    def test_load_illegal_state_rejected(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml(state="done"))
        with pytest.raises(ValidationError):
            registry.load()

    def test_load_duplicate_id_rejected(self, tmp_path):
        duplicated = _sample_yaml() + "  - id: F01\n    behavior: b\n    gate: pytest\n    state: active\n"
        registry = _write_features(tmp_path, duplicated)
        with pytest.raises(ValidationError):
            registry.load()

    def test_load_missing_top_level(self, tmp_path):
        registry = _write_features(tmp_path, "items: []\n")
        with pytest.raises(ValidationError):
            registry.load()


class TestAtomicRewrite:
    def test_state_change_preserves_other_fields(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml())
        before = registry.load().items[0]

        updated = registry.apply_state("F01", ACTIVE)

        after = registry.load().items[0]
        assert updated.state == ACTIVE
        assert after.state == ACTIVE
        # 其余字段逐字不变
        assert after.behavior == before.behavior
        assert after.gate == before.gate
        assert after.e2e == before.e2e
        assert after.precheck == before.precheck
        assert after.e2e_passed == before.e2e_passed
        assert after.note == before.note

    def test_no_tmp_leftover_after_save(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml())
        registry.apply_state("F01", ACTIVE)
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []

    def test_view_rebuilt_after_write(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml())
        registry.apply_state("F01", ACTIVE)
        view = (tmp_path / "features.md").read_text(encoding="utf-8")
        assert "F01" in view
        assert ACTIVE in view

    def test_item_not_found(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml())
        with pytest.raises(ItemNotFoundError):
            registry.apply_state("F99", ACTIVE)

    def test_stale_lock_cleared(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml())
        lock_path = tmp_path / "features.yaml.lock"
        import os
        import time as _time
        lock_path.write_text("999", encoding="utf-8")
        # 把锁文件 mtime 改到陈旧区间, 触发自动破锁
        old = _time.time() - 60
        os.utime(lock_path, (old, old))
        registry.apply_state("F01", ACTIVE)
        assert not lock_path.exists()


class TestCrlfPreservation:
    def test_crlf_kept_after_write(self, tmp_path):
        yaml_text = _sample_yaml().replace("\n", "\r\n")
        path = tmp_path / "features.yaml"
        path.write_bytes(yaml_text.encode("utf-8"))

        registry = Registry(path)
        registry.apply_state("F01", ACTIVE)

        raw = path.read_bytes()
        assert b"\r\n" in raw
        assert ACTIVE.encode() in raw


class TestBehaviorHash:
    def test_behavior_change_rolls_back_to_not_started(self, tmp_path):
        # 存量哈希与当前行为不匹配(行为被改窄) → 状态回退 not_started
        stale = _sample_yaml(state=PASSED, behavior="harness-状态机(改窄)",
                             behavior_hash_value=behavior_hash("harness-状态机"))
        feature_list = _write_features(tmp_path, stale).load()
        assert feature_list.items[0].state == NOT_STARTED

    def test_behavior_unchanged_keeps_state(self, tmp_path):
        feature_list = _write_features(tmp_path, _sample_yaml(state=PASSED)).load()
        assert feature_list.items[0].state == PASSED


class TestNote:
    def test_note_only_touches_note(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml())
        updated = registry.apply_note("F01", "卡点: 等接口冻结")
        after = registry.load().items[0]
        assert after.note == "卡点: 等接口冻结"
        assert after.state == NOT_STARTED
        assert updated.note == "卡点: 等接口冻结"


class TestRecordVerification:
    def test_active_passed_to_passed_with_evidence(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml(state=ACTIVE))
        item = registry.record_verification("F01", passed=True, evidence={"tests": 8, "coverage": 0.9})
        assert item.state == PASSED
        assert registry.load().items[0].last_verify == {"tests": 8, "coverage": 0.9}

    def test_active_failed_stays_active(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml(state=ACTIVE))
        item = registry.record_verification("F01", passed=False, evidence={"tests": 8})
        assert item.state == ACTIVE

    def test_passed_failed_to_regressed(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml(state=PASSED))
        item = registry.record_verification("F01", passed=False, evidence={"tests": 8})
        assert item.state == REGRESSED

    def test_not_started_verify_rejected(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml())
        with pytest.raises(IllegalTransitionError):
            registry.record_verification("F01", passed=True, evidence=None)


class TestFinishedTime:
    def test_active_passed_records_finished_time(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml(state=ACTIVE))
        item = registry.record_verification("F01", passed=True, evidence={"tests": 8})
        assert item.state == PASSED
        assert item.finished_time is not None
        assert len(item.finished_time) == 16  # YYYY-MM-DD HH:MM 精确到分

    def test_passed_verify_all_does_not_refresh(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml(state=PASSED))
        registry.record_verification("F01", passed=True, evidence={"tests": 8})
        registry.record_verification("F01", passed=True, evidence={"tests": 8})
        item = registry.load().items[0]
        assert item.state == PASSED
        assert item.finished_time is None  # 巡检不落完成时间

    def test_regressed_clears_finished_time(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml(state=ACTIVE))
        registry.record_verification("F01", passed=True, evidence={"tests": 8})
        registry.record_verification("F01", passed=False, evidence={"tests": 8})
        item = registry.load().items[0]
        assert item.state == REGRESSED
        assert item.finished_time is None

    def test_roundtrip_persists(self, tmp_path):
        registry = _write_features(tmp_path, _sample_yaml(state=ACTIVE))
        item = registry.record_verification("F01", passed=True, evidence={"tests": 8})
        ft = item.finished_time
        reloaded = registry.load().items[0]
        assert reloaded.finished_time == ft
        assert "finished_time:" in (tmp_path / "features.yaml").read_text(encoding="utf-8")


class TestRender:
    def test_render_markdown_golden(self, tmp_path):
        feature_list = FeatureList(
            schema=1,
            milestone="007-harness-governance",
            items=[FeatureItem(id="F01", behavior="harness-状态机", gate="uv run pytest -q")],
        )
        registry = _write_features(tmp_path, _sample_yaml())
        rendered = registry.render_markdown(feature_list)
        assert "由 harness 自动生成, 勿手改" in rendered
        assert "| ID | 行为 | 门禁验证 | 端到端验证 | 状态 | 完成时间 | 叙述 |" in rendered
        assert "| F01 | harness-状态机 | `uv run pytest -q` | `-` | not_started | - | - |" in rendered
