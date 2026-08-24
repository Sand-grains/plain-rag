"""unit：功能清单数据模型(F02)。

features.yaml 解析/校验/原子改写/渲染/行为哈希回退/CRLF 保序。只 import 公开接口
Registry/FeatureItem/FeatureList/behavior_hash, 不依赖其它 harness 组件实现内部符号。
"""
import yaml
import pytest

from harness.core.models import FeatureItem, FeatureList, behavior_hash
from harness.core.registry import ItemNotFoundError, Registry, ValidationError
from harness.core.state_machine import IllegalTransitionError
from harness.core.states import ABANDONED, ACTIVE, BLOCKED, NOT_STARTED, PASSED, REGRESSED


def _write_features(tmp_path, yaml_text: str) -> Registry:
    path = tmp_path / "features.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return Registry(path)


def _write_items(tmp_path, items) -> Registry:
    """用公开 save 接口写 features.yaml(写回时按活跃集排序)。"""
    path = tmp_path / "features.yaml"
    registry = Registry(path)
    registry.save(FeatureList(schema=1, milestone="010-obs-hardening", items=items))
    return registry


def _write_history_file(tmp_path, items, name="archive-test.yaml") -> None:
    """直接写一个 history 归档文件(与 features.yaml 同 schema)。"""
    history_dir = tmp_path / "history"
    history_dir.mkdir(exist_ok=True)
    path = history_dir / name
    payload = {
        "schema": 1,
        "milestone": "010-obs-hardening",
        "items": [{"id": item.id, "behavior": item.behavior, "gate": item.gate,
                   "state": item.state, "finished_time": item.finished_time} for item in items],
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


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


class TestLoadAll:
    def test_load_all_merges_features_and_history(self, tmp_path):
        registry = _write_items(tmp_path, [
            FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED),
            FeatureItem(id="F02", behavior="b2", gate="g", state=ACTIVE),
        ])
        registry.archive()  # F01(passed) 归档, F02(active) 保留
        feature_list = registry.load()
        feature_list.items.append(FeatureItem(id="F03", behavior="b3", gate="g", state=NOT_STARTED))
        registry.save(feature_list)

        full = registry.load_all()
        assert {item.id for item in full.items} == {"F01", "F02", "F03"}
        assert full.milestone == "010-obs-hardening"

    def test_load_all_features_wins_on_same_id(self, tmp_path):
        registry = _write_items(tmp_path, [
            FeatureItem(id="F01", behavior="b1", gate="g", state=ACTIVE),
        ])
        _write_history_file(tmp_path, [FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED)])
        full = registry.load_all()
        assert full.items[0].state == ACTIVE  # 同 id 以 features.yaml 为准

    def test_load_all_no_history_dir_degrades_to_load(self, tmp_path):
        registry = _write_items(tmp_path, [FeatureItem(id="F01", behavior="b1", gate="g", state=ACTIVE)])
        full = registry.load_all()
        assert [item.id for item in full.items] == ["F01"]


class TestArchive:
    def test_archive_moves_passed_and_abandoned_keeps_rest(self, tmp_path):
        registry = _write_items(tmp_path, [
            FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED, finished_time="2026-08-01 10:00"),
            FeatureItem(id="F02", behavior="b2", gate="g", state=ABANDONED),
            FeatureItem(id="F03", behavior="b3", gate="g", state=ACTIVE),
            FeatureItem(id="F04", behavior="b4", gate="g", state=NOT_STARTED),
        ])
        result = registry.archive()
        assert set(result["archived"]) == {"F01", "F02"}
        assert {item.id for item in registry.load().items} == {"F03", "F04"}
        history = registry.history_files()
        assert len(history) == 1
        assert history[0]["count"] == 2

    def test_archive_exclude_keeps_excluded(self, tmp_path):
        registry = _write_items(tmp_path, [
            FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED),
            FeatureItem(id="F02", behavior="b2", gate="g", state=PASSED),
        ])
        result = registry.archive(excludes={"F01"})
        assert result["archived"] == ["F02"]
        assert {item.id for item in registry.load().items} == {"F01"}

    def test_archive_nothing_to_archive(self, tmp_path):
        registry = _write_items(tmp_path, [FeatureItem(id="F01", behavior="b1", gate="g", state=ACTIVE)])
        result = registry.archive()
        assert result["archived"] == []
        assert result["batch"] is None
        assert registry.history_files() == []

    def test_archive_appends_decision(self, tmp_path):
        registry = _write_items(tmp_path, [FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED)])
        registry.archive()
        text = (tmp_path / "decisions.md").read_text(encoding="utf-8")
        assert "archive F01" in text

    def test_archive_history_records_archived_fields(self, tmp_path):
        registry = _write_items(tmp_path, [FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED)])
        registry.archive()
        history_path = list((tmp_path / "history").glob("archive-*.yaml"))[0]
        text = history_path.read_text(encoding="utf-8")
        assert "archived_at:" in text and "archived_milestone:" in text


class TestPromote:
    def test_promote_moves_back_to_features_passed(self, tmp_path):
        registry = _write_items(tmp_path, [
            FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED),
        ])
        registry.archive()
        assert registry.load().items == []
        promoted = registry.promote("F01")
        assert promoted.state == PASSED
        assert {item.id for item in registry.load().items} == {"F01"}
        assert registry.history_files()[0]["count"] == 0

    def test_promote_unknown_id_raises(self, tmp_path):
        registry = _write_items(tmp_path, [FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED)])
        registry.archive()
        with pytest.raises(ItemNotFoundError):
            registry.promote("F99")


class TestSortForWrite:
    def test_non_passed_top_passed_by_finished_time(self, tmp_path):
        registry = _write_items(tmp_path, [
            FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED, finished_time="2026-08-02 10:00"),
            FeatureItem(id="F02", behavior="b2", gate="g", state=ACTIVE),
            FeatureItem(id="F03", behavior="b3", gate="g", state=PASSED, finished_time=None),
            FeatureItem(id="F04", behavior="b4", gate="g", state=PASSED, finished_time="2026-08-01 10:00"),
        ])
        order = [item.id for item in registry.load().items]
        assert order == ["F02", "F04", "F01", "F03"]


class TestDecisionsTrim:
    def test_archive_trims_decisions_keeps_last_n(self, tmp_path, monkeypatch):
        monkeypatch.setattr("harness.core.registry.DECISIONS_KEEP", 2)
        registry = _write_items(tmp_path, [FeatureItem(id="F01", behavior="b1", gate="g", state=PASSED)])
        header = "# harness 决策日志\n\n> 由状态转移命令自动追加, 勿手改(主文件保留最近 50 条, 更早归档到 harness/history/)\n\n"
        entries = "".join(f"## 2026-08-0{i} 10:00 — verify F0{i}\nuser: sca\nreason: r\n"
                          for i in range(1, 4))
        (tmp_path / "decisions.md").write_text(header + entries, encoding="utf-8")

        registry.archive()

        text = (tmp_path / "decisions.md").read_text(encoding="utf-8")
        assert "verify F01" not in text and "verify F02" not in text
        assert "verify F03" in text and "archive" in text
        older_files = list((tmp_path / "history").glob("decisions-*.md"))
        assert len(older_files) == 1
        older = older_files[0].read_text(encoding="utf-8")
        assert "verify F01" in older and "verify F02" in older

    def test_trim_not_triggered_without_archive(self, tmp_path):
        registry = _write_items(tmp_path, [FeatureItem(id="F01", behavior="b1", gate="g", state=NOT_STARTED)])
        header = "# harness 决策日志\n\n> 由状态转移命令自动追加, 勿手改(主文件保留最近 50 条, 更早归档到 harness/history/)\n\n"
        (tmp_path / "decisions.md").write_text(header + "## 2026-08-01 10:00 — start F01\nuser: sca\nreason: r\n",
                                               encoding="utf-8")
        registry.apply_state("F01", ACTIVE)  # 非 archive 不裁剪
        text = (tmp_path / "decisions.md").read_text(encoding="utf-8")
        assert "start F01" in text
