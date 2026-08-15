"""unit：调度器(F04), 只出候选不改状态。"""
from harness.core.models import FeatureItem, FeatureList
from harness.service.scheduler import next_candidate
from harness.core.states import ACTIVE, NOT_STARTED, PASSED


def _feature_list(items: list[FeatureItem]) -> FeatureList:
    return FeatureList(schema=1, milestone="007-harness-governance", items=items)


def _item(item_id: str, state: str = NOT_STARTED) -> FeatureItem:
    return FeatureItem(id=item_id, behavior=f"行为-{item_id}", gate="pytest -q", state=state)


class TestNextCandidate:
    def test_picks_first_not_started_by_id_order(self):
        feature_list = _feature_list([
            _item("F02", PASSED),
            _item("F01"),
            _item("F03"),
        ])
        assert next_candidate(feature_list).id == "F01"

    def test_skips_started_items(self):
        feature_list = _feature_list([
            _item("F01", ACTIVE),
            _item("F02"),
        ])
        assert next_candidate(feature_list).id == "F02"

    def test_none_when_no_candidate(self):
        feature_list = _feature_list([_item("F01", PASSED), _item("F02", ACTIVE)])
        assert next_candidate(feature_list) is None

    def test_readonly(self):
        feature_list = _feature_list([_item("F01"), _item("F02")])
        next_candidate(feature_list)
        assert all(item.state == NOT_STARTED for item in feature_list.items)
