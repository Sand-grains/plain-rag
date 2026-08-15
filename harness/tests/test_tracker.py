"""unit：追踪器(F05), 健康度公式数值断言(防虚荣)。"""
import pytest

from harness.core.models import FeatureItem, FeatureList
from harness.core.states import ABANDONED, ACTIVE, NOT_STARTED, PASSED, REGRESSED
from harness.service.tracker import judge_health


def _item(item_id: str, state: str, evidence: dict | None = None) -> FeatureItem:
    return FeatureItem(id=item_id, behavior=f"行为-{item_id}", gate="pytest -q",
                       state=state, last_verify=evidence)


def _feature_list(items: list[FeatureItem]) -> FeatureList:
    return FeatureList(schema=1, milestone="007-harness-governance", items=items)


class TestDistribution:
    def test_state_distribution(self):
        feature_list = _feature_list([
            _item("F01", PASSED, {"tests": 8, "coverage": 90}),
            _item("F02", ACTIVE),
            _item("F03", NOT_STARTED),
            _item("F04", ABANDONED),
        ])
        health = judge_health(feature_list)
        assert health.distribution[PASSED] == 1
        assert health.distribution[ACTIVE] == 1
        assert health.distribution[NOT_STARTED] == 1
        assert health.distribution[ABANDONED] == 1


class TestRegressionRate:
    def test_regression_rate_formula(self):
        feature_list = _feature_list([
            _item("F01", PASSED, {"tests": 8, "coverage": 90}),
            _item("F02", REGRESSED),
        ])
        health = judge_health(feature_list)
        # regressed / (passed + regressed) = 1/2
        assert health.regression_rate == 0.5

    def test_zero_when_no_ever_passed(self):
        feature_list = _feature_list([_item("F01", ACTIVE), _item("F02", NOT_STARTED)])
        assert judge_health(feature_list).regression_rate == 0.0

    def test_abandoned_excluded_from_denominator(self):
        feature_list = _feature_list([
            _item("F01", PASSED, {"tests": 8, "coverage": 90}),
            _item("F02", REGRESSED),
            _item("F03", ABANDONED),
        ])
        # 分母只算非 abandoned, abandoned 不污染回归率
        assert judge_health(feature_list).regression_rate == 0.5


class TestPassRate:
    def test_pass_rate_includes_active(self):
        feature_list = _feature_list([
            _item("F01", PASSED, {"tests": 8, "coverage": 90}),
            _item("F02", ACTIVE),
            _item("F03", REGRESSED),
        ])
        # passed / (passed + regressed + active) = 1/3, active 计入分母
        assert judge_health(feature_list).pass_rate == pytest.approx(1 / 3)

    def test_zero_when_no_completed(self):
        feature_list = _feature_list([_item("F01", ACTIVE)])
        assert judge_health(feature_list).pass_rate == 0.0


class TestSuspicious:
    def test_passed_without_evidence_suspicious(self):
        feature_list = _feature_list([_item("F01", PASSED)])
        assert judge_health(feature_list).suspicious_ids == ["F01"]

    def test_low_test_count_suspicious(self):
        feature_list = _feature_list(
            [_item("F01", PASSED, {"tests": 3, "coverage": 90})])
        assert judge_health(feature_list).suspicious_ids == ["F01"]

    def test_low_coverage_suspicious(self):
        feature_list = _feature_list(
            [_item("F01", PASSED, {"tests": 8, "coverage": 40})])
        assert judge_health(feature_list).suspicious_ids == ["F01"]

    def test_solid_evidence_not_suspicious(self):
        feature_list = _feature_list(
            [_item("F01", PASSED, {"tests": 8, "coverage": 90})])
        assert judge_health(feature_list).suspicious_ids == []

    def test_non_passed_not_suspicious(self):
        feature_list = _feature_list(
            [_item("F01", ACTIVE, {"tests": 3, "coverage": 40})])
        assert judge_health(feature_list).suspicious_ids == []

    def test_e2e_evidence_not_suspicious(self):
        # e2e 证据无 junit/coverage 属正常, 不判可疑
        item = _item("F01", PASSED, {"exit_code": 0, "mode": "e2e"})
        item.e2e = "uv run python -m eval.runner --precheck"
        item.e2e_passed = True
        health = judge_health(_feature_list([item]))
        assert health.suspicious_ids == []


class TestE2EUnverified:
    def _item_with_e2e(self, item_id: str, state: str, e2e_passed: bool) -> FeatureItem:
        item = _item(item_id, state, {"tests": 8, "coverage": 90})
        item.e2e = "uv run python -m eval.runner --precheck"
        item.e2e_passed = e2e_passed
        return item

    def test_passed_with_unverified_e2e_listed(self):
        feature_list = _feature_list([self._item_with_e2e("F01", PASSED, e2e_passed=False)])
        assert judge_health(feature_list).e2e_unverified_ids == ["F01"]

    def test_passed_with_verified_e2e_not_listed(self):
        feature_list = _feature_list([self._item_with_e2e("F01", PASSED, e2e_passed=True)])
        assert judge_health(feature_list).e2e_unverified_ids == []

    def test_no_e2e_declared_not_listed(self):
        feature_list = _feature_list([_item("F01", PASSED, {"tests": 8, "coverage": 90})])
        assert judge_health(feature_list).e2e_unverified_ids == []

    def test_non_passed_not_listed(self):
        feature_list = _feature_list([self._item_with_e2e("F01", ACTIVE, e2e_passed=False)])
        assert judge_health(feature_list).e2e_unverified_ids == []
