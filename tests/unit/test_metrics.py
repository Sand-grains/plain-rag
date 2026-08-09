"""unit：IR 指标纯函数（eval/core/calculator/metrics.py）——空输入/精确值/k 越界。"""
import pytest

from eval.core.calculator.metrics import (
    recall_at_k, precision_at_k, hit_at_k, mrr, avg_precision, dcg_at_k, ndcg_at_k,
)


class TestRecallAtK:
    def test_standard(self):
        assert recall_at_k(["c1", "c2", "c3"], {"c1", "c4"}, k=3) == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["c1", "c2"], set(), k=3) == 0.0

    def test_k_larger_than_retrieved(self):
        assert recall_at_k(["c1"], {"c1"}, k=10) == 1.0


class TestPrecisionAtK:
    def test_standard(self):
        assert precision_at_k(["c1", "c2", "c3"], {"c1", "c4"}, k=3) == pytest.approx(1 / 3)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["c1"], {"c1"}, k=0) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert precision_at_k([], {"c1"}, k=3) == 0.0


class TestHitAtK:
    def test_hit_returns_one(self):
        assert hit_at_k(["c1", "c2"], {"c2"}, k=3) == 1

    def test_miss_returns_zero(self):
        assert hit_at_k(["c1", "c2"], {"c9"}, k=3) == 0

    def test_hit_beyond_k_ignored(self):
        assert hit_at_k(["c1", "c2", "c9"], {"c9"}, k=2) == 0

    def test_empty_returns_zero(self):
        assert hit_at_k([], {"c1"}, k=3) == 0


class TestMrr:
    def test_first_relevant_at_rank_two(self):
        assert mrr(["c1", "c2", "c3"], {"c2"}) == pytest.approx(0.5)

    def test_no_hit_returns_zero(self):
        assert mrr(["c1", "c2"], {"c9"}) == 0.0

    def test_empty_returns_zero(self):
        assert mrr([], {"c1"}) == 0.0


class TestAvgPrecision:
    def test_standard(self):
        # 相关在 rank1、rank3：P@1=1/1，P@3=2/3，均值 (1 + 2/3)/2 ≈ 0.8333
        assert avg_precision(["c1", "c2", "c3", "c4"], {"c1", "c3"}, k=4) == pytest.approx(0.83333, abs=1e-4)

    def test_no_hit_returns_zero(self):
        assert avg_precision(["c1", "c2"], {"c9"}, k=4) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert avg_precision(["c1"], set(), k=4) == 0.0


class TestDcgAtK:
    def test_standard(self):
        # (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
        expected = 7.0 + 3.0 / 1.58496 + 1.0 / 2.0
        assert dcg_at_k(["a", "b", "c"], {"a": 3, "b": 2, "c": 1}, k=3) == pytest.approx(expected, abs=1e-3)

    def test_k_beyond_list(self):
        assert dcg_at_k(["a"], {"a": 3}, k=5) == pytest.approx(7.0)


class TestNdcgAtK:
    def test_perfect_ordering_is_one(self):
        # 检索序即理想序 → NDCG=1.0
        assert ndcg_at_k(["a", "b"], {"a": 3, "b": 1}, k=2) == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert ndcg_at_k([], {}, k=3) == 0.0

    def test_nonideal_ordering_below_one(self):
        # 检索序 [b(1), a(3)] 非理想 → NDCG < 1
        value = ndcg_at_k(["b", "a"], {"a": 3, "b": 1}, k=2)
        assert 0.0 < value < 1.0
