"""unit：calibrate 校准记录选择（scripts/calibrate_auto_merge.py _select_best，G-CAL）。"""
from scripts.calibrate_auto_merge import _select_best


class TestSelectBest:
    def test_selects_max_recall(self):
        records = [
            {"boost": 1.0, "recall_at_k": 0.5},
            {"boost": 1.5, "recall_at_k": 0.8},
            {"boost": 2.0, "recall_at_k": 0.7},
        ]
        assert _select_best(records)["boost"] == 1.5

    def test_tie_returns_first(self):
        records = [
            {"boost": 1.0, "recall_at_k": 0.8},
            {"boost": 2.0, "recall_at_k": 0.8},
        ]
        assert _select_best(records)["boost"] == 1.0
