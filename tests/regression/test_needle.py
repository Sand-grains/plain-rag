"""regression：needle_test 纯函数迁移（_decide / _is_excluded / _paired_wilcoxon）+ 完整流程 slow。

needle_test.py 的判定/排除/Wilcoxon 现写为可测纯函数（F2）；完整三组索引流程（clean/control/noisy）
标 slow + skipif(manifest 缺失)，opt-in 跑（G5），真实 BGE-M3 只在 slow 流程加载，纯函数测试零重依赖。
"""
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.needle_test import _decide, _is_excluded, _paired_wilcoxon

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _PROJECT_ROOT / "data" / "synthetic" / "manifest.json"


class TestDecide:
    @pytest.mark.parametrize("drop,penetration,expected", [
        (0.00, 0.00, "INCONCLUSIVE"),
        (0.30, 0.05, "INCONCLUSIVE"),   # 渗透 < 0.10 → 先查渗透，drop 再大也 INCONCLUSIVE
        (0.30, 0.09, "INCONCLUSIVE"),
        (0.00, 0.10, "PASS"),           # 渗透 == 0.10 达标，drop < 0.05 → PASS
        (0.04, 0.20, "PASS"),
        (0.05, 0.20, "FAIL"),           # drop == 0.05 → 达标即 FAIL
        (0.10, 0.20, "FAIL"),
        (0.05, 0.50, "FAIL"),
    ])
    def test_three_way_decision(self, drop, penetration, expected):
        assert _decide(drop, penetration) == expected

    def test_custom_thresholds(self):
        # 渗透 < 0.20 → INCONCLUSIVE（先查渗透）
        assert _decide(0.20, 0.15, thresholds=(0.10, 0.20)) == "INCONCLUSIVE"
        # 渗透达标但 drop < 0.10 → PASS
        assert _decide(0.05, 0.25, thresholds=(0.10, 0.20)) == "PASS"
        # drop == 阈值 → FAIL
        assert _decide(0.10, 0.25, thresholds=(0.10, 0.20)) == "FAIL"


class TestIsExcluded:
    @pytest.mark.parametrize("relative,expected", [
        (".git/config.md", True),        # 隐藏目录段
        (".hidden.md", True),            # 隐藏文件
        ("..", True),                    # .. 段以 . 开头 → 隐藏段规则同样适用
        ("synthetic/generated.md", True),  # synthetic 子目录
        ("sub/synthetic/generated.md", True),  # 嵌套 synthetic
        ("docs/正常.md", False),
        ("docs/sub/deep.md", False),
    ])
    def test_exclusion_rules(self, tmp_path, relative, expected):
        assert _is_excluded(tmp_path / relative, tmp_path) is expected

    def test_path_outside_root_not_excluded(self, tmp_path):
        root = tmp_path / "root"  # 与 outside 无共同前缀 → relative_to ValueError → False
        outside = tmp_path / "outside.md"
        assert _is_excluded(outside, root) is False


def _result(recall: float):
    return SimpleNamespace(recall_at_k=recall)


class TestPairedWilcoxon:
    def test_scipy_missing_returns_dash(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        assert _paired_wilcoxon([_result(0.8)], [_result(0.5)]) == "—"

    def test_all_zero_diffs_returns_one(self):
        control = [_result(1.0), _result(0.5)]
        noisy = [_result(1.0), _result(0.5)]
        assert _paired_wilcoxon(control, noisy) == "1.0000"

    def test_nonzero_diffs_returns_formatted_p_value(self):
        control = [_result(recall) for recall in (0.9, 0.8, 0.7, 0.6)]
        noisy = [_result(recall) for recall in (0.1, 0.2, 0.3, 0.4)]
        result = _paired_wilcoxon(control, noisy)
        assert re.fullmatch(r"\d\.\d{4}", result)  # p 值格式化为 4 位小数


@pytest.mark.slow
@pytest.mark.skipif(not _MANIFEST.exists(),
                    reason="needle 完整流程需 data/synthetic/manifest.json（先跑 synthetic_docs_generate.py）")
class TestNeedleFullFlow:
    def test_three_group_indexing_flow_returns_verdict(self, monkeypatch):
        """G5：完整三组索引流程（clean/control/noisy + 统计判定）——opt-in 跑真实 BGE-M3。

        恢复真实 embed（autouse fake 是测试隔离用，完整流程要真实模型），跑 needle_test.main()，
        断言返回任何合法判定码 {0=PASS, 1=FAIL, 2=INCONCLUSIVE}。覆盖模型/索引/统计全链路。
        """
        import scripts.needle_test as needle_module
        import retrieval.embedding as embedding_module
        import retrieval.retriever as retriever_module
        from tests._fakes import _real

        # 恢复真实 embedding / reranker：needle 完整流程要真实模型，不能用 Fake
        monkeypatch.setattr(embedding_module, "embed", _real("retrieval.embedding", "embed"))
        monkeypatch.setattr(retriever_module, "embed", _real("retrieval.retriever", "embed"))
        monkeypatch.setattr(retriever_module, "Reranker", _real("retrieval.retriever", "Reranker"))
        monkeypatch.setattr(sys, "argv", ["needle_test.py", "--data-dir", "data"])

        assert needle_module.main() in (0, 1, 2)
