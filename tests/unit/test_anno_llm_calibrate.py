"""test_anno_llm_calibrate.py：F52 scripts/anno_llm_calibrate.py 单元测试(阈值扫描 + 操作点推荐)。

覆盖 anno_llm-semantic-calibration §8.3:
    - scan_thresholds: 扫 cos 网格, 计算正保留率/负拒绝率
    - recommend_operating_point: 保留≥90% 且 拒绝≥80% 的最低阈值
    - CLI 从 samples-file 加载并输出报告
"""
import json
from types import SimpleNamespace

import pytest

from scripts.anno_llm_calibrate import (
    DEFAULT_REJECT_RATIO,
    DEFAULT_THRESHOLDS,
    recommend_operating_point,
    scan_thresholds,
)


def test_scan_thresholds_computes_rates():
    """扫阈值: 每档正保留率(cos>t)与负拒绝率(cos<=t)。"""
    samples = [
        (0.9, True), (0.8, True), (0.7, True),   # 3 正
        (0.4, False), (0.5, False), (0.6, False),  # 3 负
    ]
    scan = scan_thresholds(samples, thresholds=[0.5, 0.7, 0.9])
    assert scan["pos_total"] == 3
    assert scan["neg_total"] == 3
    rows = {row["threshold"]: row for row in scan["rows"]}
    # t=0.5: 正保留 3/3=1.0, 负拒绝(cos<=0.5)=2/3
    assert rows[0.5]["keep_rate"] == pytest.approx(1.0)
    assert rows[0.5]["reject_rate"] == pytest.approx(2 / 3)
    # t=0.7: 正保留 2/3, 负拒绝 3/3=1.0
    assert rows[0.7]["keep_rate"] == pytest.approx(2 / 3)
    assert rows[0.7]["reject_rate"] == pytest.approx(1.0)


def test_scan_thresholds_default_grid():
    """默认网格 = cos 0.5~0.9 步 0.05(9 档)。"""
    assert DEFAULT_THRESHOLDS == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    scan = scan_thresholds([(0.8, True), (0.4, False)])
    assert len(scan["rows"]) == 9


def test_recommend_operating_point_finds_lowest_satisfying():
    """推荐操作点: 保留≥0.9 且 拒绝≥0.8 的最低阈值。"""
    samples = [
        (0.9, True), (0.85, True), (0.8, True), (0.75, True), (0.7, True),  # 5 正
        (0.4, False), (0.5, False), (0.6, False), (0.65, False), (0.68, False),  # 5 负
    ]
    scan = scan_thresholds(samples)
    rec = recommend_operating_point(scan, keep_ratio=0.9, reject_ratio=0.8)
    assert rec["found"] is True
    # 找最低满足阈值: 需 keep>=0.9(5 正全保留, cos>t 需 t<0.7) 且 reject>=0.8(负 cos<=t 需 t>=0.65)
    assert rec["threshold"] == 0.65
    assert rec["keep_rate"] == pytest.approx(1.0)
    assert rec["reject_rate"] == pytest.approx(0.8)


def test_recommend_operating_point_not_found():
    """无满足平衡约束的操作点 → found=False, threshold=None。"""
    samples = [
        (0.9, True), (0.4, True),  # 2 正(一个低余弦)
        (0.5, False), (0.6, False),  # 2 负
    ]
    scan = scan_thresholds(samples, thresholds=[0.5, 0.9])
    rec = recommend_operating_point(scan, keep_ratio=0.9, reject_ratio=0.8)
    # 保留≥0.9 需 t<0.4(保住 0.4 正), 拒绝≥0.8 需 t>=0.6 → 无交集
    assert rec["found"] is False
    assert rec["threshold"] is None


def test_cli_from_samples_file(tmp_path, capsys):
    """CLI 从 samples-file 加载并输出报告(离线/测试路径)。"""
    samples = [[0.9, True], [0.8, True], [0.4, False], [0.5, False]]
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps(samples), encoding="utf-8")

    from scripts.anno_llm_calibrate import main
    code = main(["--samples-file", str(samples_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "正 2 / 负 2" in out
    assert "推荐操作点" in out


def test_cli_requires_source_or_samples(capsys):
    """既无 --samples-file 也无 --source → 报错退 1。"""
    from scripts.anno_llm_calibrate import main
    code = main([])
    assert code == 1
    out = capsys.readouterr().out
    assert "需提供" in out


def test_compute_samples_from_benchmark(monkeypatch):
    """从 benchmark + 索引实时计算样本(方案 B, mock 依赖)。"""
    import scripts.anno_llm_calibrate as calib
    import indexing.index_store as index_store_mod
    import benchmark.anno_tool as anno_tool_mod
    import benchmark.anno_llm.grounding as grounding_mod
    import benchmark.anno_llm.validation as validation_mod

    monkeypatch.setattr(index_store_mod.IndexStore, "vector_restore",
                        classmethod(lambda cls, cache_dir: object()))
    monkeypatch.setattr(anno_tool_mod, "build_doc_index", lambda store: {
        "Crawler/0": [SimpleNamespace(chunk_id="Crawler/0:p0", content="设计模式分为三大类")]})
    monkeypatch.setattr(anno_tool_mod, "load_benchmark", lambda path: [
        {"query_id": "Q1", "query": "设计模式分几类？", "source_doc": "Crawler/0",
         "expected_parent_ids": ["Crawler/0:p0"],
         "evidence_quote": {"Crawler/0:p0": "设计模式分为三大类"}}])
    monkeypatch.setattr(grounding_mod, "verify_semantic",
                        lambda annotation, query, chunk: {"cos_sim": 0.9})
    monkeypatch.setattr(validation_mod, "_equivalence_ids", lambda *args, **kwargs: set())

    samples = calib._compute_samples_from_benchmark("src.json", "gold.json", limit=10)
    # passed_ids ∩ (gold_ids | equiv_ids) 非空 → 正样本
    assert samples == [(0.9, True)]


def test_compute_samples_skips_unmatched(monkeypatch):
    """无法对齐 gold 的条目不计入样本。"""
    import scripts.anno_llm_calibrate as calib
    import indexing.index_store as index_store_mod
    import benchmark.anno_tool as anno_tool_mod
    import benchmark.anno_llm.grounding as grounding_mod
    import benchmark.anno_llm.validation as validation_mod

    monkeypatch.setattr(index_store_mod.IndexStore, "vector_restore",
                        classmethod(lambda cls, cache_dir: object()))
    monkeypatch.setattr(anno_tool_mod, "build_doc_index", lambda store: {
        "Crawler/0": [SimpleNamespace(chunk_id="Crawler/0:p0", content="设计模式分为三大类")]})
    # source 含一条无法对齐 gold 的条目; gold 不含该 query_id
    def _fake_load(path):
        if path == "gold.json":
            return [{"query_id": "Q1", "query": "q", "source_doc": "Crawler/0",
                     "expected_parent_ids": ["Crawler/0:p0"]}]
        return [{"query_id": "unmatched", "query": "q", "source_doc": "Crawler/0",
                 "expected_parent_ids": ["Crawler/0:p0"]}]
    monkeypatch.setattr(anno_tool_mod, "load_benchmark", _fake_load)
    monkeypatch.setattr(grounding_mod, "verify_semantic",
                        lambda annotation, query, chunk: {"cos_sim": 0.9})
    monkeypatch.setattr(validation_mod, "_equivalence_ids", lambda *args, **kwargs: set())

    samples = calib._compute_samples_from_benchmark("src.json", "gold.json", limit=10)
    assert samples == []
