"""Needle Test v2：验证追加"干扰文档"后检索质量是否下降（三组索引 + 统计判定）。

v2 相对 v1 的关键改造（设计见 steps/synthetic_data.md §三.8）：
    - 三组索引：clean / +N control / +N noisy（control 与 noisy 的 N 严格相等，从 manifest 读 stratum 分组）
    - 干扰物渗透率：noisy run 中 top-5 含 ≥1 篇 noisy 文档的 query 比例，< 阈值 → inconclusive（exit 2）
    - 判定：差分 recall_control − recall_noisy ≥ 0.05 → FAIL（exit 1），否则 PASS（exit 0）
    - 参考：每 query 配对 Wilcoxon signed-rank p 值（scipy，缺失时跳过）
    - 规模效应分离：recall_clean − recall_control 是纯规模退化，单独打印不参与判定
    - clean 文档 embedding 只算一次，三索引增量复用（clean ⊂ control ⊂ noisy）
    - _load_docs 跳过隐藏目录（. 开头）与 synthetic 子目录，防 stage/语料污染 clean 组

实现说明：
    - 强制在内存模式构建临时索引，不触碰外部三库（PgSQL/ES/Milvus），幂等安全
    - 复用自己的 Router→Splitter 分块与 run_retrieval_eval 指标计算

用法:  uv run python scripts/needle_test.py [--benchmark benchmark/private_v6.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# 强制内存模式，避免污染外部三库（必须在构造 IndexStore 之前生效）
import indexing.index_store as store_mod
store_mod.STORAGE_BACKEND = "memory"

from indexing.loader import load
from indexing.router import Router
from preprocess import diagnose
from retrieval.embedding import embed
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever
from eval.core.benchmark import load_benchmark
from eval.core.retrieval.retrieval_layer import run_retrieval_eval

if TYPE_CHECKING:
    from indexing.chunk import Chunk

THRESHOLD = 0.05             # 差分 recall@5 判定阈值
PENETRATION_THRESHOLD = 0.10  # 干扰物渗透率下限（实现期标定）


def _decide(interference_drop: float, penetration: float,
            thresholds: tuple[float, float] = (THRESHOLD, PENETRATION_THRESHOLD)) -> str:
    """三路判定：先查渗透率下限（未生效 → INCONCLUSIVE），再比差分阈值（PASS/FAIL）。

    语义（F2）：干扰物渗透率 < penetration_threshold 说明干扰根本没进 top-5，
    判定不可信 → INCONCLUSIVE；渗透达标后，干扰退化 interference_drop ≥ threshold → FAIL，
    否则 PASS。

    Args:
        interference_drop: control→noisy 的 recall 差分（被测假设）。
        penetration: 干扰物渗透率（0-1）。
        thresholds: (差分阈值, 渗透率下限)，默认 (0.05, 0.10)。

    Returns:
        str：三路判定 "INCONCLUSIVE" / "PASS" / "FAIL"。
    """
    threshold, penetration_threshold = thresholds
    if penetration < penetration_threshold:
        return "INCONCLUSIVE"
    return "PASS" if interference_drop < threshold else "FAIL"


# ---- 路径与文档加载 ----

def _project_path(value: str) -> Path:
    """相对路径基于项目根解析，避免依赖 CWD（CLAUDE.md 约定）。"""
    path = Path(value)
    return path if path.is_absolute() else _PROJECT_DIR / path


def _is_excluded(path: Path, root: Path) -> bool:
    """判断路径是否应排除：任一路径段为隐藏目录（. 开头）或 synthetic 子目录。"""
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part.startswith(".") or part == "synthetic" for part in relative_parts)


def _load_docs(directory: str) -> list[Chunk]:
    """递归加载目录下 .txt/.md 文件为 Chunk，跳过隐藏目录与 synthetic 子目录。

    Args:
        directory: 文档目录路径（相对项目根解析）。

    Returns:
        list[Chunk]：目录下全部可加载文件的加载结果。
    """
    root = _project_path(directory)
    base_dir = _PROJECT_DIR / "data"
    docs: list[Chunk] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in (".txt", ".md") and not _is_excluded(path, root):
            docs.extend(load(str(path), base_dir=str(base_dir)))
    return docs


def _load_files(paths: list[Path]) -> list[Chunk]:
    """按显式文件列表加载 Chunk（供 manifest 分组的 control/noisy 精确加载）。

    Args:
        paths: 待加载文件的绝对路径列表。

    Returns:
        list[Chunk]：全部文件的加载结果。
    """
    base_dir = _PROJECT_DIR / "data"
    docs: list[Chunk] = []
    for path in paths:
        if path.is_file() and path.suffix in (".txt", ".md"):
            docs.extend(load(str(path), base_dir=str(base_dir)))
    return docs


def _index_docs(store: IndexStore, docs: list[Chunk]) -> set[str]:
    """将文档经 Router 分块、embedding 向量化后写入临时内存索引。

    Args:
        store: 内存模式 IndexStore。
        docs: 待入库的文档 Chunk 列表。

    Returns:
        set[str]：本次入库的全部父块 chunk_id（用于组归属/渗透率判定）。
    """
    router = Router()
    added_chunk_ids: set[str] = set()
    for doc in docs:
        report = diagnose(doc.content)
        splitter = router.route(report)
        base_meta = {"doc_id": doc.doc_id, "doc_meta": doc.origin_metadata}
        result = splitter.split(doc.content, base_meta)
        parents, children = result if isinstance(result, tuple) else (result, result)
        if not parents:
            continue
        vectors = embed([child.content for child in children])
        store.batch_add(parents, children, vectors)
        added_chunk_ids.update(parent.chunk_id for parent in parents)
    return added_chunk_ids


# ---- 统计判定 ----

def _paired_wilcoxon(control_results, noisy_results) -> str:
    """逐 query 配对 Wilcoxon signed-rank 检验（差分 recall_control − recall_noisy）。

    Args:
        control_results: control 组的逐 query 检索结果（与 noisy_results 按索引对齐）。
        noisy_results: noisy 组的逐 query 检索结果。

    Returns:
        str：格式化 p 值；scipy 缺失或全 0 差分时返回 "—"/"1.0000"。
    """
    try:
        import scipy.stats
    except ImportError:
        print("  ⚠ scipy 不可用，跳过 Wilcoxon（p 值记 —）")
        return "—"
    diffs = [control_result.recall_at_k - noisy_result.recall_at_k
             for control_result, noisy_result in zip(control_results, noisy_results)]
    if not any(diffs):
        return "1.0000"  # 全 0 差分 → scipy 会报 ValueError，显式兜底
    try:
        _, p_value = scipy.stats.wilcoxon(diffs)
        return f"{p_value:.4f}"
    except ValueError:
        return "—"


# ---- 主流程 ----

def main() -> int:
    """执行 needle test：clean / +control / +noisy 三组索引 recall@5 对比与统计判定。

    Returns:
        int：PASS 返回 0，FAIL（干扰显著）返回 1，INCONCLUSIVE（干扰未生效）返回 2。
    """
    parser = argparse.ArgumentParser(description="Needle Test：干扰文档对检索鲁棒性的影响（v2 三组索引）")
    parser.add_argument("--benchmark", default="benchmark/private_v6.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--synthetic-dir", default="data/synthetic")
    parser.add_argument("--penetration-threshold", type=float, default=PENETRATION_THRESHOLD)
    args = parser.parse_args()

    store = IndexStore()
    clean_docs = _load_docs(args.data_dir)
    print(f"clean 文档: {len(clean_docs)} 篇")

    print("构建纯净索引...")
    clean_chunk_ids = _index_docs(store, clean_docs)

    bench = load_benchmark(args.benchmark, valid_chunk_ids=store.chunk_ids)
    if bench.invalid_chunk_ids:
        print(f"⚠ benchmark 有 {len(bench.invalid_chunk_ids)} 条含无效 chunk_id，需先重标注后重试")
        return 1

    # needle test 测检索鲁棒性（clean/control/noisy 差分），重排无关且懒签名会随同进程语料变更过期 → 关闭 rerank
    retriever = Retriever(store, rerank_enabled=False)
    print("运行纯净索引检索评估...")
    clean_output = run_retrieval_eval(retriever, bench.valid_items)
    recall_clean = clean_output.aggregate.get("recall_at_k", 0.0)

    manifest_path = _project_path(args.synthetic_dir) / "manifest.json"
    if not manifest_path.exists():
        print(f"错误: {manifest_path} 不存在 → 无法可靠分组，请先运行 synthetic_docs_generate.py 重新生成")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    synthetic_root = _project_path(args.synthetic_dir)
    control_paths = [synthetic_root / doc["filename"] for doc in manifest["docs"] if doc["stratum"] == "control"]
    noisy_paths = [synthetic_root / doc["filename"] for doc in manifest["docs"] if doc["stratum"] == "noisy"]
    if len(control_paths) != len(noisy_paths):
        print(f"⚠ control({len(control_paths)}) 与 noisy({len(noisy_paths)}) 数量不等，差分判定可能失真（建议重造池）")

    print(f"追加 {len(control_paths)} 篇 control 文档...")
    control_chunk_ids = _index_docs(store, _load_files(control_paths))
    control_output = run_retrieval_eval(retriever, bench.valid_items)
    recall_control = control_output.aggregate.get("recall_at_k", 0.0)

    print(f"追加 {len(noisy_paths)} 篇 noisy 文档...")
    noisy_chunk_ids = _index_docs(store, _load_files(noisy_paths))
    noisy_output = run_retrieval_eval(retriever, bench.valid_items)
    recall_noisy = noisy_output.aggregate.get("recall_at_k", 0.0)

    # 干扰物渗透率：noisy run 中 top-5 含 ≥1 篇 noisy 文档的 query 比例
    penetrated = sum(
        1 for result in noisy_output.results
        if set(result.final_chunk_ids) & noisy_chunk_ids
    )
    penetration = penetrated / len(noisy_output.results) if noisy_output.results else 0.0

    # 规模效应分离 + 干扰退化
    scale_drop = recall_clean - recall_control
    interference_drop = recall_control - recall_noisy
    if recall_clean < recall_control:
        print("⚠ 真不变量被违背: recall_clean < recall_control（harness 异常，请检查索引构建）")

    print("=" * 60)
    print(f"recall@5 clean:      {recall_clean:.4f}")
    print(f"recall@5 +control:   {recall_control:.4f}  (recall_scale_control)")
    print(f"recall@5 +noisy:     {recall_noisy:.4f}")
    print(f"规模退化 (clean→control): {scale_drop:+.4f}（纯规模效应，可接受，不参与判定）")
    print(f"干扰退化 (control→noisy): {interference_drop:+.4f}（被测假设）")
    print(f"MRR clean/+control/+noisy: "
          f"{clean_output.aggregate.get('mrr', 0.0):.4f}/{control_output.aggregate.get('mrr', 0.0):.4f}/"
          f"{noisy_output.aggregate.get('mrr', 0.0):.4f}")
    print(f"diagnosis 分布 clean: {clean_output.aggregate.get('diagnosis_distribution', {})}")
    print(f"diagnosis 分布 noisy: {noisy_output.aggregate.get('diagnosis_distribution', {})}")
    print(f"干扰物渗透率: {penetration:.1%}（{penetrated}/{len(noisy_output.results)} query 的 top-5 含 ≥1 篇 noisy 文档）")
    print(f"配对 Wilcoxon p 值: {_paired_wilcoxon(control_output.results, noisy_output.results)}")

    verdict = _decide(interference_drop, penetration, (THRESHOLD, args.penetration_threshold))
    if verdict == "INCONCLUSIVE":
        print(f"判定: INCONCLUSIVE（渗透率 {penetration:.1%} < {args.penetration_threshold:.0%}，干扰未生效）→ 建议加强槽位贴近度")
        return 2
    if verdict == "PASS":
        print("判定: PASS（干扰不显著）（阈值 0.05）")
        return 0
    print("判定: FAIL（干扰显著）（阈值 0.05）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
