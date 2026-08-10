"""auto-merge boost 校准脚本：grid search 最优 MERGE_BOOST（完整管线口径，Layer 1 无 LLM）。

核心特性：
    - 在完整管线（RRF → auto-merge → CrossEncoder 重排）上对 BOOST_CANDIDATES grid search 取最佳 recall@5
    - 与基线验收口径一致（run_retrieval_eval Layer 1，无 LLM 计费）
    - Retriever(store, rerank_skip_cache=True)：跳过 rerank 缓存——不同 boost 经 auto-merge 改 rrf 序后
      候选集不同，缓存命中会拿旧 score_map 给新进候选打 0.0 沉底 → 校准系统性假阴性；跳过缓存诚实承担每 boost 全量推理
    - 进程内动态覆盖 config.MERGE_BOOST（retriever 调用时读 config 模块属性），每 boost 完整跑一遍
    - 校准结论仅对当前语料成立（auto-merge 作用面随语料扩张收缩，见 execute6.5_2.md 实测基线）

用法:  uv run python -m scripts.calibrate_auto_merge [--benchmark benchmark/private_v6.json]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import config  # noqa: E402  模块引用而非值导入——动态覆盖 MERGE_BOOST 靠 config 模块属性
from config import _PROJECT_ROOT, STORAGE_BACKEND  # noqa: E402
from eval.core.benchmark import load_benchmark  # noqa: E402
from eval.core.retrieval.retrieval_layer import run_retrieval_eval  # noqa: E402
from eval.utils import write_json  # noqa: E402
from indexing.index_store import IndexStore  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402

BOOST_CANDIDATES = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]


def _select_best(records: list[dict]) -> dict:
    """从校准记录中选出 recall@5 最高的记录（纯函数，供单测/集成复用）。

    Args:
        records: 校准记录列表（每项含 boost / recall_at_k / mrr 等键）。

    Returns:
        dict：recall_at_k 最大的记录；并列时返回首个。
    """
    return max(records, key=lambda record: record["recall_at_k"])


def main() -> int:
    """执行 auto-merge boost 校准并落盘报告。

    Returns:
        int：正常返回 0；索引缓存缺失 / benchmark 无有效条目时返回 1。
    """
    parser = argparse.ArgumentParser(description="auto-merge boost 校准（完整管线，Layer 1 无 LLM）")
    parser.add_argument("--benchmark", default="benchmark/private_v6.json")
    args = parser.parse_args()

    store = IndexStore.vector_restore()
    if store is None:
        print("错误: 索引缓存不存在")
        return 1
    retriever = Retriever(store, rerank_skip_cache=True)  # 跳过 rerank 缓存，诚实承担每 boost 全量推理

    result = load_benchmark(args.benchmark, valid_chunk_ids=store.chunk_ids)
    items = result.valid_items
    if not items:
        print("错误: benchmark 无有效条目")
        return 1
    print(f"校准样本: {len(items)} 条（{STORAGE_BACKEND} 模式）")

    # 模型预热：首次检索触发 embedding + reranker 加载，避免 Loading weights 日志打乱输出
    print("模型预热中...")
    retriever.retrieve(items[0].query, top_k=5)
    print()

    records = []
    for boost in BOOST_CANDIDATES:
        config.MERGE_BOOST = boost  # 进程内动态覆盖（retriever 调用时读 config 模块属性）
        print(f"boost={boost} ...")
        output = run_retrieval_eval(retriever, items)
        aggregate = output.aggregate
        records.append({
            "boost": boost,
            "recall_at_k": aggregate.get("recall_at_k", 0.0),
            "mrr": aggregate.get("mrr", 0.0),
            "hit_at_k": aggregate.get("hit_at_k", 0.0),
            "ndcg_at_k": aggregate.get("ndcg_at_k", 0.0),
            "diagnosis_distribution": aggregate.get("diagnosis_distribution", {}),
        })
        print(f"  recall@5={aggregate.get('recall_at_k', 0.0):.4f}  mrr={aggregate.get('mrr', 0.0):.4f}")

    config.MERGE_BOOST = 0.0  # 复位默认，防污染同进程后续调用
    best = _select_best(records)

    print("=" * 60)
    for record in records:
        marker = "  <== 最佳" if record is best else ""
        print(f"  boost={record['boost']:<4} recall@5={record['recall_at_k']:.4f} mrr={record['mrr']:.4f}{marker}")
    if best["boost"] == 1.0:
        print("结论: 无增益（最佳 boost=1.0 即无操作）→ 保持 MERGE_BOOST=0.0 关闭")
    else:
        print(f"结论: 推荐 boost={best['boost']}（recall@5 {best['recall_at_k']:.4f}），写入 config 或 .env 生效")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = _PROJECT_ROOT / "eval" / "results" / "calibrate"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"auto_merge_{timestamp}.json"
    write_json(path, {"boost_candidates": BOOST_CANDIDATES, "best_boost": best["boost"], "records": records})
    print(f"校准报告已保存: {path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for noisy in ("httpx", "openai", "jieba", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    sys.exit(main())
