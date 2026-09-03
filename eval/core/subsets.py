"""评测层多子集装载与隔离装配
把多个 benchmark 子集(私有/公开)各自加载、绑定对应检索路径、各自独立统计, 强制公开集与本地索引隔离; （合在一起算的精度/召回没意义）
缺失的子集直接跳过(skip)不炸回归保证健壮

- 私有集(private_*.json) 正常走 external, 结果反应 external 模式下 RAG 质量
- 公开集(public.json) 走 memory 独立索引(不跟本地语料共用索引), 其评测结果不代表 external 检索栈的质量, 仅反映 memory 模式下 RAG 质量
公开集的数据需先灌成 memory 缓存(.vector_cache_public), 其稠密/稀疏检索都在进程内 memory(numpy 点积 + BM25)
(不碰 Milvus dense / ES sparse)

核心机制:
    - 多子集: 多个 benchmark 文件(私有/公开)独立加载与独立统计, 合计为总规模, 不互相污染指标口径。
    - 公开集不校验本地 store: load_benchmark(valid_chunk_ids=None) 只做默认填充与结构/schema 检查。
    - 公开集独立 store 装配: 显式 memory 构造(直连 _restore_memory), 禁经 vector_restore
      (vector_restore 是 config 分发口, external 时忽略 cache_dir 直接返回本地三库实例, 会静默错库)
      装配后断言公开 store 与本地 store 的 chunk_ids 交集为空(非空即停, 防静默错库)。
    - 缺失即 skip: private_crawler.json 不进主仓库(.gitignore 排除), 缺失时跳过该子集, 不报错不炸回归。


用法示例::

    from eval.core.subsets import load_subsets, build_public_store, assert_public_store_isolated
    subsets = load_subsets(["benchmark/private_builtin.json"], local_store, public_paths=set(),
                           public_cache_dir=".vector_cache_public", skip_if_missing={"benchmark/private_crawler.json"})

公共接口:
    - BenchmarkSubset: 单个 benchmark 子集(路径 + 类型 + 检索器 + 条目 + 跳过标记)
    - build_public_store: 公开集 store 显式 memory 构造(直连 _restore_memory)
    - assert_public_store_isolated: 断言公开 store 与本地 store chunk_ids 交集为空
    - load_subsets: 加载多个 benchmark 子集(私有/公开), 缺失即 skip
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from eval.core.benchmark import BenchmarkItem, BenchmarkLoadResult, load_benchmark
from indexing.index_store import IndexStore
from retrieval.retriever import Retriever


@dataclass
class BenchmarkSubset:
    """单个 benchmark 子集: 路径 + 类型 + 检索器 + 条目 + 跳过标记。

    kind 区分私有与公开
      - private: 校验本地 IndexStore
      - public: 独立 store + valid_chunk_ids=None
    skipped 表示该子集因文件缺失而被跳过(如 private_crawler.json), 后续不参与统计。
    """
    path: str  # benchmark 文件路径
    kind: str  # "private" | "public"
    retriever: Retriever | None  # 该子集使用的检索器(私有=本地 store, 公开=独立 store)
    items: list[BenchmarkItem] = field(default_factory=list)  # 有效条目
    load_result: BenchmarkLoadResult | None = None  # 加载结果(供 runner 校验 invalid_chunk_ids)
    skipped: bool = False  # 是否因文件缺失被跳过
    skip_reason: str = ""  # 跳过原因


def build_public_store(cache_dir: str) -> IndexStore:
    """公开集 store 显式 memory 构造(直连 _restore_memory), 禁经 vector_restore。

    vector_restore 是 config 分发口: STORAGE_BACKEND=external 时忽略 cache_dir 直接返回本地三库实例,
    经它装配会静默对本地 chunk 校验与检索, 全错且无报错。公开集必须显式走 memory 构造。

    Args:
        cache_dir: 公开语料索引缓存目录(011-3 灌库落盘处)。

    Returns:
        IndexStore: 恢复成功的公开集 memory store。

    Raises:
        RuntimeError: 公开语料索引缺失(需先灌公开集)。
    """
    store = IndexStore._restore_memory(cache_dir)
    if store is None:
        raise RuntimeError(f"公开语料索引缺失(需先灌公开集): {cache_dir}")
    return store


def assert_public_store_isolated(public_store: IndexStore, local_store: IndexStore) -> None:
    """断言公开 store 与本地 store 的 chunk_ids 交集为空(非空即停, 防静默错库)。

    Args:
        public_store: 公开集独立 store。
        local_store: 本地 data/ store。

    Raises:
        RuntimeError: 交集非空(说明公开集误用了本地索引)。
    """
    intersection = public_store.chunk_ids & local_store.chunk_ids
    if intersection:
        raise RuntimeError(
            f"公开 store 与本地 store chunk_ids 交集非空({len(intersection)} 个), 防静默错库"
        )


def load_subsets(
    benchmark_paths: list[str],
    local_store: IndexStore,
    public_paths: set[str],
    public_cache_dir: str,
    skip_if_missing: set[str],
) -> list[BenchmarkSubset]:
    """加载多个 benchmark 子集(私有/公开), 缺失即 skip。

    私有子集校验本地 IndexStore(valid_chunk_ids=local_store.chunk_ids);
    公开子集走 valid_chunk_ids=None(只做默认填充与结构检查), 并装配独立 memory store + 交集断言。

    Args:
        benchmark_paths: benchmark 文件路径列表。
        local_store: 本地 data/ store(私有子集校验用)。
        public_paths: 公开子集路径集合(命中则走公开路径)。
        public_cache_dir: 公开语料索引缓存目录(公开子集 store 装配用)。
        skip_if_missing: 缺失即跳过的路径集合(如 private_crawler.json)。

    Returns:
        list[BenchmarkSubset]: 加载后的子集列表(含 skipped 标记)。
    """
    subsets: list[BenchmarkSubset] = []
    for path in benchmark_paths:
        is_public = path in public_paths
        if not Path(path).exists() and path in skip_if_missing:
            subsets.append(BenchmarkSubset(
                path=path,
                kind="public" if is_public else "private",
                retriever=None,
                skipped=True,
                skip_reason="文件缺失, 跳过该子集",
            ))
            continue
        if is_public:
            public_store = build_public_store(public_cache_dir)
            assert_public_store_isolated(public_store, local_store)
            result = load_benchmark(path, valid_chunk_ids=None)
            subsets.append(BenchmarkSubset(
                path=path, kind="public",
                retriever=Retriever(public_store),
                items=result.valid_items,
                load_result=result,
            ))
        else:
            result = load_benchmark(path, valid_chunk_ids=local_store.chunk_ids)
            subsets.append(BenchmarkSubset(
                path=path, kind="private",
                retriever=Retriever(local_store),
                items=result.valid_items,
                load_result=result,
            ))
    return subsets
