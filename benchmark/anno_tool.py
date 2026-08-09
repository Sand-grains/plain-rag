"""Benchmark 标注工具：逐块标注 expected_chunk_ids (expected_parent_ids, expected_child_ids) + relevance + difficulty。

检索/benchmark 单元为**父块**，标注父块 id。展示父块全文 + section_path。

特性：
  - 启动时可指定从第 N 条开始（断点续标）
  - 逐块展示父块全文（structured 文档渲染 section_path，flat 显示 "—"），Enter 标注 / e 跳过 / m 结束本条
  - argparse：--source 默认 benchmark/private_v5.json，--output 默认 benchmark/private_v6.json

用法: uv run python benchmark/anno_tool.py --source benchmark/private_v6.json

说明:
  当前 relevance 只能按父块打，子块轨 `expected_child_ids` 是纯二元"证据子块"勾选
  略有相关的兄弟子块无表达渠道(父块级 relevance 分级已能覆盖大部分需求)

  检索难度(difficulty)判定依据: 完整答案需要的检索单元(父块)数，而不看有证据子块数
  父块数: 1 → single_chunk，2~3（相邻）→ multi_chunk

  理由：检索单元是父块（一个父块命中即答案到齐），生成上下文也是父块（子块检索、父块作辅助上下文），
  子块级切碎是切块产物、算不改变检索难度，算 single 才避免"所有长答案都被标成 multi"的标签稀释
  另外注意 difficulty 属于 纯描述性分组字段 （eval模块中 retrieval_layer 仅用于 by_difficulty 分组统计），不影响任何实际指标计算

"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import TYPE_CHECKING

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # benchmark/ → 项目根
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from config import VECTOR_CACHE_DIR  # 先于 IndexStore 导入: 触发 load_dotenv，保证下方 os.getenv 读到 .env
from indexing.index_store import IndexStore

if TYPE_CHECKING:
    from indexing.chunk import Chunk

EXIT_KEY = "e"
DONE_KEY = "m"

VALID_DIFFICULTY = ["single_chunk", "multi_chunk"]

# 默认 source 与 output 路径
_DEFAULT_SOURCE = os.path.join(_PROJECT_DIR, "benchmark", "private_v6.json") # 当前正在标注版本
_DEFAULT_OUTPUT = os.path.join(_PROJECT_DIR, "benchmark", "private_v6.json") # 设置为当前版本


# ---- 加载 / 保存 benchmark----

def load_benchmark(path: str) -> list[dict]:
    """读取 benchmark 文件（JSON 数组）。

    Args:
        path: benchmark 文件路径。

    Returns:
        list[dict]：benchmark 条目列表。
    """
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def save_benchmark(items: list[dict], path: str) -> None:
    """写回 benchmark 文件（缩进 2, ensure_ascii=False）。

    Args:
        items: benchmark 条目列表。
        path: 输出文件路径。
    """
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(items, file_handle, ensure_ascii=False, indent=2)
    print(f"  [已保存] {path}\n")


# ---- 父块索引 ----

def _chunk_num(chunk: Chunk) -> int:
    """从 chunk_id 解析数字后缀：{doc_id}:p{i} → i，{doc_id}:{i} → i。"""
    chunk_id = chunk.chunk_id
    if ":p" in chunk_id:
        return int(chunk_id.rsplit(":p", 1)[1])
    return int(chunk_id.rsplit(":", 1)[1])


def build_doc_index(store: IndexStore) -> dict[str, list[Chunk]]:
    """doc_id → [Chunk]（父块，按 chunk_id 数字后缀排序）。

    Args:
        store: 索引存储，利用它取其中的父块集合。

    Returns:
        dict[str, list[Chunk]]：doc_id 到按数字后缀升序排列的父块列表的映射。
    """
    index: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in store.chunks:
        index[chunk.doc_id].append(chunk)
    for doc_id in index:
        index[doc_id].sort(key=_chunk_num)
    return index


def render_section_path(chunk: Chunk) -> str:
    """渲染 section_path：structured 有值 → "a / b"，flat → "—"。"""
    path = chunk.metadata.get("section_path") or []
    return " / ".join(path) if path else "—"


# ---- 逐块打印全文 ----

def print_chunks(chunks: list[Chunk], query: str, reference_facts: str, source_doc: str) -> list[dict]:
    """逐块打印父块全文，每块顶部显示 query + reference_facts + section_path 供对照。

    Enter → 标注当前块（输入 relevance 1/2/3），然后下一块
    e     → 跳过当前块，下一块
    m     → 结束本条标注，返回已标注的 chunk

    Args:
        chunks: 父块列表（按文档顺序）。
        query: 当前条目的检索问题。
        reference_facts: 参考答案事实（可为空串）。
        source_doc: 来源文档名。

    Returns:
        list[dict]：已标注的 {chunk_id, relevance}。
    """
    annotations: list[dict] = []

    index = 0
    while index < len(chunks):
        chunk = chunks[index]

        os.system("cls" if os.name == "nt" else "clear")

        print(f"Q: {query}")
        if reference_facts:
            print(f"reference_facts: {reference_facts}")
        print()

        print(f"[chunk {index}/{len(chunks) - 1}]  {chunk.chunk_id}  [section: {render_section_path(chunk)}]")
        print("─" * 70)
        print(chunk.content)
        print("─" * 70)

        if annotations:
            labeled = [f"[{annotation['chunk_id']}]= {annotation['relevance']}" for annotation in annotations]
            print(f"  已标注: {', '.join(labeled)}")

        user_input = input("  Enter=标注  e=跳过  m=结束本条  > ").strip().lower()
        if user_input == EXIT_KEY:
            index += 1
            continue
        if user_input == DONE_KEY:
            break

        # Enter / 其他 → 标注
        relevance_input = input(f"  [{chunk.chunk_id}] relevance? (1/2/3, 空=跳过)\n  > ").strip()
        if relevance_input:
            try:
                relevance_value = int(relevance_input)
                if relevance_value in (1, 2, 3):
                    annotations.append({"chunk_id": chunk.chunk_id, "relevance": relevance_value})
                    print(f"  ✓ {chunk.chunk_id}  relevance={relevance_value}")
                else:
                    print(f"  ⚠ relevance 应为 1/2/3，本次跳过")
            except ValueError:
                print(f"  ⚠ 无法解析: '{relevance_input}'，本次跳过")
        index += 1

    os.system("cls" if os.name == "nt" else "clear")
    return annotations


def annotate_children(store: IndexStore, annotations: list[dict], parent_chunks: list[Chunk]) -> list[str]:
    """在已标注的相关父块内逐个展开子块（id + 独立内容），勾选证据子块。

    flat_simple 父块（id 无 :p）无子块层级，跳过。
    交互：Enter=证据  e=跳过  m=完成该父块。

    Args:
        store: 索引存储门面，用于按 parent_id 取子块。
        annotations: 已标注的父块 {chunk_id, relevance}。
        parent_chunks: 本条标注涉及的父块列表。

    Returns:
        list[str]：勾选的证据子块 chunk_id。
    """
    expected_child_ids: list[str] = []
    for annotation in annotations:
        parent_id = annotation["chunk_id"]
        if ":p" not in parent_id:
            continue
        children = store.get_children(parent_id)
        if not children:
            continue
        print(f"\n  [{parent_id}] 展开 {len(children)} 个子块（Enter=证据  e=跳过  m=完成该父块）")
        for index, child in enumerate(children, 1):
            os.system("cls" if os.name == "nt" else "clear")
            print(f"    child {index}/{len(children)}  {child.chunk_id}")
            print("─" * 70)
            print(child.content)
            print("─" * 70)
            user_input = input("    证据子块? (Enter=证据  e=跳过  m=完成) > ").strip().lower()
            if user_input == "m":
                break
            if user_input != "e":
                expected_child_ids.append(child.chunk_id)
    return expected_child_ids


# ---- 单条标注 ----

def annotate_entry(
    entry: dict,
    chunks: list[Chunk],
    source_doc: str,
    store: IndexStore,
) -> bool:
    """标注单条 entry（父块 + 证据子块）。

    Args:
        entry: 单条 benchmark 条目（就地修改）。
        chunks: 该 entry 来源文档的父块列表。
        source_doc: 来源文档名（用于展示）。
        store: 索引存储门面。

    Returns:
        bool：False 表示用户选择退出程序。
    """
    query_id = entry.get("query_id", "?")
    existing_ids = entry.get("expected_parent_ids") or entry.get("expected_chunk_ids", [])
    reference_facts = entry.get("reference_facts") or entry.get("ground_truth", "")

    # ---- 步骤 1: 已有标注时询问是否重新输入 ----
    if existing_ids:
        print(f"\n{'=' * 60}")
        print(f"[{query_id}]  source_doc: {source_doc}")
        print(f"已有 expected_parent_ids: {existing_ids}")
        print(f"已有 relevance: {entry.get('relevance', {})}")
        print(f"{'=' * 60}")
        user_input = input("  是否重新输入? (y=覆盖 / n=跳过 / q=退出)\n  > ").strip().lower()
        if user_input == "q":
            return False
        if user_input == "n":
            return True

    # ---- 步骤 2: 逐块打印 + 边看边标 ----
    print(f"\n{'─' * 70}")
    print(f"[{query_id}]  逐块打印 {source_doc} 全文")
    print(f"  Enter=标注  e=跳过  m=结束本条")
    input("  按 Enter 开始...")

    annotations = print_chunks(chunks, entry["query"], reference_facts, source_doc)
    expected_child_ids = annotate_children(store, annotations, chunks)

    # ---- 步骤 3: 汇总 ----
    print("=" * 60)
    print(f"[{query_id}]  source_doc: {source_doc}")
    print(f"Q: {entry['query']}")
    if reference_facts:
        print(f"reference_facts: {reference_facts}")
    print("=" * 60)

    new_relevance: dict[str, int] = {}
    new_chunk_ids: list[str] = []
    for annotation in annotations:
        new_chunk_ids.append(annotation["chunk_id"])
        new_relevance[annotation["chunk_id"]] = annotation["relevance"]

    if new_chunk_ids:
        print(f"\n  本次标注父块 ({len(new_chunk_ids)} 个):")
        for chunk_id in new_chunk_ids:
            print(f"    {chunk_id}  relevance={new_relevance[chunk_id]}")
    else:
        print(f"\n  本次标注父块: 0 个")

    if expected_child_ids:
        print(f"\n  证据子块 ({len(expected_child_ids)} 个):")
        for child_id in expected_child_ids:
            print(f"    {child_id}")
    else:
        print(f"\n  证据子块: 0 个")

    # ---- 步骤 4: 询问 difficulty ----
    current_diff = entry.get("difficulty", "single_chunk")
    if current_diff not in VALID_DIFFICULTY:
        current_diff = "single_chunk"
    user_input = input(f"\n  Difficulty? ({'/'.join(VALID_DIFFICULTY)}, 空=保留 '{current_diff}')\n  > ").strip()
    if user_input and user_input in VALID_DIFFICULTY:
        current_diff = user_input
    elif user_input and user_input not in VALID_DIFFICULTY:
        print(f"  ⚠ 无效值: '{user_input}'，保留 '{current_diff}'")

    # ---- 写入 ----
    entry["expected_parent_ids"] = new_chunk_ids
    entry["expected_child_ids"] = expected_child_ids
    entry["relevance"] = new_relevance
    entry["difficulty"] = current_diff
    entry.pop("expected_chunk_ids", None)  # 迁移清理：删除旧 key，避免新旧不一致残留
    if "query_id" not in entry:
        entry["query_id"] = "Q0000"

    print(f"\n  ✓ [{query_id}] 标注完成: {len(new_chunk_ids)} chunks, difficulty={current_diff}\n")
    return True


# ---- main ----

def main() -> None:
    """标注工具主流程：加载索引 → 逐条标注 → 增量保存。"""
    parser = argparse.ArgumentParser(description="Benchmark 父块标注工具")
    parser.add_argument("--source", default=_DEFAULT_SOURCE, help="输入 benchmark 文件（默认 benchmark/private_v5.json）")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="输出 benchmark 文件（默认 benchmark/private_v6.json）")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"错误: 找不到 {args.source}")
        sys.exit(1)

    print("加载索引...")
    store = IndexStore.vector_restore(VECTOR_CACHE_DIR)
    if store is None:
        print("错误: 索引缓存不存在，请先运行 agent_pipeline.py 构建索引")
        sys.exit(1)

    doc_index = build_doc_index(store)
    items = load_benchmark(args.source)

    total = len(items)
    unannotated = sum(1 for entry in items if not entry.get("relevance"))
    annotated = total - unannotated
    print(f"总条目: {total}  已标注: {annotated}  未标注: {unannotated}")

    # 询问起始位置
    user_input = input("\n是否从头开始标注? (y=从头开始 / n=指定起始位置)\n  > ").strip().lower()
    start_idx = 0
    if user_input == "n":
        number_input = input(f"  从第几条开始? (1~{total})\n  > ").strip()
        try:
            position = int(number_input)
            if 1 <= position <= total:
                start_idx = position - 1
            else:
                print(f"  超出范围 (1~{total})，从头开始")
        except ValueError:
            print(f"  无效输入，从头开始")

    for index in range(start_idx, len(items)):
        entry = items[index]
        query_id = entry.get("query_id", f"Q{index+1:04d}")
        source_doc = entry.get("source_doc", "")

        chunks = doc_index.get(source_doc)
        if chunks is None:
            for doc_id, candidate_chunks in doc_index.items():
                if doc_id in source_doc or source_doc in doc_id:
                    chunks = candidate_chunks
                    print(f"  匹配: source_doc='{source_doc}' → doc_id='{doc_id}'")
                    break

        if chunks is None:
            print(f"\n  ⚠ [{query_id}] 未找到 source_doc='{source_doc}' 的 chunk，跳过")
            continue

        print(f"\n{'#' * 60}")
        print(f"# [{index + 1}/{total}]  {query_id}  source_doc: {source_doc}")
        print(f"# chunks: {len(chunks)}")
        print(f"{'#' * 60}")

        should_continue = annotate_entry(entry, chunks, source_doc, store)
        save_benchmark(items, args.output)

        if not should_continue:
            print("退出标注。")
            return


if __name__ == "__main__":
    main()
