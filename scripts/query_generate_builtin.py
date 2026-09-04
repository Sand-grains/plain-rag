"""private_builtin 生成 (复用 query_generate 核心)

语料为 data/ 个人笔记 (排除 data/Crawler), doc_id = 相对 data/ 路径, 输出 private_builtin.json

用法::
    uv run python scripts/query_generate_builtin.py --base benchmark/private_before.json \
        --corpus data --out benchmark/private_builtin.json --target 300
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.query_generate import call_llm, generate_multi, generate_single, split_article


def _derive_doc_id(path: Path, base_dir: Path) -> str:
    """从文件相对 data/ 的路径推导 doc_id(去后缀、斜杠归一化)。"""
    try:
        return str(path.relative_to(base_dir).with_suffix("")).replace("\\", "/")
    except ValueError:
        return path.stem


def _build_builtin_item(query_id: str, query: str, reference_facts: str, source_doc: str,
                        expected_parent_ids: list[str], difficulty: str) -> dict[str, Any]:
    """构造 private_builtin 条目(schema 对齐 private_v6, 无 question_type)。"""
    return {
        "query_id": query_id,
        "query": query,
        "reference_facts": reference_facts,
        "source_doc": source_doc,
        "category": "builtin",
        "difficulty": difficulty,
        "expected_parent_ids": expected_parent_ids,
        "expected_child_ids": [],
        "relevance": {chunk_id: 3 for chunk_id in expected_parent_ids},
        "expected_files": [source_doc],
        "expected_pages": [],
    }


def _write_checkpoint(path: Path, items: list[dict[str, Any]], processed: set[str]) -> None:
    """把当前 items + 已处理篇写入 checkpoint(增量落盘, 中断可续跑)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"items": items, "processed_files": sorted(processed)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 110 现有 + 新增补到 target, 产出 private_builtin.json。"""
    parser = argparse.ArgumentParser(description="011-1 private_builtin 生成")
    parser.add_argument("--base", default="benchmark/private_v6.json", help="种子 benchmark(现有 110 条)")
    parser.add_argument("--corpus", default="data", help="个人笔记语料目录(默认 data, 排除 Crawler)")
    parser.add_argument("--out", default="benchmark/private_builtin.json", help="输出路径")
    parser.add_argument("--target", type=int, default=300, help="目标总条数(默认 300)")
    parser.add_argument("--max-retries", type=int, default=1, help="非 JSON 输出重试次数(默认 1)")
    parser.add_argument("--checkpoint", default="benchmark/private_builtin.checkpoint.json",
                        help="增量 checkpoint 路径")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="每 N 篇写一次 checkpoint(默认 5)")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 续跑(跳过已处理篇)")
    args = parser.parse_args(argv)

    base_path = Path(args.base)
    items: list[dict[str, Any]] = []
    if base_path.exists():
        items = json.loads(base_path.read_text(encoding="utf-8"))
    start_id = len(items) + 1  # 续号 Q0111...
    target_new = max(0, args.target - len(items))

    corpus = Path(args.corpus)
    # 个人笔记 = data/ 下 .md/.txt, 排除 data/Crawler
    files = sorted(
        p for p in corpus.rglob("*")
        if p.is_file() and p.suffix.lower() in (".md", ".txt") and "Crawler" not in p.parts)
    if not files:
        print(f"个人笔记语料为空: {corpus}(排除 Crawler)")
        return 1

    checkpoint_path = Path(args.checkpoint)
    processed: set[str] = set()
    if args.resume and checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        items = data.get("items", [])
        processed = set(data.get("processed_files", []))
        start_id = len(items) + 1
        target_new = max(0, args.target - len(items))
        print(f"[resume] 从 checkpoint 恢复: {len(items)} 条, 已处理 {len(processed)} 篇")

    llm_fn: Callable[[str, str], dict[str, Any]] = functools.partial(
        call_llm, max_retries=args.max_retries)
    new_count = 0
    for index, file in enumerate(files):
        if new_count >= target_new:
            break
        if file.stem in processed:
            continue
        doc_id = _derive_doc_id(file, corpus)
        text = file.read_text(encoding="utf-8")
        parents = split_article(text, doc_id)
        if not parents:
            processed.add(file.stem)
            continue
        before = len(items)
        # 1 条 single_chunk(自动映射草稿)
        single = generate_single(parents, doc_id, 0, llm_fn, per_chunk=1, limit=1)
        for row in single:
            items.append(_build_builtin_item(
                f"Q{start_id:04d}", row["query"], row["reference_facts"], doc_id,
                [parents[0].chunk_id], "single_chunk"))
            start_id += 1
            new_count += 1
        # 2+ 父块时补 1 条 multi_chunk(保持难度混合)
        if new_count < target_new and len(parents) >= 2:
            multi = generate_multi(parents, doc_id, 0, llm_fn, pairs=1, limit=1)
            for row in multi:
                items.append(_build_builtin_item(
                    f"Q{start_id:04d}", row["query"], row["reference_facts"], doc_id,
                    [p.chunk_id for p in parents[:2]], "multi_chunk"))
                start_id += 1
                new_count += 1
        processed.add(file.stem)
        print(f"[progress] 新增 {new_count}/{target_new} (第 {index + 1}/{len(files)} 篇, 本篇 +{len(items) - before})")
        if (index + 1) % args.checkpoint_every == 0:
            _write_checkpoint(checkpoint_path, items, processed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"完成: {len(items)} query -> {out} (新增 {new_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
