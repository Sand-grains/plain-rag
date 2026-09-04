"""benchmark query 生成核心
爬取语料 -> LLM 生成 query + 证据映射 -> private_crawler.json。

流程: data/Crawler/*.md → split_article(生产 Router 切分) → 父块列表
→ single_chunk/multi_chunk → build_item 组装条目(带 expected_parent_ids / relevance)
→ 重排 query_id(crawler-0000...) → 写 private_crawler.json

问题类型策略:
  - single_chunk: factual / comparison / conditional(单块可答)
  - multi_chunk: multi_hop(跨 2+ 块可答)
难度: single_chunk / multi_chunk; multi_hop 作为问题类型标签 + 非空 gold(本期不评测)。

产物 private_crawler.json 不进主仓库(.gitignore 排除), 缺失即 skip。

用法::
    uv run python scripts/query_generate.py --corpus data/Crawler --out benchmark/private_crawler.json --target 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import QG_LLM_API_KEY, QG_LLM_BASE_URL, QG_LLM_MODEL_ID

# 问题类型
_SINGLE_TYPES = ("factual", "comparison", "conditional")
_MULTI_TYPE = "multi_hop"
_ALLOWED_TYPES = _SINGLE_TYPES + (_MULTI_TYPE,)

_PROMPT_SINGLE = (
    "你是检索 benchmark 生成器。基于给定文档片段, 生成 {num} 个检索 query。\n"
    "问题类型(选择其中几种): factual(关于文本的具体信息) / comparison(比较文本中的概念或事物) / "
    "conditional(基于文本的 if/when 条件问题)。\n"
    "要求: 每个 query 是用户会问的自然问题, 答案完全由该片段支撑, 有单句参考事实 reference_facts。\n"
    "每个query不允许太弱智, 具体形式如何参照benchmark/private_builtin.json"
    "只输出 JSON: {{\"questions\": [{{\"query\": \"...\", \"reference_facts\": \"...\", "
    "\"question_type\": \"factual|comparison|conditional\"}}]}}"
)
_PROMPT_MULTI = (
    "你是检索 benchmark 生成器。基于给定多个文档片段(用 --- 分隔), 生成 {num} 个 multi_hop 检索 query。\n"
    "multi_hop: 需综合 2+ 片段的信息才能回答的跨片段问题。\n"
    "要求: 每个 query 是用户会问的自然问题, 答案必须同时依赖这些片段, 有多句参考事实 reference_facts。\n"
    "只输出 JSON: {{\"questions\": [{{\"query\": \"...\", \"reference_facts\": \"...\", "
    "\"question_type\": \"multi_hop\"}}]}}"
)


def split_article(text: str, doc_id: str) -> list[Any]:
    """按生产 Router 切分文章, 返回父块列表(带 chunk_id)。

    Args:
        text: 文章全文。
        doc_id: 文档 id(用于 chunk_id 前缀)。

    Returns:
        list[Chunk]: 父块列表, chunk_id 形如 {doc_id}:p{i}。
    """
    from preprocess import diagnose
    from indexing.router import Router
    from indexing.chunk import DocMetadata
    report = diagnose(text)
    splitter = Router().route(report)
    metadata = {
        "doc_id": doc_id,
        "doc_meta": DocMetadata(title=doc_id, source=doc_id, doc_type=".md"),
        "chunk_meta": {},
    }
    result = splitter.split(text, metadata)
    if isinstance(result, tuple):
        parents, _ = result
    else:
        parents = result
    return parents


_JSON_ONLY_REMINDER = (
    "注意: 只输出一个 JSON 对象, 不要输出任何解释、代码审查、Markdown 代码块或其它文本。"
)


def _parse_json_object(text: str) -> Any:
    """把 LLM 输出解析为 JSON 对象; 失败返回 None(不抛)。

    Args:
        text: LLM 原始输出。

    Returns:
        Any: 解析出的 JSON 值; 无法解析时返回 None。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # LLM 常产出未转义引号等坏 JSON, 用 json_repair 兜底
        from json_repair import loads as repair_loads
        try:
            return repair_loads(text)
        except Exception:  # noqa: BLE001 - 修复失败返回 None, 由调用方重试/跳过
            return None


def call_llm(prompt: str, content: str, max_retries: int = 1) -> dict[str, Any]:
    """调用 LLM 生成 typed questions(temperature=0)。

    输出非 JSON 对象时, 追加"只输出 JSON"提醒后重试(最多 max_retries 次);
    仍失败才抛 RuntimeError(由上层按片段跳过, 不中断整轮生成)。

    Args:
        prompt: 系统提示词(已格式化问题数)。
        content: 片段内容。
        max_retries: 非 JSON 输出时的重试次数(默认 2)。

    Returns:
        dict: LLM 返回的 JSON 载荷(含 "questions" 列表)。

    Raises:
        RuntimeError: API 调用失败, 或重试后输出仍不可解析。
    """
    from openai import OpenAI
    client = OpenAI(api_key=QG_LLM_API_KEY, base_url=QG_LLM_BASE_URL)
    system_prompt = prompt
    last_text = ""
    for _attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=QG_LLM_MODEL_ID,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001 - 统一转 RuntimeError
            raise RuntimeError(f"LLM 调用失败: {exc}") from exc
        last_text = response.choices[0].message.content or ""
        payload = _parse_json_object(last_text)
        if isinstance(payload, dict):
            return payload
        # 非对象输出: 追加"只输出 JSON"提醒后重试
        system_prompt = f"{prompt}\n{_JSON_ONLY_REMINDER}"
    raise RuntimeError(f"LLM 输出不可解析(非对象): {last_text[:200]}")


def parse_questions(payload: dict[str, Any]) -> list[dict[str, str]]:
    """归一化 LLM 的 questions 列表(兼容 query/question, expected_answer 兼容)。

    Args:
        payload: LLM 返回的 JSON 载荷。

    Returns:
        list[dict]: [{query, reference_facts, question_type}], 跳过空 query,
        非法类型回退 factual。
    """
    rows = payload.get("questions")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or item.get("question") or "").strip()
        if not query:
            continue
        reference_facts = str(item.get("reference_facts")
                              or item.get("expected_answer") or "").strip()
        question_type = str(item.get("question_type") or "factual").strip().lower()
        if question_type not in _ALLOWED_TYPES:
            question_type = "factual"
        out.append({"query": query, "reference_facts": reference_facts,
                    "question_type": question_type})
    return out


def build_item(query_id: str, query: str, reference_facts: str, source_doc: str,
               expected_parent_ids: list[str], difficulty: str,
               question_type: str) -> dict[str, Any]:
    """构造 benchmark 条目(对齐 eval/core/benchmark 字段 + question_type)。"""
    return {
        "query_id": query_id,
        "query": query,
        "reference_facts": reference_facts,
        "source_doc": source_doc,
        "category": "crawler",
        "difficulty": difficulty,
        "question_type": question_type,
        "expected_parent_ids": expected_parent_ids,
        "relevance": {chunk_id: 3 for chunk_id in expected_parent_ids},
        "expected_files": [source_doc],
        "expected_pages": [],
    }


def generate_single(parents: list[Any], doc_id: str, start_index: int,
                    llm: Callable[[str, str], dict[str, Any]], per_chunk: int = 1,
                    max_failures: int = 3, limit: int | None = None) -> list[dict[str, Any]]:
    """对每个父块生成 single_chunk 类型 query(factual/comparison/conditional)。

    Args:
        parents: 父块列表。
        doc_id: 文档 id。
        start_index: query_id 起始序号。
        llm: LLM 调用函数(prompt, content) -> dict。
        per_chunk: 每个父块生成的 query 数(默认 1)。
        max_failures: 连续失败达此数则跳过本篇剩余块(默认 3, 防逐块硬磨)。
        limit: 本篇 single 最大 query 数(默认 None=不限制; 均摊模式用于控制每篇预算)。

    Returns:
        list[dict]: benchmark 条目, difficulty=single_chunk。
    """
    items: list[dict[str, Any]] = []
    prompt = _PROMPT_SINGLE.format(num=per_chunk)
    consecutive_failures = 0
    for parent in parents:
        if limit is not None and len(items) >= limit:
            break
        try:
            payload = llm(prompt, parent.content)
        except RuntimeError as exc:  # 单块失败跳过, 不中断整轮
            consecutive_failures += 1
            print(f"  [warn] 跳过 {doc_id} 父块 {parent.chunk_id}: {exc}")
            if consecutive_failures >= max_failures:
                print(f"  [warn] {doc_id} 连续 {max_failures} 块失败, 跳过本篇剩余块")
                break
            continue
        consecutive_failures = 0
        for row in parse_questions(payload):
            items.append(build_item(
                f"crawler-{start_index + len(items):04d}", row["query"],
                row["reference_facts"], doc_id, [parent.chunk_id], "single_chunk",
                row["question_type"]))
    return items


def generate_multi(parents: list[Any], doc_id: str, start_index: int,
                   llm: Callable[[str, str], dict[str, Any]], pairs: int = 2,
                   max_failures: int = 3, limit: int | None = None) -> list[dict[str, Any]]:
    """对相邻父块对生成 multi_chunk 类型 query(multi_hop 跨片段)。

    Args:
        parents: 父块列表。
        doc_id: 文档 id。
        start_index: query_id 起始序号。
        llm: LLM 调用函数(prompt, content) -> dict。
        pairs: 每篇生成的 multi_chunk 对数(默认 2)。
        max_failures: 连续失败达此数则跳过本篇剩余对(默认 3)。
        limit: 本篇 multi 最大 query 数(默认 None=不限制; 均摊模式用于控制每篇预算)。

    Returns:
        list[dict]: benchmark 条目, difficulty=multi_chunk, question_type=multi_hop。
    """
    items: list[dict[str, Any]] = []
    prompt = _PROMPT_MULTI.format(num=1)
    consecutive_failures = 0
    for i in range(0, len(parents) - 1, 2):
        if len(items) >= pairs:
            break
        if limit is not None and len(items) >= limit:
            break
        pair = parents[i:i + 2]
        content = "\n\n---\n\n".join(p.content for p in pair)
        try:
            payload = llm(prompt, content)
        except RuntimeError as exc:  # 单对失败跳过, 不中断整轮
            consecutive_failures += 1
            print(f"  [warn] 跳过 {doc_id} 父块对 {[p.chunk_id for p in pair]}: {exc}")
            if consecutive_failures >= max_failures:
                print(f"  [warn] {doc_id} 连续 {max_failures} 对失败, 跳过本篇剩余对")
                break
            continue
        consecutive_failures = 0
        for row in parse_questions(payload):
            items.append(build_item(
                f"crawler-{start_index + len(items):04d}", row["query"],
                row["reference_facts"], doc_id,
                [p.chunk_id for p in pair], "multi_chunk", "multi_hop"))
    return items


def _write_checkpoint(path: Path, items: list[dict[str, Any]], processed: set[str]) -> None:
    """把当前 items + 已处理篇写入 checkpoint(增量落盘, 中断可续跑)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"items": items, "processed_files": sorted(processed)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


def _run_multi_only(corpus_dir: str, out_path: Path, llm_fn: Callable[[str, str], dict[str, Any]],
                    multi_pairs: int) -> int:
    """只补 multi_hop: 读现有 out, 对每篇生成 1 条 multi_hop 追加写回(复用已生成 single)。

    Args:
        corpus_dir: 爬取语料目录。
        out_path: 现有 benchmark JSON 路径(读入 + 写回)。
        llm_fn: LLM 调用函数。
        multi_pairs: 每篇 multi 对数上限(此处 limit=1, 每篇只补 1 条)。

    Returns:
        int: 0=成功。
    """
    files = sorted(Path(corpus_dir).glob("*.md"))
    items: list[dict[str, Any]] = []
    if out_path.exists():
        items = json.loads(out_path.read_text(encoding="utf-8"))
    existing_docs = {item["source_doc"] for item in items}
    added = 0
    for file in files:
        doc_id = f"Crawler/{file.stem}"
        if doc_id not in existing_docs:
            continue
        text = file.read_text(encoding="utf-8")
        parents = split_article(text, doc_id)
        if not parents:
            continue
        new_items = generate_multi(parents, doc_id, len(items), llm_fn, multi_pairs, limit=1)
        for item in new_items:
            # L7: generate_multi 内部已按 start_index(=len(items)) 赋 query_id, 无需重复覆盖
            items.append(item)
            added += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"multi-only 完成: 新增 {added} 条 multi_hop, 总 {len(items)} -> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 逐文章切分 + LLM 生成 typed query, 产出 private_crawler.json。

    支持增量 checkpoint(--checkpoint/--resume)与每篇进度打印, 中断可续跑;
    --multi-only 只补 multi_hop(读现有 out, 每篇生成 1 条追加)。
    """
    import functools
    parser = argparse.ArgumentParser(description="011-2 爬取语料 query 生成")
    parser.add_argument("--corpus", required=True, help="爬取语料目录(data/Crawler)")
    parser.add_argument("--out", required=True, help="输出 benchmark JSON 路径")
    parser.add_argument("--target", type=int, default=500, help="目标 query 数(默认 500)")
    parser.add_argument("--multi-pairs", type=int, default=2, help="每篇 multi_chunk 对数(默认 2)")
    parser.add_argument("--per-chunk", type=int, default=1, help="每个父块 single query 数(默认 1)")
    parser.add_argument("--max-retries", type=int, default=1, help="非 JSON 输出重试次数(默认 1)")
    parser.add_argument("--checkpoint", default="benchmark/private_crawler.checkpoint.json",
                        help="增量 checkpoint 路径(默认 benchmark/private_crawler.checkpoint.json)")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="每 N 篇写一次 checkpoint(默认 5)")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 续跑(跳过已处理篇)")
    parser.add_argument("--multi-only", action="store_true",
                        help="只补 multi_hop(读现有 out, 对每篇生成 1 条 multi_hop 追加写回)")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    files = sorted(corpus.glob("*.md"))
    if not files:
        print(f"语料目录为空: {corpus}")
        return 1

    checkpoint_path = Path(args.checkpoint)
    items: list[dict[str, Any]] = []
    processed: set[str] = set()
    if args.resume and checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        items = data.get("items", [])
        processed = set(data.get("processed_files", []))
        print(f"[resume] 从 checkpoint 恢复: {len(items)} 条, 已处理 {len(processed)} 篇")

    llm_fn = functools.partial(call_llm, max_retries=args.max_retries)
    if args.multi_only:
        return _run_multi_only(args.corpus, Path(args.out), llm_fn, args.multi_pairs)
    # 均摊预算(方案 B): 每篇 target//篇数 条, 前 target%篇数 篇多 1 条, 覆盖全部语料
    base_budget = args.target // len(files)
    extra = args.target % len(files)
    for index, file in enumerate(files):
        if file.stem in processed:
            continue
        doc_id = f"Crawler/{file.stem}"
        text = file.read_text(encoding="utf-8")
        parents = split_article(text, doc_id)
        if not parents:
            processed.add(file.stem)
            continue
        budget = base_budget + (1 if index < extra else 0)
        # 每篇预算内给 multi_hop 留至少 1 条(避免 single 填满预算挤掉 multi)
        multi_share = min(args.multi_pairs, max(1, budget // 4))
        single_limit = budget - multi_share
        before = len(items)
        file_items = generate_single(parents, doc_id, 0, llm_fn, args.per_chunk, limit=single_limit)
        remaining = budget - len(file_items)
        if remaining > 0:
            file_items.extend(generate_multi(parents, doc_id, len(file_items), llm_fn,
                                             args.multi_pairs, limit=remaining))
        # 重新编号 query_id(全局连续)
        for item in file_items:
            item["query_id"] = f"crawler-{len(items):04d}"
            items.append(item)
        processed.add(file.stem)
        print(f"[progress] 已生成 {len(items)}/{args.target} (第 {index + 1}/{len(files)} 篇, 本篇 +{len(items) - before})")
        if (index + 1) % args.checkpoint_every == 0:
            _write_checkpoint(checkpoint_path, items, processed)
        if len(items) >= args.target:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"完成: {len(items)} query -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
