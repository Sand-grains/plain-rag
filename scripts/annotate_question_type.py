"""收口时给 private_builtin 快速补 question_type 的一次性启发式工具

规则(启发式, 非 LLM 判定, 可能会误判漏判):
  - difficulty == multi_chunk -> multi_hop
  - query 含比较标记(区别/差异/对比/比较/有何不同/异同/优劣/相比/vs/哪个更/如何选择) -> comparison
  - query 含条件标记(如果/什么情况下/何时/若/假如/一旦/需要满足什么条件/会怎样) -> conditional
  - 否则 -> factual

用法::
    uv run python scripts/annotate_question_type.py --in benchmark/private_builtin.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_COMPARISON_MARKERS = (
    "区别", "差异", "对比", "比较", "有何不同", "异同", "优劣", "优缺点",
    "相比", "vs", "VS", "哪个更", "如何选择", "分别", "关系",
)
_CONDITIONAL_MARKERS = (
    "如果", "什么情况下", "何时", "若", "假如", "一旦",
    "需要满足什么条件", "会怎样", "是否", "当",
)


def infer_question_type(query: str, difficulty: str) -> str:
    """按 query 文本 + difficulty 推断 question_type(启发式)。

    Args:
        query: 检索问题。
        difficulty: single_chunk / multi_chunk。

    Returns:
        str: factual / comparison / conditional / multi_hop。
    """
    if difficulty == "multi_chunk":
        return "multi_hop"
    if any(marker in query for marker in _COMPARISON_MARKERS):
        return "comparison"
    if any(marker in query for marker in _CONDITIONAL_MARKERS):
        return "conditional"
    return "factual"


def annotate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给每条挂 question_type(就地修改并返回)。"""
    for item in items:
        item["question_type"] = infer_question_type(
            item.get("query", ""), item.get("difficulty", "single_chunk"))
    return items


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 读 private_builtin.json, 挂 question_type, 写回。"""
    parser = argparse.ArgumentParser(description="给 private_builtin 挂 question_type(启发式)")
    parser.add_argument("--in", dest="in_path", default="benchmark/private_builtin.json",
                        help="输入/输出 JSON 路径(就地写回)")
    args = parser.parse_args(argv)

    path = Path(args.in_path)
    if not path.exists():
        print(f"文件不存在: {path}")
        return 1
    items = json.loads(path.read_text(encoding="utf-8"))
    annotate(items)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    dist = Counter(i["question_type"] for i in items)
    print(f"完成: {len(items)} 条 -> {path}")
    for key, count in dist.items():
        print(f"  {key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
