"""unit：011-1 收口 自建检索集 private_builtin。

验证内容：
- DEFAULT_BENCHMARK 已切换指向 benchmark/private_builtin.json（收口动作）。
- private_builtin.json 为合法 JSON、225 条、每条必填字段完整（query_id/reference_facts/
  source_doc/expected_parent_ids/relevance/difficulty/question_type）。
- 经 eval.core.benchmark.load_benchmark 可正常加载。
"""
import json
from pathlib import Path

import pytest

from config import DEFAULT_BENCHMARK
from eval.core.benchmark import load_benchmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_PATH = PROJECT_ROOT / "benchmark" / "private_builtin.json"

REQUIRED_FIELDS = {
    "query_id", "reference_facts", "source_doc", "expected_parent_ids",
    "relevance", "difficulty", "question_type",
}


def test_default_benchmark_switched_to_private_builtin():
    """011-1 收口：默认检索 benchmark 必须指向 private_builtin（而非旧 private_v6）。"""
    assert DEFAULT_BENCHMARK == "benchmark/private_builtin.json"


def test_private_builtin_parses_225_with_required_fields():
    """重标收口结果：合法 JSON、225 条、必填字段齐全。"""
    if not BENCH_PATH.exists():
        pytest.skip("缺少 benchmark/private_builtin.json（未重标）")
    raw = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    assert len(raw) == 225, f"期望 225 条，实际 {len(raw)} 条"
    for item in raw:
        missing = REQUIRED_FIELDS - set(item.keys())
        assert not missing, f"[{item.get('query_id')}] 缺字段: {sorted(missing)}"


def test_private_builtin_loads_via_benchmark_module():
    """经 load_benchmark 全量可加载，无缺 query_id 的条目。"""
    if not BENCH_PATH.exists():
        pytest.skip("缺少 benchmark/private_builtin.json（未重标）")
    result = load_benchmark(str(BENCH_PATH))
    assert len(result.valid_items) == 225
    assert result.missing_query_id == []
