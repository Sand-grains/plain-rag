"""根级 conftest：sys.path 注入项目根 + 共享 fixture（memory_store / mini_corpus / semantic_fakes）+ 测试报告钩子。

核心特性：
    - sys.path 注入项目根，使各测试可直接 `import config` / `from tests._fakes import ...`
    - memory_store：强制 memory 后端的 IndexStore（STORAGE_BACKEND 由 autouse install_all_fakes 打成 memory）
    - mini_corpus：含 h1/代码块/纯文本 3 篇 fixture 语料
    - semantic_fakes：安装带语义分组的 Fake 层（集成/回归断言"query 命中预期 chunk"用）
    - pytest_sessionfinish：每次 pytest 后渲染 tests/report.md（三层统计 + 模块覆盖表 + 失败明细）
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from tests._fakes import install_all_fakes


# ---- 共享 fixture ----

@pytest.fixture
def memory_store():
    """强制 memory 后端的 IndexStore 实例（autouse install_all_fakes 已把 STORAGE_BACKEND 打成 memory）。"""
    from indexing.index_store import IndexStore
    store = IndexStore()
    yield store


MINI_CORPUS = {
    "doc_rag": (
        "# RAG 基础\n\n"
        "RAG 是检索增强生成（Retrieval-Augmented Generation）。\n\n"
        "```python\nfrom rag import pipeline\n```\n\n"
        "RAG 结合检索与生成两大能力。"
    ),
    "doc_code": (
        "# Python 教程\n\n"
        "## 列表\n\n列表是 Python 的容器。\n\n"
        "```python\na = [1, 2, 3]\n```\n\n"
        "## 字典\n\n字典是键值对容器。"
    ),
    "doc_plain": (
        "纯文本笔记。\n\n"
        "没有标题结构，只有段落。\n\n"
        "第二段。"
    ),
}


@pytest.fixture
def mini_corpus() -> dict[str, str]:
    """3 篇 fixture 语料：h1 结构 / 含代码块 / 纯文本无标题。"""
    return dict(MINI_CORPUS)


@pytest.fixture
def semantic_fakes(monkeypatch):
    """安装带语义分组的 Fake 层。用法：`semantic_fakes({"rag": ["RAG 检索增强生成"]})`。"""
    def _install(groups: dict[str, list[str]]) -> dict[str, list[str]]:
        install_all_fakes(monkeypatch, groups=groups)
        return groups
    return _install


# ---- 测试报告钩子 ----

_LAYERS = ("unit", "integration", "regression")


def _group_of(nodeid: str) -> str:
    """按 nodeid 前缀归层：tests/unit/... → unit，其余 → other。"""
    for layer in _LAYERS:
        if nodeid.startswith(f"tests/{layer}/"):
            return layer
    return "other"


_REPORTS: dict[str, object] = {}  # nodeid → call 阶段 report（sessionfinish 渲染用）


def pytest_runtest_logreport(report) -> None:
    """收集每个测试的 call 阶段 report（pytest 9 移除 session.stats，改用本钩子）。"""
    if report.when == "call":
        _REPORTS[report.nodeid] = report


def _collect_layer_stats(session) -> dict[str, dict]:
    """收集三层（unit/integration/regression）统计：总数/通过/失败/跳过/耗时。"""
    stats = {layer: {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0,
                     "duration": 0.0, "other": 0}
             for layer in _LAYERS}
    stats["other"] = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "error": 0,
                      "duration": 0.0, "other": 0}
    for item in session.items:
        stats[_group_of(item.nodeid)]["total"] += 1
    for report in _REPORTS.values():
        group = stats[_group_of(report.nodeid)]
        outcome = report.outcome  # passed / failed / skipped
        if outcome in group:
            group[outcome] += 1
        group["duration"] += float(getattr(report, "duration", 0.0) or 0.0)
    return stats


def _collect_failures(session) -> list[tuple[str, str]]:
    """收集失败明细：[(nodeid, longrepr)]。"""
    failures = []
    for report in _REPORTS.values():
        if report.failed:
            longrepr = getattr(report, "longrepr", None)
            text = str(longrepr) if longrepr is not None else ""
            failures.append((report.nodeid, text))
    return failures


def _collect_coverage(session) -> list[tuple[str, int, int, int, int]]:
    """从 pytest-cov 插件取覆盖数据：[(模块相对路径, 行覆盖, 行总数, 分支覆盖, 分支总数)]。

    pytest-cov 的覆盖对象挂在 `_cov` 插件的 `cov_controller.cov`（非 session.cov），
    插件经 config.pluginmanager 访问。
    """
    plugin = session.config.pluginmanager.getplugin("_cov") if session.config.pluginmanager.hasplugin("_cov") else None
    if plugin is None or plugin.cov_controller is None:
        return []
    cov = plugin.cov_controller.cov
    rows = []
    try:
        data = cov.get_data()
        for filename in sorted(data.measured_files()):
            try:
                analysis = cov._analyze(filename)
            except Exception:
                continue
            statements = set(analysis.statements)
            missing = set(analysis.missing)
            line_total = len(statements)
            line_covered = line_total - len(missing)
            try:
                branch_total = len(analysis.arc_possibilities)
                branch_covered = branch_total - len(analysis.missing_branch_arcs())
            except Exception:
                branch_total, branch_covered = 0, 0
            try:
                module = os.path.relpath(filename, str(_PROJECT_ROOT)).replace(os.sep, "/")
            except Exception:
                module = filename
            rows.append((module, line_covered, line_total, branch_covered, branch_total))
    except Exception:
        return rows
    return rows


def pytest_sessionfinish(session, exitstatus) -> None:
    """会话结束：渲染 tests/report.md（失败不影响报告生成）。"""
    try:
        _write_report(session, exitstatus)
    except Exception as error:  # 报告生成失败只警告，不影响测试结果
        print(f"\n[report] report.md 生成失败: {error}")


def _write_report(session, exitstatus) -> None:
    if not _REPORTS:  # 无 call 阶段报告（如 --collect-only）→ 不渲染，防用零值覆盖真实报告
        return
    stats = _collect_layer_stats(session)
    coverage_rows = _collect_coverage(session)
    failures = _collect_failures(session)

    lines = [
        "# 测试报告",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> pytest 退出码: {exitstatus}",
        "",
        "## 分层统计",
        "",
        "| 层 | 总数 | 通过 | 失败 | 跳过 | 错误 | 耗时(s) |",
        "|----|------|------|------|------|------|---------|",
    ]
    for layer in _LAYERS:
        layer_stats = stats[layer]
        lines.append(
            f"| {layer} | {layer_stats['total']} | {layer_stats['passed']} | "
            f"{layer_stats['failed']} | {layer_stats['skipped']} | {layer_stats['error']} | "
            f"{layer_stats['duration']:.2f} |"
        )
    other_stats = stats["other"]
    lines.append(
        f"| other | {other_stats['total']} | {other_stats['passed']} | "
        f"{other_stats['failed']} | {other_stats['skipped']} | {other_stats['error']} | "
        f"{other_stats['duration']:.2f} |"
    )
    lines.append(f"| **合计** | {sum(s['total'] for s in stats.values())} | "
                 f"{sum(s['passed'] for s in stats.values())} | "
                 f"{sum(s['failed'] for s in stats.values())} | "
                 f"{sum(s['skipped'] for s in stats.values())} | "
                 f"{sum(s['error'] for s in stats.values())} | "
                 f"{sum(s['duration'] for s in stats.values()):.2f} |")

    lines.extend(["", "## 模块覆盖", "", "| 模块 | 行覆盖 | 分支覆盖 |", "|------|--------|----------|"])
    if coverage_rows:
        for module, line_covered, line_total, branch_covered, branch_total in coverage_rows:
            line_pct = f"{line_covered / line_total * 100:.1f}%" if line_total else "—"
            branch_pct = f"{branch_covered / branch_total * 100:.1f}%" if branch_total else "—"
            lines.append(f"| {module} | {line_pct} ({line_covered}/{line_total}) | "
                         f"{branch_pct} ({branch_covered}/{branch_total}) |")
    else:
        lines.append("| （无覆盖数据，需 --cov 运行时收集） | | |")

    lines.extend(["", "## 失败明细", ""])
    if failures:
        for nodeid, text in failures:
            lines.append(f"### {nodeid}")
            lines.append("")
            lines.append("```")
            lines.append(text[:2000])
            lines.append("```")
            lines.append("")
    else:
        lines.append("无失败。")

    # 追加持久已知问题记录（tests/known_issues.md，存在才嵌入；reporter 每次重生成 report.md，故手工记录走此文件）
    known_issues_path = Path(__file__).resolve().parent / "known_issues.md"
    if known_issues_path.exists():
        lines.extend(["", known_issues_path.read_text(encoding="utf-8").rstrip(), ""])

    report_path = Path(__file__).resolve().parent / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] 测试报告已生成: {report_path}")
