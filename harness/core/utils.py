"""harness 包内共享的纯工具函数: 跨模块复用的无状态 helpers。

markdown 表格转义与功能项排序被 registry/reporter/scheduler 各自手写过,
集中一处避免漂移。无状态、无 I/O, 供只读渲染与调度复用。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.core.models import FeatureItem


def md_escape(text: str) -> str:
    """Markdown 表格单元格转义: 竖线拆列、换行破行, 必须转义保表格结构。"""
    return text.replace("|", "\\|").replace("\n", " ")


def sorted_items(items: list[FeatureItem]) -> list[FeatureItem]:
    """按 id 升序返回功能项, 调度候选与报告表格共用同一顺序。"""
    return sorted(items, key=lambda item: item.id)
