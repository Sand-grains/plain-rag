"""Precheck 摄入文档预检 数据模型
含 DispatchDecision 决策枚举 + PrecheckResult 结果

008-1 预检回答"这份文档该走哪条解析分支"：整篇产出一个 doc_decision,
不产逐页决策（逐页/逐 slide 图文分类由 loader 自行判定，F2）。
质量细分（文本页占比不足等）不是决策值，记入 degraded_flags 质量档位。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DispatchDecision(Enum):
    """预检分流决策枚举值
    根据采样窗口的情况回答 "这份文档该走哪条解析分支？"
    每个决策值 映射 唯一 loader 分支
    - WHOLE_TEXT_PIPELINE: 采样窗口有文本单元（无论占比）: 整篇进文本管线, 图文页由 loader 单跳过
    - HTML_TEXT:  仅 HTML 有效，走正文抽取（HTML 的解析路径和 PDF/DOCX/PPTX 本质不同: 必须先剥掉 nav/header/sidebar/script/style, 留正文）
    - SKIP_TEXT_PIPELINE: 采样窗口内无文本单元（全图片/扫描件/封面/版权页/纯图目录等）-> 不进文本管线（loader 返回空, v1 不索引该文档）; 记为 vlm_candidate 预留给 v2 视觉路由（VLM/ColPali）。(v2 接视觉后端后可改名为视觉语义相关)
    """
    WHOLE_TEXT_PIPELINE = "WHOLE_TEXT"
    HTML_TEXT = "HTML_TEXT"
    SKIP_TEXT_PIPELINE = "SKIP"


@dataclass
class PrecheckResult:
    """预检采样统计与分流决策（纯接口，无共享状态）。

    Attributes:
        doc_decision: 整篇分流决策（WHOLE_TEXT_PIPELINE / HTML_TEXT / SKIP_TEXT_PIPELINE）。
        empty_page_ratio: 空页占比（0.0 ~ 1.0）。
        vlm_candidate_count: 图文页/纯图页候选数（聚合计数，非逐页锚点）。
        degraded_flags: 质量降级标记（text_page_ratio_below_threshold / multi_column / low_text_tag_ratio / precheck_disabled 等; 非决策值）。
        sampling_format_stats: 每格式采样统计（字符数/图片数/坐标等，格式相关）。
    """
    doc_decision: DispatchDecision  # 整篇分流决策(WHOLE_TEXT_PIPELINE / HTML_TEXT / SKIP_TEXT_PIPELINE)
    empty_page_ratio: float = 0.0  # 空页占比(0.0 ~ 1.0)
    vlm_candidate_count: int = 0  # 非文本页候选数(聚合计数, 非逐页锚点)
    degraded_flags: list[str] = field(default_factory=list)  # 质量降级标记(非决策值)
    sampling_format_stats: dict[str, Any] = field(default_factory=dict)  # 每格式采样统计(格式相关)
