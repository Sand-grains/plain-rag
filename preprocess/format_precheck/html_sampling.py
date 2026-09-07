"""HTML 采样预检：摄入 HTML 文件路径，产出 PrecheckResult（分流决策 + 正文统计[字符数/text_tag_ratio/图片数] + 质量降级标记[low_text_tag_ratio]）。

决策域: {HTML_TEXT, VLM_TEXT_PIPELINE, SKIP_TEXT_PIPELINE}
  - 正文 >= HTML_MIN_TEXT -> HTML_TEXT
  - 正文 < 阈值 且含图 -> VLM_TEXT_PIPELINE
  - 正文 < 阈值 且无图 -> SKIP_TEXT_PIPELINE
text_tag_ratio 低 -> degraded_flag("low_text_tag_ratio")。
"""
from pathlib import Path

from config import HTML_LOW_TEXT_TAG_RATIO, HTML_MIN_TEXT
from .result import DispatchDecision, PrecheckResult

# 正文抽取时剔除的标签(样板/导航容器)
_SKIP_TAGS = ["nav", "aside", "footer", "script", "style", "header", "form", "noscript"]


def precheck_html(path: Path) -> PrecheckResult:
    """对 HTML 做预检：抽正文后统计 text_tag_ratio 并产出分流决策。

    Args:
        path: HTML 文件路径。

    Returns:
        PrecheckResult：HTML_TEXT / VLM_TEXT_PIPELINE / SKIP_TEXT_PIPELINE 决策与质量信号。
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(_SKIP_TAGS):
        tag.decompose()

    images = soup.find_all("img")
    main_text = soup.get_text(" ", strip=True)
    tag_count = len(soup.find_all())
    text_tag_ratio = (len(main_text) / tag_count) if tag_count else 0.0

    degraded_flags: list[str] = []
    if tag_count and text_tag_ratio < HTML_LOW_TEXT_TAG_RATIO:
        degraded_flags.append("low_text_tag_ratio")

    if len(main_text) >= HTML_MIN_TEXT:
        decision = DispatchDecision.HTML_TEXT
        vlm_candidates = 0
    elif len(images) > 0:
        decision = DispatchDecision.VLM_TEXT_PIPELINE
        vlm_candidates = len(images)
    else:
        decision = DispatchDecision.SKIP_TEXT_PIPELINE
        vlm_candidates = 0

    return PrecheckResult(
        doc_decision=decision,
        empty_page_ratio=0.0,
        vlm_candidate_count=vlm_candidates,
        degraded_flags=degraded_flags,
        sampling_format_stats={
            "main_text_chars": len(main_text),
            "tag_count": tag_count,
            "text_tag_ratio": round(text_tag_ratio, 4),
            "image_count": len(images),
        },
    )
