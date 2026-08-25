"""preprocess/format_precheck：预检分流模块。

明确不处理 .md/.txt，只针对 .pdf/.html/.docx/.pptx

对外入口 precheck(file_path) -> PrecheckResult，按文件后缀分发到四格式采样器。
只回答"这份文档该走哪条解析分支"，不解析正文；逐页/逐 slide 图文分类由 loader 自判。

决策域（每个决策值映射唯一一个 loader 分支）:
- PDF/DOCX/PPTX: {WHOLE_TEXT_PIPELINE, SKIP_TEXT_PIPELINE}
- HTML:          {HTML_TEXT, SKIP_TEXT_PIPELINE}（无 WHOLE_TEXT_PIPELINE 分支, 一律走正文抽取）

config 开关接线：PRECHECK_ENABLED 为总开关，各 *_PRECHECK_ENABLED 为格式级开关。
开关关闭时跳过采样，回退"默认文本决策"（pdf/docx/pptx→WHOLE_TEXT_PIPELINE, html→HTML_TEXT）,
由 loader 逐页/逐 slide 自判兜底，保证格式仍可加载。
"""
from pathlib import Path

from config import (
    DOCX_PRECHECK_ENABLED,
    HTML_PRECHECK_ENABLED,
    PDF_PRECHECK_ENABLED,
    PPTX_PRECHECK_ENABLED,
    PRECHECK_ENABLED,
)
from .result import DispatchDecision, PrecheckResult

__all__ = ["DispatchDecision", "PrecheckResult", "precheck", "SUFFIX_TO_PARSER_KIND"]

# 格式后缀 路由(映射) 对应解析器名（单源真值：precheck / loaders / loader.py 共用）
SUFFIX_TO_PARSER_KIND = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html", # HTML 短扩展名
    ".docx": "docx",
    ".pptx": "pptx",
}

# 各格式预检门控: 解析器名(kind) -> 格式级开关(默认开) (关掉时跳过采样, 由 _default_text_decision 回退)
_FORMAT_PRECHECK_GATES = {
    "pdf": PDF_PRECHECK_ENABLED,
    "html": HTML_PRECHECK_ENABLED,
    "docx": DOCX_PRECHECK_ENABLED,
    "pptx": PPTX_PRECHECK_ENABLED,
}


def _default_text_decision(kind: str) -> PrecheckResult:
    """预检被关闭时的回退策略：假设有文本，交由 loader 逐页/逐 slide 自判兜底。"""
    decision = DispatchDecision.HTML_TEXT if kind == "html" else DispatchDecision.WHOLE_TEXT_PIPELINE
    return PrecheckResult(
        doc_decision=decision,
        degraded_flags=["precheck_disabled"],
        sampling_format_stats={"precheck_disabled": True},
    )


def precheck(file_path: str | Path) -> PrecheckResult:
    """按文件后缀(仅.pdf/.html/.docx/.pptx)分发到对应格式的预检采样器，产出 PrecheckResult。

    Args:
        file_path: 文件路径（对应 pdf/html/docx/pptx 格式）。

    Returns:
        PrecheckResult：整篇分流决策与采样统计。

    Raises:
        ValueError: 后缀不在预检支持范围内（.md/.txt 不走 precheck）。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    kind = SUFFIX_TO_PARSER_KIND.get(suffix)
    if kind is None:
        raise ValueError(f"precheck 不支持的文件类型: {suffix}")

    if not PRECHECK_ENABLED or not _FORMAT_PRECHECK_GATES[kind]:
        return _default_text_decision(kind)

    if kind == "pdf":
        from .pdf_sampling import precheck_pdf
        return precheck_pdf(path)
    if kind == "html":
        from .html_sampling import precheck_html
        return precheck_html(path)
    if kind == "docx":
        from .docx_sampling import precheck_docx
        return precheck_docx(path)
    from .pptx_sampling import precheck_pptx
    return precheck_pptx(path)
