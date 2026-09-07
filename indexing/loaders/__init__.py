"""indexing/loaders：四格式 (.pdf/.html/.docx/.pptx) 的轻量级解析归一方案。

注意: loaders 和 parse_backends 当前是地位平行的解析后端

对外入口 load_multiformat(file_path, precheck_result) -> (markdown, format_meta),
按后缀分发到对应 loader: 各 loader 逐页/逐 slide 自判图文分类（F2）, 图文单元单跳过，文本单元拼入归一化 MD。

config 开关接线: 各 *_LOADER_ENABLED 为格式级开关，
关闭时该格式整篇跳过（返回空md + disabled 标记），材料不参与归一化。
"""
from pathlib import Path

from config import (
    DOCX_LOADER_ENABLED,
    HTML_LOADER_ENABLED,
    PDF_LOADER_ENABLED,
    PPTX_LOADER_ENABLED,
)
from preprocess.format_precheck import PrecheckResult, SUFFIX_TO_PARSER_KIND

__all__ = ["load_multiformat", "skip_result", "vlm_result"]

# 各格式 loader 门控: 解析器名(kind) -> 格式级开关(默认开); 关掉时整篇跳过, 返回空 MD + disabled 标记
_FORMAT_LOADER_GATES = {
    "pdf": PDF_LOADER_ENABLED,
    "html": HTML_LOADER_ENABLED,
    "docx": DOCX_LOADER_ENABLED,
    "pptx": PPTX_LOADER_ENABLED,
}


def load_multiformat(file_path: str | Path, precheck_result: PrecheckResult) -> tuple[str, dict]:
    """按格式后缀分发到对应 loader, 产出 (归一化 Markdown, format_meta)。

    Args:
        file_path: 新格式文件路径（pdf/html/docx/pptx）。
        precheck_result: precheck 产出的分流决策。

    Returns:
        (markdown, format_meta)：归一化 MD 与格式元数据；SKIP_TEXT_PIPELINE/空文档/loader 关闭时
        markdown 为空串。

    Raises:
        ValueError: 后缀不在多格式 loader 支持范围内。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    kind = SUFFIX_TO_PARSER_KIND.get(suffix)
    if kind is None:
        raise ValueError(f"loaders 不支持的文件类型: {suffix}")

    if not _FORMAT_LOADER_GATES[kind]:
        return skip_result(suffix, 0, disabled=True)

    if kind == "pdf":
        from .pdf import pdf_loader
        return pdf_loader(path, precheck_result)
    if kind == "html":
        from .html import html_loader
        return html_loader(path, precheck_result)
    if kind == "docx":
        from .docx import docx_loader
        return docx_loader(path, precheck_result)
    from .pptx import pptx_loader
    return pptx_loader(path, precheck_result)


def skip_result(doc_type: str, vlm_candidate_count: int, disabled: bool = False) -> tuple[str, dict]:
    """SKIP_TEXT_PIPELINE / loader 关闭时的统一返回: 返回空 MD + 标记 skipped 的 元信息。
    (这里 loader 关闭是指通过设置环境变量把某个格式的 *_LOADER_ENABLED 设成非 "1")
    这种情况四种格式实际上不被真正解析, 只有 vlm_candidate_count 被带出来留作候选计数

    Args:
        doc_type: 文件后缀（如 ".pdf"）。
        vlm_candidate_count: 图文/纯图候选聚合计数。
        disabled: 是否因 loader 开关关闭而跳过（True 时补 disabled 标记）。

    Returns:
        (markdown, format_meta)：markdown 恒为空串，format_meta 记 skipped=True；
        disabled=True 时额外记 disabled=True。
    """
    meta = {
        "doc_type": doc_type,
        "vlm_candidate_count": vlm_candidate_count,
        "skipped": True,
    }
    if disabled:
        meta["disabled"] = True
    return "", meta


def vlm_result(doc_type: str, vlm_candidate_count: int) -> tuple[str, dict]:
    """VLM_TEXT_PIPELINE 的统一返回: 返回空 MD + route_decision=vlm(不落轻量)。

    视觉/扫描类文档由 precheck 路由到 VLM 管线;
    这是因为轻量 loader 不解析视觉/扫描类文档, 返回空 MD + route_decision="vlm", 由调用方进 VLM 管线或失败清单

    Args:
        doc_type: 文件后缀（如 ".pdf"）。
        vlm_candidate_count: 图文/纯图候选聚合计数。

    Returns:
        (markdown, format_meta)：markdown 恒为空串，format_meta 记 skipped=True + route_decision="vlm"。
    """
    return "", {
        "doc_type": doc_type,
        "vlm_candidate_count": vlm_candidate_count,
        "skipped": True,
        "route_decision": "vlm",
    }
