"""页级重试管线: 按页拆开, 逐页重试, 坏页 VLM 修正, 跨页合并
把粒度降到单页: 只对坏页重试/VLM 修正, 好页直接用; 最后再合并

整篇先 Docling;
整篇未达标时进入页级:
  - PDF 按页拆分(每页独立 pdf), PPTX 按 slide 拆分(每 slide 文本文件);
  - 逐页跑文本链(Docling -> MinerU -> MarkItDown), 页级质量不达标 -> 下一后端;
  - 页级全链仍不达标 -> VLM 修正: 渲染该页为图像, 以图像为 source of truth 重写 Markdown;
  - 各页结果经 merge_cross_page 合并回整篇(跨页表格/列表锚点)。
  - DOCX/HTML 无稳定页边界, 不拆: 整篇重试 + 整篇 VLM 修正。

页级拆分/渲染/链/VLM 均以可注入 callable 提供(默认实现用 PyMuPDF/python-pptx),
便于单元测试用 fake 覆盖, 不依赖真实重型后端。
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from indexing.parse_backends.merge import merge_cross_page

# 页级拆分器: (doc_path, workdir) -> 页文件路径列表 | None(无稳定页边界)
PageSplitter = Callable[[str, str], list[str] | None]

# 页渲染器: (page_path, workdir) -> 图像路径 | None(无法渲染)
PageRenderer = Callable[[str, str], str | None]

# 文本链: (doc_path) -> (markdown, meta); 失败抛异常
ChainFn = Callable[[str], tuple[str, dict[str, Any]]]

# 质量判定: (markdown, original_table_count, expected_code_blocks) -> (ok, reasons, signals)
QualityFn = Callable[[str, int | None, int | None], tuple[bool, list[str], dict[str, Any]]]

# VLM 修正: (image_path) -> (markdown, meta)
VlmFn = Callable[[str], tuple[str, dict[str, Any]]]


def split_pdf_pages(doc_path: str, workdir: str) -> list[str]:
    """用 PyMuPDF 把 PDF 拆为逐页独立 pdf 文件。

    Args:
        doc_path: PDF 文件路径。
        workdir: 临时工作目录。

    Returns:
        list[str]: 每页一个独立 pdf 文件路径(按页序)。
    """
    import pymupdf
    source = pymupdf.open(doc_path)
    paths: list[str] = []
    for index in range(len(source)):
        out = Path(workdir) / f"page_{index:03d}.pdf"
        single = pymupdf.open()
        single.insert_pdf(source, from_page=index, to_page=index)
        single.save(str(out))
        single.close()
        paths.append(str(out))
    source.close()
    return paths


def split_pptx_slides(doc_path: str, workdir: str) -> list[str]:
    """用 python-pptx 把 PPTX 拆为逐 slide 文本文件(供文本链重试)。

    Args:
        doc_path: PPTX 文件路径。
        workdir: 临时工作目录。

    Returns:
        list[str]: 每 slide 一个文本文件路径(按页序)。
    """
    from pptx import Presentation
    prs = Presentation(doc_path)
    paths: list[str] = []
    for index, slide in enumerate(prs.slides):
        texts = [shape.text for shape in slide.shapes
                 if getattr(shape, "text", "") and shape.text.strip()]
        out = Path(workdir) / f"slide_{index:03d}.txt"
        out.write_text("\n\n".join(texts), encoding="utf-8")
        paths.append(str(out))
    return paths


def default_page_splitter(doc_path: str, workdir: str) -> list[str] | None:
    """按格式返回页级拆分; DOCX/HTML 无稳定页边界返回 None。

    Args:
        doc_path: 文档路径。
        workdir: 临时工作目录。

    Returns:
        list[str] | None: 页文件路径列表; 无稳定页边界返回 None。
    """
    suffix = Path(doc_path).suffix.lower()
    if suffix == ".pdf":
        return split_pdf_pages(doc_path, workdir)
    if suffix == ".pptx":
        return split_pptx_slides(doc_path, workdir)
    return None


def render_pdf_page(page_path: str, workdir: str) -> str | None:
    """用 PyMuPDF 把单页 pdf 渲染为 PNG(供 VLM 修正)。

    Args:
        page_path: 单页 pdf 文件路径。
        workdir: 临时工作目录。

    Returns:
        str | None: 渲染出的 PNG 路径; 渲染失败返回 None。
    """
    import pymupdf
    doc = pymupdf.open(page_path)
    if not doc:
        return None
    pix = doc[0].get_pixmap(dpi=150)
    out = Path(workdir) / f"{Path(page_path).stem}.png"
    pix.save(str(out))
    doc.close()
    return str(out)


def default_page_renderer(page_path: str, workdir: str) -> str | None:
    """按格式渲染页为图像; 非 PDF(如 PPTX 文本文件)无法渲染返回 None。

    Args:
        page_path: 页文件路径。
        workdir: 临时工作目录。

    Returns:
        str | None: 图像路径; 无法渲染返回 None。
    """
    if Path(page_path).suffix.lower() == ".pdf":
        return render_pdf_page(page_path, workdir)
    return None


def _parse_page(page_path: str, chain: ChainFn, quality_fn: QualityFn,
                vlm_extract: VlmFn, page_renderer: PageRenderer, workdir: str,
                original_table_count: int | None,
                expected_code_blocks: int | None) -> tuple[str, dict[str, Any]]:
    """单页解析: 文本链重试 -> 仍不达标则 VLM 修正。

    Args:
        page_path: 页文件路径。
        chain: 文本链(整篇/单页通用)。
        quality_fn: 质量判定函数。
        vlm_extract: VLM 修正函数(输入图像路径)。
        page_renderer: 页渲染函数。
        workdir: 临时工作目录。
        original_table_count: 原始表格数(可判时传入)。
        expected_code_blocks: 预期代码块数(可判时传入)。

    Returns:
        tuple[str, dict]: (页 Markdown, 页元信息{backend, vlm_correction_applied})。
    """
    try:
        markdown, meta = chain(page_path)
        ok, _reasons, _signals = quality_fn(markdown, original_table_count, expected_code_blocks)
        if ok:
            return markdown, dict(meta)
    except Exception:  # 页级链失败, 进入 VLM 修正
        pass
    image = page_renderer(page_path, workdir)
    if image:
        try:
            markdown, meta = vlm_extract(image)
            meta = dict(meta)
            meta["vlm_correction_applied"] = True
            return markdown, meta
        except Exception:  # VLM 修正失败, 返回空页
            pass
    return "", {"vlm_correction_applied": False}


def _whole_doc_vlm_correct(doc_path: str, chain: ChainFn, quality_fn: QualityFn,
                           vlm_extract: VlmFn, page_renderer: PageRenderer, workdir: str,
                           original_table_count: int | None,
                           expected_code_blocks: int | None) -> tuple[str, dict[str, Any]]:
    """无稳定页边界(DOCX/HTML): 整篇重试 + 整篇 VLM 修正。

    Args:
        doc_path: 文档路径。
        chain: 文本链。
        quality_fn: 质量判定函数。
        vlm_extract: VLM 修正函数。
        page_renderer: 页渲染函数(整篇渲染用)。
        workdir: 临时工作目录。
        original_table_count: 原始表格数。
        expected_code_blocks: 预期代码块数。

    Returns:
        tuple[str, dict]: (整篇 Markdown, 元信息)。
    """
    try:
        markdown, meta = chain(doc_path)
        ok, _reasons, _signals = quality_fn(markdown, original_table_count, expected_code_blocks)
        if ok:
            return markdown, dict(meta)
    except Exception:  # noqa: BLE001 - 整篇链失败, 进入 VLM 修正
        pass
    image = page_renderer(doc_path, workdir)
    if image:
        try:
            markdown, meta = vlm_extract(image)
            meta = dict(meta)
            meta["vlm_correction_applied"] = True
            return markdown, meta
        except Exception:  # noqa: BLE001 - VLM 修正失败
            pass
    return "", {"vlm_correction_applied": False}


def parse_text_pipeline_paged(
    doc_path: str,
    original_table_count: int | None,
    expected_code_blocks: int | None,
    chain: ChainFn,
    quality_fn: QualityFn,
    vlm_extract: VlmFn,
    page_splitter: PageSplitter | None = None,
    page_renderer: PageRenderer | None = None,
    page_chain: ChainFn | None = None,
) -> tuple[str, dict[str, Any]]:
    """页级混合重试 + VLM 修正 + 跨页合并(012-2 2.4)。

    Args:
        doc_path: 文档路径。
        original_table_count: 原始表格数(可判时传入)。
        expected_code_blocks: 预期代码块数(可判时传入)。
        chain: 文本链(整篇重试/单页通用)。
        quality_fn: 质量判定函数。
        vlm_extract: VLM 修正函数(输入图像路径)。
        page_splitter: 页级拆分器; 缺省按格式(PDF/PPTX 拆, DOCX/HTML 不拆)。
        page_renderer: 页渲染器; 缺省 PDF 渲染, 其余不渲染。
        page_chain: 页级文本链(缺省复用 chain); 页级应关闭表格/代码保留率判定(M2)。

    Returns:
        tuple[str, dict]: (合并后整篇 Markdown, 元信息
            {page_retry_stats, cross_page_merge_*, vlm_correction_applied})。
    """
    page_splitter = page_splitter or default_page_splitter
    page_renderer = page_renderer or default_page_renderer
    page_chain = page_chain or chain
    with tempfile.TemporaryDirectory() as workdir:
        pages = page_splitter(doc_path, workdir)
        if not pages:
            return _whole_doc_vlm_correct(
                doc_path, chain, quality_fn, vlm_extract, page_renderer, workdir,
                original_table_count, expected_code_blocks)
        page_results: list[str] = []
        meta: dict[str, Any] = {"page_retry_stats": {"pages": len(pages), "vlm_corrected": 0}}
        for page_path in pages:
            # M2: 页级判定关闭表格/代码保留率(单页数远小于整篇, 用整篇基准会系统性假性降级)
            markdown, page_meta = _parse_page(
                page_path, page_chain, quality_fn, vlm_extract, page_renderer, workdir,
                None, None)
            page_results.append(markdown)
            if page_meta.get("vlm_correction_applied"):
                meta["page_retry_stats"]["vlm_corrected"] += 1
        merged, merge_meta = merge_cross_page(page_results)
        meta.update(merge_meta)
        return merged, meta
