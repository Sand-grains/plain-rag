"""文档加载器：按文件后缀分发，产出未切分的原始 Chunk 列表。

从文件加载，走完"预检→归一化→清洗"这段摄入管线, 返回一个 List[Chunk] (列表能表达"被跳过"语义, 实际上这里 Chunk 数量只有0或1)

核心特性：
    - 支持 .txt / .md 纯文本格式，一次性读取全文
    - 支持 .pdf / .html / .docx / .pptx 新格式
    precheck 分流 → loaders 归一化 Markdown → cleaner 清洗 → 产出单 Chunk（origin_metadata.protect_tables=True）
    - doc_id 由文件相对于 data/ 的路径推导（去后缀、斜杠归一化）
    - 每个文件产出一个 Chunk，后续由 Router → Splitter 分块

用法示例::

    from indexing import load
    chunks = load("data/Agent/README.md")  # → [Chunk(doc_id="Agent/README", ...)]

公共接口：
    - load: 按后缀分发加载，返回 Chunk 列表
"""

from pathlib import Path
from typing import List
from config import CLEAN_NEW_FORMAT, CLEAN_PLAIN_TEXT
from preprocess.format_precheck import SUFFIX_TO_PARSER_KIND
from .chunk import Chunk, DocMetadata

# 新格式后缀（单源：preprocess/format_precheck.SUFFIX_TO_PARSER_KIND）
_NEW_FORMAT_SUFFIXES = tuple(SUFFIX_TO_PARSER_KIND)


def load(file_path: str, base_dir: str = "data") -> List[Chunk]:
    """根据文件后缀分发到对应的加载器，返回该文件产出的 Chunk 列表"""
    path = Path(file_path)             # 字符串路径 → Path 对象, 便于对其操作(如取后缀)
    suffix = path.suffix.lower()       # 后缀统一小写, 便于识别
    base = Path(base_dir)

    if suffix in (".txt", ".md"):# 仅支持.md/.txt(它们走同一个加载逻辑)
        return _load_text(path, base)
    if suffix in _NEW_FORMAT_SUFFIXES:
        return _load_multiformat(path, base)
    raise ValueError(f"不支持的文件类型: {suffix}")


def _derive_doc_id(path: Path, base_dir: Path) -> str:
    """从文件相对 data/ 的路径推导 doc_id（去后缀、斜杠归一化）。"""
    try:
        relative = path.relative_to(base_dir)
        return str(relative.with_suffix("")).replace("\\", "/")
    except ValueError:
        return path.stem


def _load_multiformat(path: Path, base_dir: Path) -> List[Chunk]:
    """新格式路径：precheck 顶层路由 → 重型/VLM 管线 → cleaner → 单 Chunk。

    非 .md/.txt 文档按 precheck 决策分流:
    - WHOLE_TEXT_PIPELINE / HTML_TEXT -> 重型文本管线(parse_text_pipeline)
    - VLM_TEXT_PIPELINE -> VLM 后端
    - SKIP_TEXT_PIPELINE -> 空(跳过)
    目标管线不可用/全部失败 -> 进失败清单, 不落轻量(返回空列表)。
    colpali_triggered 并行触发: 标记进 format_meta; ColPali 不可用记失败清单但不阻塞文本。
    """
    from preprocess.cleaner import clean
    from preprocess.format_precheck import DispatchDecision, precheck
    from config import CLEAN_NEW_FORMAT
    from indexing.parse_backends.failure_list import failure_list

    precheck_result = precheck(path)
    decision = precheck_result.doc_decision

    if decision == DispatchDecision.SKIP_TEXT_PIPELINE:
        return []

    if decision == DispatchDecision.VLM_TEXT_PIPELINE:
        markdown, format_meta = _run_vlm_pipeline(path, precheck_result)
    else:  # WHOLE_TEXT_PIPELINE / HTML_TEXT
        markdown, format_meta = _run_text_pipeline(path, precheck_result)

    if CLEAN_NEW_FORMAT:
        markdown = clean(markdown)
    if not markdown.strip():
        return []

    doc_id = _derive_doc_id(path, base_dir)
    origin_metadata = DocMetadata(
        title=path.stem,
        source=str(path),
        doc_type=path.suffix.lower(),
        protect_tables=True,
    )

    return [Chunk(
        content=markdown,
        doc_id=doc_id,
        origin_metadata=origin_metadata,
        metadata={"format_meta": format_meta},
    )]


def _record_failure_and_empty(path: Path, route_decision: str, exc: Exception,
                              version_string: str) -> str:
    """管线失败统一处理: 记失败清单 + 返回空 Markdown(不落轻量)。

    Args:
        path: 文档路径。
        route_decision: 路由决策(text/vlm)。
        exc: 捕获的异常。
        version_string: 后端版本串(溯源)。

    Returns:
        str: 空 Markdown。调用方以 ("", {}) 返回, meta 恒被 _load_multiformat 在空输出时丢弃(L2)。
    """
    from indexing.parse_backends.failure_list import failure_list
    failure_list.record(str(path), route_decision, f"{type(exc).__name__}: {exc}",
                        version_string=version_string)
    return ""


def _run_text_pipeline(path: Path, precheck_result) -> tuple[str, dict]:
    """重型文本管线: parse_text_pipeline; 失败进失败清单, 不落轻量。"""
    from indexing.parse_backends import backend_version, parse_text_pipeline
    from indexing.parse_backends import colpali

    stats = precheck_result.sampling_format_stats or {}
    original_table_count = stats.get("table_count")
    route_decision = "text"
    try:
        markdown, meta = parse_text_pipeline(
            str(path), original_table_count=original_table_count)
        meta["route_decision"] = route_decision
        meta["colpali_triggered"] = precheck_result.colpali_triggered
        if precheck_result.colpali_triggered:
            from indexing.parse_backends import run_stats
            run_stats.record_colpali_trigger()  # L6: 接线 ColPali 触发率统计
            if not colpali.is_available():
                from indexing.parse_backends.failure_list import failure_list
                failure_list.record(str(path), "colpali", "colpali_unavailable",
                                    version_string=colpali.version())
        return markdown, meta
    except Exception as exc:  # noqa: BLE001 - 失败进清单, 不落轻量
        # M4: 记录真实失败后端版本(链式异常携带 backend_name), 而非固定 docling
        backend_name = getattr(exc, "backend_name", "docling")
        return _record_failure_and_empty(path, route_decision, exc,
                                         backend_version(backend_name)), {}


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _vlm_render_pages(path: Path, workdir: str) -> list[str]:
    """把文档渲染为图像列表, 供 VLM 逐图解析。

    - 已是图像(png/jpg/jpeg/webp): 单图直通。
    - PDF: 逐页渲染为 PNG(复用 paged_pipeline 渲染逻辑)。
    - 其余(PPTX/DOCX/HTML): 无稳定渲染器, 抛 VlmUnavailableError(进失败清单, 不发无效 data URL)。

    Args:
        path: 文档路径。
        workdir: 临时工作目录(渲染中间产物)。

    Returns:
        list[str]: 图像路径列表(按页序)。

    Raises:
        VlmUnavailableError: 无法渲染为图像。
    """
    from indexing.parse_backends.vlm import VlmUnavailableError
    suffix = path.suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        return [str(path)]
    if suffix == ".pdf":
        from indexing.parse_backends.paged_pipeline import render_pdf_page, split_pdf_pages
        images: list[str] = []
        for page in split_pdf_pages(str(path), workdir):
            image = render_pdf_page(page, workdir)
            if image:
                images.append(image)
        if not images:
            raise VlmUnavailableError(f"VLM 无法渲染 PDF 页面: {path}")
        return images
    raise VlmUnavailableError(f"VLM 路由仅支持图像/PDF, 无法渲染为图像: {path}")


def _run_vlm_pipeline(path: Path, precheck_result) -> tuple[str, dict]:
    """VLM 管线: 先按页渲染为图像, 再逐图经 _run_backend(子进程隔离 + 熔断 + 统计)解析并合并;
    失败进失败清单, 不落轻量。"""
    import tempfile
    from indexing.parse_backends import _run_backend, colpali, vlm
    from indexing.parse_backends.paged_pipeline import merge_cross_page

    route_decision = "vlm"
    try:
        with tempfile.TemporaryDirectory() as workdir:
            images = _vlm_render_pages(path, workdir)
            page_markdowns: list[str] = []
            page_meta: dict = {}
            for image in images:
                md, meta = _run_backend("vlm", image)
                page_markdowns.append(md)
                page_meta = meta  # 各页 backend/vlm_provider/vlm_model 一致, 取末页
            markdown, merge_meta = merge_cross_page(page_markdowns)
        meta = dict(page_meta)
        meta["vlm_pages"] = len(images)
        meta.update(merge_meta)
        meta["route_decision"] = route_decision
        meta["colpali_triggered"] = precheck_result.colpali_triggered
        if precheck_result.colpali_triggered:
            from indexing.parse_backends import run_stats
            run_stats.record_colpali_trigger()  # L6: 接线 ColPali 触发率统计
            if not colpali.is_available():
                from indexing.parse_backends.failure_list import failure_list
                failure_list.record(str(path), "colpali", "colpali_unavailable",
                                    version_string=colpali.version())  # M3: 与文本路由一致
        return markdown, meta
    except Exception as exc:  # noqa: BLE001 - 失败进清单, 不落轻量
        return _record_failure_and_empty(path, route_decision, exc, vlm.version()), {}


def _load_text(path: Path, base_dir: Path = Path("data")) -> List[Chunk]:
    """一次性读取全文件，内容作为单个大 Chunk 对象返回
    文档 -> Chunk对象就是在这里被抽象出来的
    """
    content = path.read_text(encoding="utf-8")
    if CLEAN_PLAIN_TEXT:
        from preprocess.cleaner import clean
        content = clean(content)
    doc_id = _derive_doc_id(path, base_dir)
    origin_metadata = DocMetadata(
        title=path.stem,
        author="sd", # 本地文档默认作者
        source=str(path),
        doc_type=path.suffix.lower(),
    )
    return [Chunk(
        content=content,
        doc_id=doc_id,
        origin_metadata=origin_metadata,
    )]
