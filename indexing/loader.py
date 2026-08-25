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
    """新格式路径：precheck 分流 → loaders 归一化 MD → cleaner → 单 Chunk。

    SKIP_TEXT_PIPELINE/空文档返回空列表；新格式 Chunk 的 protect_tables=True（表格保护只对新格式开）。
    """
    from preprocess.cleaner import clean
    from preprocess.format_precheck import precheck
    from indexing.loaders import load_multiformat

    precheck_result = precheck(path)
    markdown, format_meta = load_multiformat(path, precheck_result)
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
