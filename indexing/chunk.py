"""管线流转的通用文本数据模型。

核心特性：
    - Chunk 承载一段文本及其来源元数据，贯穿索引 → 检索 → 生成全管线
    - DocMetadata 作为共享引用挂在每个 Chunk 上，避免文档级信息冗余存储
    - chunk_id 确定性生成（父块 {doc_id}:p{i}，子块 {doc_id}:p{i}:c{j}），父子层级可定位

用法示例::

    from indexing import Chunk, DocMetadata
    meta = DocMetadata(title="README", doc_type=".md")
    c = Chunk(doc_id="docs/readme", chunk_id="docs/readme:p0", content="# Hello", origin_metadata=meta)

公共接口：
    - DocMetadata: 文档级元数据（title / author / source / source_url / doc_type / language / chunk_level）
    - Chunk: 通用文本容器（doc_id / chunk_id / content / origin_metadata / metadata / created_at）
"""

from dataclasses import dataclass, field


@dataclass
class DocMetadata:
    """文档级元数据（一篇文档一份，chunk 共享引用）"""
    title: str = ""              # 文档标题
    author: str = ""             # 作者（本地文档默认 "sd"）
    source: str = ""             # 来源路径/URL
    source_url: str = ""         # 原始链接（Web 来源溯源）
    doc_type: str = ""           # .txt / .md / .pdf / .docx / .xlsx
    language: str = "zh"         # 语言（预留，当前默认中文）
    chunk_level: str = "child"   # "parent" / "child"（父块为检索的目标上下文，子块为检索目标）
    protect_tables: bool = False # 是否开启表格保护 (仅新格式归一化 得出的md 置 True; 原 .md/.txt 保持 False 保 private_v6 字节级不变)

    # ---- 后续父子检索时新增 ----
    # page_number: int | None        # PDF 页码
    # page_start: int | None
    # page_end: int | None


@dataclass
class Chunk:
    """管线流转的通用文本容器，承载一段文本及其来源元数据"""
    doc_id: str = ""             # 来源文档 ID (如 Knowledge/MainLine/RAG基础架构)
    chunk_id: str = ""           # 分块 ID, 格式:父块 {doc_id}:p{i}, 子块 {doc_id}:p{i}:c{j}
    content: str = ""            # chunk 原始文本（统一用于 embedding、检索、LLM 上下文）
    origin_metadata: DocMetadata = field(default_factory=DocMetadata)  # 文档级元数据（共享引用）
    metadata: dict = field(default_factory=dict)  # chunk 级元数据: parent_id / section_path / start_char_index
    created_at: str = ""         # 创建时间, 增量索引用
