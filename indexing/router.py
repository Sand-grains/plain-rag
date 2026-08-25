"""Router：依据 DocQualityReport 路由到分块策略。"""
from config import (
    CHILD_CHUNK_SIZE,
    CHILD_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_OVERLAP,
    PARENT_MAX_CHARS,
    HEADING_SPLIT_RULES,
)
from indexing.splitter import (
    ParentChildMappingWrapper,
    RecursiveCharacterTextSplitter,
    HeadingSplitter,
)


class Router:
    """根据质量诊断结果，将文档路由到四类分块策略之一（按文档形态选切分器）。

    决策键：有 h1 时看 connection（标题层级规整度，即 heading_continuous 且 heading_density_ok）；无 h1 时看 too_fragmented（碎片度）

    有 h1 且层级规整       → structured_clear：父块=标题树章节（HeadingSplitter，≤8000字），子块=300字
    有 h1 但层级不规整     → flat_parent_child：父块=1200字 / 子块=300字（不信任标题树，退回字符切分）
    无 h1 且文本过碎       → flat_simple：单层 300字，父=子（无值得当上下文的父块，不建父子分块）
    无 h1 且文本不过碎     → flat_parent_child：父块=1200字 / 子块=300字（与第二条配置相同）
    """

    def route(self, doc_quality_report):
        if (doc_quality_report.has_h1
                and doc_quality_report.heading_continuous
                and doc_quality_report.heading_density_ok):
            return ParentChildMappingWrapper(
                HeadingSplitter(HEADING_SPLIT_RULES, PARENT_MAX_CHARS),
                RecursiveCharacterTextSplitter(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
            )
        if doc_quality_report.has_h1:
            return ParentChildMappingWrapper(
                RecursiveCharacterTextSplitter(PARENT_CHUNK_SIZE, PARENT_OVERLAP, chunk_level="parent"),
                RecursiveCharacterTextSplitter(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
            )
        if doc_quality_report.too_fragmented:
            return RecursiveCharacterTextSplitter(CHILD_CHUNK_SIZE, CHILD_OVERLAP, chunk_level="parent")
        return ParentChildMappingWrapper(
            RecursiveCharacterTextSplitter(PARENT_CHUNK_SIZE, PARENT_OVERLAP, chunk_level="parent"),
            RecursiveCharacterTextSplitter(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
        )
