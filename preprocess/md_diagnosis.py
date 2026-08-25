"""文档质量诊断：DocQualityReport + diagnose。

路由键仅 has_h1 / heading_continuous / heading_density_ok / too_fragmented；
exceeds_embed_token_limit / text_ratio / has_encoding_issues 仅诊断记录，不参与路由。
heading_connection_standard 已拆为 heading_continuous + heading_density_ok 两字段（条例七），
路由处取"两者都真"，路由结果不变。
"""
from dataclasses import dataclass

from config import CHUNK_TOO_FRAGMENTED_THRESHOLD, EMBEDDING_MODEL_TOKEN_CONSTRAINT
from .md_struct_analysis import (
    check_heading_continuity,
    token_estimate,
    detect_encoding_issues,
    heading_density_ok,
    parse_headings,
    section_token_statistics,
    text_ratio,
)


@dataclass
class DocQualityReport:
    has_h1: bool # 是否有 h1 标题
    heading_continuous: bool # 标题层级无跳跃(向下钻一次跨两级以上则 False)
    heading_density_ok: bool # 平均每 HEADING_DENSITY_THRESHOLD 字符至少一个标题
    too_fragmented: bool # 是否过碎片化
    exceeds_embed_token_limit: bool # doc是否超过嵌入模型 token 限制
    text_ratio: float # 真实文本占比
    has_encoding_issues: bool # 是否有编码问题
    n_headings: int # 标题数量
    max_section_tokens: int # 最长章节 token 数
    median_section_tokens: int # 章节 token 数的中位数
    total_tokens: int # 总 token 数


def diagnose(text: str) -> DocQualityReport:
    headings = parse_headings(text)
    section_max_token, section_median_token = section_token_statistics(text, headings)
    return DocQualityReport(
        has_h1=any(heading.level == 1 for heading in headings),
        heading_continuous=check_heading_continuity(headings),
        heading_density_ok=heading_density_ok(headings, text),
        too_fragmented=section_median_token < CHUNK_TOO_FRAGMENTED_THRESHOLD,
        exceeds_embed_token_limit=section_max_token > EMBEDDING_MODEL_TOKEN_CONSTRAINT,
        text_ratio=text_ratio(text),
        has_encoding_issues=detect_encoding_issues(text),
        n_headings=len(headings),
        max_section_tokens=section_max_token,
        median_section_tokens=section_median_token,
        total_tokens=token_estimate(text),
    )
