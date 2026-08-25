"""项目文档预处理治理层：归一化前后的决策与质检，不负责格式解析本体。

由三块组成：
- precheck: 预检分流。采样下 doc 级决策（WHOLE_TEXT_PIPELINE/HTML_TEXT/SKIP），是跑解析后端前的廉价闸门
- cleaner: 确定性清洗。对归一化 Markdown 去噪（样板行/URL/HTML 残留/控制字符），与具体后端解耦
- md_diagnosis: md质量 结构诊断。产出 DocQualityReport 供 Router 路由与展示

本包只做"解析路由决策"与"md文档质检"，不渗入任何格式解析逻辑。
归一化本体（格式 -> Markdown）由专业后端承担：轻量 indexing/loaders 或重型 indexing/parse_backends（Docling, MinerU 等）。
"""

from .cleaner import clean
from .format_precheck import PrecheckResult, precheck
from .md_diagnosis import DocQualityReport, diagnose

__all__ = ["DocQualityReport", "diagnose", "precheck", "PrecheckResult", "clean"]
