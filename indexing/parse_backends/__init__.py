"""indexing/parse_backends：重型解析后端 adapter 桩（008-2 条例三）。

定义抽象接口 ParseBackend + ParseResult + 注册表 + 配置门控。
轻量后端（pdfplumber/bs4/python-docx/python-pptx）在 indexing/loaders 真接入；
重型后端（Docling/MinerU/Marker/LlamaParse/VLM/ColPali）(当前只做了协议与门控, v2 逐个接入)，
默认全关，门控关时桩只返回"未启用/降级"信号，不写任何 fake 实现。

"""
from dataclasses import dataclass, field
from typing import Any, Protocol

from config import (
    COLPALI_ENABLED,
    DOCLING_ENABLED,
    LLAMAPARSE_ENABLED,
    MARKER_ENABLED,
    MINERU_ENABLED,
    VLM_ENABLED,
)

# 重型后端注册表：parser 名 -> 后端实例（本期全为空，v2 填充）
_BACKEND_REGISTRY: dict[str, "ParseBackend"] = {}

# 门控映射：parser 名 -> 开关
_BACKEND_GATES: dict[str, bool] = {
    "docling": DOCLING_ENABLED,
    "mineru": MINERU_ENABLED,
    "marker": MARKER_ENABLED,
    "llamaparse": LLAMAPARSE_ENABLED,
    "vlm": VLM_ENABLED,
    "colpali": COLPALI_ENABLED,
}


@dataclass
class ParseResult:
    """重型后端解析结果（协议返回，v2 接入时消费）。"""
    markdown: str = ""  # 归一化 Markdown 输出(v2 接入时消费)
    format_meta: dict[str, Any] = field(default_factory=dict)  # 格式元信息(v2 接入时消费)


class ParseBackend(Protocol):
    """重型后端抽象接口：输入文档路径，输出解析结果。"""

    def extract(self, doc_path: str) -> ParseResult: ...


def register_backend(name: str, backend: "ParseBackend") -> None:
    """按 parser 名注册后端到注册表（v2 接入时调用）。"""
    _BACKEND_REGISTRY[name] = backend


def is_enabled(name: str) -> bool:
    """查询某后端是否被配置门控开启（默认全关）。"""
    return _BACKEND_GATES.get(name, False)


def resolve(name: str) -> ParseBackend | None:
    """解析后端：已注册且门控开启才返回实例；否则返回 None（未启用信号）。"""
    if not is_enabled(name):
        return None
    return _BACKEND_REGISTRY.get(name)


def enabled_backends() -> list[str]:
    """返回当前门控开启的后端名列表（用于诊断/日志）。"""
    return [name for name, enabled in _BACKEND_GATES.items() if enabled]
