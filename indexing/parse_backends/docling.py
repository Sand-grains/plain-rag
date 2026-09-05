"""Docling 重型文本解析后端薄封装 adapter。
把第三方 Docling 库包成项目统一的 ParseBackend 接口

负责"依赖门控 + 环境准备 + 调 Docling 解析 + 归一化对齐, 并注册到重型后端注册表

Docling 布局模型经 HF 缓存(本机 HF_HOME), 已缓存时可离线解析（首次需联网下载）;
Docling 依赖未安装时,它只是"不注册、不启用",让链跳过这个后端。

"""
from __future__ import annotations

import importlib.util
import os

from indexing.parse_backends import ParseResult, register_backend
from indexing.parse_backends._normalize import normalize_to_contract

_NAME = "docling"


def is_available() -> bool:
    """docling 依赖是否可导入。"""
    return importlib.util.find_spec("docling") is not None


def version() -> str:
    """返回 docling 版本; 未安装返回 'not-installed'。"""
    if not is_available():
        return "not-installed"
    from importlib.metadata import version as _version
    return _version("docling")


def _prepare_env() -> None:
    """转换前准备环境: 禁用 XetHub 下载(镜像下 401), 其余沿用项目 .env。"""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class DoclingBackend:
    """Docling 解析后端: 输入文档路径, 输出归一化 ParseResult。"""

    name = _NAME

    def extract(self, doc_path: str) -> ParseResult:
        """用 Docling 解析文档并归一化对齐。

        Args:
            doc_path: 文档路径(pdf/html/docx/pptx)。

        Returns:
            ParseResult: 归一化 Markdown + 后端/版本元信息。

        Raises:
            RuntimeError: docling 未安装时抛出(由链路上层捕获降级)。
        """
        if not is_available():
            raise RuntimeError(f"{_NAME} 未安装, 无法解析 {doc_path}")
        _prepare_env()
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(doc_path)
        markdown = result.document.export_to_markdown()
        return ParseResult(
            markdown=normalize_to_contract(markdown),
            format_meta={"backend": _NAME, "version": version()},
        )


def register() -> None:
    """注册 Docling 后端 (依赖未安装时不注册)"""
    if is_available():
        register_backend(_NAME, DoclingBackend())
