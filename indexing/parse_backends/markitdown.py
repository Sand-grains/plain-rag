"""MarkItDown 重型文本解析后端 adapter。

在函数内做 import 守卫: 未安装 markitdown 时 is_available=False, 不注册后端,
resolve 返回 None。

MarkItDown(MIT) 是文本链的本地兜底: 在 Docling/MinerU 不可用或质量不达标时兜底, 本身只读文本层, 无版式/OCR。
"""
from __future__ import annotations

import importlib.util

from indexing.parse_backends import ParseResult, register_backend
from indexing.parse_backends._normalize import normalize_to_contract

_NAME = "markitdown"


def is_available() -> bool:
    """markitdown 依赖是否可导入。"""
    return importlib.util.find_spec("markitdown") is not None


def version() -> str:
    """返回 markitdown 版本; 未安装返回 'not-installed'。"""
    if not is_available():
        return "not-installed"
    from importlib.metadata import version as _version
    return _version("markitdown")


class MarkItDownBackend:
    """MarkItDown 解析后端: 输入文档路径, 输出归一化 ParseResult。"""

    name = _NAME

    def extract(self, doc_path: str) -> ParseResult:
        """用 MarkItDown 解析文档并归一化对齐 008 契约。

        Args:
            doc_path: 文档路径(pdf/html/docx/pptx)。

        Returns:
            ParseResult: 归一化 Markdown + 后端/版本元信息。

        Raises:
            RuntimeError: markitdown 未安装时抛出(由链路上层捕获降级)。
        """
        if not is_available():
            raise RuntimeError(f"{_NAME} 未安装, 无法解析 {doc_path}")
        from markitdown import MarkItDown
        result = MarkItDown().convert(doc_path)
        markdown = result.text_content or ""
        return ParseResult(
            markdown=normalize_to_contract(markdown),
            format_meta={"backend": _NAME, "version": version()},
        )


def register() -> None:
    """注册 MarkItDown 后端; 依赖未安装时不注册(不写任何 fake)。"""
    if is_available():
        register_backend(_NAME, MarkItDownBackend())
