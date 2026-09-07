"""vlm.py：VLM 视觉/扫描解析后端薄封装。

统一 OpenAI 兼容接口 + base64 data URL, 直接调用 官方 API, 不配置本地 VLM

视觉/扫描类文档(纯图/扫描件)由 precheck 路由到 VLM: 把页面/文档图像 base64 编码为 data URL, 以图像为 source of truth 提取 Markdown。

import 守卫在函数内: 未装 openai 或未配 VLM_API_KEY 时 is_available=False, 不注册后端;
VLM 不可用时视觉类文档进失败清单(不落轻量)。
"""
from __future__ import annotations

import base64
import importlib.util
import mimetypes
from pathlib import Path

from config import (
    VLM_API_KEY,
    VLM_BASE_URL,
    VLM_ENABLED,
    VLM_MODEL,
    VLM_PROVIDER,
    VLM_TIMEOUT_S,
)
from indexing.parse_backends import ParseResult, register_backend

_NAME = "vlm"

# 提取 Markdown 的提示词(以图像为 source of truth)
_PROMPT = (
    "你是文档解析器。请以这张图像为唯一事实来源, 提取其中的全部文字内容, "
    "输出为 Markdown: 标题用 # 层级, 表格用 pipe-table, 代码用 fenced block。"
    "不要添加图像中不存在的内容, 不要输出解释。"
)


class VlmUnavailableError(RuntimeError):
    """VLM 不可用(未启用/未配 key/未装 openai)。"""


def is_available() -> bool:
    """VLM 是否可用: 门控开启 + 已配 API key + openai 可导入。"""
    if not VLM_ENABLED or not VLM_API_KEY:
        return False
    return importlib.util.find_spec("openai") is not None


def version() -> str:
    """返回 VLM 模型名(版本冻结用); 不可用返回 'not-installed'。"""
    return VLM_MODEL if is_available() else "not-installed"


def _image_to_data_url(doc_path: str) -> str:
    """把图像文件编码为 base64 data URL。

    Args:
        doc_path: 图像文件路径(png/jpg/jpeg/webp)。

    Returns:
        str: data URL(如 data:image/png;base64,...)。

    Raises:
        VlmUnavailableError: 文件不存在或非图像。
    """
    path = Path(doc_path)
    if not path.exists():
        raise VlmUnavailableError(f"VLM 输入文件不存在: {doc_path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _call_vlm(data_url: str) -> str:
    """调用 DeepSeek 官方 API(OpenAI 兼容)提取 Markdown。

    Args:
        data_url: 图像 base64 data URL。

    Returns:
        str: 模型返回的 Markdown 文本。

    Raises:
        VlmUnavailableError: API 调用失败。
    """
    from openai import OpenAI
    client = OpenAI(api_key=VLM_API_KEY, base_url=VLM_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            timeout=VLM_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - 统一转 VlmUnavailableError
        raise VlmUnavailableError(f"VLM API 调用失败: {exc}") from exc
    content = response.choices[0].message.content or ""
    return content


class VlmBackend:
    """VLM 解析后端: 输入图像路径, 输出归一化 ParseResult。"""

    name = _NAME

    def extract(self, doc_path: str) -> ParseResult:
        """用 DeepSeek VLM 解析图像并返回 Markdown。

        Args:
            doc_path: 图像文件路径。

        Returns:
            ParseResult: Markdown + 元信息(backend/vlm_provider/vlm_model)。

        Raises:
            VlmUnavailableError: VLM 不可用或调用失败(由链路上层进失败清单)。
        """
        if not is_available():
            raise VlmUnavailableError(f"VLM 不可用(未启用/未配 key/未装 openai): {doc_path}")
        data_url = _image_to_data_url(doc_path)
        markdown = _call_vlm(data_url)
        return ParseResult(
            markdown=markdown,
            format_meta={
                "backend": _NAME,
                "vlm_provider": VLM_PROVIDER,
                "vlm_model": VLM_MODEL,
            },
        )


def register() -> None:
    """注册 VLM 后端; 不可用时不注册(不写任何 fake)。"""
    if is_available():
        register_backend(_NAME, VlmBackend())
