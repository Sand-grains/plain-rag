"""colpali.py：ColPali/ColQwen 视觉检索薄封装。

ColPali 管线 = 页面/图像 -> 现成 ColPali/ColQwen 编码 -> multi-vector 索引 -> 查询时 MaxSim 检索。
012 只做现成库/服务薄封装, 不实现训练; 索引与检索接口单独登记, 不与文本链混用。

import 守卫在函数内: 未装 colpali_engine 或未配 COLPALI_MODEL_PATH 时 is_available=False;
ColPali 不可用时, 触发 ColPali 的文档进失败清单(不静默降级为文本链)。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from config import COLPALI_ENABLED, COLPALI_MODEL_PATH, COLPALI_TIMEOUT_S

_NAME = "colpali"


class ColPaliUnavailableError(RuntimeError):
    """ColPali 不可用(未启用/未装依赖/未配权重路径)。"""


def is_available() -> bool:
    """ColPali 是否可用: 门控开启 + 已配权重路径 + colpali_engine 可导入。"""
    if not COLPALI_ENABLED or not COLPALI_MODEL_PATH:
        return False
    return importlib.util.find_spec("colpali_engine") is not None


def version() -> str:
    """返回 ColPali 权重路径(版本冻结用); 不可用返回 'not-installed'。"""
    return COLPALI_MODEL_PATH if is_available() else "not-installed"


def triggered(precheck_result) -> bool:
    """ColPali 触发判定: 复用 precheck 的 colpali_triggered(图表面积/表格占比)。

    Args:
        precheck_result: PrecheckResult 实例。

    Returns:
        bool: 是否触发 ColPali 视觉检索管线。
    """
    return bool(getattr(precheck_result, "colpali_triggered", False))


def encode_image(image_path: str) -> list[list[float]]:
    """把图像编码为 multi-vector 嵌入(薄封装 colpali_engine)。

    Args:
        image_path: 图像文件路径。

    Returns:
        list[list[float]]: 每 patch 一个向量(多向量, 供 MaxSim 检索)。

    Raises:
        ColPaliUnavailableError: ColPali 不可用或编码失败。
    """
    if not is_available():
        raise ColPaliUnavailableError(
            f"ColPali 不可用(未启用/未装 colpali_engine/未配 COLPALI_MODEL_PATH): {image_path}")
    path = Path(image_path)
    if not path.exists():
        raise ColPaliUnavailableError(f"ColPali 输入文件不存在: {image_path}")
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor
        from colpali_engine.utils.processing_utils import process_images
        import torch
        model = ColPali.from_pretrained(COLPALI_MODEL_PATH, torch_dtype=torch.bfloat16).eval()
        processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_PATH)
        images = [path]
        batch = process_images(images, processor)
        with torch.no_grad():
            embeddings = model(**batch)
        return embeddings[0].tolist()
    except Exception as exc:  # noqa: BLE001 - 统一转 ColPaliUnavailableError
        raise ColPaliUnavailableError(f"ColPali 编码失败: {exc}") from exc
