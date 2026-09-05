"""indexing/parse_backends: 重型解析后端注册表和编排器
定义抽象接口、管理注册与门控、包一层运行时健壮性
对外提供"重型文本链"和"质量驱动文本管线"两个解析入口

Marker 因许可/依赖冲突移出默认链, LlamaParse 排除
默认全关, 门控关或依赖未装时 resolve 返回 None（未启用/降级信号），不写任何 fake 实现。

"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from config import (
    CIRCUIT_BREAKER_MAX_FAILURES,
    DOCLING_ENABLED,
    DOCLING_TIMEOUT_S,
    HEAVY_ENABLED,
    LLAMAPARSE_ENABLED,
    MARKER_ENABLED,
    MARKITDOWN_ENABLED,
    MARKITDOWN_TIMEOUT_S,
    MINERU_ENABLED,
    MINERU_TIMEOUT_S,
    VLM_ENABLED,
    VLM_TIMEOUT_S,
)

# 重型后端注册表：parser 名 -> 后端实例（resolve/enabled_backends 首次调用时惰性填充）
_BACKEND_REGISTRY: dict[str, "ParseBackend"] = {}

# 已知后端全集的注册表
# 门控映射：parser 名 -> 开关(重型文本链受 HEAVY_ENABLED 总开关约束; VLM 独立)
# 注: ColPali 是独立视觉检索管线(非 ParseBackend), 不注册进本表, 由 loader 直接调用 colpali.is_available()
_BACKEND_GATES: dict[str, bool] = {
    "docling": HEAVY_ENABLED and DOCLING_ENABLED,
    "mineru": HEAVY_ENABLED and MINERU_ENABLED,
    "marker": HEAVY_ENABLED and MARKER_ENABLED,
    "markitdown": HEAVY_ENABLED and MARKITDOWN_ENABLED,
    "llamaparse": LLAMAPARSE_ENABLED,
    "vlm": VLM_ENABLED,
}

# 各后端单文档超时
_BACKEND_TIMEOUTS: dict[str, float] = {
    "docling": DOCLING_TIMEOUT_S,
    "mineru": MINERU_TIMEOUT_S,
    "markitdown": MARKITDOWN_TIMEOUT_S,
    "vlm": VLM_TIMEOUT_S,
}

# 运行时健壮性单例（熔断 + run 级统计）
from indexing.parse_backends.robustness import CircuitBreaker, RunStats, classify_error

circuit_breaker = CircuitBreaker(max_failures=CIRCUIT_BREAKER_MAX_FAILURES)
run_stats = RunStats()

# 惰性注册守卫：只初始化一次
_REGISTRY_INITIALIZED = False


@dataclass
class ParseResult:
    """重型后端解析结果。"""
    markdown: str = ""  # 归一化 Markdown 输出(v2 接入时消费)
    format_meta: dict[str, Any] = field(default_factory=dict)  # 格式元信息(v2 接入时消费)


class ParseBackend(Protocol):
    """重型后端抽象接口：输入文档路径，输出解析结果。"""

    def extract(self, doc_path: str) -> ParseResult: ...


def register_backend(name: str, backend: "ParseBackend") -> None:
    """按 parser 名注册后端到注册表（adapter 在 register() 中调用）。"""
    _BACKEND_REGISTRY[name] = backend


def _ensure_registered() -> None:
    """惰性初始化注册表：首次调用时让各 adapter 自行注册。

    依赖未装的 adapter 不注册（resolve 返回 None），保证"依赖缺失自动禁用"。
    """
    global _REGISTRY_INITIALIZED
    if _REGISTRY_INITIALIZED:
        return
    # 惰性导入避免顶层拉入重型依赖; 各 register() 内部做 importlib 守卫
    from indexing.parse_backends import docling, markitdown, mineru, vlm
    docling.register()
    mineru.register()
    markitdown.register()
    vlm.register()
    _REGISTRY_INITIALIZED = True


def is_enabled(name: str) -> bool:
    """查询某后端是否被配置门控开启（默认全关）。"""
    return _BACKEND_GATES.get(name, False)


def resolve(name: str) -> ParseBackend | None:
    """解析后端：已注册且门控开启才返回实例；否则返回 None（未启用信号）。

    惰性触发注册表初始化（仅当门控开启后才真正 import 各 adapter 依赖）。
    """
    if not is_enabled(name):
        return None
    _ensure_registered()
    return _BACKEND_REGISTRY.get(name)


def enabled_backends() -> list[str]:
    """返回当前门控开启且已注册的后端名列表（用于诊断/日志）。"""
    if any(_BACKEND_GATES.values()):
        _ensure_registered()
    return [name for name, enabled in _BACKEND_GATES.items() if enabled]


def backend_version(name: str) -> str:
    """按后端名取版本号(溯源用); 未启用/不可用时返回 'unknown'。"""
    backend = resolve(name)
    if backend is None:
        return "unknown"
    return getattr(backend, "version", lambda: "unknown")()


def reset_runtime() -> None:
    """清空熔断与 run 级统计(新 run 开始时调用)。"""
    circuit_breaker.reset()
    run_stats.reset()


def _run_backend(name: str, doc_path: str) -> tuple[str, dict[str, Any]]:
    """执行单个后端 extract(经 subprocess_runner + 熔断 + 统计)。

    熔断开启时直接抛 RuntimeError(不执行);
    成功/失败都记入 run_stats, 失败按分类记入熔断并重抛(由链路上层降级/进失败清单)。

    Args:
        name: 后端名。
        doc_path: 文档路径。

    Returns:
        tuple[str, dict]: (归一化 Markdown, 元信息)。

    Raises:
        RuntimeError/TimeoutError: 后端不可用/失败/超时/已熔断。
    """
    from indexing.parse_backends.subprocess_runner import run_extract
    if circuit_breaker.is_open(name):
        raise RuntimeError(f"{name} 已熔断(本次运行禁用)")
    timeout_s = _BACKEND_TIMEOUTS.get(name, 60.0)
    start = time.monotonic()
    try:
        markdown, meta = run_extract(name, doc_path, timeout_s)
        circuit_breaker.record_success(name)
        run_stats.record(name, time.monotonic() - start, ok=True)
        return markdown, meta
    except Exception as exc:  # noqa: BLE001 - 统一分类记熔断后重抛
        category = classify_error(exc, name)
        circuit_breaker.record_failure(name, category)
        run_stats.record(name, time.monotonic() - start, ok=False, reason=category)
        raise


def heavy_chain_extract(doc_path: str) -> tuple[str, dict[str, Any]]:
    """默认重型文本链解析：Docling -> MinerU -> MarkItDown。

    任一后端门控关闭/依赖未装/解析失败时自动落到链内下一个后端；
    全链失败则抛 RuntimeError（调用方负责进失败清单, 不允许落轻量）。

    Args:
        doc_path: 文档路径。

    Returns:
        tuple[str, dict]: (归一化 Markdown, 使用的后端元信息 {backend, version})。

    Raises:
        RuntimeError: 文本链全部后端不可用或均解析失败。
    """
    _ensure_registered()
    chain = ("docling", "mineru", "markitdown")
    last_error: Exception | None = None
    last_backend: str | None = None
    for name in chain:
        if resolve(name) is None or circuit_breaker.is_open(name):
            continue
        last_backend = name
        try:
            markdown, meta = _run_backend(name, doc_path)
            if markdown:
                return markdown, meta
            last_error = RuntimeError(f"{name} 产出为空")
        except Exception as exc:  # noqa: BLE001 - 链式兜底, 失败降级
            last_error = exc
    enabled_names = [name for name in chain if is_enabled(name)]
    detail = f"失败清单, 不允许落轻量: {doc_path}"
    if last_error is not None:
        detail = f"{detail}({type(last_error).__name__}: {last_error})"
    error = RuntimeError(f"重型文本链({enabled_names or '全部关闭'})不可用, {detail}")
    error.backend_name = last_backend or "docling"  # 溯源: 记录真实失败后端
    raise error


def _run_whole_doc_chain(doc_path: str, original_table_count: int | None,
                         expected_code_blocks: int | None) -> tuple[str, dict[str, Any]]:
    """整篇质量驱动文本链: Docling -> MinerU -> MarkItDown, 按质量代理信号降级。

    与 heavy_chain_extract 的区别: 不只按异常/空产出降级, 还按质量代理信号(表格丢失/代码 fence 丢失/标题不连续)判定是否换后端。
    质量达标即停。

    Args:
        doc_path: 文档路径。
        original_table_count: 原始表格数(precheck 阶段统计, 可判时传入)。
        expected_code_blocks: 预期代码块数(可判时传入)。

    Returns:
        tuple[str, dict]: (归一化 Markdown, 元信息 {backend, version, quality, retry_stats})。

    Raises:
        RuntimeError: 全链不可用或均未通过质量判定(调用方进失败清单, 不落轻量)。
    """
    from indexing.parse_backends.quality import quality_ok
    _ensure_registered()
    chain = ("docling", "mineru", "markitdown")
    retry_stats: dict[str, Any] = {"attempted": [], "quality_reasons": {}}
    last_error: Exception | None = None
    last_backend: str | None = None
    for name in chain:
        if resolve(name) is None or circuit_breaker.is_open(name):
            continue
        last_backend = name
        retry_stats["attempted"].append(name)
        try:
            markdown, meta = _run_backend(name, doc_path)
            ok, reasons, signals = quality_ok(
                markdown, original_table_count, expected_code_blocks)
            retry_stats["quality_reasons"][name] = reasons
            if ok:
                meta = dict(meta)
                meta["quality"] = signals
                meta["retry_stats"] = retry_stats
                return markdown, meta
            last_error = RuntimeError(f"{name} 质量未达标: {reasons}")
        except Exception as exc:  # 链式兜底, 失败降级
            last_error = exc
            retry_stats["quality_reasons"][name] = [f"{type(exc).__name__}"]
    enabled_names = [name for name in chain if is_enabled(name)]
    detail = f"失败清单, 不允许落轻量: {doc_path}"
    if last_error is not None:
        detail = f"{detail}({type(last_error).__name__}: {last_error})"
    error = RuntimeError(f"文本管线({enabled_names or '全部关闭'})不可用, {detail}")
    error.backend_name = last_backend or "docling"  # 溯源: 记录真实失败后端
    raise error


def parse_text_pipeline(doc_path: str, original_table_count: int | None = None,
                        expected_code_blocks: int | None = None) -> tuple[str, dict[str, Any]]:
    """质量驱动的文本管线: 整篇链 -> 未达标则页级重试 + VLM 修正。

    先跑整篇质量驱动链(Docling -> MinerU -> MarkItDown);
    整篇未达标时进入页级: PDF/PPTX 按页/slide 拆分逐页重试, 仍不达标页 VLM 修正, 跨页合并;
    DOCX/HTML 无稳定页边界, 整篇重试 + 整篇 VLM 修正。

    Args:
        doc_path: 文档路径。
        original_table_count: 原始表格数(precheck 阶段统计, 可判时传入)。
        expected_code_blocks: 预期代码块数(可判时传入)。

    Returns:
        tuple[str, dict]: (归一化 Markdown, 元信息
        {backend, version, quality, retry_stats, page_retry_stats, cross_page_merge_*, vlm_correction_applied})。

    Raises:
        RuntimeError: 整篇链与页级修正均未产出达标 Markdown(调用方进失败清单, 不落轻量)。
    """
    try:
        return _run_whole_doc_chain(doc_path, original_table_count, expected_code_blocks)
    except RuntimeError:
        pass  # 整篇未达标, 进入页级混合重试 + VLM 修正
    from indexing.parse_backends.paged_pipeline import parse_text_pipeline_paged
    from indexing.parse_backends.quality import quality_ok

    def _chain(page_path: str) -> tuple[str, dict[str, Any]]:
        # 整篇重试(DOCX/HTML 无页边界): 保留整篇表格/代码基准
        return _run_whole_doc_chain(page_path, original_table_count, expected_code_blocks)

    def _page_chain(page_path: str) -> tuple[str, dict[str, Any]]:
        # 页级重试: 单页表格/代码数远小于整篇, 关闭表格/代码保留率判定, 避免系统性假性降级(M2)
        return _run_whole_doc_chain(page_path, None, None)

    def _vlm_extract(image_path: str) -> tuple[str, dict[str, Any]]:
        # 经 _run_backend 走门控 + 熔断 + 统计: vlm 未启用/不可用/已熔断时抛错, 由页级管线跳过 VLM 修正
        return _run_backend("vlm", image_path)

    try:
        markdown, meta = parse_text_pipeline_paged(
            doc_path, original_table_count, expected_code_blocks,
            chain=_chain, page_chain=_page_chain, quality_fn=quality_ok, vlm_extract=_vlm_extract)
        if markdown.strip():
            return markdown, meta
    except Exception as exc:  # noqa: BLE001 - 页级修正失败, 统一转"不允许落轻量"
        error = RuntimeError(
            f"文本管线整篇与页级修正均未达标, 不允许落轻量: {doc_path}"
            f"({type(exc).__name__}: {exc})")
        error.backend_name = getattr(exc, "backend_name", None)  # 溯源: 透传真实失败后端
        raise error from exc
    error = RuntimeError(f"文本管线整篇与页级修正均未达标, 不允许落轻量: {doc_path}")
    raise error
