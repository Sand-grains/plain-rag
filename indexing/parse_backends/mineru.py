"""mineru.py：MinerU 重型解析后端 adapter。
以独立 subprocess 跑 MinerU 的 CLI(并非薄封装), 支持可靠超时

在函数内做 import 守卫: 未安装 mineru 且未配独立发行版时 is_available=False, 不注册后端。

MinerU 需本地模型权重 + 较充分算力; 模型未就绪/GPU 不足时子进程返回非零, extract() 抛 MinerUError
(异常由链路上层捕获并降级到 MarkItDown)。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

from config import MINERU_PYTHON, MINERU_ROOT
from indexing.parse_backends import ParseResult, register_backend
from indexing.parse_backends._normalize import normalize_to_contract

_NAME = "mineru"


class MinerUError(RuntimeError):
    """MinerU 解析失败(子进程异常 / 未产出 MD / 超时)。"""


def _python_executable() -> str:
    """返回用于调 mineru CLI 的 python: 配了 MINERU_PYTHON 用独立发行版, 否则项目 venv。"""
    return MINERU_PYTHON or sys.executable


def _mineru_env() -> dict[str, str] | None:
    """构造独立发行版运行环境(复刻一键启动Webui.bat 的环境变量)。

    Returns:
        dict | None: 配了 MINERU_ROOT 时返回环境变量覆盖, 否则 None(用项目 venv 默认环境)。
    """
    if not MINERU_ROOT:
        return None
    root = Path(MINERU_ROOT)
    return {
        "PYTHONPATH": str(root / "src"),
        "HF_HOME": str(root / "models"),
        "HF_HUB_CACHE": str(root / "models" / "huggingface"),
        "MODELSCOPE_CACHE": str(root / "models" / "modelscope"),
        "MINERU_TOOLS_CONFIG_JSON": str(root / "config" / "mineru.json"),
        "MINERU_API_OUTPUT_ROOT": str(root / "output"),
        "CUDA_PATH": str(root / "cuda"),
        "HF_HUB_OFFLINE": "1",
        "MINERU_MODEL_SOURCE": "modelscope",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }


def _ensure_config() -> None:
    """确保独立发行版 config/mineru.json 已生成(缺则跑 src/generate_config.py)。"""
    if not MINERU_ROOT:
        return
    root = Path(MINERU_ROOT)
    config_file = root / "config" / "mineru.json"
    if config_file.exists():
        return
    generate_script = root / "src" / "generate_config.py"
    if not generate_script.exists():
        return
    subprocess.run(
        [_python_executable(), str(generate_script)],
        capture_output=True, text=True, timeout=120, check=False,
    )


def is_available() -> bool:
    """mineru 是否可用: 当前解释器可导入, 或已配独立发行版 python。"""
    if MINERU_PYTHON:
        return True
    return importlib.util.find_spec("mineru") is not None


def version() -> str:
    """返回 mineru 版本(按实际解释器读取); 不可用返回 'not-installed'。"""
    if not is_available():
        return "not-installed"
    if MINERU_PYTHON:
        try:
            completed = subprocess.run(
                [MINERU_PYTHON, "-c",
                 "import importlib.metadata as m; print(m.version('mineru'))"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout.strip()
        except subprocess.SubprocessError:
            pass
        return "standalone-unknown"
    from importlib.metadata import version as _version
    return _version("mineru")


def _run_cli(doc_path: str, output_dir: str, timeout: float) -> None:
    """以子进程方式运行 mineru CLI(`mineru.cli.client`)。

    Args:
        doc_path: 待解析文档路径。
        output_dir: 子进程写出目录。
        timeout: 超时秒数(主文档条例八: 可靠超时)。

    Raises:
        MinerUError: 子进程返回非零或超时。
    """
    cmd = [_python_executable(), "-m", "mineru.cli.client", "-p", doc_path, "-o", output_dir]
    _ensure_config()
    env = _mineru_env()
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise MinerUError(f"{_NAME} 解析超时: {doc_path}") from exc
    if completed.returncode != 0:
        hint = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise MinerUError(f"{_NAME} 子进程失败(如模型未就绪/算力不足): {doc_path}\n{hint}")


def _discover_markdown(output_dir: str, doc_path: str) -> str:
    """从 mineru 输出目录找出主 MD 文件并读取内容。

    Args:
        output_dir: mineru 写出目录。
        doc_path: 原文档路径(用于按 stem 优先匹配)。

    Returns:
        str: 主 Markdown 文本。

    Raises:
        MinerUError: 未发现任何 .md 产物。
    """
    root = Path(output_dir)
    candidates = sorted(root.rglob("*.md"))
    if not candidates:
        raise MinerUError(f"{_NAME} 未产出 Markdown: {doc_path}")
    stem = Path(doc_path).stem
    # 优先生成目录下与文档同名的 MD; 否则取内容最长的(主正文通常最长)。
    primary = next((c for c in candidates if c.stem == stem), max(candidates, key=lambda c: c.stat().st_size))
    return primary.read_text(encoding="utf-8", errors="replace")


class MinerUBackend:
    """MinerU 解析后端: 输入文档路径, 输出归一化 ParseResult。"""

    name = _NAME

    def __init__(self, timeout: float = 600.0) -> None:
        """构建 MinerU 后端。

        Args:
            timeout: 单文档子进程超时秒数(默认 600)。
        """
        self.timeout = timeout

    def extract(self, doc_path: str) -> ParseResult:
        """用 MinerU CLI 解析文档并归一化。

        Args:
            doc_path: 文档路径(pdf/html/docx/pptx)。

        Returns:
            ParseResult: 归一化 Markdown + 后端/版本元信息。

        Raises:
            MinerUError: 子进程失败 / 超时 / 未产出 MD(由链路上层捕获降级)。
        """
        if not is_available():
            raise MinerUError(f"{_NAME} 未安装, 无法解析 {doc_path}")
        with tempfile.TemporaryDirectory() as tmp:
            _run_cli(doc_path, tmp, self.timeout)
            markdown = _discover_markdown(tmp, doc_path)
        return ParseResult(
            markdown=normalize_to_contract(markdown),
            format_meta={"backend": _NAME, "version": version()},
        )


def register() -> None:
    """注册 MinerU 后端; 依赖未安装时不注册(不写任何 fake)。"""
    if is_available():
        register_backend(_NAME, MinerUBackend())
