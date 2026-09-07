"""subprocess_runner.py：重型/视觉后端子进程隔离执行。
隔离执行主要是重型文本链(docling/mineru/markitdown) + VLM这些"可能挂起/崩溃/吃大内存"的后端

子进程通过 JSON 回传 (markdown, format_meta);
超时/异常统一抛错, 由链路上层进失败清单(不落轻量)。

SUBPROCESS_EXECUTION=0(默认, 测试/调试)时退化为进程内直接调 resolve().extract(), 保证单元测试可用 fake 后端; =1(生产)时走子进程隔离。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from config import SUBPROCESS_EXECUTION, _PROJECT_ROOT

_WORKER = r'''
import json, sys
def _main():
    mode, backend_name, doc_path, project_root = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    sys.path.insert(0, project_root)  # 项目根由父进程显式传入, 不依赖 CWD(硬约束 #7)
    from indexing.parse_backends import resolve
    backend = resolve(backend_name)
    if backend is None:
        raise RuntimeError(f"{backend_name} 不可用")
    if mode == "probe":
        ver = getattr(backend, "version", lambda: "unknown")()
        print(json.dumps({"version": ver}, ensure_ascii=False))
        return
    result = backend.extract(doc_path)
    print(json.dumps({"markdown": result.markdown, "format_meta": result.format_meta},
                     ensure_ascii=False))
if __name__ == "__main__":
    _main()
'''


def _run_worker(mode: str, backend_name: str, doc_path: str, timeout_s: float) -> dict[str, Any]:
    """在子进程跑 worker, 超时 kill, 返回 JSON 载荷。

    Args:
        mode: "extract" 或 "probe"。
        backend_name: 后端名。
        doc_path: 文档路径。
        timeout_s: 超时(秒)。

    Returns:
        dict: worker 回传的 JSON 载荷。

    Raises:
        TimeoutError: 子进程超时。
        RuntimeError: 子进程非零退出或输出不可解析。
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _WORKER, mode, backend_name, str(doc_path), str(_PROJECT_ROOT)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{backend_name} 子进程超时({timeout_s}s): {doc_path}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"{backend_name} 子进程失败: {proc.stderr.strip() or 'unknown'}")
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"{backend_name} 子进程输出不可解析: {proc.stdout.strip()}") from exc
    return payload


def run_extract(backend_name: str, doc_path: str, timeout_s: float,
                use_subprocess: bool | None = None) -> tuple[str, dict[str, Any]]:
    """执行后端 extract, 返回 (markdown, format_meta)。

    Args:
        backend_name: 后端名。
        doc_path: 文档路径。
        timeout_s: 超时(秒)。
        use_subprocess: 是否走子进程; None 时取 config.SUBPROCESS_EXECUTION。

    Returns:
        tuple[str, dict]: (归一化 Markdown, 元信息)。

    Raises:
        RuntimeError/TimeoutError: 后端不可用/失败/超时(由链路上层进失败清单)。
    """
    if use_subprocess is None:
        use_subprocess = SUBPROCESS_EXECUTION
    if use_subprocess:
        payload = _run_worker("extract", backend_name, doc_path, timeout_s)
        return payload["markdown"], payload["format_meta"]
    from indexing.parse_backends import resolve
    backend = resolve(backend_name)
    if backend is None:
        raise RuntimeError(f"{backend_name} 不可用")
    result = backend.extract(doc_path)
    return result.markdown, dict(result.format_meta)


def probe(backend_name: str, timeout_s: float,
          use_subprocess: bool | None = None) -> str:
    """探活: 返回后端版本串(import/版本检查, 失败抛异常)。

    Args:
        backend_name: 后端名。
        timeout_s: 超时(秒)。
        use_subprocess: 是否走子进程; None 时取 config.SUBPROCESS_EXECUTION。

    Returns:
        str: 版本串。

    Raises:
        RuntimeError/TimeoutError: 后端不可用/失败/超时。
    """
    if use_subprocess is None:
        use_subprocess = SUBPROCESS_EXECUTION
    if use_subprocess:
        payload = _run_worker("probe", backend_name, "", timeout_s)
        return payload["version"]
    from indexing.parse_backends import resolve
    backend = resolve(backend_name)
    if backend is None:
        raise RuntimeError(f"{backend_name} 不可用")
    return getattr(backend, "version", lambda: "unknown")()
