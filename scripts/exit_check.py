"""会话退出检查脚本: 编排已有门禁 + 清洁检查 + 回归计数, 按会话类型三档运行。

规则:
    - full(默认): 干净退出, 全门禁硬检查 + 清洁/git, 退出码 0 全绿 / 1 有 FAIL
    - partial: 跨会话续做, 只跑单元测试 + git 摘要, 恒退 0(信息性)
    - info: 探索/计划型, 只输出状态摘要 + git 摘要, 恒退 0(给下个会话交接)

用法::

    uv run python scripts/exit_check.py                       # full(默认)
    uv run python scripts/exit_check.py --mode partial        # 跨会话续做
    uv run python scripts/exit_check.py --mode info           # 探索/计划型

交互触发: 在 Claude Code 提示符输入 /checkout (命令正文在 .claude/commands/checkout.md,
由 agent 判断会话任务类型 full/partial/info 后分级检查 + note 交接)

与上层的关系: 脚本只检查不修复不写状态;
交接 note 由 agent 按 /checkout 指令执行 `harness note` 留痕。
所有子进程经 _run 执行, cwd=项目根, 不依赖 CWD(硬约束 7)。
回归计数读 features.yaml 权威状态文件, 补 verify-all 只巡 passed 项的盲区。
"""
import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "harness" / "features.yaml"

# ---- 常量 ----
# 源码 TODO 扫描范围(目录 + 单文件), 排除数据/虚拟/文档目录
_SOURCE_DIRS = ("indexing", "retrieval", "eval", "agent", "harness", "obs", "scripts")
_SOURCE_FILES = ("config.py",)
# rglob 递归时跳过的非源码目录名(数据产物/缓存/虚拟环境)
_SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", "data", "results", "logs", "htmlcov", ".vector_cache", ".synthetic_cache"}
# harness/本脚本自产文件, git 检查提示时过滤(避免每次退出必 warn)
_SELF_PRODUCED_NAMES = {"features.yaml", "features.md", "PROGRESS.md"}
# 大写精确匹配, 避免命中占位符 "xxx"(如 eval/__init__.py 的 from eval.xxx import)
_TODO_PATTERN = re.compile(r"\b(?:TODO|FIXME|XXX|HACK)\b")
_TAIL_LINES = 12


@dataclass
class ExitReport:
    """full 运行汇总: 结果列表 + 退出码(0 无 FAIL / 1 FAIL)。"""
    exit_code: int
    results: list[tuple[str, bool | None, str]] = field(default_factory=list)  # (检查名, 是否通过, 明细)


# 单项检查结果: (passed, detail); passed=None 记 warn(提示级, 不影响退出码)
CheckOutcome = tuple[bool | None, str]
# 检查项: (显示名, 检查函数)
Check = tuple[str, Callable[[], CheckOutcome]]


# ---- 子进程与通用工具 ----
def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    """执行命令并返回 (退出码, 输出), 不抛异常。

    子进程输出显式按 utf-8 解码且 errors="replace", 并把 PYTHONIOENCODING 设为 utf-8,
    防 Windows GBK 控制台下不可编码字符使捕获崩溃。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output
    except subprocess.TimeoutExpired as exc:
        partial_output = exc.output or ""
        if isinstance(partial_output, bytes):
            partial_output = partial_output.decode("utf-8", "replace")
        return 1, f"timeout {timeout}s\n{_tail(str(partial_output), _TAIL_LINES)}"
    except OSError as exc:
        return 1, f"无法启动命令: {exc}"


def _tail(output: str, line_count: int) -> str:
    """取输出末尾 line_count 行, 每行前加 '  | ' 缩进(对齐 harness 诊断风格)。"""
    lines = output.strip().splitlines()[-line_count:]
    return "\n".join(f"  | {line}" for line in lines)


def _load_features(features_path: Path | None = None) -> dict | None:
    """读 features.yaml 为 dict; 缺失/损坏返回 None(信息性输出不因读取失败中断)。"""
    path = features_path or FEATURES_PATH
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, yaml.YAMLError):
        return None


def _count_regressed(features_path: Path | None = None) -> int:
    """统计 features.yaml 中 state=='regressed' 的项数(权威状态文件, 补 verify-all 盲区)。"""
    data = _load_features(features_path)
    if data is None:
        return 0
    items = data.get("items", [])
    return sum(1 for item in items if isinstance(item, dict) and item.get("state") == "regressed")


def _parse_porcelain(output: str) -> list[str]:
    """解析 git status --porcelain 输出为改动路径列表(重命名只取新路径)。"""
    paths: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[-1]
        if path:
            paths.append(path)
    return paths


def _is_self_produced(path: str) -> bool:
    """判断改动路径是否为本脚本/harness 自产文件(features.yaml/features.md/PROGRESS.md)。"""
    return Path(path).name in _SELF_PRODUCED_NAMES


def _collect_todo_hits(files: list[Path]) -> list[str]:
    """扫描给定文件, 返回含大写 TODO/FIXME/XXX/HACK 的 '路径:行号' 列表(小写占位符不误报)。"""
    hits: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _TODO_PATTERN.search(line):
                label = str(path)
                try:
                    label = str(path.relative_to(PROJECT_ROOT))
                except ValueError:
                    pass
                hits.append(f"{label}:{lineno}")
    return hits


def _build_source_files() -> list[Path]:
    """收集源码目录下待扫描 TODO 的 .py 文件(排除非源码子目录与本脚本自身)。"""
    files: list[Path] = []
    for directory_name in _SOURCE_DIRS:
        base = PROJECT_ROOT / directory_name
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if path.resolve() == Path(__file__).resolve():
                continue  # 跳过工具自身, 其源码含匹配模式定义
            if any(part in _SKIP_DIR_NAMES for part in path.relative_to(base).parts):
                continue
            files.append(path)
    for file_name in _SOURCE_FILES:
        path = PROJECT_ROOT / file_name
        if path.exists():
            files.append(path)
    return files


# ---- full 模式检查项 ----
def _check_deps_sync() -> CheckOutcome:
    """依赖一致: uv sync --check(环境与 lockfile 同步)。"""
    code, output = _run(["uv", "sync", "--check"], timeout=300)
    return code == 0, "" if code == 0 else _tail(output, _TAIL_LINES)


def _check_pytest() -> CheckOutcome:
    """测试全绿: 全量 pytest(addopts 自带 --cov, 生成的 .coverage 属 gitignore 常规产物)。"""
    code, output = _run(["uv", "run", "pytest", "-q"], timeout=900)
    return code == 0, "" if code == 0 else _tail(output, _TAIL_LINES)


def _check_verify_all() -> CheckOutcome:
    """治理巡检: harness verify-all 刷新 last_verify 并捕行为哈希漂移, 退出码作巡检参考。"""
    code, output = _run(["uv", "run", "python", "-m", "harness", "verify-all"], timeout=1800)
    return code == 0, "" if code == 0 else _tail(output, _TAIL_LINES)


def _check_regressed_count() -> CheckOutcome:
    """治理无回归-存量: 读 features.yaml 数 regressed 项(verify-all 只巡 passed, 存量回归靠此项兜底)。"""
    count = _count_regressed()
    detail = "" if count == 0 else f"存在 {count} 个 regressed 项, 需 reactivate 后 verify 恢复"
    return count == 0, detail


def _check_report() -> CheckOutcome:
    """进度快照: harness report 重新生成 PROGRESS.md(verify-all 后执行, 快照反映最终状态)。"""
    code, output = _run(["uv", "run", "python", "-m", "harness", "report"], timeout=120)
    return code == 0, "" if code == 0 else _tail(output, _TAIL_LINES)


def _check_status() -> CheckOutcome:
    """启动可用: harness status 健康视图(status 恒退 0, 输出健康度供退出前确认)。"""
    code, output = _run(["uv", "run", "python", "-m", "harness", "status"], timeout=120)
    return code == 0, "" if code == 0 else _tail(output, _TAIL_LINES)


def _check_tracked_bytecode() -> CheckOutcome:
    """清洁检查-编译残留: git 跟踪的 *.pyc/*.pyo 非空判 FAIL(__pycache__ 属 gitignore 不扫)。"""
    code, output = _run(["git", "ls-files"], timeout=60)
    found = [path for path in output.splitlines() if path.endswith(".pyc") or path.endswith(".pyo")]
    if found:
        return False, "git 跟踪了编译产物: " + ", ".join(found)
    return True, ""


def _check_source_todos() -> CheckOutcome:
    """清洁检查-源码残留: 源码目录大写 TODO/FIXME/XXX/HACK 命中记 warn(可能故意保留)。"""
    hits = _collect_todo_hits(_build_source_files())
    if hits:
        return None, "源码含 TODO/FIXME/XXX/HACK: " + ", ".join(hits[:20])
    return True, ""


def _check_git_dirty() -> CheckOutcome:
    """git 检查: 未提交改动(过滤自产文件)记 warn, 指示 agent 提交。"""
    code, output = _run(["git", "status", "--porcelain"], timeout=60)
    remaining = [path for path in _parse_porcelain(output) if not _is_self_produced(path)]
    if remaining:
        return None, "未提交改动: " + ", ".join(remaining[:20])
    return True, ""


# ---- 编排 ----
def build_checks() -> list[Check]:
    """构建 full 模式检查项: 复用已有门禁 + 回归计数 + 清洁/git, 列表顺序即执行顺序。"""
    return [
        ("依赖一致", _check_deps_sync),
        ("测试全绿", _check_pytest),
        ("治理无回归", _check_verify_all),
        ("治理无回归-存量", _check_regressed_count),
        ("进度快照", _check_report),
        ("启动可用", _check_status),
        ("清洁检查-编译残留", _check_tracked_bytecode),
        ("清洁检查-源码残留", _check_source_todos),
        ("git 检查", _check_git_dirty),
    ]


def _format_result(name: str, outcome: CheckOutcome) -> str:
    """格式化单项结果行为 [PASS]/[FAIL]/[warn] + 明细。"""
    passed, detail = outcome
    label = "[PASS]" if passed is True else ("[FAIL]" if passed is False else "[warn]")
    return f"{label} {name}  {detail}" if detail else f"{label} {name}"


def run_checks(checks: list[Check]) -> ExitReport:
    """逐项执行检查, 打印结果行, 汇总退出码(任一 FAIL 退 1, warn 不影响)。"""
    results: list[tuple[str, bool | None, str]] = []
    failed = 0
    for name, run_check in checks:
        try:
            outcome = run_check()
        except Exception as exc:  # noqa: BLE001 - 单项检查异常不应中断整个仪式
            outcome = (False, f"检查自身异常: {exc!r}")
        results.append((name, outcome[0], outcome[1]))
        print(_format_result(name, outcome))
        if outcome[0] is False:
            failed += 1
    print()
    if failed == 0:
        print("全部通过, 可以提交并退出")
    else:
        print(f"存在 {failed} 项 FAIL, 修复后重跑; warn 为提示级不影响退出")
    return ExitReport(exit_code=1 if failed else 0, results=results)


def _print_git_status_summary() -> None:
    """打印 git status 摘要(改动/未跟踪路径, 供 partial/info 交接)。"""
    code, output = _run(["git", "status", "--porcelain"], timeout=60)
    paths = _parse_porcelain(output)
    if paths:
        print("未提交/未跟踪路径:")
        for path in paths:
            print(f"  - {path}")
    else:
        print("工作树干净")


def _print_state_summary() -> None:
    """打印 features.yaml 状态分布与含 note 的项(info 模式交接信息)。"""
    data = _load_features()
    if data is None:
        print("无法读取 features.yaml")
        return
    items = data.get("items", [])
    distribution: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            state = str(item.get("state", "unknown"))
            distribution[state] = distribution.get(state, 0) + 1
    print("状态分布:")
    for state, count in sorted(distribution.items()):
        print(f"  {state}: {count}")
    noted = [item for item in items if isinstance(item, dict) and item.get("note")]
    if noted:
        print("含交接 note 的项:")
        for item in noted:
            print(f"  {item.get('id')}: {item.get('note')}")


def run_partial() -> int:
    """partial 模式: 跨会话续做, 只跑单元测试 + git 摘要, 恒退 0(信息性, 不拦退出)。"""
    print("== partial 模式: 跨会话续做 ==")
    code, output = _run(["uv", "run", "pytest", "tests/unit", "-q"], timeout=900)
    print(f"单元测试: 退出码 {code}")
    if code != 0:
        print(_tail(output, _TAIL_LINES))
    _print_git_status_summary()
    print("partial 为信息性, 不判失败; 请执行 harness note 留痕后退出")
    return 0


def run_info() -> int:
    """info 模式: 探索/计划型, 只输出状态摘要 + git 摘要, 恒退 0(供下个会话交接)。"""
    print("== info 模式: 探索/计划型交接 ==")
    _print_state_summary()
    _print_git_status_summary()
    print("info 不设门禁; 请执行 harness note 留痕, 供下个会话恢复上下文")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: --mode full|partial|info, 默认 full; full 退 0/1, partial/info 恒退 0。"""
    parser = argparse.ArgumentParser(prog="exit_check", description="会话退出检查")
    parser.add_argument("--mode", choices=("full", "partial", "info"), default="full")
    args = parser.parse_args(argv)
    if args.mode == "partial":
        return run_partial()
    if args.mode == "info":
        return run_info()
    summary = run_checks(build_checks())
    return summary.exit_code


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
