"""验证器: 执行功能项验证命令
退出码 + junitxml 双重判定, 是 features.yaml 条例的 active -> passed 唯一路径

gate 和 e2e 是 Features.yaml 里每个功能项的两个同级验证命令字段
gate: 默认跑, 本质是验证"实现逻辑正确"的 pytest 测试 (附加 --junitxml 与 --cov-report=xml 采集证据)
e2e: 显式触发的、验证"全链路在真实模型上跑通"的重验证 (仅 --e2e 显式触发, 是 eval/benchmark(计费, 慢), 以退出码判定)
另外, e2e 可选 precheck 进行前置自检 (如 eval.runner --precheck), 未满足则报"前置未满足"不跑 e2e。


关键机制:
    - 通过 = 退出码 0 且 junitxml 测试数 > 0, 不单靠退出码 5 的语义(无测试 = 无验证)
    - e2e/precheck 仅看退出码(test_count=None), 不附加 junitxml/coverage, 不校验 gate 前缀白名单
    - 超时(默认 60s, per-item 可覆盖)视为失败
    - gate 前缀白名单(uv run pytest / python -m pytest / pytest), 拒绝任意命令当 gate
    - gate 自带 --cov= 作用域时中和全局 addopts 的裸 --cov, 让覆盖率按模块计(可选 --cov-fail-under 作为覆盖率门禁)
    - 验证证据(command/测试名/测试数/coverage/耗时/mode)进 VerificationResult, 供报告器与可疑项标记

与上层的关系: cli 调 verify() 拿结果, 再经 registry.record_verification 落状态与证据。
"""
import shlex
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

from harness.core.errors import HarnessError
from harness.core.models import FeatureItem

# gate 命令前缀白名单: 可扩展允许列表, 防"任意命令当 gate"
DEFAULT_GATE_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("uv", "run", "pytest"),
    ("python", "-m", "pytest"),
    ("pytest",),
)


@dataclass(frozen=True)
class VerificationResult:
    """单次验证的完整证据: 命令/退出码/输出/测试名/测试数/coverage/耗时/结论。"""
    item_id: str                         # 功能项编号, 标识本次验证对象
    command: str                         # 实际执行的验证命令(不含附加的 junitxml/coverage 参数), 供证据回放
    exit_code: int | None                # 命令退出码, 超时场景为 None
    timed_out: bool                      # 是否超时, 超时一律视为验证失败
    output: str                          # 合并的 stdout+stderr 输出, 供失败诊断与可疑项判定
    test_names: list[str]                # classname::name 格式的测试名列表, e2e/precheck 路径为空
    test_count: int | None               # junit 解析出的测试数, e2e/precheck 为 None(仅看退出码)
    coverage: float | None               # 覆盖率百分比(0-100), 非 pytest 路径为 None
    duration_seconds: float              # 命令耗时(秒), 供巡检与超时分析
    passed: bool                         # 最终通过结论, gate 为双重判定, e2e/precheck 为仅退出码判定
    precheck_passed: bool | None = None  # None=gate; False=precheck 未满足; True=e2e 正常跑完
    mode: str = "gate"                   # "gate" / "e2e"

    def to_evidence(self) -> dict:
        """压缩成写入 features.yaml 的 last_verify 证据。"""
        return {
            "exit_code": self.exit_code,
            "tests": self.test_count,
            "coverage": self.coverage,
            "duration": round(self.duration_seconds, 2),
            "mode": self.mode,
        }


class Verifier:
    """门禁/完整验证命令执行器, 返回可回放的 VerificationResult。"""

    def __init__(self, timeout: float = 60.0,
                 gate_prefixes: tuple[tuple[str, ...], ...] = DEFAULT_GATE_PREFIXES) -> None:
        self._timeout = timeout
        self._gate_prefixes = gate_prefixes

    def verify(self, item: FeatureItem, e2e: bool = False) -> VerificationResult:
        """验证功能项: 跑 gate(默认)或 e2e(--e2e), 返回证据与结论。

        e2e 路径: 若 item.precheck 非空则先跑前置自检, 非零退出 → precheck_passed=False 返回(不跑 e2e);
        前置通过后跑 item.e2e, 仅退出码判定。e2e 与 precheck 均绕过 gate
        白名单(白名单只在 gate 分支生效)。

        Args:
            item: 待验证功能项。
            e2e: True 时跑 item.e2e(先跑 item.precheck); 该字段为空时拒绝执行。

        Returns:
            VerificationResult: 验证证据(命令输出/测试名/测试数/coverage/通过与否)。

        Raises:
            HarnessError: gate 前缀不在白名单, 或 --e2e 但 e2e 字段为空。
        """
        if e2e:
            return self._verify_e2e(item)
        return self._verify_gate(item)

    def _verify_gate(self, item: FeatureItem) -> VerificationResult:
        """gate 路径: 附加 junitxml/coverage 证据, 双重判定, 白名单校验。"""
        command = self._resolve_command(item, e2e=False)
        tokens = shlex.split(command)
        self._check_gate_prefix(tokens)
        timeout = item.timeout if item.timeout else self._timeout

        with tempfile.TemporaryDirectory(prefix="harness_verify_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            junit_path = tmp_path / "junit.xml"
            coverage_path = tmp_path / "coverage.xml"
            augmented = list(tokens)
            # 覆盖配置隔离到临时 data_file: 防存量 .coverage 污染 gate 判定
            augmented += [
                f"--junitxml={junit_path}",
                f"--cov-report=xml:{coverage_path}",
                f"--cov-config={self._write_cov_config(tmp_path)}",
            ]
            if any(token.startswith("--cov=") for token in tokens):
                # 中和 addopts 的裸 --cov, 让 gate 自带 --cov= 作用域生效
                augmented += ["-o", "addopts="]

            started = time.monotonic()
            exit_code, timed_out, output = self._run(augmented, timeout)
            duration_seconds = time.monotonic() - started

            test_count, test_names = self._parse_junit(junit_path)
            coverage = self._parse_coverage(coverage_path)
            passed = self._is_passed(exit_code, test_count, timed_out)
            return VerificationResult(
                item_id=item.id,
                command=command,
                exit_code=exit_code,
                timed_out=timed_out,
                output=output,
                test_names=test_names,
                test_count=test_count,
                coverage=coverage,
                duration_seconds=duration_seconds,
                passed=passed,
                precheck_passed=None,
                mode="gate",
            )

    def _verify_e2e(self, item: FeatureItem) -> VerificationResult:
        """e2e 路径: precheck 前置自检 → 跑 item.e2e, 仅凭退出码判定, 不附加 junit/coverage。"""
        command = self._resolve_command(item, e2e=True)
        tokens = shlex.split(command)
        timeout = item.timeout if item.timeout else self._timeout

        if item.precheck:
            precheck_tokens = shlex.split(item.precheck)
            started = time.monotonic()
            exit_code, timed_out, output = self._run(precheck_tokens, timeout)
            duration_seconds = time.monotonic() - started
            if exit_code != 0 or timed_out:
                return VerificationResult(
                    item_id=item.id,
                    command=item.precheck,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    output=output,
                    test_names=[],
                    test_count=None, # e2e 不附加junitxml/coverage,所以只能显式填"此证据缺席"的哨兵值
                    coverage=None, # e2e 不采集junit/coverage"这个决策在每个构造点都可见
                    duration_seconds=duration_seconds,
                    passed=False,
                    precheck_passed=False,
                    mode="e2e",
                )

        started = time.monotonic()
        exit_code, timed_out, output = self._run(tokens, timeout)
        duration_seconds = time.monotonic() - started
        passed = not timed_out and exit_code == 0
        return VerificationResult(
            item_id=item.id,
            command=command,
            exit_code=exit_code,
            timed_out=timed_out,
            output=output,
            test_names=[],
            test_count=None,
            coverage=None,
            duration_seconds=duration_seconds,
            passed=passed,
            precheck_passed=True,
            mode="e2e",
        )

    def _write_cov_config(self, tmp_path: Path) -> Path:
        """写临时 coveragerc: data_file 隔离到本次验证的临时目录。

        不加 source, 避免继承 pyproject 的全局 [tool.coverage.run] source,
        保证 gate 自带 --cov= 作用域是唯一测量源, 且不与存量 .coverage 合并。
        """
        config_path = tmp_path / "coveragerc"
        config_path.write_text(
            f"[run]\ndata_file = {(tmp_path / 'cov.data').as_posix()}\n",
            encoding="utf-8",
        )
        return config_path

    def _resolve_command(self, item: FeatureItem, e2e: bool) -> str:
        """解析待执行命令: gate 分支取 item.gate, e2e 分支取 item.e2e。

        带出 --e2e 但 item.e2e 为空时在此抛 HarnessError 拒绝, 是 e2e 字段必填的强制点。

        Args:
            item: 待验证功能项, 从中取 gate/e2e 命令字段。
            e2e: True 走 e2e 分支(校验 item.e2e 非空), False 走 gate 分支。

        Returns:
            str: 实际要执行的命令字符串。

        Raises:
            HarnessError: e2e=True 但 item.e2e 为空。
        """
        if not e2e:
            return item.gate
        if not item.e2e:
            raise HarnessError(f"{item.id} 未配置 e2e 验证命令, 不能 --e2e")
        return item.e2e

    def _check_gate_prefix(self, tokens: list[str]) -> None:
        """校验命令前辍命中白名单, 否则抛 HarnessError(防任意命令当 gate)。

        Args:
            tokens: shlex.split 后的命令词元, 取前 N 个与白名单前缀比对。

        Raises:
            HarnessError: 命令前缀不在白名单。
        """
        for prefix in self._gate_prefixes:
            if tuple(tokens[: len(prefix)]) == prefix:
                return
        raise HarnessError(
            f"gate 命令前缀不在白名单: {' '.join(tokens[:3])}; "
            "只允许 uv run pytest / python -m pytest / pytest"
        )

    def _run(self, command: list[str], timeout: float) -> tuple[int | None, bool, str]:
        """执行子进程并合并采集 stdout+stderr: 返回 (退出码, 是否超时, 输出)。

        超时返回 exit_code=None 与 timed_out=True, 并带出超时前的部分输出作为诊断证据。

        Args:
            command: 待执行的完整命令词元列表(不经 shell)。
            timeout: 超时秒数, 超时视为失败。

        Returns:
            tuple[int | None, bool, str]: (退出码, 是否超时, 合并后的 stdout+stderr 输出)。
        """
        try:
            completed = subprocess.run(
                command, capture_output=True, timeout=timeout, shell=False, check=False,
            )
            output = completed.stdout.decode("utf-8", errors="replace")
            if completed.stderr:
                output += completed.stderr.decode("utf-8", errors="replace")
            return completed.returncode, False, output
        except subprocess.TimeoutExpired as exc:
            partial = exc.output or b""
            return None, True, partial.decode("utf-8", errors="replace")

    def _parse_junit(self, junit_path: Path) -> tuple[int | None, list[str]]:
        """解析 pytest 的 junitxml: 返回 (测试数, 测试名列表), 无文件时 (None, [])。

        根节点是 <testsuites>, tests 计数在子 <testsuite> 上, 逐 suite 累加防多包并跑漏计。

        Args:
            junit_path: pytest --junitxml 产出的 JUnit XML 文件路径。

        Returns:
            tuple[int | None, list[str]]: (测试总数, classname::name 格式的测试名列表)。
        """
        if not junit_path.exists():
            return None, []
        root = ElementTree.parse(junit_path).getroot()
        # pytest 的 junit 根节点是 <testsuites>, tests 属性在子 <testsuite> 上
        test_count = 0
        test_names = []
        for suite in root.iter("testsuite"):
            test_count += int(suite.get("tests", 0))
            for case in suite.iter("testcase"):
                classname = case.get("classname") or ""
                name = case.get("name") or ""
                test_names.append(f"{classname}::{name}" if classname else name)
        return test_count, test_names

    def _parse_coverage(self, coverage_path: Path) -> float | None:
        """解析 cobertura coverage.xml: 取根节点 line-rate 转百分比, 无文件返回 None。

        Args:
            coverage_path: --cov-report=xml 产出的 coverage.xml 文件路径。

        Returns:
            float | None: 覆盖率百分比(0-100), 文件缺失或无 line-rate 时返回 None。
        """
        if not coverage_path.exists():
            return None
        root = ElementTree.parse(coverage_path).getroot()
        line_rate = root.get("line-rate")
        if line_rate is None:
            return None
        return float(line_rate) * 100.0

    def _is_passed(self, exit_code: int | None, test_count: int | None, timed_out: bool) -> bool:
        """双重判定通过: 未超时 && 退出码 0 && 测试数 > 0 才算通过 (三者缺一即失败)。

        test_count == 0 单独拦截 pytest 空收集(退出码 5): 无测试 = 无验证;
        test_count 为 None(非 pytest 命令)时回退到仅看退出码。

        Args:
            exit_code: 命令退出码, 超时场景为 None。
            test_count: 解析 junit 得到的测试数, 非 pytest 命令时为 None。
            timed_out: 是否超时。

        Returns:
            bool: 是否通过。
        """
        if timed_out:
            return False
        if exit_code != 0:
            return False
        if test_count == 0:
            return False  # pytest 空收集(退出码 5): 0 个测试 = 无验证
        return True  # test_count 为 None(非 pytest 命令)时回退到仅看退出码
