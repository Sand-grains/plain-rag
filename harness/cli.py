"""Harness治理命令全集入口
每个子命令是一个合法转移的入口, 参数被翻译为 registry 的状态改写调用
决策日志的触发入口, 指标门结果的展示层

核心子命令: start / verify / block / unblock / reactivate / abandon / note
完整子命令(next/status/verify-all/report)在 P4 追加
所有状态改写都经 registry 原子落盘, 非法转移抛 HarnessError 并被本层转成
非零退出码; verify 的退出码即验证结论(0 = 通过), 供脚本 / CI 消费

与上层的关系: 每个命令即一个合法转移的触发点, 命令集与 state_machine 转移表
对账(条例七); 缺省 features.yaml 路径由 __file__ 推导, 不依赖 CWD。
"""
import argparse
import sys
from pathlib import Path

from harness.core.errors import HarnessError
from harness.core.registry import DECISION_ACTIONS, ItemNotFoundError, Registry
from harness.service.reporter import Reporter
from harness.service.scheduler import next_candidate
from harness.core.states import ABANDONED, ACTIVE, BLOCKED, PASSED, VALID_STATES
from harness.service.tracker import judge_health
from harness.service.verifier import Verifier


def _default_features_path() -> Path:
    """features.yaml 绝对路径: 由本文件位置推导, 不依赖 CWD。

    Returns:
        Path: harness/features.yaml 的绝对路径。
    """
    return Path(__file__).resolve().parent / "features.yaml"


def _default_progress_path() -> Path:
    """PROGRESS.md 绝对路径(项目根), 由本文件位置推导, 不依赖 CWD。

    Returns:
        Path: 项目根下 PROGRESS.md 的绝对路径。
    """
    return Path(__file__).resolve().parent.parent / "PROGRESS.md"


def _find_item(registry: Registry, item_id: str):
    """按 id 在清单快照中查找功能项, 未命中抛 ItemNotFoundError。

    Args:
        registry: 清单读写入口(Registry 实例)。
        item_id: 功能项编号。

    Returns:
        FeatureItem: 命中的功能项。

    Raises:
        ItemNotFoundError: 清单中不存在该 item_id。
    """
    for item in registry.load().items:
        if item.id == item_id:
            return item
    raise ItemNotFoundError(f"功能项不存在: {item_id}")


def _print_state_change(item_id: str, new_state: str) -> None:
    """打印状态变更行: 统一 start/block/unblock/reactivate/abandon 的输出格式。

    Args:
        item_id: 功能项编号。
        new_state: 变更后的目标状态。
    """
    print(f"{item_id} 状态: {new_state}")


def _print_verification(item_id: str, result, new_state: str) -> None:
    """打印验证结果行: 通过态带判定依据, 失败态带退出码/超时与输出尾部。

    通过时区分端到端(仅退出码)与 gate(测试数/覆盖率); 失败时打印最近 12 行输出供诊断。

    Args:
        item_id: 功能项编号。
        result: VerificationResult 验证证据。
        new_state: record_verification 后的目标状态。
    """
    if result.passed:
        if result.test_count is None:
            detail = "端到端验证(退出码判定)" if result.mode == "e2e" else "退出码判定"
        else:
            detail = f"{result.test_count} 测试"
            if result.coverage is not None:
                detail += f", 覆盖率 {result.coverage:.1f}%"
        print(f"验证 {item_id}: 通过 ({detail}) -> {new_state}")
        if result.metric_gate is not None:
            report = result.metric_gate
            print(f"  指标门: 通过 (run_id={report.get('run_id')}, {len(report.get('checks', []))} 项)")
    else:
        reason = f"退出码 {result.exit_code}" if result.exit_code is not None else "超时"
        detail = f"{result.test_count or 0} 测试" if result.test_count is not None else "端到端验证"
        print(f"验证 {item_id}: 未通过 ({reason}, {detail}) -> {new_state}")
        if result.metric_gate is not None:
            _print_metric_gate_detail(result.metric_gate)
        else:
            tail = result.output.strip().splitlines()[-12:]
            if tail:
                print("  | " + "\n  | ".join(tail))


def _print_metric_gate_detail(report: dict) -> None:
    """指标门失败明细(计划 §7): 打印不达标指标与原因。"""
    reason = report.get("reason") or "阈值/delta 不达标"
    print(f"  指标门: 未通过 ({reason})")
    for check in report.get("checks", []):
        if not check["passed"]:
            rule = check.get("rule") or {}
            baseline = f"基线 {check['baseline']}, " if "baseline" in check else ""
            print(f"    FAIL {check['name']}: value={check.get('value')}, "
                  f"rule={rule}, {baseline}{check.get('reason') or ''}")
    if report.get("command_output_tail"):
        print("  | " + "\n  | ".join(report["command_output_tail"]))

def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器: 每个子命令即是一个合法状态转移的触发点, 命令集与转移表对账。

    start/block/unblock/reactivate/abandon/note 为核心态转移子命令,
    verify 带 --e2e/--note;
    next/status/verify-all/report 为只读/巡检子命令。

    Returns:
        argparse.ArgumentParser: 配置好全部子命令的解析器。
    """
    parser = argparse.ArgumentParser(prog="harness", description="命令是裁判的验收状态机")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_start = subparsers.add_parser("start", help="not_started -> active")
    parser_start.add_argument("item_id")
    parser_start.add_argument("-m", "--note", default=None, help="开始理由(可选, 写决策日志)")

    parser_verify = subparsers.add_parser("verify", help="验证门禁, 决定 active -> passed / passed -> regressed")
    parser_verify.add_argument("item_id")
    parser_verify.add_argument("--e2e", action="store_true", help="跑该功能项的端到端验证(eval/benchmark)")
    parser_verify.add_argument("-m", "--note", default=None, help="验证失败时的卡点说明")

    for name, help_text in (("block", "active -> blocked"), ("unblock", "blocked -> active"),
                            ("reactivate", "regressed -> active, 用户执行, 强制带 note")):
        parser_block = subparsers.add_parser(name, help=help_text)
        parser_block.add_argument("item_id")
        parser_block.add_argument("-m", "--note", required=True)

    parser_abandon = subparsers.add_parser("abandon", help="任意态 -> abandoned")
    parser_abandon.add_argument("item_id")
    parser_abandon.add_argument("-m", "--note", default=None, help="放弃理由(可选, 写决策日志)")

    parser_note = subparsers.add_parser("note", help="写叙述字段, 不碰状态(可 --action/--reason 结构化决策)")
    parser_note.add_argument("item_id")
    parser_note.add_argument("note_text", nargs="?", default=None, help="自由文本叙述")
    parser_note.add_argument("--action", choices=DECISION_ACTIONS, default=None, help="决策 action")
    parser_note.add_argument("--reason", default=None, help="决策理由(结构化)")
    parser_note.add_argument("--evidence", default=None, help="证据引用(如 verify 日志文件名)")

    subparsers.add_parser("next", help="调度器出候选(只读), 用户拍板")
    subparsers.add_parser("status", help="状态分布 + 健康度(只读)")
    subparsers.add_parser("verify-all", help="全部 passed 项跑门禁, 失败者转 regressed")
    subparsers.add_parser("report", help="重新生成 PROGRESS.md 快照")
    return parser


def _run_core(args, features_path: Path) -> int:
    """核心子命令分发: 每个命令一个合法转移, 返回进程退出码。

    含 verify 的 precheck 短路分支: 前置未满足时不落证据、不转移状态, 退 2。

    Args:
        args: 解析后的命令行参数。
        features_path: features.yaml 绝对路径。

    Returns:
        int: 进程退出码, 0=成功/1=验证未通过/2=前置未满足。
    """
    registry = Registry(features_path)
    verifier = Verifier()

    if args.command == "start":
        item = registry.apply_state(args.item_id, ACTIVE, note=args.note, action="start")
        _print_state_change(item.id, item.state)
        return 0

    if args.command == "verify":
        item = _find_item(registry, args.item_id)
        result = verifier.verify(item, e2e=args.e2e)
        if result.precheck_passed is False:
            # 前置未满足: 打印原因, 不 record_verification, 不转移状态, 退 2
            print(f"前置未满足: {item.id} 的 precheck 未通过, 请先重标注/入库", file=sys.stderr)
            tail = result.output.strip().splitlines()[-8:]
            if tail:
                print("  | " + "\n  | ".join(tail), file=sys.stderr)
            return 2
        updated = registry.record_verification(item.id, result.passed, result.to_evidence(), note=args.note,
                                               metric_baseline=result.metric_baseline_run_id)
        _print_verification(item.id, result, updated.state)
        return 0 if result.passed else 1

    if args.command == "block":
        item = registry.apply_state(args.item_id, BLOCKED, note=args.note, action="block")
        _print_state_change(item.id, item.state)
        return 0

    if args.command == "unblock":
        item = registry.apply_state(args.item_id, ACTIVE, note=args.note, action="unblock")
        _print_state_change(item.id, item.state)
        return 0

    if args.command == "reactivate":
        item = registry.apply_state(args.item_id, ACTIVE, note=args.note, action="reactivate")
        _print_state_change(item.id, item.state)
        return 0

    if args.command == "abandon":
        item = registry.apply_state(args.item_id, ABANDONED, note=args.note, action="abandon")
        _print_state_change(item.id, item.state)
        return 0

    if args.command == "note":
        if args.action is not None:
            if not (args.reason or args.note_text):
                raise HarnessError("结构化决策必须给 --reason 或自由文本")
            reason = args.reason or args.note_text
            item = registry.apply_note(args.item_id, args.note_text,
                                       action=args.action, reason=reason, evidence=args.evidence)
            print(f"{item.id} 决策已记录: {args.action} ({reason})")
        else:
            if args.note_text is None:
                raise HarnessError("note 需要自由文本或 --action/--reason 结构化输入")
            item = registry.apply_note(args.item_id, args.note_text)
            print(f"{item.id} 叙述: {item.note}")
        return 0

    raise HarnessError(f"未知命令: {args.command}")


def _run_full(args, features_path: Path, progress_path: Path) -> int:
    """完整子命令分发: next/status/verify-all/report, 返回进程退出码。

    verify-all 对全部 passed 项巡检, 任一失败者转 regressed 并返回 1。

    Args:
        args: 解析后的命令行参数。
        features_path: features.yaml 绝对路径。
        progress_path: PROGRESS.md 绝对路径。

    Returns:
        int: 进程退出码, 0=成功/1=存在回归。
    """
    registry = Registry(features_path)

    if args.command == "next":
        feature_list = registry.load()
        candidate = next_candidate(feature_list)
        if candidate is None:
            print("无候选: 没有 not_started 项")
        else:
            print(f"下一个候选: {candidate.id}")
            print(f"  行为: {candidate.behavior}")
            print(f"  点位: {feature_list.milestone}")
            if candidate.note:
                print(f"  叙述: {candidate.note}")
        return 0

    if args.command == "status":
        feature_list = registry.load()
        health = judge_health(feature_list)
        print("状态分布:")
        for state in VALID_STATES:
            print(f"  {state}: {health.distribution[state]}")
        print(f"回归率: {health.regression_rate:.2f}")
        print(f"验证通过率: {health.pass_rate:.2f}")
        if health.suspicious_ids:
            print(f"可疑通过项: {', '.join(health.suspicious_ids)}")
        if health.e2e_unverified_ids:
            print(f"端到端未验: {', '.join(health.e2e_unverified_ids)}")
        return 0

    if args.command == "verify-all":
        verifier = Verifier()
        feature_list = registry.load()
        passed_items = [item for item in feature_list.items if item.state == PASSED]
        if not passed_items:
            print("无 passed 项可巡检")
            return 0
        regressed_ids = []
        for item in passed_items:
            result = verifier.verify(item)
            # 巡检非决策: record_decision=False, 不写决策日志防批量噪音
            updated = registry.record_verification(item.id, result.passed, result.to_evidence(),
                                                   metric_baseline=result.metric_baseline_run_id,
                                                   record_decision=False)
            _print_verification(item.id, result, updated.state)
            if not result.passed:
                regressed_ids.append(item.id)
        if regressed_ids:
            print(f"回归: {', '.join(regressed_ids)}")
            return 1
        return 0

    if args.command == "report":
        feature_list = registry.load()
        reporter = Reporter(features_path, progress_path)
        reporter.write(feature_list, judge_health(feature_list))
        print(f"PROGRESS.md 已更新: {progress_path}")
        return 0

    raise HarnessError(f"未知命令: {args.command}")


def main(argv: list[str] | None = None, features_path: Path | None = None,
         progress_path: Path | None = None) -> int:
    """CLI 入口: 返回进程退出码, 失败(含非法转移/清单错误/验证未通过)返回 1。

    Args:
        argv: 命令行参数列表, 缺省取 sys.argv。
        features_path: features.yaml 路径, 测试注入用; 缺省由 __file__ 推导。
        progress_path: PROGRESS.md 路径, 测试注入用; 缺省由 __file__ 推导。

    Returns:
        int: 进程退出码, 0=成功, 1=失败 (非法转移/清单错误/验证未通过)。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    target = features_path if features_path is not None else _default_features_path()
    target_progress = progress_path if progress_path is not None else _default_progress_path()
    try:
        if args.command in ("next", "status", "verify-all", "report"):
            return _run_full(args, target, target_progress)
        return _run_core(args, target)
    except HarnessError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
