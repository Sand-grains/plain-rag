"""unit：六态状态机转移矩阵(F01)。

36 个 (from, to) 组合精确断言: 转移表硬编码, 非法转移抛 IllegalTransitionError,
同态转移视为无操作放行, 未知状态报错。只 import 公开接口 ALLOWED_TRANSITIONS/transition。
"""
import pytest

from harness.core.state_machine import ALLOWED_TRANSITIONS, IllegalTransitionError, transition
from harness.core.states import VALID_STATES


class TestTransitionMatrix:
    def test_six_state_matrix(self):
        # 全 36 组合逐一断言: 合法表内转移放行, 表外拒绝
        for current in VALID_STATES:
            for target in VALID_STATES:
                # 同态为无操作放行; 否则必须落在转移表内
                allowed = (target == current) or (target in ALLOWED_TRANSITIONS[current])
                if allowed:
                    assert transition(current, target) == target
                else:
                    with pytest.raises(IllegalTransitionError):
                        transition(current, target)

    def test_same_state_is_noop(self):
        # 同态转移为无操作, 任何态都放行(verify 对 passed 项重复通过的场景)
        for state in VALID_STATES:
            assert transition(state, state) == state

    def test_table_covers_all_states(self):
        # 转移表键集与状态集一一对应, 不漏不加
        assert set(ALLOWED_TRANSITIONS.keys()) == set(VALID_STATES)


class TestIllegalTransitionRejected:
    def test_verify_without_start_rejected(self):
        # not_started 不能直接 verify 通过(必须先 start)
        with pytest.raises(IllegalTransitionError):
            transition("not_started", "passed")

    def test_blocked_cannot_pass(self):
        # blocked 项不能直接 verify 通过, 必须 unblock
        with pytest.raises(IllegalTransitionError):
            transition("blocked", "passed")

    def test_abandoned_is_terminal(self):
        # abandoned 是终态, 无出边, 不可复活(同态无操作除外)
        for target in VALID_STATES:
            if target == "abandoned":
                continue
            with pytest.raises(IllegalTransitionError):
                transition("abandoned", target)

    def test_unknown_state(self):
        with pytest.raises(IllegalTransitionError):
            transition("mystery", "active")
        with pytest.raises(IllegalTransitionError):
            transition("active", "mystery")


class TestCommandReachability:
    def test_every_legal_transition_reachable(self):
        # 条例七对账: 每个合法转移至少被一个命令覆盖, 命令集映射到转移表
        command_reachable = {
            ("not_started", "active"),      # start
            ("not_started", "abandoned"),   # abandon
            ("active", "blocked"),          # block
            ("active", "passed"),          # verify(退出码 0)
            ("active", "abandoned"),        # abandon
            ("blocked", "active"),          # unblock
            ("blocked", "abandoned"),       # abandon
            ("passed", "regressed"),       # verify(非0)/verify-all
            ("passed", "abandoned"),       # abandon
            ("regressed", "active"),        # reactivate
            ("regressed", "abandoned"),     # abandon
        }
        for current in VALID_STATES:
            for target in ALLOWED_TRANSITIONS[current]:
                assert (current, target) in command_reachable, (current, target)
