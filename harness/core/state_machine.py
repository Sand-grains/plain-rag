"""任务状态机(六态): 硬编码转移表 + 中央 transition() 拒绝非法转移。
state_machine 只回答"这个转移合不合法", 不执行验证也不读写清单

命令是裁判: 状态只由 harness 命令决定, 合法转移集硬编码为 dict
任何不在表内的 (from, to) 组合一律抛 IllegalTransitionError 拒绝。

与上层的关系: cli 在改写状态前先调 transition() 校验; verifier 用其推导
verify 的目标态; tracker/registry 不直接依赖本模块。
"""
from harness.core.errors import HarnessError
from harness.core.states import (
    ABANDONED,
    ACTIVE,
    BLOCKED,
    NOT_STARTED,
    PASSED,
    REGRESSED,
)

# 硬编码转移表: 从态 -> 合法目标态集合。
# abandoned 是终态。
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    NOT_STARTED: frozenset({ACTIVE, ABANDONED}),
    ACTIVE: frozenset({BLOCKED, PASSED, ABANDONED}),
    BLOCKED: frozenset({ACTIVE, ABANDONED}),
    PASSED: frozenset({REGRESSED, ABANDONED}),
    REGRESSED: frozenset({ACTIVE, ABANDONED}),
    ABANDONED: frozenset(),
}


class IllegalTransitionError(HarnessError):
    """非法状态转移: 从态到目标态不在转移表中。"""


def transition(current_state: str, target_state: str) -> str:
    """校验从态到目标态的转移是否合法, 合法则返回目标态。

    与当前态相同的目标态视为无操作直接返回(如对已 passed 项重复 verify 通过)。
    其余必须落在 ALLOWED_TRANSITIONS[current_state] 内, 否则抛 IllegalTransitionError
    并给出当前态的合法去向, 便于命令提示正确路径。

    Args:
        current_state: 功能项当前状态, 必须是 VALID_STATES 之一。
        target_state: 期望转移到的状态, 必须是 VALID_STATES 之一。

    Returns:
        str: 转移后的目标态(与入参相同)。

    Raises:
        IllegalTransitionError: 非法转移或当前态/目标态不在状态集内。
    """
    if current_state == target_state:
        return target_state
    allowed = ALLOWED_TRANSITIONS.get(current_state) # 从当前态映射出对应的"可达目标态集合"
    if allowed is None:
        raise IllegalTransitionError(f"未知状态: {current_state!r}")
    if target_state not in allowed: # 目标态不在可达集合里 → 非法转移
        raise IllegalTransitionError(
            f"非法转移: {current_state} -> {target_state}; 从 {current_state} 只允许到 {sorted(allowed)}"
        )
    return target_state
