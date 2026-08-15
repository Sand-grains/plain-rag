"""功能项六态枚举: 全 harness 共享的状态常量, 不藏在状态机内部。

状态常量同时被 state_machine(转移校验)、registry(YAML 解析校验)、
tracker(健康度统计)与 cli(命令路由)引用。独立成模块避免 F01/F02 互相依赖,
也避免各组件各自硬编码字符串造成漂移。

与上层的关系: 任何写 features.yaml 状态的地方都必须取这里的常量,
保证"6 个枚举 token 仅此一处定义"。
"""
NOT_STARTED = "not_started"
ACTIVE = "active"
BLOCKED = "blocked"
PASSED = "passed"
REGRESSED = "regressed"
ABANDONED = "abandoned"

# 全量合法状态, 用于 registry 解析校验与 CLI 参数校验
VALID_STATES: tuple[str, ...] = (NOT_STARTED, ACTIVE, BLOCKED, PASSED, REGRESSED, ABANDONED)
