"""harness 治理核心: 数据模型 + 六态 + 转移表 + 权威清单读写。

core 是"命令是裁判"的裁判逻辑, 全 harness 的基石, 供 service 层与 cli 依赖。
所有 import 用完整路径 harness.core.*, 不在本包做 re-export。
"""
