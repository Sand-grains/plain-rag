"""harness 服务层: 对 core 做事的执行/分析/视图组件。

verifier(验证执行)、tracker(健康度)、scheduler(调度候选)、reporter(PROGRESS.md 快照)。
依赖方向单向: service → core, 不反向依赖。
"""
