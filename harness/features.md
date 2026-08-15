# 功能清单

> 由 harness 自动生成, 勿手改; 改状态请用 CLI 命令

| ID | 行为 | 门禁验证 | 端到端验证 | 状态 | 完成时间 | 叙述 |
| --- | --- | --- | --- | --- | --- | --- |
| F01 | harness-状态机: 编码转移表并拒绝非法转移 | `uv run pytest harness/tests/test_state_machine.py -q --cov=harness.core.state_machine --cov-fail-under=90` | `-` | passed | - | 回归修复: 还原 F01 门禁 |
| F02 | harness-清单解析: features.yaml 解析/校验/渲染 | `uv run pytest harness/tests/test_registry.py -q --cov=harness.core.registry --cov-fail-under=75` | `-` | passed | - | - |
| F03 | harness-验证器: 退出码判定 + 两级验证 | `uv run pytest harness/tests/test_verifier.py -q --cov=harness.service.verifier --cov-fail-under=90` | `-` | passed | - | - |
| F04 | harness-调度器: 只出候选 | `uv run pytest harness/tests/test_scheduler.py -q --cov=harness.service.scheduler --cov-fail-under=90` | `-` | passed | - | - |
| F05 | harness-追踪器: 状态分布 + 健康度 | `uv run pytest harness/tests/test_tracker.py -q --cov=harness.service.tracker --cov-fail-under=90` | `-` | passed | - | - |
| F06 | harness-报告器: PROGRESS.md 快照 | `uv run pytest harness/tests/test_reporter.py -q --cov=harness.service.reporter --cov-fail-under=80` | `-` | passed | - | - |
| F07a | harness-CLI 核心: start/verify/reactivate/abandon/block/note 子命令, 不依赖 F04-F06 | `uv run pytest harness/tests/test_cli_core.py -q --cov=harness.cli --cov-fail-under=55` | `-` | passed | - | - |
| F07b | harness-CLI 完整: next/status/verify-all/report 子命令, 依赖 F04-F06 | `uv run pytest harness/tests/test_cli_full.py -q --cov=harness.cli --cov-fail-under=55` | `-` | passed | - | - |
| F08 | harness-端到端验证: full 字段/--full 旗标改 e2e/--e2e, precheck 前置自检, 按功能项声明端到端测评命令 (含eval --precheck/--smoke 模式) | `uv run pytest harness/tests/test_e2e.py tests/unit/test_eval_runner_modes.py -q --cov=harness.service.verifier --cov=harness.core.registry --cov=harness.cli --cov-fail-under=80` | `-` | passed | - | - |
| F09 | harness-完成时间与状态语义: FeatureItem 增 finished_time 字段(active->passed 记录/巡检不刷新/regressed 清空/渲染展示), 状态 token passing 改 passed | `uv run pytest harness/tests/test_registry.py harness/tests/test_state_machine.py harness/tests/test_reporter.py -q --cov=harness.core.registry --cov=harness.core.state_machine --cov=harness.service.reporter --cov-fail-under=80` | `-` | passed | 2026-08-14 18:54 | - |
