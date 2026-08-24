# harness 决策日志

> 由状态转移命令自动追加, 勿手改(主文件保留最近 50 条, 更早归档到 harness/history/)

## 2026-08-19 09:27 — verify F27
user: sca
reason: verify 通过
evidence: verify-20260819-092738-F27.log
## 2026-08-19 09:51 — reactivate F03
user: sca
reason: F26/F27 给 verifier.py 增码, 门禁纳入 metric_gate/decision_record 测试补覆盖
## 2026-08-19 09:52 — verify F03
user: sca
reason: verify 通过
evidence: verify-20260819-095208-F03.log
## 2026-08-19 09:52 — reactivate F08
user: sca
reason: F26/F27 给 verifier/registry/cli 增码, 门禁纳入全模块测试补覆盖
## 2026-08-19 09:53 — verify F08
user: sca
reason: verify 未通过
evidence: verify-20260819-095321-F08.log
## 2026-08-19 09:54 — verify F08
user: sca
reason: verify 未通过
evidence: verify-20260819-095450-F08.log
## 2026-08-19 09:56 — verify F08
user: sca
reason: verify 通过
evidence: verify-20260819-095619-F08.log
## 2026-08-20 12:54 — verify F22
user: sca
reason: verify 通过
evidence: verify-20260820-125441-F22.log
## 2026-08-20 13:59 — verify F26
user: sca
reason: verify 通过
evidence: verify-20260820-135936-F26.log
## 2026-08-22 21:00 — archive F01,F02,F03,F04,F05,F06,F07a,F07b,F08,F09,F10,F11,F12,F13,F14,F15
user: sca
reason: 归档到 harness/history/archive-20260822-210037.yaml
## 2026-08-22 21:44 — reactivate F07a
user: sca
reason: 修复 F07a 覆盖率: test_cli_core.py 补 archive/history 测试
## 2026-08-22 21:44 — verify F07a
user: sca
reason: verify 通过
evidence: verify-20260822-214452-F07a.log
## 2026-08-22 21:45 — reactivate F18
user: sca
reason: F18 gate 超时为瞬时慢(直接跑 31.5s 通过), 恢复
## 2026-08-22 21:45 — verify F18
user: sca
reason: verify 通过
evidence: verify-20260822-214549-F18.log
