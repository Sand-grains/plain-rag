# harness 决策日志

> 由状态转移命令自动追加, 勿手改(主文件保留最近 50 条, 更早归档到 harness/history/)

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
## 2026-08-24 15:18 — archive F19,F20,F17,F21,F23,F25,F26,F27,F07a,F18
user: sca
reason: 归档到 harness/history/archive-20260824-151808.yaml
## 2026-08-24 15:20 — start F16
user: sca
reason: start -> active
## 2026-08-24 15:21 — start F22
user: sca
reason: start -> active
## 2026-08-24 15:21 — start F24
user: sca
reason: start -> active
## 2026-08-24 15:21 — verify F16
user: sca
reason: verify 通过
evidence: verify-20260824-152136-F16.log
## 2026-08-24 15:22 — verify F22
user: sca
reason: verify 通过
evidence: verify-20260824-152205-F22.log
## 2026-08-24 15:22 — verify F24
user: sca
reason: verify 通过
evidence: verify-20260824-152237-F24.log
## 2026-08-24 15:35 — start F29
user: sca
reason: 008-3 cleaner+字段拆解完成, 开始门禁验证
## 2026-08-24 15:36 — verify F29
user: sca
reason: verify 通过
evidence: verify-20260824-153606-F29.log
## 2026-08-24 15:46 — start F28
user: sca
reason: 008-1 precheck 四格式预检完成, 开始门禁验证
## 2026-08-24 15:46 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260824-154643-F28.log
## 2026-08-24 15:56 — start F30
user: sca
reason: 008-2 loader+splitter 表格保护完成, 开始门禁验证
## 2026-08-24 15:57 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260824-155702-F30.log
## 2026-08-24 16:02 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260824-160222-F30.log
## 2026-08-24 16:45 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260824-164537-F28.log
## 2026-08-24 16:46 — verify F29
user: sca
reason: verify 通过
evidence: verify-20260824-164606-F29.log
## 2026-08-24 16:46 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260824-164642-F30.log
## 2026-08-24 18:47 — start F28
user: sca
reason: DispatchMode 改名 DispatchDecision, 复验
## 2026-08-24 18:48 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260824-184806-F28.log
## 2026-08-24 18:53 — start F28
user: sca
reason: sampling_stats 改名 sampling_format_stats, 复验
## 2026-08-24 18:53 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260824-185356-F28.log
## 2026-08-25 10:08 — start F28
user: sca
reason: start -> active
## 2026-08-25 10:08 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260825-100844-F28.log
## 2026-08-25 10:09 — verify F29
user: sca
reason: verify 通过
evidence: verify-20260825-100926-F29.log
## 2026-08-25 10:10 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260825-101005-F30.log
## 2026-08-25 10:16 — start F28
user: sca
reason: start -> active
## 2026-08-25 10:17 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260825-101727-F28.log
## 2026-08-25 10:17 — verify F29
user: sca
reason: verify 通过
evidence: verify-20260825-101756-F29.log
## 2026-08-25 10:18 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260825-101832-F30.log
## 2026-08-25 10:27 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260825-102707-F28.log
## 2026-08-25 10:27 — verify F29
user: sca
reason: verify 通过
evidence: verify-20260825-102745-F29.log
## 2026-08-25 10:28 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260825-102822-F30.log
## 2026-08-25 10:35 — start F28
user: sca
reason: start -> active
## 2026-08-25 10:36 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260825-103617-F28.log
## 2026-08-25 10:36 — verify F29
user: sca
reason: verify 通过
evidence: verify-20260825-103648-F29.log
## 2026-08-25 10:37 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260825-103724-F30.log
## 2026-08-25 11:19 — verify F28
user: sca
reason: verify 通过
evidence: verify-20260825-111936-F28.log
## 2026-08-25 11:20 — verify F29
user: sca
reason: verify 通过
evidence: verify-20260825-112004-F29.log
## 2026-08-25 11:20 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260825-112040-F30.log
## 2026-08-25 14:42 — verify F30
user: sca
reason: verify 通过
evidence: verify-20260825-144248-F30.log
## 2026-08-25 15:31 — archive F16,F24,F22,F29,F30,F28
user: sca
reason: 归档到 harness/history/archive-20260825-153141.yaml
