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
## 2026-08-29 21:09 — start F31
user: sca
reason: 011-5 语料版本化落地
## 2026-08-29 21:09 — verify F31
user: sca
reason: verify 通过
evidence: verify-20260829-210958-F31.log
## 2026-08-29 21:21 — start F32
user: sca
reason: 011-0 评测基建适配落地
## 2026-08-29 21:22 — verify F32
user: sca
reason: verify 通过
evidence: verify-20260829-212234-F32.log
## 2026-08-29 21:23 — verify F32
user: sca
reason: verify 通过
evidence: verify-20260829-212326-F32.log
## 2026-08-29 21:29 — start F33
user: sca
reason: 011-4 非 pandoc 脚手架落地(样本选择/gold/pptx/快照/检测)
## 2026-08-29 21:29 — verify F33
user: sca
reason: verify 通过
evidence: verify-20260829-212921-F33.log
## 2026-08-29 21:32 — start F34
user: sca
reason: 011-6 指标模块+基线脚本脚手架落地
## 2026-08-29 21:32 — verify F34
user: sca
reason: verify 通过
evidence: verify-20260829-213215-F34.log
## 2026-08-29 21:44 — start F33
user: sca
reason: 011-4 完整转制落地(51篇x4格式+gold+快照)
## 2026-08-29 21:44 — verify F33
user: sca
reason: verify 通过
evidence: verify-20260829-214450-F33.log
## 2026-08-29 21:44 — start F34
user: sca
reason: 011-6 真实轻量基线落地(204样本)
## 2026-08-29 21:44 — verify F34
user: sca
reason: verify 通过
evidence: verify-20260829-214453-F34.log
## 2026-08-29 21:59 — start F34
user: sca
reason: 011-6 真实轻量基线(修正 pdf 后重跑)
## 2026-08-29 21:59 — verify F34
user: sca
reason: verify 通过
evidence: verify-20260829-215942-F34.log
## 2026-08-30 15:00 — start F35
user: sca
reason: 012-1 重型链 adapter 层 + 路由验收落地
## 2026-08-30 15:00 — verify F35
user: sca
reason: verify 通过
evidence: verify-20260830-150014-F35.log
## 2026-08-30 15:03 — verify F35
user: sca
reason: verify 未通过
evidence: verify-20260830-150356-F35.log
## 2026-08-30 15:04 — reactivate F35
user: sca
reason: 补 MINERU_PYTHON 分支测试, 覆盖率回 80%+
## 2026-08-30 15:04 — verify F35
user: sca
reason: verify 通过
evidence: verify-20260830-150431-F35.log
## 2026-08-30 15:11 — start F36
user: sca
reason: 012-1 合成扫描 gold 集 + 差分报告框架落地
## 2026-08-30 15:12 — verify F36
user: sca
reason: verify 通过
evidence: verify-20260830-151201-F36.log
## 2026-08-30 15:24 — verify F35
user: sca
reason: verify 通过
evidence: verify-20260830-152408-F35.log
## 2026-08-30 16:44 — start F37
user: sca
reason: 012-2 precheck 顶层路由升级
## 2026-08-30 16:45 — verify F37
user: sca
reason: verify 通过
evidence: verify-20260830-164516-F37.log
## 2026-08-30 16:50 — start F38
user: sca
reason: 012-2 文本管线质量驱动 + 跨页合并
## 2026-08-30 16:50 — verify F38
user: sca
reason: verify 通过
evidence: verify-20260830-165052-F38.log
## 2026-08-30 16:52 — start F39
user: sca
reason: 012-2 VLM 薄封装 + 视觉类路由
## 2026-08-30 16:52 — verify F39
user: sca
reason: verify 未通过
evidence: verify-20260830-165225-F39.log
## 2026-08-30 16:53 — verify F39
user: sca
reason: verify 通过
evidence: verify-20260830-165332-F39.log
## 2026-08-30 16:57 — verify F40
user: sca
reason: verify 通过
evidence: verify-20260830-165701-F40.log
## 2026-08-30 17:18 — verify F41
user: sca
reason: verify 通过
evidence: verify-20260830-171833-F41.log
## 2026-08-30 17:20 — verify F42
user: sca
reason: verify 通过
evidence: verify-20260830-172003-F42.log
## 2026-08-30 17:26 — verify F42
user: sca
reason: verify 通过
evidence: verify-20260830-172645-F42.log
## 2026-08-30 17:34 — verify F43
user: sca
reason: verify 通过
evidence: verify-20260830-173419-F43.log
## 2026-08-30 17:34 — verify F44
user: sca
reason: verify 通过
evidence: verify-20260830-173426-F44.log
## 2026-08-31 06:57 — verify F43
user: sca
reason: verify 通过
evidence: verify-20260831-065729-F43.log
## 2026-08-31 07:15 — verify F44
user: sca
reason: verify 通过
evidence: verify-20260831-071559-F44.log
## 2026-08-31 07:25 — verify F44
user: sca
reason: verify 通过
evidence: verify-20260831-072538-F44.log
## 2026-08-31 08:49 — verify F44
user: sca
reason: verify 通过
evidence: verify-20260831-084901-F44.log
## 2026-08-31 09:29 — verify F35
user: sca
reason: verify 未通过
evidence: verify-20260831-092911-F35.log
## 2026-08-31 09:29 — verify F36
user: sca
reason: verify 通过
evidence: verify-20260831-092922-F36.log
## 2026-08-31 09:30 — verify F37
user: sca
reason: verify 通过
evidence: verify-20260831-093000-F37.log
## 2026-08-31 09:30 — verify F38
user: sca
reason: verify 未通过
evidence: verify-20260831-093005-F38.log
## 2026-08-31 09:30 — verify F39
user: sca
reason: verify 通过
evidence: verify-20260831-093013-F39.log
## 2026-08-31 09:30 — verify F41
user: sca
reason: verify 通过
evidence: verify-20260831-093051-F41.log
## 2026-08-31 09:30 — verify F44
user: sca
reason: verify 通过
evidence: verify-20260831-093058-F44.log
## 2026-08-31 09:31 — verify F40
user: sca
reason: verify 通过
evidence: verify-20260831-093143-F40.log
## 2026-08-31 09:32 — reactivate F35
user: sca
reason: 补 paged_pipeline 测试入 gate, 覆盖率回 80%+
## 2026-08-31 09:32 — reactivate F38
user: sca
reason: 补 paged_pipeline 测试入 gate, 覆盖率回 80%+
## 2026-08-31 09:32 — verify F35
user: sca
reason: verify 未通过
evidence: verify-20260831-093212-F35.log
## 2026-08-31 09:32 — verify F38
user: sca
reason: verify 未通过
evidence: verify-20260831-093216-F38.log
## 2026-08-31 09:32 — verify F35
user: sca
reason: verify 未通过
evidence: verify-20260831-093257-F35.log
## 2026-08-31 09:33 — verify F38
user: sca
reason: verify 未通过
evidence: verify-20260831-093302-F38.log
## 2026-08-31 09:35 — verify F35
user: sca
reason: verify 通过
evidence: verify-20260831-093500-F35.log
## 2026-08-31 09:35 — verify F38
user: sca
reason: verify 通过
evidence: verify-20260831-093516-F38.log
## 2026-08-31 13:16 — verify F44
user: sca
reason: verify 通过
evidence: verify-20260831-131649-F44.log
## 2026-08-31 14:22 — verify F44
user: sca
reason: verify 未通过
evidence: verify-20260831-142235-F44.log
## 2026-08-31 14:24 — reactivate F44
user: sca
reason: 补 _run_multi_only 测试, 覆盖率回 80%+
## 2026-08-31 14:24 — verify F44
user: sca
reason: verify 通过
evidence: verify-20260831-142425-F44.log
## 2026-09-02 17:42 — start F45
user: sca
reason: 011-1 收口: private_builtin 重标完成+默认路径切换+225条全有效
## 2026-09-02 17:43 — verify F45
user: sca
reason: verify 通过
evidence: verify-20260902-174321-F45.log
