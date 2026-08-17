---
description: 会话退出检查仪式: 按会话类型分级检查 + note 交接
---

# 会话退出检查(/checkout)

把"退出前必须完成"变成可执行仪式。脚本只检查不修复不写状态; 修复、交接 note、提交由你按本指令完成。

## 1. 判断会话类型并向用户确认模式

- 功能已完成: `full`(全门禁硬检查 + 提交)
- 跨会话续做(做了一半): `partial`(只跑单元测试 + note 留痕, 恒退 0)
- 探索/计划型(无可测产出): `info`(只留交接信息, 不设门禁, 恒退 0)

## 2. 跑检查脚本

```
uv run python scripts/exit_check.py --mode <full|partial|info>
```

## 3. 写交接 note(所有模式, 用户拍板)

```
harness note <最相关项ID> "<做了什么 / 剩什么 / 风险>"
```

## 4. full 专用: 修复直至退出码 0

- 测试/依赖失败: 修代码 -> 重跑
- verify-all 报回归(某项被翻成 regressed): 先修代码, 再依次执行
  ```
  harness reactivate <ID> -m "<note>"   # regressed -> active
  harness verify <ID>                    # active -> passed
  ```
  (重跑 verify-all 会跳过 regressed 项, 不能自愈; verify 只允许 active/passed)

## 5. full 专用: 提交已完成工作

```
git add <已完成文件>
git commit -m "<做了什么与为什么, 含治理文件 features.yaml/features.md 的更新>"
```

无改动则跳过本步。

## 6. 提示用户

退出前状态已交接, 可以 exit。
