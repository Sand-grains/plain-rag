"""日志/trace 保留清理脚本: 供 Windows 计划任务每 3 天调用。

规则:
    - trace 文件超过 100 个 → 保留最近 100 个(按天分包 logs/traces/YYYY-MM-DD/)
    - logs/ 下全部文件超过 200 个 → 保留最近 200 个

用法::

    uv run python scripts/prune_logs.py
    .venv\\Scripts\\python.exe scripts\\prune_logs.py   # 计划任务直接调 venv python

计划任务注册(schtasks, 每 3 天)::

    schtasks /create /tn "plain_rag_prune_logs" /tr "\"D:\\Pycharm\\plain_rag\\.venv\\Scripts\\python.exe\" \"D:\\Pycharm\\plain_rag\\scripts\\prune_logs.py\"" /sc DAILY /mo 3 /f
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 独立运行(计划任务直调)时确保能 import config/obs

from obs.retention_policy import prune_logs


def main() -> None:
    """执行保留清理, 打印删除数(计划任务退出码 0 即成功)。"""
    pruned = prune_logs(max_traces=100, max_all_logs=200)
    print(f"prune_logs: 删除 {pruned} 个过旧文件")


if __name__ == "__main__":
    main()
