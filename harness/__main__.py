"""python -m harness 入口: 委托给 cli.main(), 以退出码收尾。"""
import sys

if sys.platform == "win32":
    # Windows 管道默认按本地代码页输出, 中文会乱码; 强制 UTF-8 让 CLI 可读
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

from harness.cli import main

if __name__ == "__main__":
    sys.exit(main())
