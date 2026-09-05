"""重型解析链的归一化 Markdown 整型层

它把 Docling/MinerU/MarkItDown 各自原生输出的 Markdown, 统一整形
后端原生 Markdown → 统一行尾(\n) → 标题补空格 → 表头分隔行统一(---) → 行尾去空白 → 折叠空行 → 对齐契约 Markdown 格式

本身只依赖 re 与自身正则, 无业务依赖(纯函数文件)。
仅作整形, 不改变语义内容, 也不判断指令, 不合并, 不校验(这些是quality.py, merge.py的工作)
"""
from __future__ import annotations

import re

# 行首 1-6 个 # 后缺空格的标题(如 "#foo" / "##foo"), 补一个空格
_HEADING_NO_SPACE_RE = re.compile(r"(?m)^(#{1,6})(?=[^#\s])")
# 连续 3 个及以上空行折叠为 2 个
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
# 分隔行判定: 行首/行尾为 | 且所有非空单元格仅由短横构成
_SEPARATOR_LINE_RE = re.compile(r"^\|[\s\-|]*\|\s*$")
# 分隔单元格短横串(如 "-" / "----")
_DASH_CELL_RE = re.compile(r"^-+$")

# 共享正则(012-1 收敛: quality/merge/parse_metrics 复用, 不写三份)
PIPE_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")          # pipe-table 行
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-|]+\|\s*$")   # 表头分隔行(如 |---|---|)
FENCED_CODE_RE = re.compile(r"```\w*\n.*?```", re.DOTALL)   # fenced code 块


def normalize_to_contract(markdown: str) -> str:
    """把后端原生 Markdown 对齐 008 归一化契约。

    Args:
        markdown: 后端输出的原始 Markdown 文本。

    Returns:
        str: 对齐契约后的 Markdown(标题补空格 / 空行收敛 / 表头分隔统一; 行结尾统一为 \\n)。
    """
    if not markdown:
        return ""
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    # 标题补空格: "#x" -> "# x"(避免被指标当成非法标题)
    text = _HEADING_NO_SPACE_RE.sub(r"\1 ", text)
    # 表头分隔行: 把短横单元格统一为 "---"(markitdown 个别情况输出 "-" 或 "----")
    text = _normalize_separator_rows(text)
    # 去掉每行行尾空白
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # 折叠多余空行并去首尾空行
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text).strip("\n")
    return text


def _normalize_separator_rows(text: str) -> str:
    """把 pipe-table 表头分隔行内每个短横单元格统一为 '---'。

    Args:
        text: 逐行处理的 Markdown 文本。

    Returns:
        str: 分隔行统一后的文本。
    """
    lines = []
    for line in text.split("\n"):
        if _SEPARATOR_LINE_RE.match(line):
            cells = line.split("|")
            normalized = ["---" if _DASH_CELL_RE.match(cell.strip()) else cell for cell in cells]
            line = "|".join(normalized)
        lines.append(line)
    return "\n".join(lines)
