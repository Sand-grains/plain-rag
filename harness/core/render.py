"""通用 Markdown 表格渲染基元: features.md 与 PROGRESS.md 共用, 消除表格行格式漂移。

registry 渲染 features.md、reporter 渲染 PROGRESS.md,
两者的行格式 (空值回退/反引号包裹/叙述转义/行拼接)抽到此处归一
"""
from harness.core.utils import md_escape


def cell(value: object, code: bool = False) -> str:
    """单元格: 空值回退 '-' ; code=True 时反引号包裹(命令列用, 不转义)。

    Args:
        value: 单元格值, None 渲染为 "-"。
        code: 是否反引号包裹。

    Returns:
        str: 单元格文本。
    """
    text = "-" if value is None else str(value)
    return f"`{text}`" if code else text


def note_cell(note: str | None) -> str:
    """叙述列: 空值回退 '-' 并转义(叙述可含竖线/换行, 直接拼接会破坏表格)。

    Args:
        note: 功能项叙述, 可为 None。

    Returns:
        str: 转义后的叙述单元格。
    """
    return md_escape("-" if note is None else note)


def table_row(cells: list[str]) -> str:
    """拼接一行 Markdown 表格: ['a','b'] -> '| a | b |'。

    Args:
        cells: 已格式化的各列文本。

    Returns:
        str: 单行表格行。
    """
    return "| " + " | ".join(cells) + " |"
