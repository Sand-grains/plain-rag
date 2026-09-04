"""多格式解析质量指标
把 某份文档被 parse 得好不好 量化成可复现的数值:
它测 转制+解析 管线保真度, 不读成绝对解析质量: 因为喂给解析器的是转制后产物(md→html→pdf/docx/pptx)。转制本身就引入了损失

- 文本维度:
  - text_recall(parsed, gold): 解析出的内容有没有丢(重叠 n-gram / gold n-gram)
  - text_precision(parsed, gold): 有没有多灌垃圾(重叠 n-gram / parsed n-gram, 防单侧拉高召回)
- 结构维度: structure_fidelity → 标题(层级+文本)、表格(行×列)、代码块(语言+内容) 三类保真度 + overall

核心机制:
    - 文本召回率: parsed 与 gold 空白归一化后的 char n-gram 重叠度(不复用 text_ratio, 它量的是单篇文档代码块占比, 与召回率量纲/输入/语义全不同; 而 text_ratio 只作报告诊断字段)。
    - 文本精确率: parsed 中与 gold 重叠的 n-gram 占比, 防解析器倾倒多余文本单侧拉高召回。
    - 结构保真度: 标题(# 层级+文本) / 表格(行/列结构) / 代码块(语言+内容) 与原 md 的重合。
    - 同源一致性 = 端到端管线保真度(转制 + 解析), 不读成绝对解析质量。

用法示例::

    from eval.core.parse_metrics import text_recall, text_precision, structure_fidelity
    recall = text_recall(parsed_md, gold_md)
    fidelity = structure_fidelity(parsed_md, gold_md)

公共接口:
    - text_recall: 文本召回率(char n-gram 重叠度)
    - text_precision: 文本精确率(parsed 中与 gold 重叠占比)
    - heading_fidelity / table_fidelity / code_fidelity: 三类结构保真度
    - structure_fidelity: 结构保真度聚合(三类均值)
"""
from __future__ import annotations

import re

from indexing.parse_backends._normalize import PIPE_TABLE_LINE_RE, TABLE_SEPARATOR_RE

# 空白归一化: 任意连续空白(含换行/全角空格)折叠为单个半角空格
_WHITESPACE_RE = re.compile(r"\s+")
# 标题行: 行首 1-6 个 # + 空格 + 文本
_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.*)$")
# fenced code 块: ```lang\n...```(需捕获语言与内容, 故本地保留带分组版本)
_FENCED_CODE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def _normalize(text: str) -> str:
    """空白归一化: 折叠连续空白为单空格并去首尾空白。

    Args:
        text: 原始文本。

    Returns:
        str: 归一化后的文本。
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def _ngrams(text: str, ngram_size: int = 2) -> set[str]:
    """生成 char n-gram 集合(默认 bigram)。

    Args:
        text: 归一化后的文本。
        ngram_size: n-gram 长度。

    Returns:
        set[str]: n-gram 集合。
    """
    return {text[index:index + ngram_size] for index in range(len(text) - ngram_size + 1)}


def text_recall(parsed: str, gold: str, ngram_size: int = 2) -> float:
    """文本召回率: parsed 与 gold 空白归一化后的 char n-gram 重叠度。

    Args:
        parsed: 解析出的归一化 MD。
        gold: 原 md gold。
        ngram_size: n-gram 长度(默认 2)。

    Returns:
        float: 0~1, gold 为空时按 parsed 是否为空判定(空对空=1, 非空对空=0)。
    """
    parsed_ngrams = _ngrams(_normalize(parsed), ngram_size)
    gold_ngrams = _ngrams(_normalize(gold), ngram_size)
    if not gold_ngrams:
        return 1.0 if not parsed_ngrams else 0.0
    return len(parsed_ngrams & gold_ngrams) / len(gold_ngrams)


def text_precision(parsed: str, gold: str, ngram_size: int = 2) -> float:
    """文本精确率: parsed 中与 gold 重叠的 n-gram 占比(防倾倒多余文本单侧拉高召回)。

    Args:
        parsed: 解析出的归一化 MD。
        gold: 原 md gold。
        ngram_size: n-gram 长度(默认 2)。

    Returns:
        float: 0~1, parsed 为空时返回 1.0。
    """
    parsed_ngrams = _ngrams(_normalize(parsed), ngram_size)
    gold_ngrams = _ngrams(_normalize(gold), ngram_size)
    if not parsed_ngrams:
        return 1.0
    return len(parsed_ngrams & gold_ngrams) / len(parsed_ngrams)


def _headings(text: str) -> list[tuple[int, str]]:
    """提取标题树: (层级, 归一化文本) 列表。

    Args:
        text: md 文本。

    Returns:
        list[tuple[int, str]]: 标题列表(层级 + 文本)。
    """
    return [(len(match.group(1)), match.group(2).strip())
            for match in _HEADING_RE.finditer(text)]


def heading_fidelity(parsed: str, gold: str) -> float:
    """标题保真度: 解析出的 # 层级+文本 与原 md 标题树的重合。

    Args:
        parsed: 解析出的归一化 MD。
        gold: 原 md gold。

    Returns:
        float: 0~1, gold 无标题时返回 1.0。
    """
    parsed_headings = set(_headings(parsed))
    gold_headings = _headings(gold)
    if not gold_headings:
        return 1.0
    matched = sum(1 for heading in gold_headings if heading in parsed_headings)
    return matched / len(gold_headings)


def _tables(text: str) -> list[tuple[int, int]]:
    """提取 pipe-table 结构: 每张表 (数据行数, 列数) 列表。

    Args:
        text: md 文本。

    Returns:
        list[tuple[int, int]]: 表格结构列表(行数, 列数)。
    """
    tables: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if PIPE_TABLE_LINE_RE.match(line):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    result: list[tuple[int, int]] = []
    for table in tables:
        data_rows = [line for line in table if not TABLE_SEPARATOR_RE.match(line)]
        columns = len([cell for cell in table[0].split("|") if cell.strip()]) if table else 0
        result.append((len(data_rows), columns))
    return result


def table_fidelity(parsed: str, gold: str) -> float:
    """表格保真度: 解析出表格的行/列结构与原 md pipe-table 的重合。

    Args:
        parsed: 解析出的归一化 MD。
        gold: 原 md gold。

    Returns:
        float: 0~1, gold 无表格时返回 1.0。
    """
    parsed_tables = set(_tables(parsed))
    gold_tables = _tables(gold)
    if not gold_tables:
        return 1.0
    matched = sum(1 for table in gold_tables if table in parsed_tables)
    return matched / len(gold_tables)


def _code_blocks(text: str) -> list[tuple[str, str]]:
    """提取 fenced code 块: (语言, 内容) 列表。

    Args:
        text: md 文本。

    Returns:
        list[tuple[str, str]]: 代码块列表(语言, 内容)。
    """
    return [(language, content) for language, content in _FENCED_CODE_RE.findall(text)]


def code_fidelity(parsed: str, gold: str) -> float:
    """代码块保真度: 解析出 fenced code(语言+内容)与原 md 的重合。

    Args:
        parsed: 解析出的归一化 MD。
        gold: 原 md gold。

    Returns:
        float: 0~1, gold 无代码块时返回 1.0。
    """
    parsed_blocks = set(_code_blocks(parsed))
    gold_blocks = _code_blocks(gold)
    if not gold_blocks:
        return 1.0
    matched = sum(1 for block in gold_blocks if block in parsed_blocks)
    return matched / len(gold_blocks)


def structure_fidelity(parsed: str, gold: str) -> dict[str, float]:
    """结构保真度聚合: 标题/表格/代码块三类保真度 + 均值。

    Args:
        parsed: 解析出的归一化 MD。
        gold: 原 md gold。

    Returns:
        dict[str, float]: {"heading", "table", "code", "overall"}。
    """
    heading = heading_fidelity(parsed, gold)
    table = table_fidelity(parsed, gold)
    code = code_fidelity(parsed, gold)
    return {"heading": heading, "table": table, "code": code,
            "overall": (heading + table + code) / 3.0}
