"""HTML loader：按 precheck 决策走正文抽取（HTML 决策域无 WHOLE_TEXT_PIPELINE 分支，B4）。

HTML_TEXT: 抽正文，标题->#, 表格->pipe-table, 代码->fenced；SKIP_TEXT_PIPELINE: 返回空。
"""
from pathlib import Path

from bs4 import Tag

from preprocess.format_precheck import DispatchDecision, PrecheckResult
from indexing.loaders import skip_result
from ._md import count_degraded_tables, heading_md, to_pipe_table

_SKIP_TAGS = {"nav", "aside", "footer", "script", "style", "header", "form", "noscript"}
_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _normalize_body(soup) -> str:
    """把 body 归一化为类 Markdown（标题/表格/代码/段落）。"""
    body = soup.body or soup
    lines: list[str] = []

    def walk(element):
        if not isinstance(element, Tag):
            return
        if element.name in _SKIP_TAGS:
            return
        if element.name in _HEADING_TAGS:
            text = element.get_text(" ", strip=True)
            if text:
                lines.append(heading_md(_HEADING_TAGS[element.name], text))
            return
        if element.name == "table":
            rows = []
            for tr in element.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            table_md = to_pipe_table(rows)
            if table_md:
                lines.append(table_md)
            return
        if element.name == "pre":
            code = element.get_text("\n")
            lines.append("```\n" + code.strip("\n") + "\n```")
            return
        # 其余容器递归
        for child in element.children:
            if isinstance(child, Tag):
                walk(child)
            else:
                text = str(child).strip()
                if text:
                    lines.append(text)

    for child in body.children:
        walk(child)
    return "\n\n".join(line for line in lines if line.strip())


def html_loader(path: Path, precheck: PrecheckResult) -> tuple[str, dict]:
    """把 HTML 归一化为 Markdown + format_meta。

    Args:
        path: HTML 文件路径。
        precheck: precheck 产出的分流决策。

    Returns:
        (markdown, format_meta)：归一化 MD（SKIP_TEXT_PIPELINE 时为空串）+ 格式元数据。
    """
    from bs4 import BeautifulSoup

    format_meta: dict = {"doc_type": ".html"}

    if precheck.doc_decision is DispatchDecision.SKIP_TEXT_PIPELINE:
        return skip_result(".html", precheck.vlm_candidate_count)

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    markdown = _normalize_body(soup)
    format_meta.update({
        "title": (soup.title.get_text(strip=True) if soup.title else ""),
        "degraded_table_count": count_degraded_tables(markdown),
    })
    return markdown, format_meta
