"""语料转制: 将 data/ 路径下的 .md 转制为 pdf/html/docx/pptx
本文件只转制 + 写 gold + 保留快照, 不作任何解析, 也不作保真度计算(计算和指标实现在 scripts/parse_baseline.py + eval/core/parse_metrics.py)

gold 定义: 把 data/ 里选中的每篇 md 原样复制到 artifacts/gold/; gold属于是派生工件 (data/ 是唯一源文档)
gold 的意义是给解析质量对比提供一个"同源可重建、可校验"的本地基准 (即 转制再解析后 得到的md 对比 原md, 这个原md就是gold)
样本按内容形态分层 (纯文本/表格/代码/图文), 不按 "格式 x 数量" 硬凑。

转制工具:
    - md -> html: pandoc
    - md -> pdf: md -> html -> Edge/Chromium headless --print-to-pdf
    - md -> docx: pandoc
    - md -> pptx: python-pptx

转制产物为快照化工件: SHA 固化, 以内容为比较基准, 不挂活体 data/ 路径 (与 data/ 重组解耦)。

用法示例::

    uv run python scripts/corpus_convert.py select --out artifacts/samples.json
    uv run python scripts/corpus_convert.py convert --samples artifacts/samples.json --out artifacts/corpus_multiformat/
    uv run python scripts/corpus_convert.py snapshot --corpus artifacts/corpus_multiformat/

公共接口:
    - classify_md: 按内容形态分类单篇 md(plain/table/code/mixed)
    - select_samples: 按内容形态分层选择样本
    - generate_gold: 生成 gold(原 md 内容)
    - convert_md_to_html / convert_md_to_pdf / convert_md_to_docx / convert_md_to_pptx: 各格式转制
    - snapshot_converted: 转制语料 SHA 快照
    - main: argparse CLI 入口
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from config import _PROJECT_ROOT

# 内容形态正则
_PIPE_TABLE_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_FENCED_CODE_RE = re.compile(r"```")

# 分层样本默认规模
PLAIN_SAMPLE_N = 10      # 纯文本抽样
CODE_SAMPLE_N = 20       # 代码抽样(代码是主流, 不全量)
TABLE_ALL = True         # 表格全量(保非空)

# Edge/Chromium headless 候选路径(可被 config 覆盖)
_EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


@dataclass
class SampleSelection:
    """分层样本选择结果: 各形态选中的 md 相对路径列表。"""
    plain: list[str] = field(default_factory=list)      # 纯文本
    table: list[str] = field(default_factory=list)      # 表格(全量)
    code: list[str] = field(default_factory=list)       # 代码(抽样)
    mixed: list[str] = field(default_factory=list)      # 图文混合(外部补样例, 本地近无)

    def all_paths(self) -> list[str]:
        """全部选中样本(去重保序)。"""
        seen: set[str] = set()
        result: list[str] = []
        for path in self.plain + self.table + self.code + self.mixed:
            if path not in seen:
                seen.add(path)
                result.append(path)
        return result


def classify_md(content: str) -> str:
    """按内容形态分类单篇 md: plain(纯文本) / table(含 pipe-table) / code(含 fenced code) / mixed(表格+代码)。

    Args:
        content: md 文件内容。

    Returns:
        str: 内容形态分类(plain/table/code/mixed)。
    """
    has_table = bool(_PIPE_TABLE_RE.search(content))
    has_code = bool(_FENCED_CODE_RE.search(content))
    if has_table and has_code:
        return "mixed"
    if has_table:
        return "table"
    if has_code:
        return "code"
    return "plain"


def select_samples(data_dir: str, plain_n: int = PLAIN_SAMPLE_N,
                   code_n: int = CODE_SAMPLE_N, table_all: bool = TABLE_ALL) -> SampleSelection:
    """按内容形态分层选择样本(确定性: 按相对路径排序后取前 N)。

    表格全量: 所有含 pipe-table 的文档(含表格+代码混合)都进 table, 保证表格覆盖非空;
    code 只收"纯代码"(含代码且无表格)的抽样, 避免与 table 重复计入样本集。

    Args:
        data_dir: data/ 目录。
        plain_n: 纯文本抽样数。
        code_n: 代码抽样数。
        table_all: 表格是否全量(默认 True, 保非空)。

    Returns:
        SampleSelection: 各形态选中的 md 相对路径列表。
    """
    root = Path(data_dir)
    table_docs: list[str] = []   # 含 pipe-table(纯表格 + 表格+代码混合)
    code_only: list[str] = []    # 含代码且无表格
    plain_docs: list[str] = []   # 两者皆无
    mixed_docs: list[str] = []   # 表格+代码混合(报告用, 是 table 的子集)
    for path in sorted(root.rglob("*.md")):
        rel = str(path.relative_to(root)).replace("\\", "/")
        content = path.read_text(encoding="utf-8", errors="replace")
        has_table = bool(_PIPE_TABLE_RE.search(content))
        has_code = bool(_FENCED_CODE_RE.search(content))
        if has_table and has_code:
            table_docs.append(rel)
            mixed_docs.append(rel)
        elif has_table:
            table_docs.append(rel)
        elif has_code:
            code_only.append(rel)
        else:
            plain_docs.append(rel)
    return SampleSelection(
        plain=plain_docs[:plain_n],
        table=table_docs if table_all else table_docs[:plain_n],
        code=code_only[:code_n],
        mixed=mixed_docs,
    )


def generate_gold(md_path: str) -> str:
    """生成 gold: 原 md 内容(端到端管线保真度的基准, 转制 + 解析后与原 md 对比)。

    Args:
        md_path: 原 md 文件路径。

    Returns:
        str: gold 文本(原 md 内容)。
    """
    return Path(md_path).read_text(encoding="utf-8", errors="replace")


def _find_browser() -> str | None:
    """查找可用的 Edge/Chromium headless 可执行文件(供 print-to-pdf)。

    Returns:
        str | None: 浏览器可执行文件路径; 未找到返回 None。
    """
    for candidate in _EDGE_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return shutil.which("chrome") or shutil.which("msedge")


def _require_pandoc() -> str:
    """定位 pandoc 可执行文件, 未安装时抛明确错误。

    优先用 config.PANDOC_PATH(显式路径, 未加 PATH 时用), 否则回退 shutil.which("pandoc")。

    Returns:
        str: pandoc 可执行文件路径。

    Raises:
        RuntimeError: pandoc 未安装(提示用户安装)。
    """
    from config import PANDOC_PATH
    if PANDOC_PATH and Path(PANDOC_PATH).exists():
        return PANDOC_PATH
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError(
            "pandoc 未安装, 无法转制 html/docx。请先安装: winget install pandoc 或 choco install pandoc, "
            "或在 .env 配置 PANDOC_PATH 指向 pandoc.exe"
        )
    return pandoc


def convert_md_to_html(md_path: str, out_path: str) -> None:
    """md -> html(经 pandoc, 保留标题/表格/代码块结构)。

    Args:
        md_path: 原 md 文件路径。
        out_path: 输出 html 文件路径。

    Raises:
        RuntimeError: pandoc 未安装。
    """
    pandoc = _require_pandoc()
    subprocess.run([pandoc, md_path, "-o", out_path, "--standalone"], check=True)


def convert_md_to_pdf(md_path: str, out_path: str) -> None:
    """md -> pdf: md -> html -> Edge/Chromium headless --print-to-pdf。

    --print-to-pdf 必须用绝对路径(相对路径会让 Edge 静默失败, 返回 0 但无输出);
    中间 html 写入系统临时目录并在转完后清理, 避免污染输出目录。

    Args:
        md_path: 原 md 文件路径。
        out_path: 输出 pdf 文件路径。

    Raises:
        RuntimeError: pandoc 未安装或浏览器未找到。
    """
    pandoc = _require_pandoc()
    browser = _find_browser()
    if browser is None:
        raise RuntimeError("未找到 Edge/Chromium, 无法 print-to-pdf")
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix="corpus_pdf_"))
    html_path = temp_dir / f"{Path(out_path).stem}.html"
    convert_md_to_html(md_path, str(html_path))
    pdf_abs = str(Path(out_path).resolve())
    try:
        subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-sandbox",
             f"--print-to-pdf={pdf_abs}",
             f"file:///{html_path.resolve().as_posix()}"],
            check=True,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def convert_md_to_docx(md_path: str, out_path: str) -> None:
    """md -> docx(经 pandoc, 保留标题层级/表格/代码块/列表)。

    Args:
        md_path: 原 md 文件路径。
        out_path: 输出 docx 文件路径。

    Raises:
        RuntimeError: pandoc 未安装。
    """
    pandoc = _require_pandoc()
    subprocess.run([pandoc, md_path, "-o", out_path], check=True)


def convert_md_to_pptx(md_path: str, out_path: str) -> None:
    """md -> pptx(经 python-pptx): 标题 -> slide 标题, 段落 -> 文本框, 代码块 -> 文本框。

    pptx 会把 md 层级拍平成 slide, 结构保真度按格式定义可达上限(见 011-4 2.5)。

    Args:
        md_path: 原 md 文件路径。
        out_path: 输出 pptx 文件路径。
    """
    from pptx import Presentation
    from pptx.util import Inches

    content = Path(md_path).read_text(encoding="utf-8", errors="replace")
    presentation = Presentation()
    blank = presentation.slide_layouts[6]  # 空白版式
    current_slide = None
    current_text = []
    in_code = False

    def flush() -> None:
        """把当前收集的文本写入当前 slide 的文本框。"""
        nonlocal current_text
        if current_slide is None or not current_text:
            current_text = []
            return
        textbox = current_slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(6))
        textbox.text_frame.text = "\n".join(current_text)
        current_text = []

    for line in content.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            current_text.append(line)
            continue
        if line.startswith("#"):
            flush()
            current_slide = presentation.slides.add_slide(blank)
            title = current_slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
            title.text_frame.text = line.lstrip("#").strip()
            continue
        if line.strip():
            current_text.append(line)
    flush()
    presentation.save(out_path)


def snapshot_converted(corpus_dir: str, out_manifest: str) -> dict:
    """转制语料 SHA 快照(相对路径 -> sha256), 供 011-6/012-1 复用与校验。

    Args:
        corpus_dir: 转制语料目录。
        out_manifest: 输出 manifest 文件路径。

    Returns:
        dict: {"version", "created_at", "files": {相对路径: sha256}}。
    """
    root = Path(corpus_dir)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root)).replace("\\", "/")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
        files[rel] = digest.hexdigest()
    manifest = {
        "version": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }
    Path(out_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(out_manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器: select / convert / snapshot 三个子命令。"""
    parser = argparse.ArgumentParser(prog="corpus_convert", description="多格式解析语料转制(011-4)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_select = subparsers.add_parser("select", help="按内容形态分层选择样本")
    parser_select.add_argument("--data", default=str(_PROJECT_ROOT / "data"), help="data/ 目录")
    parser_select.add_argument("--out", default="artifacts/samples.json", help="样本清单输出路径")

    parser_convert = subparsers.add_parser("convert", help="把选中样本转制为 4 格式")
    parser_convert.add_argument("--samples", default="artifacts/samples.json", help="样本清单路径")
    parser_convert.add_argument("--out", default="artifacts/corpus_multiformat", help="转制语料输出目录")
    parser_convert.add_argument("--gold", default="artifacts/gold", help="gold 输出目录(原 md)")

    parser_snapshot = subparsers.add_parser("snapshot", help="转制语料 SHA 快照")
    parser_snapshot.add_argument("--corpus", default="artifacts/corpus_multiformat", help="转制语料目录")
    parser_snapshot.add_argument("--out", default="artifacts/corpus_manifest.json", help="manifest 输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 返回进程退出码(0=成功, 1=失败)。

    Args:
        argv: 命令行参数列表, 缺省取 sys.argv。

    Returns:
        int: 进程退出码。
    """
    args = _build_parser().parse_args(argv)
    if args.command == "select":
        selection = select_samples(args.data)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"plain": selection.plain, "table": selection.table,
                        "code": selection.code, "mixed": selection.mixed},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"样本已选择: plain={len(selection.plain)} table={len(selection.table)} "
              f"code={len(selection.code)} mixed={len(selection.mixed)} -> {args.out}")
        return 0
    if args.command == "convert":
        samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))
        # 去重保序(mixed 是 table 的子集, 避免重复转换)
        seen: set[str] = set()
        all_paths: list[str] = []
        for rel in (samples.get("plain", []) + samples.get("table", []) +
                    samples.get("code", []) + samples.get("mixed", [])):
            if rel not in seen:
                seen.add(rel)
                all_paths.append(rel)
        out_root = Path(args.out)
        gold_root = Path(args.gold)
        out_root.mkdir(parents=True, exist_ok=True)
        gold_root.mkdir(parents=True, exist_ok=True)
        converters = (("html", convert_md_to_html), ("pdf", convert_md_to_pdf),
                      ("docx", convert_md_to_docx), ("pptx", convert_md_to_pptx))
        for rel in all_paths:
            md_path = _PROJECT_ROOT / "data" / rel
            stem = Path(rel).stem
            # gold: 原 md 内容(端到端保真度基准)
            (gold_root / f"{stem}.md").write_text(
                md_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            for fmt, convert_fn in converters:
                fmt_dir = out_root / fmt
                fmt_dir.mkdir(parents=True, exist_ok=True)
                convert_fn(str(md_path), str(fmt_dir / f"{stem}.{fmt}"))
        print(f"已转制 {len(all_paths)} 篇 x 4 格式 -> {out_root}, gold -> {gold_root}")
        return 0
    if args.command == "snapshot":
        manifest = snapshot_converted(args.corpus, args.out)
        print(f"转制语料快照: {len(manifest['files'])} 文件 -> {args.out}")
        return 0
    return 1  # pragma: no cover  # argparse required subparsers 保证 command 恒为三者之一


if __name__ == "__main__":
    sys.exit(main())
