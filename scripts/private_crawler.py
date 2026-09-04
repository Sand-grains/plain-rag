"""定向爬虫: 爬取用户提供的指定的URL, 正文抽取, 清洗落盘(本地存储不发布)

流程: 逐 URL -> requests 抓取(UA+超时) -> trafilatura 抽正文 -> cleaner.clean 清洗
      -> 存 data/Crawler/<slug>.md -> 写 manifest.json(url/path/source/title/chars)。
可导入本地 html

用法::
    uv run python scripts/private_crawler.py --urls urls.txt --out data/Crawler --delay 2

产物: data/Crawler/*.md + data/Crawler/manifest.json(本地存储, 不发布)。
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocess.cleaner import clean

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "Crawler"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_TIMEOUT_S = 20.0

# 文件名安全: 去非法字符, 保留中文/字母/数字/连字符
_SLUG_RE = re.compile(r"[^\w\u4e00-\u9fff-]+")


def slugify(url: str, index: int) -> str:
    """从 URL 生成安全文件名(去 query/fragment, 取路径末段, 非法字符替换为 -)。

    Args:
        url: 文章 URL。
        index: 序号(避免同路径冲突)。

    Returns:
        str: 安全文件名(不含后缀)。
    """
    path = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    stem = path.rsplit("/", 1)[-1] or "article"
    stem = _SLUG_RE.sub("-", stem).strip("-")
    return f"{index:04d}-{stem or 'article'}"


def fetch_html(url: str, timeout_s: float = _TIMEOUT_S) -> str:
    """抓取 URL 返回 HTML 文本(带 UA, 超时)。

    Args:
        url: 文章 URL。
        timeout_s: 超时(秒)。

    Returns:
        str: HTML 文本。

    Raises:
        RuntimeError: 抓取失败(网络/非 2xx)。
    """
    import requests
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"抓取失败 {url}: HTTP {resp.status_code}")
    return resp.text


def extract_main_text(html: str) -> str:
    """用 trafilatura 抽取正文(去导航/侧栏/页脚噪声)。

    Args:
        html: HTML 文本。

    Returns:
        str: 抽取的正文文本(失败返回空串)。
    """
    import trafilatura
    return trafilatura.extract(html) or ""


def _save_article(text: str, out_dir: Path, index: int) -> dict[str, Any]:
    """清洗并落盘一篇文章, 返回 manifest 条目(文件名按序号, 避免跨源冲突)。"""
    cleaned = clean(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{index:04d}.md"
    path.write_text(cleaned, encoding="utf-8")
    return {
        "url": "",
        "path": str(path),
        "source": "local",
        "title": path.stem,
        "chars": len(cleaned),
    }


def crawl_one(url: str, out_dir: Path, index: int, delay_s: float, source: str | None = None) -> dict[str, Any]:
    """抓取并清洗单篇文章, 落盘 .md, 返回 manifest 条目。

    Args:
        url: 文章 URL。
        out_dir: 输出目录。
        index: 序号。
        delay_s: 抓取前限频延迟(秒)。
        source: 来源标签(缺省按域名判定)。

    Returns:
        dict: manifest 条目 {url, path, source, title, chars}。

    Raises:
        RuntimeError: 抓取/抽取失败或正文为空。
    """
    if delay_s > 0:
        time.sleep(delay_s)
    html = fetch_html(url)
    text = extract_main_text(html)
    if not text.strip():
        raise RuntimeError(f"正文为空: {url}")
    entry = _save_article(text, out_dir, index)
    entry["url"] = url
    entry["source"] = source or _source_of(url)
    entry["title"] = slugify(url, index)  # L4: title 与文件名语义一致(序号+slug)
    return entry


def import_html_file(html_path: str | Path, out_dir: Path, index: int) -> dict[str, Any]:
    """导入本地 HTML 文件: 读文件 -> 抽正文 -> 清洗 -> 落盘 .md。

    用于复用已下载的本地页面(如 benchmark/route_sample/*.html), 免去网络抓取。

    Args:
        html_path: 本地 HTML 文件路径。
        out_dir: 输出目录。
        index: 序号。

    Returns:
        dict: manifest 条目 {url, path, source, title, chars}。

    Raises:
        RuntimeError: 抽取正文为空。
    """
    html = Path(html_path).read_text(encoding="utf-8")
    text = extract_main_text(html)
    if not text.strip():
        raise RuntimeError(f"正文为空: {html_path}")
    entry = _save_article(text, out_dir, index)
    entry["source"] = Path(html_path).parent.name or "local"
    entry["title"] = Path(html_path).stem  # L4: 本地导入用源文件名作 title
    return entry


def _source_of(url: str) -> str:
    """按域名判定来源(掘金/CSDN/知乎/其他)。"""
    if "juejin" in url:
        return "juejin"
    if "csdn" in url:
        return "csdn"
    if "zhihu" in url:
        return "zhihu"
    return "other"


def read_urls(path: str | Path) -> list[str]:
    """读取 URL 列表文件(每行一个, 去空行/注释)。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def write_manifest(out_dir: Path, entries: list[dict[str, Any]]) -> None:
    """把 manifest 条目落盘为 JSON。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        __import__("json").dumps({"articles": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 先导入本地 HTML(--import-dir, 可选), 再逐 URL 爬取(--urls)。"""
    parser = argparse.ArgumentParser(description="011-2 定向爬取检索集")
    parser.add_argument("--urls", help="URL 列表文件(每行一个); 与 --import-dir 至少一个")
    parser.add_argument("--import-dir", help="本地 HTML 目录(如 benchmark/route_sample), 导入免网络")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="输出目录(默认 data/Crawler)")
    parser.add_argument("--delay", type=float, default=2.0, help="限频延迟秒(默认 2)")
    args = parser.parse_args(argv)

    if not args.urls and not args.import_dir:
        print("错误: 需提供 --urls 或 --import-dir")
        return 2

    out_dir = Path(args.out)
    entries: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []

    # 1) 先导入本地 HTML(免网络, index 从 0 起)
    if args.import_dir:
        html_files = sorted(Path(args.import_dir).glob("*.html"))
        for index, html_file in enumerate(html_files):
            try:
                entries.append(import_html_file(html_file, out_dir, index))
                print(f"[import-ok] {html_file.name}")
            except Exception as exc:  # noqa: BLE001 - 单条失败不阻塞
                failed.append((str(html_file), str(exc)))
                print(f"[import-fail] {html_file.name}: {exc}")

    # 2) 再爬取 URL(从 import 数量后继续编号)
    base = len(entries)
    if args.urls:
        urls = read_urls(args.urls)
        for offset, url in enumerate(urls):
            index = base + offset
            try:
                entries.append(crawl_one(url, out_dir, index, args.delay))
                print(f"[ok] {url}")
            except Exception as exc:  # noqa: BLE001 - 单条失败不阻塞, 记入 failed
                failed.append((url, str(exc)))
                print(f"[fail] {url}: {exc}")

    write_manifest(out_dir, entries)
    print(f"完成: {len(entries)} 成功, {len(failed)} 失败 -> {out_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
