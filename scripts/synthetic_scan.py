""" 扫描件合成: 原 gold md 渲染成图, 加噪, 歪斜 （gold 仍是原 md）
复用已产出的 pdf (原 md -> html -> Edge print-to-pdf) 作为"渲染成图"的中间件:
pdf 首页 -> pymupdf 渲染 png -> PIL 加高斯噪声 + 轻微旋转(模拟扫描件) -> 扫描图。

输出 manifest 记录 扫描图 -> gold md 映射

样本量 >= 20 篇, 覆盖中文 OCR 与结构可读性; 作为计入主指标的独立分层

用法示例::

    uv run python scripts/synthetic_scan.py --gold artifacts/gold --pdf artifacts/corpus_multiformat/pdf --out artifacts/scan_gold

公共接口:
    - render_pdf_page_to_png: pdf 指定页渲染为 png
    - add_scan_noise: 对 png 加高斯噪声 + 旋转(模拟扫描件)
    - build_scan_gold: 遍历 gold 生成扫描图 + manifest
    - main: argparse CLI 入口
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 独立运行(uv run python scripts/synthetic_scan.py)时确保能 import config(硬约束 7: __file__ 推导)
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from config import _PROJECT_ROOT


def render_pdf_page_to_png(pdf_path: str, out_png: str, page: int = 0, dpi: int = 150) -> None:
    """把 pdf 指定页渲染为 png(经 pymupdf)。

    Args:
        pdf_path: pdf 文件路径。
        out_png: 输出 png 路径。
        page: 页索引(默认 0)。
        dpi: 渲染分辨率(默认 150)。
    """
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pix = doc[page].get_pixmap(dpi=dpi)
    pix.save(out_png)
    doc.close()


def add_scan_noise(png_path: str, out_path: str, noise: float = 0.02, skew: float = 1.0) -> None:
    """对 png 加高斯噪声 + 轻微旋转, 模拟扫描件。

    Args:
        png_path: 输入 png 路径。
        out_path: 输出 png 路径。
        noise: 高斯噪声标准差(相对 255 的比例, 默认 0.02)。
        skew: 旋转角度(度, 默认 1.0; 0 表示不旋转)。
    """
    import numpy as np
    from PIL import Image
    img = Image.open(png_path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.normal(0.0, noise * 255.0, arr.shape)
    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    if skew:
        img = img.rotate(skew, resample=Image.BICUBIC, fillcolor=255)
    img.save(out_path)


def build_scan_gold(gold_dir: str, pdf_dir: str, out_dir: str,
                    limit: int | None = None, dpi: int = 150,
                    noise: float = 0.02, skew: float = 1.0) -> dict:
    """遍历 gold 生成扫描图 + manifest。

    Args:
        gold_dir: gold 目录(原 md)。
        pdf_dir: pdf 目录(与 gold 同名)。
        out_dir: 扫描图输出目录。
        limit: 最多生成篇数(默认全部; 用于测试)。
        dpi: 渲染分辨率。
        noise: 噪声强度。
        skew: 旋转角度。

    Returns:
        dict: manifest {"samples": [{scan, gold, stem}], "num_samples", "params"}。
    """
    gold_root = Path(gold_dir)
    pdf_root = Path(pdf_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    md_files = sorted(gold_root.glob("*.md"))
    if limit is not None:
        md_files = md_files[:limit]
    for md_path in md_files:
        stem = md_path.stem
        pdf_path = pdf_root / f"{stem}.pdf"
        if not pdf_path.exists():
            continue
        scan_path = out_root / f"{stem}.png"
        render_pdf_page_to_png(str(pdf_path), str(scan_path), dpi=dpi)
        add_scan_noise(str(scan_path), str(scan_path), noise=noise, skew=skew)
        samples.append({"scan": scan_path.name, "gold": md_path.name, "stem": stem})

    manifest = {
        "samples": samples,
        "num_samples": len(samples),
        "params": {"dpi": dpi, "noise": noise, "skew": skew, "gold_dir": gold_dir, "pdf_dir": pdf_dir},
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器: --gold / --pdf / --out / --limit / --dpi / --noise / --skew。"""
    parser = argparse.ArgumentParser(prog="synthetic_scan", description="合成扫描件 gold 集生成(012-1)")
    parser.add_argument("--gold", default="artifacts/gold", help="gold 目录(原 md)")
    parser.add_argument("--pdf", default="artifacts/corpus_multiformat/pdf", help="pdf 目录(与 gold 同名)")
    parser.add_argument("--out", default="artifacts/scan_gold", help="扫描图输出目录")
    parser.add_argument("--limit", type=int, default=None, help="最多生成篇数(默认全部)")
    parser.add_argument("--dpi", type=int, default=150, help="渲染分辨率")
    parser.add_argument("--noise", type=float, default=0.02, help="高斯噪声强度")
    parser.add_argument("--skew", type=float, default=1.0, help="旋转角度(度)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 返回进程退出码(0=成功, 1=失败)。"""
    args = _build_parser().parse_args(argv)
    manifest = build_scan_gold(args.gold, args.pdf, args.out,
                               limit=args.limit, dpi=args.dpi, noise=args.noise, skew=args.skew)
    print(f"合成扫描 gold 集已生成: {args.out} (样本 {manifest['num_samples']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
