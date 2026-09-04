"""多格式解析质量基线评测: 格式解析质量评测的统一 runner
得到的数据表示端到端管线保真度(转制 + 解析), 不读成绝对解析质量。(因为转制本身会损失一定质量)

意义不在于 artifacts/baseline.json 中的数据, 而在于暂停和校准:
- 在重型后端崩溃/超时/空产出/依赖缺失时, 触发暂停，而不是静默产出烂结果（运行时健壮性代码在 indexing/parse_backends/内, 本方仅为调用者）
- 确保在重型接入前 parse_metrics sane, 且 语料/gold 可用

parse_baseline 不进 eval/runner 主链路 (它属于产出语料质量评测, 与 retrieval/full 语义本质不同)

核心机制:
    - 对每篇样本: 用对应后端解析转制文件 -> 归一化 MD, 与原 md gold 对比 (同源一致性 = 端到端保真度)。
    - 指标: 文本召回率 / 文本精确率 / 结构保真度(标题/表格/代码块), 见 eval/core/parse_metrics。
    - 报告落盘: 记录格式 / 内容形态 / 指标 / 语料版本 / cleaner 配置(两侧对称) / 格式可达上限语义。默认路径 artifacts/baseline.json
    - 差分有效前提: 同 gold + 同指标实现 + 同 cleaner 对称配置。

用法示例::

    uv run python scripts/parse_baseline.py --corpus artifacts/corpus_multiformat/ --gold artifacts/gold/ --out artifacts/baseline.json
    uv run python scripts/parse_baseline.py --backend heavy-chain --corpus ... --gold ... --out artifacts/diff.json

公共接口:
    - compute_sample_metrics: 单篇样本指标(文本召回/精确 + 结构保真度)
    - aggregate_report: 按格式/内容形态聚合指标
    - run_baseline: 跑完整基线(解析 + 指标 + 聚合 + 落盘)
    - main: argparse CLI 入口
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from config import _PROJECT_ROOT
from eval.core.parse_metrics import text_recall, text_precision, structure_fidelity

# 支持的解析后端
SUPPORTED_BACKENDS = ("lightweight", "docling", "mineru", "marker", "markitdown",
                      "heavy-chain", "vlm", "colpali")


def _make_parse_fn(backend: str) -> Callable[[str], str]:
    """按后端名构造解析函数（签名 (file_path) -> str，返回归一化 MD）。

    Args:
        backend: 后端名（lightweight / 单个重型 / heavy-chain）。

    Returns:
        Callable[[str], str]: 解析函数。

    Raises:
        ValueError: 后端名不支持。
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"不支持的 backend: {backend}, 可选 {SUPPORTED_BACKENDS}")
    if backend == "lightweight":
        return _lightweight_parse
    if backend == "heavy-chain":
        return _heavy_chain_parse
    if backend == "colpali":
        raise ValueError("colpali 是视觉检索后端, 不产出 Markdown, 不能作为解析后端")
    # 其余(docling/mineru/markitdown/vlm/marker)统一走注册表: 门控关/依赖未装时 resolve 返回 None
    def _single(path: str) -> str:
        from indexing.parse_backends import resolve
        resolved = resolve(backend)
        if resolved is None:
            raise RuntimeError(f"后端 {backend} 未启用或依赖未装")
        result = resolved.extract(path)
        return result.markdown
    return _single


def compute_sample_metrics(parsed_md: str, gold_md: str) -> dict:
    """单篇样本指标: 文本召回率/精确率 + 结构保真度。

    Args:
        parsed_md: 解析出的归一化 MD。
        gold_md: 原 md gold。

    Returns:
        dict: {"text_recall", "text_precision", "structure": {heading/table/code/overall}}。
    """
    structure = structure_fidelity(parsed_md, gold_md)
    return {
        "text_recall": text_recall(parsed_md, gold_md),
        "text_precision": text_precision(parsed_md, gold_md),
        "structure": structure,
    }


def _mean(values: list[float]) -> float:
    """列表均值(空列表返回 0.0)。"""
    return statistics.mean(values) if values else 0.0


def _aggregate_group(items: list[dict]) -> dict:
    """单组聚合(均值 + 样本数)。"""
    return {
        "text_recall": _mean([item["text_recall"] for item in items]),
        "text_precision": _mean([item["text_precision"] for item in items]),
        "structure_overall": _mean([item["structure"]["overall"] for item in items]),
        "num_samples": len(items),
    }


def aggregate_report(per_sample: list[dict]) -> dict:
    """按格式/内容形态聚合指标(均值)。

    Args:
        per_sample: 每篇样本的指标 dict(含 format/morphology 字段)。

    Returns:
        dict: {"overall", "by_format", "by_morphology"} 聚合指标。
    """
    overall = {
        "text_recall": _mean([item["text_recall"] for item in per_sample]),
        "text_precision": _mean([item["text_precision"] for item in per_sample]),
        "structure_overall": _mean([item["structure"]["overall"] for item in per_sample]),
        "num_samples": len(per_sample),
    }
    by_format: dict[str, dict] = {}
    by_morphology: dict[str, dict] = {}
    for item in per_sample:
        for bucket, key in ((by_format, item.get("format", "unknown")),
                            (by_morphology, item.get("morphology", "unknown"))):
            bucket.setdefault(key, []).append(item)
    return {
        "overall": overall,
        "by_format": {key: _aggregate_group(items) for key, items in by_format.items()},
        "by_morphology": {key: _aggregate_group(items) for key, items in by_morphology.items()},
    }


def run_baseline(corpus_dir: str, gold_dir: str, backend: str = "lightweight",
                 parse_fn=None) -> dict:
    """跑完整基线: 遍历样本 -> 解析 -> 指标 -> 聚合。

    Args:
        corpus_dir: 转制语料目录(按格式子目录组织, 如 pdf/xxx.pdf)。
        gold_dir: gold 目录(原 md, 与转制样本同名)。
        backend: 解析后端(默认 lightweight)。
        parse_fn: 可注入的解析函数(测试用), 签名 (file_path) -> str(归一化 MD);
            缺省用 indexing.loaders 轻量解析。

    Returns:
        dict: 基线报告(overall + by_format + by_morphology + 版本头)。
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"不支持的 backend: {backend}, 可选 {SUPPORTED_BACKENDS}")
    if parse_fn is None:
        parse_fn = _make_parse_fn(backend)
    per_sample: list[dict] = []
    failures: list[dict] = []
    corpus_root = Path(corpus_dir)
    gold_root = Path(gold_dir)
    for format_dir in sorted(corpus_root.iterdir()):
        if not format_dir.is_dir():
            continue
        for converted in sorted(format_dir.glob("*")):
            if not converted.is_file():
                continue
            stem = converted.stem
            gold_path = gold_root / f"{stem}.md"
            if not gold_path.exists():
                continue
            gold_md = gold_path.read_text(encoding="utf-8", errors="replace")
            morphology = _morphology(gold_md)
            try:
                parsed_md = parse_fn(str(converted))
            except Exception as exc:  # noqa: BLE001 - 失败记账, 不落轻量
                failures.append({
                    "file": str(converted), "format": format_dir.name,
                    "morphology": morphology, "backend": backend,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            metrics = compute_sample_metrics(parsed_md, gold_md)
            metrics["format"] = format_dir.name
            metrics["morphology"] = morphology
            per_sample.append(metrics)
    report = aggregate_report(per_sample)
    total = len(per_sample) + len(failures)
    report["header"] = {
        "backend": backend,
        "corpus_dir": corpus_dir,
        "gold_dir": gold_dir,
        "cleaner_symmetric": _cleaner_symmetric(),  # 011-4 2.2: 两侧 cleaner 对称(默认都关, 禁止单侧开)
        "num_success": len(per_sample),
        "num_failed": len(failures),
        "total": total,
        "success_rate": (len(per_sample) / total) if total else 0.0,
    }
    if failures:
        report["failures"] = failures
    return report


def _morphology(gold_md: str) -> str:
    """按 gold 内容形态分类(复用 corpus_convert.classify_md 语义, 避免循环依赖)。"""
    from scripts.corpus_convert import classify_md
    return classify_md(gold_md)


def _cleaner_symmetric() -> bool:
    """011-4 2.2: 两侧 cleaner 是否对称(默认都关, 禁止单侧开)。

    Returns:
        bool: CLEAN_PLAIN_TEXT 与 CLEAN_NEW_FORMAT 取值一致时为 True。
    """
    from config import CLEAN_NEW_FORMAT, CLEAN_PLAIN_TEXT
    return CLEAN_PLAIN_TEXT == CLEAN_NEW_FORMAT


def _lightweight_parse(file_path: str) -> str:
    """轻量后端解析: 经 indexing.loaders 归一化 MD(008 已接的轻量 loader)。

    Args:
        file_path: 转制文件路径。

    Returns:
        str: 归一化 MD。
    """
    from indexing.loaders import load_multiformat
    from preprocess.format_precheck import precheck
    from pathlib import Path as _Path
    path = _Path(file_path)
    precheck_result = precheck(path)
    markdown, _ = load_multiformat(path, precheck_result)
    return markdown


def _heavy_chain_parse(file_path: str) -> str:
    """重型文本链解析: Docling -> MinerU -> MarkItDown(归一化 MD)。

    Args:
        file_path: 转制文件路径。

    Returns:
        str: 归一化 MD。

    Raises:
        RuntimeError: 全链不可用/失败(不落轻量)。
    """
    from indexing.parse_backends import heavy_chain_extract
    markdown, _ = heavy_chain_extract(file_path)
    return markdown


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器: --corpus / --gold / --backend / --out。"""
    parser = argparse.ArgumentParser(prog="parse_baseline", description="多格式解析质量基线(011-6)")
    parser.add_argument("--corpus", default="artifacts/corpus_multiformat", help="转制语料目录")
    parser.add_argument("--gold", default="artifacts/gold", help="gold 目录(原 md)")
    parser.add_argument("--backend", default="lightweight", choices=SUPPORTED_BACKENDS,
                        help="解析后端(默认 lightweight)")
    parser.add_argument("--out", default="artifacts/baseline.json", help="基线报告输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 返回进程退出码(0=成功, 1=失败)。

    Args:
        argv: 命令行参数列表, 缺省取 sys.argv。

    Returns:
        int: 进程退出码。
    """
    args = _build_parser().parse_args(argv)
    report = run_baseline(args.corpus, args.gold, backend=args.backend)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = report["overall"]
    print(f"基线报告已落盘: {args.out}")
    print(f"  backend={args.backend} 样本={overall['num_samples']} "
          f"text_recall={overall['text_recall']:.4f} "
          f"text_precision={overall['text_precision']:.4f} "
          f"structure_overall={overall['structure_overall']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
