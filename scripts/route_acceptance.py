"""路由验收: 在真实爬虫 HTML 小样本上验证 precheck 路由的四类 precision/recall/误判代价。

路由验收专用小样本(20~50 篇真实 HTML, 覆盖导航/广告/正文混杂)。
gold 为人工标注的"应走文本 / 应走 VLM / 应走 ColPali / 应跳过"四类(文件名 -> 标签), 禁止默认全 text(否则指标失去意义)。

预测标签由 precheck 决策 + colpali_triggered 映射:
  - colpali_triggered -> "colpali"(图表面积/表格占比触发, 并行路由)
  - WHOLE_TEXT_PIPELINE / HTML_TEXT -> "text"
  - VLM_TEXT_PIPELINE -> "vlm"
  - SKIP_TEXT_PIPELINE -> "skip"

误判代价: 把文本页误判为 skip(丢弃)最贵(w_fn=2), 其余误判 w=1。

用法示例::

    uv run python scripts/route_acceptance.py --sample-dir benchmark/route_sample \
        --gold route_gold.json --out artifacts/route_report.json

公共接口:
    - route_acceptance: 跑路由验收(遍历样本 -> precheck -> 四类指标 -> 落盘)
    - _route_label: 决策枚举 + colpali_triggered -> 路由标签
    - _metrics: 四类 precision/recall/f1/accuracy/误判代价
    - main: argparse CLI 入口
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from config import _PROJECT_ROOT
from preprocess.format_precheck import DispatchDecision, precheck

# 误判代价权重: 把文本页误判为 skip(丢弃)比把 skip 误判为 text(多解析)更贵
_W_FN = 2.0
_W_FP = 1.0

# 四类路由标签
_LABELS = ("text", "vlm", "colpali", "skip")


def _route_label(decision: DispatchDecision, colpali_triggered: bool = False) -> str:
    """把 precheck 决策 + colpali_triggered 映射为四类路由标签。

    Args:
        decision: precheck 分流决策。
        colpali_triggered: 是否触发 ColPali 视觉检索管线(并行触发, 优先于文本/VLM 决策)。

    Returns:
        str: "text" / "vlm" / "colpali" / "skip"。
    """
    if colpali_triggered:
        return "colpali"
    if decision in (DispatchDecision.HTML_TEXT, DispatchDecision.WHOLE_TEXT_PIPELINE):
        return "text"
    if decision == DispatchDecision.VLM_TEXT_PIPELINE:
        return "vlm"
    return "skip"


def _metrics(gold: list[str], predicted: list[str]) -> dict:
    """四类路由指标: 每类 precision/recall/f1 + 整体 accuracy + 误判代价。

    Args:
        gold: 每篇 gold 标签(text/vlm/colpali/skip)。
        predicted: 每篇预测标签(text/vlm/colpali/skip)。

    Returns:
        dict: {"accuracy", "per_class", "misroute_cost"}。
    """
    labels = sorted(set(gold) | set(predicted))
    per_class: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
    accuracy = (sum(1 for g, p in zip(gold, predicted) if g == p) / len(gold)) if gold else 0.0
    cost = 0.0
    for g, p in zip(gold, predicted):
        if g == p:
            continue
        cost += _W_FN if (g == "text" and p == "skip") else _W_FP
    return {
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "misroute_cost": round(cost, 4),
    }


def route_acceptance(sample_dir: str, gold: dict[str, str]) -> dict:
    """跑路由验收: 遍历样本目录顶层 HTML -> precheck -> 四类指标。

    Args:
        sample_dir: 样本目录(只取顶层 .html, 忽略 *_files 资源子目录)。
        gold: 文件名 -> 标签 映射(text/vlm/colpali/skip), 必须提供(人工标注)。

    Returns:
        dict: 路由验收报告(逐篇决策表 + 四类指标)。

    Raises:
        ValueError: 样本目录无顶层 HTML, 或未提供 gold。
    """
    root = Path(sample_dir)
    html_files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".html", ".htm"))
    if not html_files:
        raise ValueError(f"样本目录无顶层 HTML: {sample_dir}")
    if not gold:
        raise ValueError("路由验收需要人工标注 gold(文件名 -> text/vlm/colpali/skip), 禁止默认全 text")

    rows: list[dict] = []
    gold_labels: list[str] = []
    predicted_labels: list[str] = []
    for path in html_files:
        result = precheck(path)
        predicted = _route_label(result.doc_decision, result.colpali_triggered)
        label = gold.get(path.name)
        if label is None:
            raise ValueError(f"gold 缺失样本: {path.name}(需标注 text/vlm/colpali/skip)")
        gold_labels.append(label)
        predicted_labels.append(predicted)
        rows.append({
            "file": path.name,
            "gold": label,
            "predicted": predicted,
            "hit": label == predicted,
            "decision": result.doc_decision.value,
            "colpali_triggered": result.colpali_triggered,
            "main_text_chars": result.sampling_format_stats.get("main_text_chars", 0),
            "image_count": result.sampling_format_stats.get("image_count", 0),
            "degraded_flags": result.degraded_flags,
        })

    metrics = _metrics(gold_labels, predicted_labels)
    return {
        "sample_dir": sample_dir,
        "num_samples": len(html_files),
        "gold_source": "provided",
        "metrics": metrics,
        "per_sample": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    """构建 CLI 解析器: --sample-dir / --gold / --out。"""
    parser = argparse.ArgumentParser(prog="route_acceptance", description="precheck 路由验收(012-1)")
    parser.add_argument("--sample-dir", default="benchmark/route_sample", help="真实 HTML 样本目录")
    parser.add_argument("--gold", required=True, help="gold 标注 JSON(文件名->text/vlm/colpali/skip)")
    parser.add_argument("--out", default="artifacts/route_report.json", help="路由验收报告输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口: 返回进程退出码(0=成功, 1=失败)。"""
    args = _build_parser().parse_args(argv)
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    report = route_acceptance(args.sample_dir, gold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = report["metrics"]
    print(f"路由验收报告已落盘: {args.out}")
    print(f"  样本={report['num_samples']} gold来源={report['gold_source']}")
    print(f"  accuracy={metrics['accuracy']} 误判代价={metrics['misroute_cost']}")
    for label, m in metrics["per_class"].items():
        print(f"  {label}: precision={m['precision']} recall={m['recall']} f1={m['f1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
