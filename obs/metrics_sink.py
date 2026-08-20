"""跨 run 指标持久化库(run 结束后 JSONL 落盘, 累计成历史): 把每次 eval run 的指标落成可查、可对比的指标库 (补"跑完即失"短板)。

核心特性:
    - JSONL 落盘 eval/results/metrics/metrics.jsonl, 每行一次 run 的记录(与 timeline/<ts>/ 并存:
      timeline 是单次运行详细报告, sink 是跨 run 趋势数据, 是唯一跨 run 数据源)
    - run 行含 summary 全量 + per_query 纯指标(不含正文) + attribution 归因 + signatures(query 级对齐签名)
    - record_run 由 runner 侧 try/except log-only 包裹(指标库是旁路, 写失败不打断 eval 主流程)
    - 同 run_id 冲突时追加单调后缀(-1/-2, 尽力与 timeline 目录同名), 超过 OBS_METRICS_MAX_RUNS 截尾防膨胀
    - 纯 dict 消费, 不 import eval(保 obs -> eval 分层单向), 文件 I/O 基于 config 路径推导不依赖 CWD

与上层的关系: 数据流由 monitor_metrics(进程内累积) 单向流入 metrics_sink.record_run(落盘 metrics.jsonl)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from datetime import datetime
from pathlib import Path

import config as config_module

logger = logging.getLogger(__name__)


def _metrics_path() -> Path:
    """指标库文件路径(每次实时读 config, 测试可 monkeypatch 目录)。"""
    return Path(config_module.OBS_METRICS_DIR) / "metrics.jsonl"


def query_signature(query_id: str, expected_ids: list[str]) -> str:
    """query 级对齐签名: expected_ids 排序后与 query_id 拼接做 sha256 前 16 位。

    签名只随 (query_id, expected 标注) 变化, 与 expected_ids 顺序无关; 重标注 → 签名变 → 跨 run 自动失配。

    Args:
        query_id: 查询标识。
        expected_ids: benchmark 中标注的 expected_parent_ids。

    Returns:
        str: 16 位十六进制签名。
    """
    material = f"{query_id}|{'|'.join(sorted(expected_ids))}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def record_run(run_id: str, benchmark: str, config: dict, corpus_signature: str,
               test_mode: str, num_queries: int, expected_parent_ids_by_query: dict,
               summary: dict, per_query: dict, attribution: dict,
               timestamp: str | None = None) -> Path:
    """记录一次 run 的指标到指标库(JSONL 追加一行)。

    Args:
        run_id: 运行标识(%Y-%m-%d_%H%M%S, 尽力与 timeline 目录同名)。
        benchmark: benchmark 文件路径。
        config: build_run_info 快照(chunk_size/chunk_overlap/top_k/model_id/git_commit), 供 compare 归因配置变更。
        corpus_signature: 语料签名, compare 时作可对比性校验。
        test_mode: retrieval / full。
        num_queries: query 数。
        expected_parent_ids_by_query: query_id -> expected_parent_ids, 用于算 query 级对齐签名。
        summary: 聚合指标(aggregate + by_category + by_difficulty + layer2(可选) + cost(可选))。
        per_query: query_id -> 纯指标记录(不含 final_context_text/query 正文)。
        attribution: query_id -> {failure_type, evidence}。
        timestamp: 运行时间 ISO 字符串, 缺省取当前时间。

    Returns:
        Path: metrics.jsonl 落盘路径。
    """
    file_path = _metrics_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    line = {
        "run_id": run_id,
        "timestamp": timestamp or datetime.now().isoformat(),
        "test_mode": test_mode,
        "benchmark": benchmark,
        "config": config,
        "corpus_signature": corpus_signature,
        "num_queries": num_queries,
        "signatures": {
            query_id: query_signature(query_id, expected_ids)
            for query_id, expected_ids in expected_parent_ids_by_query.items()
        },
        "summary": summary,
        "per_query": per_query,
        "attribution": attribution,
    }

    existing = load_all()
    used_run_ids = {record["run_id"] for record in existing if "run_id" in record}
    if line["run_id"] in used_run_ids:
        suffix = 1
        while f"{line['run_id']}-{suffix}" in used_run_ids:
            suffix += 1
        line["run_id"] = f"{line['run_id']}-{suffix}"

    with open(file_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")

    existing.append(line)
    max_runs = int(config_module.OBS_METRICS_MAX_RUNS)
    if len(existing) > max_runs:
        _rewrite(file_path, existing[-max_runs:])

    logger.info("指标落库: %s (run_id=%s)", file_path, line["run_id"])
    return file_path


def list_runs() -> list[dict]:
    """列出全部 run 记录, 新到旧(文件是追加的, 倒序即最新在前)。"""
    return list(reversed(load_all()))


def get_run(run_id: str) -> dict | None:
    """按 run_id 查单条 run 记录; 未命中返回 None。

    Args:
        run_id: 运行标识(含同秒冲突时的 -N 后缀)。

    Returns:
        dict | None: 命中返回 run 记录, 未命中返回 None。
    """
    for record in load_all():
        if record.get("run_id") == run_id:
            return record
    return None


def load_all() -> list[dict]:
    """读全部 run 记录(旧到新); 逐行 try/except, 坏行跳过不中断。

    返回:
        list[dict]: 全部可解析的 run 记录; 文件不存在或全坏行时为空列表。
    """
    file_path = _metrics_path()
    if not file_path.exists():
        return []
    records = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                logger.warning("指标库坏行跳过: %s", stripped[:80])
    return records


def _rewrite(file_path: Path, records: list[dict]) -> None:
    """整文件重写(截尾用), 只写给定记录。"""
    with open(file_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---- compare_runs 跨 run 对齐与显著性 ----

# paired(连续/比率项): 在 matched 子集上按 query 配对, 报 Wilcoxon 符号秩 p + Holm 校正
_PAIRED_METRICS = (
    "recall_at_k", "precision_at_k", "mrr", "map_at_k", "ndcg_at_k",
    "child_recall_at_k",
    "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    "answer_correctness",
    "retrieve_ms", "generate_ms",
)

# 稳定子集最小样本数: 低于此值样本过少, 配对检验退化(降纯 delta)
_MIN_PAIRED_SAMPLES = 6

# 显著性阈值(Holm 校正后)
_ALPHA = 0.05


def compare_runs(run_a: dict, run_b: dict) -> dict:
    """跨 run 对齐与显著性检验: 双 run 按 signatures 交集配对, 稳定子集上做 Wilcoxon 配对检验。


    对齐规则:
        - 配对子集 = 双 run 共有的 query 且 query 级对齐签名一致(重标注 → 签名变 → 自动失配)
        - matched < 6 或签名全不匹配 → degenerated=True, 全部降纯 delta 不报 p 值
        - benchmark 追加新 query(旧标注不变)时旧子集仍配对, 不因整体 size 不同而退化

    可对比性: 双 run corpus_signature 不等 → comparable=False + WARN, 语料已变不报 p 降 delta。

    显著性(Wilcoxon 符号秩 + Holm 校正):
        - paired 项: n>=6 才报 p(校正后), 否则 p 留 None
        - delta-only 项: 真二元/聚合计数, 无 per-query 数组, a/b 取 summary 全量, 永不报 p
        - 跨模式(如 retrieval vs full): 某 query 缺该字段即不进配对数组, 缺则自然 delta-only 不崩

    Args:
        run_a: 第一次 run 记录(get_run 的返回 dict)。
        run_b: 第二次 run 记录。

    Returns:
        dict: 含对齐信息与指标对比表(run_a/run_b/alignable/comparable/matched/total_a/total_b/degenerated/metrics)。
    """
    matched_ids, matched = _matched_query_ids(run_a, run_b)
    total_a = len(run_a.get("per_query", {}) or {})
    total_b = len(run_b.get("per_query", {}) or {})
    comparable = run_a.get("corpus_signature") == run_b.get("corpus_signature")
    if not comparable:
        logger.warning("语料签名不一致: %s != %s, 视为语料已变, 不报显著性(降 delta)",
                       run_a.get("corpus_signature"), run_b.get("corpus_signature"))
    degenerated = matched < _MIN_PAIRED_SAMPLES

    paired_rows = _paired_rows(run_a, run_b, matched_ids)
    if paired_rows and not degenerated and comparable:
        value_arrays = _paired_value_arrays(run_a, run_b, matched_ids)
        _apply_holm_correction(paired_rows, value_arrays)
    delta_rows = _delta_only_rows(run_a, run_b)

    return {
        "run_a": run_a.get("run_id"),
        "run_b": run_b.get("run_id"),
        "alignable": matched > 0,
        "comparable": comparable,
        "matched": matched,
        "total_a": total_a,
        "total_b": total_b,
        "degenerated": degenerated,
        "metrics": paired_rows + delta_rows,
    }


def _matched_query_ids(run_a: dict, run_b: dict) -> tuple[list[str], int]:
    """双 run 中签名一致的 query 子集(matched 即交集数)。

    Args:
        run_a: 第一次 run 记录。
        run_b: 第二次 run 记录。

    Returns:
        tuple[list[str], int]: 配对 query_id 列表与交集数。
    """
    sig_a = run_a.get("signatures", {}) or {}
    sig_b = run_b.get("signatures", {}) or {}
    per_query_b = run_b.get("per_query", {}) or {}
    matched_ids = [
        query_id for query_id in (run_a.get("per_query", {}) or {})
        if query_id in per_query_b
        and sig_a.get(query_id) and sig_a.get(query_id) == sig_b.get(query_id)
    ]
    return matched_ids, len(matched_ids)


def _paired_rows(run_a: dict, run_b: dict, matched_ids: list[str]) -> list[dict]:
    """构造 paired 指标行: 在签名稳定的子集上按字段存在性配对, 只收都有值的 query。

    a/b 取该指标有效配对子集的均值, 与 delta/p 同口径对齐(计划 §6 R2)。

    Args:
        run_a: 第一次 run 记录。
        run_b: 第二次 run 记录。
        matched_ids: 签名一致的配对 query 子集。

    Returns:
        list[dict]: 含 name/a/b/delta/n/mode="paired" 的行, p_value/significant 先置 None 待 Holm 校正回填。
    """
    rows = []
    per_query_a = run_a.get("per_query", {}) or {}
    per_query_b = run_b.get("per_query", {}) or {}
    for name in _PAIRED_METRICS:
        pairs = []
        for query_id in matched_ids:
            value_a = per_query_a.get(query_id, {}).get(name)
            value_b = per_query_b.get(query_id, {}).get(name)
            if value_a is None or value_b is None:
                continue
            pairs.append((value_a, value_b))
        if not pairs:
            continue
        a_vals = [pair[0] for pair in pairs]
        b_vals = [pair[1] for pair in pairs]
        rows.append({
            "name": name,
            "a": statistics.mean(a_vals),
            "b": statistics.mean(b_vals),
            "delta": statistics.mean(b_vals) - statistics.mean(a_vals),
            "p_value": None,
            "significant": None,
            "mode": "paired",
            "n": len(pairs),
        })
    return rows


def _apply_holm_correction(paired_rows: list[dict], value_arrays: dict) -> None:
    """对 n>=6 的 paired 行做 Wilcoxon 符号秩检验 + Holm-Bonferroni 校正, 就地回填 p_value/significant。

    Args:
        paired_rows: _paired_rows 产出的行列表(就地修改)。
        value_arrays: name -> (a_vals, b_vals) 配对值数组, 供显著性检验。
    """
    estimable = [row for row in paired_rows if row["n"] >= _MIN_PAIRED_SAMPLES]
    if not estimable:
        return
    for row in estimable:
        a_vals, b_vals = value_arrays[row["name"]]
        row["p_value"] = _wilcoxon_p(a_vals, b_vals)
    raw_p = [row["p_value"] for row in estimable]
    corrected = _holm_correction([value if value is not None else 1.0 for value in raw_p])
    for row, corrected_p in zip(estimable, corrected):
        row["p_value"] = corrected_p
        row["significant"] = corrected_p < _ALPHA


def _paired_value_arrays(run_a: dict, run_b: dict, matched_ids: list[str]) -> dict:
    """paired 指标的配对值数组(name -> (a_vals, b_vals)), 供 Wilcoxon 显著性检验。

    Args:
        run_a: 第一次 run 记录。
        run_b: 第二次 run 记录。
        matched_ids: 签名一致的配对 query 子集。

    Returns:
        dict: name -> (a_vals, b_vals); 无双 run 都有的值则不收录。
    """
    per_query_a = run_a.get("per_query", {}) or {}
    per_query_b = run_b.get("per_query", {}) or {}
    arrays: dict[str, tuple[list[float], list[float]]] = {}
    for name in _PAIRED_METRICS:
        pairs = []
        for query_id in matched_ids:
            value_a = per_query_a.get(query_id, {}).get(name)
            value_b = per_query_b.get(query_id, {}).get(name)
            if value_a is None or value_b is None:
                continue
            pairs.append((value_a, value_b))
        if pairs:
            arrays[name] = ([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    return arrays


def _wilcoxon_p(a_vals: list[float], b_vals: list[float]) -> float | None:
    """Wilcoxon 符号秩双侧检验 p 值; 全 0 差或 scipy 异常时返回 1.0/None 不崩。

    Args:
        a_vals: 第一次 run 的配对值。
        b_vals: 第二次 run 的配对值。

    Returns:
        float | None: 双侧 p 值; 全部差异为 0 时恒 1.0(无差异无从拒绝)。
    """
    from scipy.stats import wilcoxon
    if all(b == a for a, b in zip(a_vals, b_vals)):
        return 1.0
    try:
        p_value = wilcoxon(a_vals, b_vals, alternative="two-sided").pvalue
    except Exception:
        return None
    if p_value is None or math.isnan(float(p_value)):
        return 1.0
    return float(p_value)


def _holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni 校正: 升序排列后 p_i * (N - i + 1) 截断到 1, 多重比较下控假阳性。

    Args:
        p_values: 未校正的原始 p 值列表(顺序与 paired_rows 对齐)。

    Returns:
        list[float]: 校正后 p 值(顺序与输入对齐)。
    """
    length = len(p_values)
    if length == 0:
        return []
    order = sorted(range(length), key=lambda index: p_values[index])
    corrected = [0.0] * length
    for rank, index in enumerate(order, start=1):
        corrected[index] = min(1.0, p_values[index] * (length - rank + 1))
    return corrected


def _delta_only_rows(run_a: dict, run_b: dict) -> list[dict]:
    """构造 delta-only 指标行: 真二元项与聚合计数, a/b 取 summary 全量(无 per-query 数组)。

    只对比双 run 都存在的字段, 缺侧不报(防 0 污染)。永不报 p。

    Args:
        run_a: 第一次 run 记录。
        run_b: 第二次 run 记录。

    Returns:
        list[dict]: 含 name/a/b/delta/mode="delta" 的行, p_value/significant/n 为 None。
    """
    metrics_a = _summary_scalar_metrics(run_a)
    metrics_b = _summary_scalar_metrics(run_b)
    rows = []
    for name in sorted(set(metrics_a) & set(metrics_b)):
        value_a = metrics_a[name]
        value_b = metrics_b[name]
        rows.append({
            "name": name,
            "a": value_a,
            "b": value_b,
            "delta": value_b - value_a,
            "p_value": None,
            "significant": None,
            "mode": "delta",
            "n": None,
        })
    return rows


def _summary_scalar_metrics(run: dict) -> dict[str, float]:
    """从 run 的 summary/attribution 提取全部标量 delta-only 指标(按名称索引)。

    hit/child_hit 与 diagnosis 分布取自 aggregate, 成本/缓存/错误/阶段分位取自 cost,
    verdict 分布取自 layer2, 归因各型计数自 attribution 现场统计。

    Args:
        run: run 记录。

    Returns:
        dict[str, float]: 指标名(带 cost./diagnosis./verdict./attribution. 前缀) -> 标量值。
    """
    result: dict[str, float] = {}
    aggregate = run.get("summary", {}).get("aggregate", {}) or {}
    for name in ("hit_at_k", "child_hit_at_k"):
        if name in aggregate:
            result[name] = float(aggregate[name])
    diagnosis_dist = aggregate.get("diagnosis_distribution") or {}
    for name, count in diagnosis_dist.items():
        result[f"diagnosis.{name}"] = float(count)
    cost = run.get("summary", {}).get("cost", {}) or {}
    for name, value in cost.items():
        if isinstance(value, (int, float)):
            result[f"cost.{name}"] = float(value)
    layer2 = run.get("summary", {}).get("layer2", {}) or {}
    verdict_dist = layer2.get("verdict_distribution") or {}
    for name, count in verdict_dist.items():
        result[f"verdict.{name}"] = float(count)
    attribution_counts: dict[str, int] = {}
    for record in (run.get("attribution") or {}).values():
        if isinstance(record, dict) and record.get("failure_type"):
            failure_type = record["failure_type"]
            attribution_counts[failure_type] = attribution_counts.get(failure_type, 0) + 1
    for name, count in attribution_counts.items():
        result[f"attribution.{name}"] = float(count)
    return result
