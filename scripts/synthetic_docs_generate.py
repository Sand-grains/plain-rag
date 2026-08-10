"""批量生成中文技术笔记干扰语料（needle_test 配套生成器）。

v2 相对 v1 的关键改造（设计见 guide/synthetic_data.md §三）：
    - 主题池：模板 × 槽位池组合式展开（TOPIC_TEMPLATES），按目标总量 2N 的 1.5× 造头寸
    - topic 级预筛：与 benchmark 全部 query+reference_facts 的 embedding 距离，剔除"贴脸"主题
    - 两分层：同一主题池按距离分 noisy(近) / control(远)，N 严格相等
    - 采样：random.shuffle + slice 前 N（prefix 稳定，增量补跑命中缓存）
    - 并发：OpenAI(max_retries=0) + 线程池 + 全局信号量 + 429 突发冷却
    - 缓存：sha256(topic|prompt|model|temperature)，失效即删，--cache-list / --cache-purge
    - 替换：同卷 staging + 3 步 rename + --min-success-rate 阈值保护
    - 质量建模：按 (tech, feature) 对统计通过率，<30% 预警
    - 模型固定为 LLM_MODEL_ID（.env 配置，设计拍板"仅 deepseek-v4-flash"）

用法::
    uv run python scripts/synthetic_docs_generate.py --noisy 500 --control 500 --workers 8 --seed 42
    uv run python scripts/synthetic_docs_generate.py --cache-list
    uv run python scripts/synthetic_docs_generate.py --cache-purge  # 全清
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

import numpy as np
from openai import OpenAI, RateLimitError

from config import LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL, TEXT_RATIO_WARN_THRESHOLD
from eval.core.benchmark import load_benchmark
from preprocess import diagnose
from retrieval.embedding import embed

# ---- 常量 ----

GUARD_WINDOW = 20            # 答案隔离短语粗筛的连续子串最小长度
POOL_HEADROOM = 1.5          # 主题池头寸系数（目标总量 2N 的 1.5×）
GENERATION_TEMPERATURE = 0.8  # 生成温度（缓存键组成部分）
RATE_LIMIT_BASE_DELAY = 2.0   # 429 冷却初始退避（秒），复用 JUDGE_BASE_DELAY 模式
RATE_LIMIT_MAX_DELAY = 30.0   # 429 冷却退避上限（秒）
QUALITY_WARN_RATE = 0.30      # (tech, feature) 对通过率预警阈值
CACHE_DIR = _PROJECT_DIR / ".synthetic_cache"  # 缓存目录在 data 树外，加 .gitignore

# ---- 主题池：模板 × 槽位池 ----
# 槽位内容是"仅剩人工审"的调节旋钮：近域槽位贴近 data/ 语料域（noisy 侧主力），
# 域外槽位（OS/网络协议/编译原理等）不贴查询域，天然落入 control（远）侧。

TECH_SLOTS_NEAR = [
    "LLM", "大模型", "Agent", "RAG", "检索", "向量数据库", "Embedding", "提示工程",
    "Java", "Spring", "Spring Boot", "Spring Cloud", "微服务", "分布式事务",
    "Redis", "MySQL", "Kafka", "消息队列", "线程池", "并发编程", "JVM",
    "缓存", "分布式锁", "接口设计", "中间件", "Elasticsearch",
]
TECH_SLOTS_FAR = [
    "操作系统", "Linux 内核", "进程调度", "内存管理", "文件系统",
    "网络协议", "TCP/IP", "HTTP 协议", "编译原理", "语法分析",
    "代码生成", "正则文法", "汇编语言", "信号处理", "计算机体系结构",
]
FEATURE_SLOTS = [
    "原理", "核心机制", "设计", "实现细节", "调优", "故障排查",
    "最佳实践", "性能分析", "架构", "源码解读", "应用场景", "面试要点",
]

TOPIC_TEMPLATES: list[tuple[str, dict[str, list[str]]]] = [
    ("{tech} {feature} 详解", {"tech": TECH_SLOTS_NEAR + TECH_SLOTS_FAR, "feature": FEATURE_SLOTS}),
    ("{tech} {feature} 从原理到实战", {"tech": TECH_SLOTS_NEAR + TECH_SLOTS_FAR, "feature": FEATURE_SLOTS}),
    ("深入理解{tech} {feature}", {"tech": TECH_SLOTS_NEAR + TECH_SLOTS_FAR, "feature": FEATURE_SLOTS}),
    ("{tech} {feature} 常见问题与排查", {"tech": TECH_SLOTS_NEAR + TECH_SLOTS_FAR, "feature": FEATURE_SLOTS}),
    ("{tech} {feature} 核心知识总结", {"tech": TECH_SLOTS_NEAR + TECH_SLOTS_FAR, "feature": FEATURE_SLOTS}),
]

_GENERATION_PROMPT = """请写一篇中文技术笔记，主题为：{topic}。

要求：
1. 使用 Markdown 标题结构：1 个一级标题（#）+ 2~3 个二级标题（##）+ 若干三级标题（###） （四级标题不强制）
2. 全文约 2000 字，每个 ## section 正文至少 500 字（信息量需要充实）
3. 至少包含两个 fenced code block（```...```）展示代码示例; 但也别泛滥
4. 内容具体、像真实的技术笔记，不要空泛
5. 直接输出 Markdown 正文，不要任何额外说明
"""


# ---- 主题池构建 ----

def _build_topic_pool() -> list[tuple[str, str, str]]:
    """组合式展开主题池：模板 × 槽位池笛卡尔积，frozenset 语义去重。

    Returns:
        list[tuple[str, str, str]]：[(topic, tech, feature)]，携带 (tech, feature)
        来源供质量建模按对统计通过率。
    """
    seen: set[str] = set()
    pool: list[tuple[str, str, str]] = []
    for template, slots in TOPIC_TEMPLATES:
        tech_list = slots["tech"]
        feature_list = slots["feature"]
        for tech in tech_list:
            for feature in feature_list:
                topic = template.format(tech=tech, feature=feature)
                if topic in seen:
                    continue
                seen.add(topic)
                pool.append((topic, tech, feature))
    return pool


# ---- 答案隔离守卫 ----

def _build_guard_substrings(items) -> tuple[set[str], list[str]]:
    """收集全部 reference_facts 的 20 字滑动窗口（<20 字 fact 用整串兜底）。

    Args:
        items: benchmark valid_items，每条的 reference_facts 为单句字符串。

    Returns:
        tuple[set[str], list[str]]：(20 字窗口集合, 短于 20 字的 fact 整串列表)。
        文档含任一 ≥20 字连续子串即判命中——等价于含某个 20 字窗口。
    """
    windows: set[str] = set()
    short_facts: list[str] = []
    for item in items:
        fact = item.reference_facts
        if len(fact) >= GUARD_WINDOW:
            windows.update(fact[index:index + GUARD_WINDOW] for index in range(len(fact) - GUARD_WINDOW + 1))
        elif fact:
            short_facts.append(fact)
    return windows, short_facts


def contains_guard_substring(text: str, guard_windows: set[str], short_facts: list[str]) -> bool:
    """判断生成文本是否命中答案隔离守卫（参考事实的 ≥20 字连续子串）。

    用文档的 20 字窗口集合与守卫窗口集合做交集，O(文档长度)；短 fact 整串做子串匹配。

    Args:
        text: 生成文档全文。
        guard_windows: 全部参考事实的 20 字窗口集合。
        short_facts: 短于 20 字的参考事实整串列表。

    Returns:
        bool：命中任一守卫子串返回 True（该文档应丢弃）。
    """
    if len(text) >= GUARD_WINDOW:
        doc_windows = {text[index:index + GUARD_WINDOW] for index in range(len(text) - GUARD_WINDOW + 1)}
        if doc_windows & guard_windows:
            return True
    return any(fact in text for fact in short_facts)


# ---- topic 距离计算（预筛 + 两分层） ----

def _topic_distances_file() -> Path:
    """topic↔查询空间距离缓存文件路径。"""
    return CACHE_DIR / "topic_distances.json"


def _load_or_compute_distances(pool: list[tuple[str, str, str]], benchmark_path: str) -> dict[str, float]:
    """计算每个 topic 到查询空间（query+reference_facts）的最小距离，结果落盘可复用。

    Args:
        pool: 主题池（(topic, tech, feature) 三元组）。
        benchmark_path: benchmark 文件路径（相对路径基于项目根解析）。

    Returns:
        dict[str, float]：topic → min_distance（1 − 最大余弦相似度）。
    """
    topics = [topic for topic, _, _ in pool]
    benchmark_key = _benchmark_key(benchmark_path)
    pool_hash = hashlib.sha256("|".join(sorted(topics)).encode()).hexdigest()[:12]

    cache_file = _topic_distances_file()
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("pool_hash") == pool_hash and cached.get("benchmark_key") == benchmark_key:
                distances = cached["distances"]
                if len(distances) == len(pool):
                    print(f"复用 topic 距离缓存（{len(distances)} 条）")
                    return distances
        except (json.JSONDecodeError, KeyError):
            pass  # 缓存损坏 → 重算

    query_texts = []
    bench = load_benchmark(benchmark_path)
    for item in bench.valid_items:
        query_texts.append(item.query)
        query_texts.append(item.reference_facts)
    if not query_texts:
        print(f"错误: benchmark 无有效条目: {benchmark_path}")
        sys.exit(1)

    print(f"计算 {len(topics)} 个 topic × {len(query_texts)} 条查询文本的 embedding 距离...")
    topic_vectors = np.asarray(embed(topics))
    query_vectors = np.asarray(embed(query_texts))
    similarities = topic_vectors @ query_vectors.T  # 归一化向量，点积即余弦
    min_distances = 1.0 - similarities.max(axis=1)

    distances = {topic: float(distance) for topic, distance in zip(topics, min_distances)}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"pool_hash": pool_hash, "benchmark_key": benchmark_key, "distances": distances},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return distances


def _benchmark_key(benchmark_path: str) -> str:
    """benchmark 身份指纹（文件名 + mtime），用于距离缓存失效判断。"""
    path = Path(benchmark_path)
    if not path.is_absolute():
        path = _PROJECT_DIR / path
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return f"{path.name}:{mtime:.0f}"


def _print_distance_quantiles(distances: dict[str, float]) -> None:
    """打印距离分位数，供预筛/分层阈值标定留痕（设计 §三.1 验证清单 2）。"""
    values = sorted(distances.values())
    if not values:
        return
    def percentile(ratio: float) -> float:
        return values[min(len(values) - 1, int(ratio * len(values)))]
    print(f"topic 最小距离分位数 P5={percentile(0.05):.3f}  P50={percentile(0.50):.3f}  P95={percentile(0.95):.3f}")


# ---- 采样 ----

def _slug(topic: str) -> str:
    """将主题转为文件名安全片段（去空格、斜杠换下划线，限长 12）。"""
    return topic.replace(" ", "").replace("/", "_")[:12]


def _make_filename(seq: int, topic: str) -> str:
    """生成唯一文件名：{seq:04d}_{slug[:12]}_{topic_hash4}.md。"""
    topic_hash = hashlib.sha256(topic.encode()).hexdigest()[:4]
    return f"{seq:04d}_{_slug(topic)}_{topic_hash}.md"


# ---- 并发：429 冷却 + 单循环失败处理 ----

class RateLimitCooling:
    """全局 429 冷却状态：收到 429 → 设置冷却截止时刻，所有 worker 等待。线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until = 0.0
        self._level = 0

    def wait_if_cooling(self) -> None:
        """等待当前冷却窗口结束（若在冷却中）。"""
        with self._lock:
            until = self._until
        remaining = until - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def trigger(self) -> float:
        """触发冷却：指数退避 2s→4s→...→30s 上限，返回本次延迟秒数。"""
        with self._lock:
            now = time.monotonic()
            if self._until <= now:
                self._level = 0  # 冷却窗口已过 → 重置指数
            delay = min(RATE_LIMIT_MAX_DELAY, RATE_LIMIT_BASE_DELAY * (2 ** self._level))
            self._level += 1
            self._until = now + delay
            return delay


# ---- 质量门槛 ----

def passes_quality(text: str) -> bool:
    """判断生成文本是否过 Markdown 结构质量门槛（guide §2.14）。"""
    report = diagnose(text)
    return (
        report.has_h1
        and report.heading_connection_standard
        and not report.too_fragmented
        and report.text_ratio >= TEXT_RATIO_WARN_THRESHOLD
        and not report.has_encoding_issues
    )


def _call_llm(client: OpenAI, topic: str) -> str:
    """调用 LLM 生成一篇指定主题的中文技术笔记（不处理重试，由调用方状态机统一处理）。"""
    response = client.chat.completions.create(
        model=LLM_MODEL_ID,
        messages=[
            {"role": "system", "content": "你是一位技术文档撰写专家，产出高质量的中文技术笔记。"},
            {"role": "user", "content": _GENERATION_PROMPT.format(topic=topic)},
        ],
        temperature=GENERATION_TEMPERATURE,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


# ---- 缓存（失效即删 + 可管理） ----

def _cache_key(topic: str) -> str:
    """缓存键：sha256(topic | PROMPT 全文 | model | temperature)，任一变更自动全 miss。"""
    payload = "|".join([topic, _GENERATION_PROMPT, LLM_MODEL_ID, str(GENERATION_TEMPERATURE)])
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_path(cache_dir: Path, topic: str) -> Path:
    return cache_dir / f"{_cache_key(topic)}.md"


def _read_cached_document(cache_dir: Path, topic: str) -> str | None:
    """读缓存文档；缓存文件缺失或为空返回 None。"""
    path = _cache_path(cache_dir, topic)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _write_cache_entry(cache_dir: Path, topic: str, content: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, topic).write_text(content, encoding="utf-8")


def _delete_cache_entry(cache_dir: Path, topic: str) -> None:
    path = _cache_path(cache_dir, topic)
    if path.exists():
        path.unlink()


def _read_h1(path: Path) -> str:
    """读缓存文档的 H1 标题（约等于 topic），供 --cache-list 显示。"""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def _cache_list(cache_dir: Path) -> None:
    """列出缓存条目：key + H1 标题。"""
    if not cache_dir.exists():
        print("缓存为空（目录不存在）")
        return
    for path in sorted(cache_dir.glob("*.md")):
        print(f"{path.stem}  {_read_h1(path)}")


def _cache_purge(cache_dir: Path, query: str | None) -> None:
    """按 topic 子串/key 删除缓存条目；query 为 None 时全清。"""
    if query is None:
        removed = 0
        for path in cache_dir.glob("*.md") if cache_dir.exists() else []:
            path.unlink()
            removed += 1
        print(f"全清缓存：移除 {removed} 条")
        return
    removed = 0
    for path in cache_dir.glob("*.md"):
        if query in path.stem or query in _read_h1(path):
            path.unlink()
            removed += 1
    print(f"按 '{query}' 移除 {removed} 条缓存")


# ---- 质量建模（按 (tech, feature) 对） ----

class QualityTracker:
    """按 (tech, feature) 对统计主题通过率，线程安全。"""

    def __init__(self) -> None:
        self._attempts: Counter = Counter()
        self._passes: Counter = Counter()
        self._lock = threading.Lock()

    def record_attempt(self, tech: str, feature: str) -> None:
        with self._lock:
            self._attempts[(tech, feature)] += 1

    def record_pass(self, tech: str, feature: str) -> None:
        with self._lock:
            self._passes[(tech, feature)] += 1

    def report(self) -> None:
        """打印通过率 < 30% 的槽位对，建议修剪。"""
        print("\n质量建模（按 tech × feature 对）:")
        low_pairs = []
        for pair, attempts in self._attempts.items():
            if attempts < 5:
                continue  # 样本太少不预警，避免噪音
            passes = self._passes[pair]
            rate = passes / attempts
            if rate < QUALITY_WARN_RATE:
                low_pairs.append((pair, attempts, passes, rate))
        if not low_pairs:
            print("  无低于 30% 的槽位对")
            return
        for pair, attempts, passes, rate in sorted(low_pairs, key=lambda item: item[3]):
            print(f"  ⚠ {pair[0]} × {pair[1]}: {passes}/{attempts} = {rate:.0%}（建议修剪该组合）")


# ---- 单篇生成（缓存查询 + 生成状态机） ----

def _generate_document(
    task: tuple,
    client: OpenAI,
    cooling: RateLimitCooling,
    guard_windows: set[str],
    short_facts: list[str],
    cache_dir: Path,
    use_cache: bool,
    retry: int,
    tracker: QualityTracker,
) -> tuple[str, str]:
    """为单个主题生成文档：先查缓存，miss 后走"429/质量门槛/短语粗筛"统一状态机。

    Args:
        task: (topic, tech, feature, stratum, filename)。
        client: 共享 OpenAI 客户端（max_retries=0，线程安全）。
        cooling: 全局 429 冷却状态。
        guard_windows / short_facts: 答案隔离守卫（见 _build_guard_substrings）。
        cache_dir: 缓存目录。
        use_cache: 是否启用缓存（--no-cache 关闭）。
        retry: 单主题未过门槛的最大重试次数。
        tracker: (tech, feature) 对质量统计器。

    Returns:
        tuple[str, str]：(文档全文, 状态)。状态 ∈ {"cache_hit", "generated", "failed"}。
    """
    topic, tech, feature, _, filename = task
    tracker.record_attempt(tech, feature)

    if use_cache:
        cached = _read_cached_document(cache_dir, topic)
        if cached is not None:
            if not contains_guard_substring(cached, guard_windows, short_facts) and passes_quality(cached):
                tracker.record_pass(tech, feature)
                return cached, "cache_hit"
            _delete_cache_entry(cache_dir, topic)  # 失效条目即删，根治坏缓存死循环

    for _ in range(retry + 1):
        cooling.wait_if_cooling()
        try:
            text = _call_llm(client, topic)
        except RateLimitError:
            delay = cooling.trigger()
            print(f"    429 冷却 {delay:.0f}s（{filename}）")
            continue
        except Exception as exc:  # 网络/鉴权等其他异常按失败重试
            print(f"    ⚠ LLM 调用异常: {exc}")
            continue
        if contains_guard_substring(text, guard_windows, short_facts):
            continue  # 短语命中 → 计入质量失败，绝不写缓存
        if passes_quality(text):
            tracker.record_pass(tech, feature)
            if use_cache:
                _write_cache_entry(cache_dir, topic, text)
            return text, "generated"
    return "", "failed"


# ---- 目录替换（同卷 staging + 3 步 rename + 阈值保护） ----

def _clean_directory(directory: Path) -> None:
    """递归删除目录（清理 stage / stale backup），目录不存在时静默。"""
    if directory.exists():
        shutil.rmtree(directory)


def _swap_directories(active_dir: Path, stage_dir: Path) -> None:
    """3 步 rename 原子换盘：active→backup → stage→active → 删 backup。

    失败时回滚（backup→active），backup 作为唯一旧语料副本绝不自动删（active 完好时除外）。

    Args:
        active_dir: 生效目录（data/synthetic）。
        stage_dir: 暂存目录（data/.synthetic_stage）。

    Raises:
        OSError: 换盘失败（已尝试回滚）。
    """
    backup_dir = active_dir.parent / ".synthetic_backup"
    had_active = active_dir.exists()
    if had_active:
        if backup_dir.exists():
            _clean_directory(backup_dir)  # active 完好 → backup 为 stale，可安全删
        os.replace(active_dir, backup_dir)
    try:
        os.replace(stage_dir, active_dir)
    except OSError:
        if had_active and backup_dir.exists():
            os.replace(backup_dir, active_dir)  # 回滚：还原旧语料
        raise
    if backup_dir.exists():
        _clean_directory(backup_dir)  # 换盘成功 → 删除旧副本


# ---- 主流程 ----

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek 批量生成中文技术笔记（needle_test 干扰语料，v2）")
    parser.add_argument("--noisy", type=int, default=50, help="noisy（近查询域）目标篇数")
    parser.add_argument("--control", type=int, default=50, help="control（远查询域）目标篇数")
    parser.add_argument("--workers", type=int, default=8, help="并发 worker 数")
    parser.add_argument("--retry", type=int, default=3, help="单篇未过门槛的最大重试次数")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子（prefix 稳定）")
    parser.add_argument("--min-success-rate", type=float, default=0.8, help="换盘最低成功率，低于则保留旧语料")
    parser.add_argument("--output", default=str(_PROJECT_DIR / "data" / "synthetic"), help="生效目录（data/synthetic）")
    parser.add_argument("--benchmark", default="benchmark/private_v6.json", help="benchmark 文件（查询空间）")
    parser.add_argument("--prefilter-threshold", type=float, default=0.15,
                        help="topic 预筛阈值：min_distance 低于此值剔除（贴脸，实现期标定）")
    parser.add_argument("--stratum-threshold", type=float, default=0.5,
                        help="两分层阈值：min_distance ≥ 此值 → control（远），否则 noisy（实现期标定）")
    parser.add_argument("--no-cache", action="store_true", help="关闭 LLM 输出缓存")
    parser.add_argument("--cache-list", action="store_true", help="列出缓存条目（key + H1 标题）")
    parser.add_argument("--cache-purge", nargs="?", const="__all__", default=None,
                        help="按 topic 子串/key 删除缓存，省略参数 = 全清")
    return parser.parse_args()


def _validate_count(noisy: int, control: int) -> None:
    """计数接口放行条件：N 严格相等，或 min 侧为 0（纯侧冒烟）。"""
    if noisy == control or noisy == 0 or control == 0:
        return
    print("错误: 计数接口要求 --noisy 与 --control 严格相等, 或 min 侧为 0（纯侧冒烟不作 needle_test）")
    sys.exit(1)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> int:
    """执行干扰语料生成：建池 → 预筛分层 → 采样 → 并发生成 → 换盘。

    Returns:
        int：成功 0；缓存管理/成功率不足 1。
    """
    args = _parse_args()

    if args.cache_list:
        _cache_list(CACHE_DIR)
        return 0
    if args.cache_purge is not None:
        query = None if args.cache_purge == "__all__" else args.cache_purge
        _cache_purge(CACHE_DIR, query)
        return 0

    _validate_count(args.noisy, args.control)
    if not LLM_API_KEY:
        print("错误: 未配置 LLM_API_KEY（.env）")
        return 1

    # ---- 主题池 + 预筛 + 两分层 ----
    pool = _build_topic_pool()
    target_total = args.noisy + args.control
    print(f"构建主题池: {len(pool)} 个唯一主题（目标 2N={target_total}，头寸需 ≥{int(target_total * POOL_HEADROOM)}）")
    if len(pool) < target_total:
        print(f"错误: 主题池不足以支撑目标篇数 {target_total}，请扩展 TOPIC_TEMPLATES/槽位池")
        return 1
    if len(pool) < target_total * POOL_HEADROOM:
        print(f"⚠ 主题池未达 1.5× 头寸（{len(pool)} < {int(target_total * POOL_HEADROOM)}），两侧候选可能不足")

    distances = _load_or_compute_distances(pool, args.benchmark)
    _print_distance_quantiles(distances)

    filtered = [(topic, tech, feature) for topic, tech, feature in pool
                if distances[topic] >= args.prefilter_threshold]
    noisy_candidates = [(topic, tech, feature) for topic, tech, feature in filtered
                        if distances[topic] < args.stratum_threshold]
    control_candidates = [(topic, tech, feature) for topic, tech, feature in filtered
                          if distances[topic] >= args.stratum_threshold]
    print(f"预筛剔除 {len(pool) - len(filtered)} 个贴脸主题; "
          f"分层后 noisy 候选 {len(noisy_candidates)} / control 候选 {len(control_candidates)}")

    # ---- 采样（shuffle + slice，prefix 稳定） ----
    rng = random.Random(args.seed)
    rng.shuffle(noisy_candidates)
    rng.shuffle(control_candidates)
    selected_noisy = noisy_candidates[:args.noisy]
    selected_control = control_candidates[:args.control]
    if len(selected_noisy) < args.noisy or len(selected_control) < args.control:
        print(f"⚠ 候选不足，截断: noisy {len(selected_noisy)}/{args.noisy}, control {len(selected_control)}/{args.control}（宁少勿重）")

    tasks: list[tuple] = []
    seq = 0
    for stratum, selected in (("noisy", selected_noisy), ("control", selected_control)):
        for topic, tech, feature in selected:
            seq += 1
            tasks.append((topic, tech, feature, stratum, _make_filename(seq, topic)))
    planned_total = len(tasks)
    print(f"采样 {len(selected_noisy)} noisy + {len(selected_control)} control = {planned_total} 篇")

    # ---- 并发生成 ----
    active_dir = Path(args.output)
    stage_dir = active_dir.parent / ".synthetic_stage"
    _clean_directory(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=0)
    bench = load_benchmark(args.benchmark)
    guard_windows, short_facts = _build_guard_substrings(bench.valid_items)
    print(f"答案隔离守卫: {len(guard_windows)} 个 20 字窗口 + {len(short_facts)} 条短事实")

    cooling = RateLimitCooling()
    tracker = QualityTracker()
    semaphore = threading.BoundedSemaphore(args.workers)
    generated_records: list[dict] = []

    def _bounded_worker(task: tuple) -> tuple:
        with semaphore:
            return task, *_generate_document(
                task, client, cooling, guard_windows, short_facts,
                CACHE_DIR, not args.no_cache, args.retry, tracker,
            )

    print(f"开始生成（workers={args.workers}, retry={args.retry}, seed={args.seed}, cache={'on' if not args.no_cache else 'off'}）")
    started_at = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_bounded_worker, task): task for task in tasks}
        for future in as_completed(futures):
            task, content, status = future.result()
            topic, tech, feature, stratum, filename = task
            if status != "failed":
                (stage_dir / filename).write_text(content, encoding="utf-8")
            generated_records.append({
                "seed": args.seed,
                "stratum": stratum,
                "status": status,
                "filename": filename,
                "topic": topic,
                "timestamp": _now_iso(),
            })
            print(f"  [{status:>9}] {filename}  {topic}")
    generation_seconds = time.monotonic() - started_at
    print(f"生成耗时 {generation_seconds:.1f}s")

    success = sum(1 for record in generated_records if record["status"] != "failed")
    success_rate = success / planned_total if planned_total else 0.0
    tracker.report()
    print(f"成功率: {success}/{planned_total} = {success_rate:.0%}（目标 ≥ {args.min_success_rate:.0%}）")

    if success == 0 or success_rate < args.min_success_rate:
        print(f"✗ 成功率不足或成功 0 篇 → 保留旧语料 {active_dir}")
        _clean_directory(stage_dir)
        print(f"  stage 已清理; 旧语料（若存在）保持不变")
        return 1

    # ---- 换盘 + manifest ----
    manifest = {
        "seed": args.seed,
        "noisy_target": args.noisy,
        "control_target": args.control,
        "generated_total": len(generated_records),
        "created_at": _now_iso(),
        "docs": generated_records,
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _swap_directories(active_dir, stage_dir)
    _clean_directory(stage_dir)  # finally 语义：stage 残留清理（换盘成功后 stage 已被 rename 走）
    print(f"✓ 换盘成功: 新语料 {success} 篇 + manifest 已就位于 {active_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
