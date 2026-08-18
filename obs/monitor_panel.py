"""MonitorPanel：后台 daemon 线程终端实时面板(自 eval/monitor 迁入, 逻辑与类名不变)。

核心特性：
    - 后台 daemon 线程每 2 秒持锁浅拷贝 metrics → 格式化面板 → 清屏重绘（ANSI）或追加（plain）
    - ANSI/plain 双模式，通过 .env 中 MONITOR_PANEL_MODE 切换，兼容 PyCharm embedded terminal
    - 面板显示：进度条 + Layer 1/Layer 2 指标 vs 上次 delta + 阶段延迟 P50/P75/P95 + 成本/缓存命中率 + 异常事件滚动区
    - 事件区仅显示异常（429 / parse_error / judge_error），不含 pass——成功的 query 不产生事件
    - final_report() 跑完后打印完整最终报告，替代老旧散落 print
    - stop() 幂等，finally 块安全调用
    - 模块级单例 get_panel() / set_panel()，深层 judge 代码直接 alert() 避免参数层层透传
    - alert() 无锁（deque.appendleft 在 CPython GIL 下原子）

用法示例::

    from obs.monitor_panel import MonitorPanel, set_panel
    panel = MonitorPanel(metrics, previous_per_query)
    set_panel(panel)
    panel.set_total(len(items))
    panel.set_meta("bench.json", generator_model="deepseek-v4-flash", judge_model="deepseek-v4-pro")
    panel.start()
    # ... as_completed loop ...
    panel.stop()
    panel.final_report()

公共接口：
    - MonitorPanel: 终端面板类（start / stop / query_done / alert / final_report）
    - get_panel: 模块级单例获取
    - set_panel: 模块级单例设置
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from obs._stats import format_time, p95

if TYPE_CHECKING:
    from obs.monitor_metrics import MonitorMetrics

# Windows: 显式开启 ANSI 转义码支持（Python 默认不会设置此标志）
if sys.platform == "win32":
    import os as _os
    # os.system("") 会触发 CRT 初始化控制台并打开 VT 处理，比 ctypes 更稳
    _os.system("")
    import ctypes
    try:
        _kernel32 = ctypes.windll.kernel32
        _kernel32.GetStdHandle.restype = ctypes.c_void_p
        _kernel32.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        _kernel32.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        _STD_OUTPUT_HANDLE = -11
        _ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        _handle = _kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
        _mode = ctypes.c_uint32()
        if _kernel32.GetConsoleMode(_handle, ctypes.byref(_mode)):
            _kernel32.SetConsoleMode(_handle, _mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except OSError:
        pass  # 非控制台环境（如管道重定向），ANSI 不可用

DEFAULT_REFRESH_SEC = 2
ALERT_QUEUE_MAXLEN = 20
ALERT_DISPLAY_N = 5
PROGRESS_BAR_WIDTH = 30

# ANSI escape codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_CYAN = "\033[36m"
CLEAR_SCREEN = "\033[H\033[2J\033[3J"
CURSOR_HOME = "\033[H"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"


class MonitorPanel:
    """后台 daemon 线程终端面板。持有 MonitorMetrics 引用 + 上次运行基线。"""

    def __init__(self, metrics: MonitorMetrics, previous_per_query: dict | None = None):
        self.metrics = metrics
        self.previous_per_query = previous_per_query  # query_id -> per_query dict

        self.query_count: int = 0
        self.total_queries: int = 0
        self.start_time: float = 0.0

        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._alerts: deque[str] = deque(maxlen=ALERT_QUEUE_MAXLEN)
        self._crashed: bool = False

        self._benchmark_name: str = ""
        self._generator_model: str = ""
        self._judge_model: str = ""
        self._eval_mode: str = "full"

        from config import MONITOR_PANEL_MODE
        self._mode: str = MONITOR_PANEL_MODE

        self._redis_available: bool = self._check_redis()
        self._first_render: bool = True

    @staticmethod
    def _check_redis() -> bool:
        try:
            from infra.cache import get_cache
            from infra.cache.noop_backend import NoopBackend
            return not isinstance(get_cache(), NoopBackend)
        except Exception:
            return False

    def set_total(self, total: int) -> None:
        self.total_queries = total

    def set_meta(self, benchmark_name: str, generator_model: str = "", judge_model: str = "",
                 eval_mode: str = "full") -> None:
        self._benchmark_name = benchmark_name
        self._generator_model = generator_model
        self._judge_model = judge_model
        self._eval_mode = eval_mode

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """幂等停止。第二次调用直接返回。"""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._terminal_reset()

    def query_done(self) -> None:
        with self._lock:
            self.query_count += 1

    def alert(self, query_id: str, message: str) -> None:
        """入队一条警告事件。无锁——deque.appendleft 在 CPython GIL 下原子。"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._alerts.appendleft(f"{timestamp}  {query_id}  {message}")

    # ---- 渲染循环 ----

    def _render_loop(self) -> None:
        try:
            while self._running:
                with self._lock:
                    layer1 = self.metrics.layer1_avgs()
                    layer2 = self.metrics.layer2_avgs()
                    stages = self.metrics.stage_percentiles()
                    verdicts = self.metrics.verdict_distribution()
                    deltas = self._compute_deltas(self.metrics)
                    query_count = self.query_count
                    alerts = list(self._alerts)[:ALERT_DISPLAY_N]
                    alert_total = len(self._alerts)
                    overflow = max(0, alert_total - ALERT_QUEUE_MAXLEN)
                self._render(layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow)
                time.sleep(DEFAULT_REFRESH_SEC)
        except Exception:
            self._crashed = True
            self._running = False
            import traceback
            traceback.print_exc()

    def _render(self, layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow) -> None:
        if self._mode == "plain":
            self._render_plain(layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow)
        else:
            self._render_ansi(layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow)

    # ---- ANSI 模式 ----

    def _render_ansi(self, layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow) -> None:
        lines = self._build_panel_lines(layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow)

        if self._first_render:
            sys.stdout.write(CURSOR_HIDE)
            self._first_render = False

        # \033[H\033[2J\033[3J 光标归位 + 清可见屏 + 清 scrollback，一次 write 到 stdout
        sys.stdout.write(CLEAR_SCREEN + "\n".join(lines) + "\n")
        sys.stdout.flush()

    # ---- plain 模式 ----

    def _render_plain(self, layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow) -> None:
        lines = ["\n" + "=" * 76]
        lines.extend(self._build_panel_lines(layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow))
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    # ---- 面板内容构建 ----

    def _build_panel_lines(self, layer1, layer2, stages, verdicts, deltas, query_count, alerts, overflow) -> list[str]:
        lines = []

        # Header
        redis_status = "connected" if self._redis_available else f"{C_RED}unavailable (no cache){C_RESET}"
        lines.append(f"{C_BOLD}{'═' * 76}{C_RESET}")
        lines.append(f"  plain-rag eval · {self._eval_mode} mode · {self._benchmark_name} · {self.total_queries} queries")
        lines.append(f"  GeneratorModel: {self._generator_model}  |  JudgeModel: {self._judge_model}")
        lines.append(f"  Redis: {redis_status}")
        lines.append(f"{C_BOLD}{'═' * 76}{C_RESET}")
        lines.append("")

        # Progress
        elapsed = time.time() - self.start_time
        done = min(query_count, self.total_queries)
        ratio = done / self.total_queries if self.total_queries > 0 else 0
        filled = int(ratio * PROGRESS_BAR_WIDTH)
        bar = "█" * filled + "░" * (PROGRESS_BAR_WIDTH - filled)
        lines.append(f"  进度  {bar}  {done}/{self.total_queries} ({ratio:.0%})  [{format_time(elapsed)}]")
        lines.append("")

        # Layer 1 + Layer 2 side by side
        valid_l2_count = len([result for result in list(self.metrics.layer2_results) if result.faithfulness is not None])
        delta_label = ""
        if deltas and "_matching" in deltas:
            delta_label = f"Δ (N={deltas['_matching']} matching)"

        lines.append("  Layer 1                                 Layer 2"
                     + (f" ({valid_l2_count} valid)" if valid_l2_count else ""))
        lines.append("  ────────────                            ────────────")
        lines.append("           本次      vs 上次                        本次      vs 上次")

        layer1_names = ["Recall@5", "Prec@5", "Hit@5", "MRR", "MAP@5", "NDCG@5"]
        layer1_keys = ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "map_at_k", "ndcg_at_k"]
        layer2_names = ["Faithfulness", "Answer Relev", "Context Prec", "Context Rec", "Answer Corr"]
        layer2_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]

        for index in range(6):
            layer1_value = self._fmt_val(layer1.get(layer1_keys[index]))
            layer1_delta = self._fmt_delta(deltas.get(layer1_keys[index]))
            left = f"  {layer1_names[index]:<10s}  {layer1_value:>8s}  {layer1_delta:>8s}"

            right = ""
            if index < 5 and layer2:
                layer2_value = self._fmt_val(layer2.get(layer2_keys[index]))
                layer2_delta = self._fmt_delta(deltas.get(layer2_keys[index]))
                right = f"  {layer2_names[index]:<14s}  {layer2_value:>8s}  {layer2_delta:>8s}"
            elif index == 5 and layer2:
                # Verdict line
                right = (f"  {C_GREEN}pass:{verdicts.get('pass', 0)}{C_RESET}"
                         f"  {C_YELLOW}partial:{verdicts.get('partial', 0)}{C_RESET}"
                         f"  {C_RED}fail:{verdicts.get('fail', 0)}{C_RESET}"
                         f"  error:{verdicts.get('error', 0)}")

            if right:
                lines.append(f"{left}    {right}")
            else:
                lines.append(left)

        if delta_label:
            lines.append(f"  {' ' * 62}{delta_label}")
        lines.append("")

        # Stage latency + Cost side by side
        lines.append("  阶段延迟 (P50 / P75 / P95)                   成本")
        lines.append("  ────────────────────────────               ──────────────────")

        stage_labels = ["Retrieve", "Generate", "Judge Faithfulness", "Judge Quality"]
        stage_keys = ["retrieve", "generate", "judge_faithfulness", "judge_quality"]
        cost_labels = ["Token", "Cost", "Generator Cache", "Judge Cache"]
        cost_values = [
            f"{self.metrics.total_input_tokens:,} in / {self.metrics.total_output_tokens:,} out",
            f"¥{self.metrics.estimated_cost:.2f}",
            (f"{self.metrics.generator_cache_hits} hit "
             f"({self.metrics.generator_cache_hit_rate:.1%})"),
            (f"{self.metrics.judge_cache_hits} hit "
             f"({self.metrics.judge_cache_hit_rate:.1%})"),
        ]

        for index in range(4):
            stage_stats = stages.get(stage_keys[index], {})
            stage_data = f"{int(stage_stats.get('p50', 0))} / {int(stage_stats.get('p75', 0))} / {int(stage_stats.get('p95', 0))} ms"
            left = f"  {stage_labels[index]:<20s}  {stage_data:<24s}"

            # Cost: only show meaningful values if Redis unavailable
            if not self._redis_available and index in (2, 3):
                cost_str = "—"
            else:
                cost_str = cost_values[index]
            right = f"  {cost_labels[index]:<16s}  {cost_str}"

            lines.append(f"{left} {right}")
        lines.append("")

        # Alerts
        alert_total = len(self._alerts)
        if overflow > 0:
            lines.append(f"  {C_RED}⚠ 事件 ({alert_total}/{ALERT_QUEUE_MAXLEN})  {overflow} 条未显示{C_RESET}")
        elif alert_total > 0:
            lines.append(f"  {C_YELLOW}⚠ 事件 ({alert_total}/{ALERT_QUEUE_MAXLEN}){C_RESET}")
        else:
            lines.append("  ⚠ 事件 (0)")
        for alert in alerts:
            lines.append(f"  {alert}")

        lines.append("")
        lines.append(f"{C_BOLD}{'═' * 76}{C_RESET}")
        return lines

    # ---- 最终报告 ----

    def final_report(self) -> None:
        """跑完后打印最终报告。从 metrics 读取全部数据。"""
        try:
            metrics = self.metrics
            layer1 = metrics.layer1_avgs()
            layer2 = metrics.layer2_avgs()
            stages = metrics.stage_percentiles()
            verdicts = metrics.verdict_distribution()
            deltas = self._compute_deltas(metrics)
            elapsed = time.time() - self.start_time if self.start_time > 0 else 0
            run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

            valid = [result for result in list(metrics.layer2_results) if result.faithfulness is not None]
            valid_n = len(valid)
            total_n = len(list(metrics.layer2_results))

            # Diagnosis distribution from layer1 results
            diag_counts: dict[str, int] = {}
            for result in list(metrics.layer1_results):
                diagnosis = getattr(result, "diagnosis", "unknown")
                diag_counts[diagnosis] = diag_counts.get(diagnosis, 0) + 1

            print()
            print(f"{'═' * 76}")
            print(f"  Run ID: {run_id}    Mode: {self._eval_mode}    {self.total_queries} queries    耗时: {format_time(elapsed)}")
            print(f"{'═' * 76}")
            print()
            print("  Layer 1                                             vs 上次")
            print("  ────────────")
            for name, key in [("Recall@5", "recall_at_k"), ("Precision@5", "precision_at_k"),
                              ("Hit@5", "hit_at_k"), ("MRR", "mrr"),
                              ("MAP@5", "map_at_k"), ("NDCG@5", "ndcg_at_k")]:
                value = self._fmt_val(layer1.get(key))
                delta = self._fmt_delta(deltas.get(key))
                print(f"  {name:<14s}  {value:>8s}                             {delta:>8s}")

            print()
            if diag_counts:
                diag_parts = "   ".join(f"{key}: {value}" for key, value in sorted(diag_counts.items()))
                print(f"  诊断分布:  {diag_parts}")

            print()
            delta_label = f"Δ (N={deltas.get('_matching', 0)} matching)" if deltas else ""
            print(f"  Layer 2 ({valid_n} valid / {total_n} total)                     vs 上次")
            print("  ────────────")
            for name, key in [("Faithfulness", "faithfulness"), ("Answer Relev", "answer_relevancy"),
                              ("Context Prec", "context_precision"), ("Context Rec", "context_recall"),
                              ("Answer Corr", "answer_correctness")]:
                value = self._fmt_val(layer2.get(key))
                delta = self._fmt_delta(deltas.get(key))
                print(f"  {name:<14s}  {value:>8s}                             {delta:>8s}")
            if delta_label:
                print(f"  {' ' * 50}{delta_label}")

            print()
            print(f"  Verdict:  pass: {verdicts.get('pass', 0)}   partial: {verdicts.get('partial', 0)}"
                  f"   fail: {verdicts.get('fail', 0)}   error: {verdicts.get('error', 0)}")

            print()
            print("  阶段延迟 (ms)           P50     P75     P95")
            print("  ────────────────")
            for label, key in [("Retrieve", "retrieve"), ("Generate", "generate"),
                               ("Judge Faithfulness", "judge_faithfulness"),
                               ("Judge Quality", "judge_quality")]:
                stage_stats = stages.get(key, {})
                p50 = int(stage_stats.get("p50", 0))
                p75 = int(stage_stats.get("p75", 0))
                p95 = int(stage_stats.get("p95", 0))
                print(f"  {label:<20s}  {p50:>6d}  {p75:>6d}  {p95:>6d}")

            end_to_end = stages.get("end_to_end", {})
            print(f"  {'End-to-end':<20s}  {int(end_to_end.get('p50', 0)):>6d}  {int(end_to_end.get('p75', 0)):>6d}  {int(end_to_end.get('p95', 0)):>6d}")

            print()
            print("  LLM 调用统计")
            print("  ────────────")
            print(f"  Generator: {metrics.generator_llm_calls} 次  |  "
                  f"Judge Faithfulness: {metrics.judge_faithfulness_calls} 次  |  "
                  f"Judge Quality: {metrics.judge_quality_calls} 次")
            print(f"  Token: {metrics.total_input_tokens:,} in / {metrics.total_output_tokens:,} out")
            print(f"  Input P50/P95:  {int(p95([metrics.total_input_tokens // max(1, metrics.total_llm_calls)])) if metrics.total_llm_calls > 0 else 0}"
                  f" / {int(metrics.input_tokens_p95)} tok  |  "
                  f"Output P50/P95: {int(p95([metrics.total_output_tokens // max(1, metrics.total_llm_calls)])) if metrics.total_llm_calls > 0 else 0}"
                  f" / {int(metrics.output_tokens_p95)} tok")
            print(f"  重试: {metrics.retry_count} 次  |  解析失败: {metrics.parse_error_count} 次")

            print()
            print("  缓存 / 成本")
            print("  ────────────")
            generator_rate = f"{metrics.generator_cache_hits} hit / {metrics.generator_cache_misses} miss ({metrics.generator_cache_hit_rate:.1%})"
            judge_rate = f"{metrics.judge_cache_hits} hit / {metrics.judge_cache_misses} miss ({metrics.judge_cache_hit_rate:.1%})"
            rerank_hit_rate = metrics.rerank_cache_hit_rate
            if rerank_hit_rate is None:
                rerank_line = "—"  # 无 rerank 调用（rerank 关 / 基线 / 控制组）
            else:
                from retrieval.reranker import get_rerank_cache_statistics
                rerank_stats = get_rerank_cache_statistics()
                rerank_line = f"{rerank_stats['hits']} hit / {rerank_stats['misses']} miss ({rerank_hit_rate:.1%})"
            print(f"  Generator 缓存: {generator_rate}")
            print(f"  Judge 缓存:     {judge_rate}")
            print(f"  Rerank 缓存:     {rerank_line}")
            print(f"  估算成本:       ¥{metrics.estimated_cost:.2f}")

            print()
            print("  错误明细")
            print("  ────────────")
            if metrics.error_types:
                for error_type, count in sorted(metrics.error_types.items(), key=lambda item: -item[1]):
                    print(f"  {error_type}: {count}")
            else:
                print("  (无错误)")

            print()
            print(f"{'═' * 76}")
        except Exception:
            print("\n(最终报告生成失败)")

    # ---- 辅助方法 ----

    def _compute_deltas(self, metrics: MonitorMetrics) -> dict:
        """对已完成的 query 计算与上次运行的差值均值。

        Args:
            metrics: 本次运行的指标容器。

        Returns:
            dict：各指标 delta 均值 + "_matching" 匹配 query 数。
        """
        if not self.previous_per_query:
            return {}
        deltas_accum: dict[str, list[float]] = {}
        matching = 0
        for result in list(metrics.layer2_results):
            query_id = getattr(result, "query_id", None)
            if query_id not in self.previous_per_query:
                continue
            previous = self.previous_per_query[query_id]
            matching += 1
            for field in ["faithfulness", "answer_relevancy", "context_precision",
                          "context_recall", "answer_correctness"]:
                current_value = getattr(result, field, None)
                previous_value = previous.get(field)
                if current_value is not None and previous_value is not None:
                    deltas_accum.setdefault(field, []).append(current_value - previous_value)

        # 同时从 layer1_results 计算 Layer 1 delta
        layer1_deltas: dict[str, list[float]] = {}
        for result in list(metrics.layer1_results):
            query_id = getattr(result, "query_id", None)
            if not query_id or query_id not in self.previous_per_query:
                continue
            previous = self.previous_per_query[query_id]
            for field in ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "map_at_k", "ndcg_at_k"]:
                current_value = getattr(result, field, None)
                previous_value = previous.get(field)
                if current_value is not None and previous_value is not None:
                    layer1_deltas.setdefault(field, []).append(current_value - previous_value)

        result = {}
        for field, values in {**deltas_accum, **layer1_deltas}.items():
            if values:
                result[field] = sum(values) / len(values)
        result["_matching"] = matching
        return result

    @staticmethod
    def _fmt_val(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    @staticmethod
    def _fmt_delta(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, (int, float)):
            sign = "+" if value >= 0 else "−"
            return f"{sign}{abs(value):.4f}"
        return str(value)

    def _terminal_reset(self) -> None:
        if self._mode == "ansi":
            sys.stdout.write(C_RESET + CURSOR_SHOW)
            sys.stdout.flush()


# ---- 模块级单例 ----

_panel: MonitorPanel | None = None


def get_panel() -> MonitorPanel | None:
    """模块级单例。未初始化时返回 None，调用方自行判空。"""
    return _panel


def set_panel(panel: MonitorPanel) -> None:
    global _panel
    _panel = panel
