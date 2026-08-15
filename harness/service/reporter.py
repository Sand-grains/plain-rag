"""报告器: 从清单 + 健康度生成 PROGRESS.md 快照。

叙述字段只来自 features.yaml 的 note, 报告器本身绝不自己造句;
<!-- manual --> 之后为人工保护区, 每次重写原样保留(供手写说明/临时路由入口)。
"""
from datetime import datetime
from pathlib import Path

from harness.core.models import FeatureItem, FeatureList
from harness.core.registry import Registry
from harness.core.render import cell, note_cell, table_row
from harness.core.states import ACTIVE, BLOCKED, VALID_STATES
from harness.service.tracker import HealthMetrics
from harness.core.utils import sorted_items

_MANUAL_MARKER = "<!-- manual -->"


class Reporter:
    """PROGRESS.md 生成器: render 出快照文本, write 落盘并保留人工区。"""

    def __init__(self, features_path: Path, progress_path: Path) -> None:
        self._registry = Registry(features_path)
        self._progress_path = Path(progress_path)

    def render(self, feature_list: FeatureList, health: HealthMetrics,
               generated_at: str | None = None) -> str:
        """渲染和组装 PROGRESS.md 快照文本: 自动区(点位/状态分布/健康度/状态表/叙述/验证结果)以人工保护区标记收尾。

        叙述列只取 item.note, 报告器不自行造句; 健康度缺失的可疑/未验列表整行省略。

        Args:
            feature_list: 清单快照(registry.load() 的结果)。
            health: 健康度快照(judge_health 的结果)。
            generated_at: 生成时间字符串, 缺省取当前时间。

        Returns:
            str: PROGRESS.md 快照全文, 行间以 \n 连接。
        """
        timestamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# 项目进度",
            "",
            f"> harness 自动生成, 勿手改; 生成时间: {timestamp}",
            "",
            "## 当前点位",
            "",
            feature_list.milestone,
            "",
            "## 状态分布",
            "",
        ]
        for state in VALID_STATES:
            lines.append(f"- {state}: {health.distribution[state]}")
        lines += ["", "## 健康度", ""]
        lines.append(f"- 回归率: {health.regression_rate:.2f}")
        lines.append(f"- 验证通过率: {health.pass_rate:.2f}")
        if health.suspicious_ids:
            lines.append(f"- 可疑通过项: {', '.join(health.suspicious_ids)}")
        if health.e2e_unverified_ids:
            lines.append(f"- 端到端未验: {', '.join(health.e2e_unverified_ids)}")
        lines += ["", "## 功能项状态表", "", "| ID | 行为 | 状态 | 完成时间 | 叙述 |", "| --- | --- | --- | --- | --- |"]
        for item in sorted_items(feature_list.items):
            lines.append(self._row(item))
        lines += ["", "## 活跃与阻塞项叙述", ""]
        narrated = [item for item in feature_list.items
                    if item.state in (ACTIVE, BLOCKED) and item.note]
        if narrated:
            for item in narrated:
                lines.append(f"- {item.id} ({item.state}): {item.note}")
        else:
            lines.append("(无)")
        lines += ["", "## 验证结果", ""]
        evidenced = [item for item in feature_list.items if item.last_verify]
        if evidenced:
            for item in evidenced:
                lines.append(f"- {item.id}: {self._evidence_text(item)}")
        else:
            lines.append("(无)")
        lines += ["", _MANUAL_MARKER, ""]
        return "\n".join(lines)

    def write(self, feature_list: FeatureList, health: HealthMetrics,
              generated_at: str | None = None) -> None:
        """落盘 PROGRESS.md (自动区快照与人工保护区拼接写回, 人工区原样保留。)

        Args:
            feature_list: 清单快照。
            health: 健康度快照。
            generated_at: 生成时间字符串, 缺省取当前时间。
        """
        auto = self.render(feature_list, health, generated_at).rstrip("\n")
        manual = self._manual_zone()
        content = auto + ("\n\n" + manual if manual else "\n")
        self._progress_path.write_text(content, encoding="utf-8")

    def _row(self, item: FeatureItem) -> str:
        """渲染功能项状态表的一行: ID/行为/状态/完成时间/叙述。

        列渲染统一走 render helpers(cell/note_cell/table_row), 与 registry 的 features.md 格式一致。

        Args:
            item: 待渲染的功能项。

        Returns:
            str: '| a | b | ...' 格式的表格行。
        """
        return table_row([
            cell(item.id),
            cell(item.behavior),
            cell(item.state),
            cell(item.finished_time),
            note_cell(item.note),
        ])

    def _evidence_text(self, item: FeatureItem) -> str:
        """把 last_verify 证据压缩成验证结果列表的一行摘要文本。

        Args:
            item: 待展示的功能项, 取其 last_verify。

        Returns:
            str: 'exit_code=.. tests=.. coverage=.. duration=..s' 格式的摘要行。
        """
        evidence = item.last_verify or {}
        return (f"exit_code={evidence.get('exit_code')} "
                f"tests={evidence.get('tests')} "
                f"coverage={evidence.get('coverage')} "
                f"duration={evidence.get('duration')}s")

    def _manual_zone(self) -> str:
        """抽取人工保护区: <!-- manual --> 标记之后的内容, 无标记/文件缺失返回空串。

        依赖 self._progress_path; 读取失败按无人工区处理, 不抛异常。

        Returns:
            str: 人工保护区文本(已去首尾空行), 文件不存在或无标记时为空串。
        """
        if not self._progress_path.exists():
            return ""
        try:
            text = self._progress_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        marker_index = text.find(_MANUAL_MARKER)
        if marker_index == -1:
            return ""
        return text[marker_index + len(_MANUAL_MARKER):].lstrip("\n").rstrip("\n")
