"""features.yaml 的唯一读写入口: 解析/校验/原子改写/渲染视图。

features.md 是给人看的渲染视图, 每次写回后重建, 不手改。
registry 是唯一写 features.yaml 的入口。而 features.yaml 是唯一权威, 状态仅由 CLI 命令经 registry 改写;

关键机制:
    - 原子改写 = 临时文件 + os.replace + 文件锁(O_EXCL), 防并发写坏文件
    - CRLF 保序: 首次读取探测换行风格, 写回时保持(Windows 上避免整文件改行)
    - 行为哈希: behavior 字段存 sha256 前缀, 行为描述变更 → 状态回退 not_started, 防"行为悄悄改窄, 测试跟着改窄"
    - milestone 是指针, items 跨 milestone 累积不清空, 任务返工用 abandoned 表达

与上层的关系: cli 通过 apply_state/apply_note/record_verification 改动状态,
只读命令 (调度/追踪/报告) 用 load() 读快照; 各组件不直接写 features.yaml。

模块划分: 数据模型在 models, 文件锁/原子写在 persistence, 表格渲染在 render,
本模块只保留 Registry 的清单语义操作。
"""
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from harness.core import persistence, render
from harness.core.errors import HarnessError
from harness.core.models import FeatureItem, FeatureList, behavior_hash
from harness.core.state_machine import IllegalTransitionError, transition
from harness.core.states import ACTIVE, NOT_STARTED, PASSED, REGRESSED, VALID_STATES

# e2e 验证命令的最小超时(秒) floor: eval 类命令慢, 单一 floor 防超时诡异失败;
# 不解析 e2e 命令串, registry 不耦合 eval.runner 语法, 具体值由 feature 声明。
E2E_MIN_TIMEOUT = 600


def _now_minute() -> str:
    """当前时间精确到分(YYYY-MM-DD HH:MM), 供 finished_time 落值。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


class ValidationError(HarnessError):
    """features.yaml 结构校验失败(缺字段/非法状态/重复 id)。"""


class ItemNotFoundError(HarnessError):
    """按 id 查找功能项时不存在。"""


class Registry:
    """ features.yaml 的唯一读写入口: load 读快照, 各 apply_* 原子改写状态。
    所有写操作经 _commit(加锁读-改-写)串行化, 写回后立即重建 features.md 视图。
    """

    def __init__(self, features_path: Path) -> None:
        self._features_path = Path(features_path)
        self._view_path = self._features_path.with_suffix(".md")
        self._lock_path = Path(str(self._features_path) + ".lock")

    # ---- 读 ----
    def load(self) -> FeatureList:
        """读取并校验 features.yaml, 返回快照(行为哈希回退在内存生效)。"""
        return self._load_unlocked()

    def _load_unlocked(self) -> FeatureList:
        if not self._features_path.exists():
            raise HarnessError(f"清单不存在: {self._features_path}")
        with open(self._features_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValidationError("features.yaml 根节点必须是映射")
        for key in ("schema", "milestone", "items"):
            if key not in data:
                raise ValidationError(f"features.yaml 缺字段: {key}")

        items: list[FeatureItem] = []
        seen: set[str] = set()
        for raw in data["items"]:
            item = self._parse_item(raw, seen)
            items.append(self._reconcile_item(item))
        return FeatureList(schema=int(data["schema"]), milestone=str(data["milestone"]), items=items)

    def _parse_item(self, raw: object, seen: set[str]) -> FeatureItem:
        if not isinstance(raw, dict):
            raise ValidationError("items 中每项必须是映射")
        for key in ("id", "behavior", "gate", "state"):
            if key not in raw:
                raise ValidationError(f"功能项缺字段: {key}")
        item_id = str(raw["id"])
        if item_id in seen:
            raise ValidationError(f"重复的功能项 id: {item_id}")
        seen.add(item_id)
        state = raw["state"]
        if state not in VALID_STATES:
            raise ValidationError(f"非法状态 {state!r}, 只允许 {list(VALID_STATES)}")
        return FeatureItem(
            id=item_id,
            behavior=str(raw["behavior"]),
            gate=str(raw["gate"]),
            state=state,
            e2e=raw.get("e2e"),
            precheck=raw.get("precheck"),
            e2e_passed=bool(raw.get("e2e_passed", False)),
            finished_time=raw.get("finished_time"),
            note=raw.get("note"),
            timeout=raw.get("timeout"),
            behavior_hash=str(raw.get("behavior_hash") or ""),
            last_verify=raw.get("last_verify"),
        )

    def _reconcile_item(self, item: FeatureItem) -> FeatureItem:
        """行为哈希归一: 存量哈希与当前行为不匹配 → 状态回退 not_started 并更新哈希。

        行为变更视为功能重定义, e2e 通过证据一并失效, 防"旧 e2e 通过"延续到新行为。
        """
        current_hash = behavior_hash(item.behavior)
        if item.behavior_hash and item.behavior_hash != current_hash:
            item.state = NOT_STARTED
            item.e2e_passed = False
            item.finished_time = None
        item.behavior_hash = current_hash
        return item

    # ---- 写 ----
    def apply_state(self, item_id: str, target_state: str, note: str | None = None) -> FeatureItem:
        """状态转移写回: 锁内读-改-写, 转移合法性由状态机校验。

        与当前态相同视为无意义转移直接拒绝(start 一个已是 active 的项报错,
        保证"仅 not_started 可 start"之类的命令语义); verify 的同态保持
        走 record_verification, 不经过本方法。
        """
        def mutate(feature_list: FeatureList) -> FeatureItem:
            """锁内变更: 拒绝同态转移, 校验后执行状态转移, 附带给 note。"""
            item = self._find_item(feature_list, item_id)
            if item.state == target_state:
                raise IllegalTransitionError(f"{item_id} 已是 {target_state}, 无需转移")
            transition(item.state, target_state)
            item.state = target_state
            if note is not None:
                item.note = note
            return item

        return self._commit(mutate)

    def apply_note(self, item_id: str, note: str | None) -> FeatureItem:
        """只写 note 字段, 不碰 state, 经原子改写(agent 不手改 yaml)。"""
        def mutate(feature_list: FeatureList) -> FeatureItem:
            """锁内变更: 只覆盖 note 字段, 不碰 state。"""
            item = self._find_item(feature_list, item_id)
            item.note = note
            return item

        return self._commit(mutate)

    def record_verification(self, item_id: str, passed: bool,
                            evidence: dict | None, note: str | None = None) -> FeatureItem:
        """verify 结果写回: 依据当前态 + 通过与否, 推导目标态(单锁内完成状态+证据)。

        转移语义(条例七): active+通过→passed, active+失败→保持 active;
        passed+失败→regressed, passed+通过→保持 passed; 其它态拒绝并提示正确路径。

        e2e_passed 持久旗标: evidence["mode"]=="e2e" 且通过 → 置 True; 转 regressed
        → 置 False。gate 通过不改动(verify-all 巡检 passed 项不误清 e2e 证据)。
        finished_time: 仅 active->passed 落当前时间(精确到分), 巡检不刷新;
        转 regressed 或行为哈希重置清空。

        Args:
            item_id: 功能项 id。
            passed: 验证是否通过(退出码 0 且测试数 > 0)。
            evidence: 验证证据 dict, 存入 last_verify。
            note: 可选, 传入时覆盖 note(如 verify 失败时记录卡点)。

        Returns:
            FeatureItem: 更新后的功能项。

        Raises:
            IllegalTransitionError: 非 active/passed 态执行 verify, 或推导出的转移非法。
        """
        def mutate(feature_list: FeatureList) -> FeatureItem:
            """锁内变更: 按当前态 + passed 推导目标态, 同步 last_verify/e2e_passed/finished_time。"""
            item = self._find_item(feature_list, item_id)
            if item.state not in (ACTIVE, PASSED):
                raise IllegalTransitionError(
                    f"verify 只允许 active/passed 项, {item_id} 当前是 {item.state}; "
                    "not_started 需先 start, blocked 需先 unblock, regressed 需先 reactivate"
                )
            was_active = item.state == ACTIVE
            if item.state == ACTIVE:
                target_state = PASSED if passed else ACTIVE
            else:
                target_state = REGRESSED if not passed else PASSED
            transition(item.state, target_state)
            item.state = target_state
            item.last_verify = evidence
            if (evidence or {}).get("mode") == "e2e" and passed:
                item.e2e_passed = True
            if was_active and passed:
                # active -> passed 才算完成, 巡检(passed->passed)不刷新完成时间
                item.finished_time = _now_minute()
            if target_state == REGRESSED:
                item.e2e_passed = False
                item.finished_time = None
            if note is not None:
                item.note = note
            return item

        return self._commit(mutate)

    def _find_item(self, feature_list: FeatureList, item_id: str) -> FeatureItem:
        """按 id 在清单快照中线性查找功能项。

        Args:
            feature_list: 清单快照。
            item_id: 功能项 id。

        Returns:
            FeatureItem: 命中的功能项。

        Raises:
            ItemNotFoundError: id 不存在于清单。
        """
        for item in feature_list.items:
            if item.id == item_id:
                return item
        raise ItemNotFoundError(f"功能项不存在: {item_id}")

    def _commit(self, mutate_fn: Callable[[FeatureList], FeatureItem]) -> FeatureItem:
        """加锁读-改-写: 锁内 load → mutate → save, 防并发读-改-写覆盖。"""
        with persistence._FileLock(self._lock_path):
            feature_list = self._load_unlocked()
            updated = mutate_fn(feature_list)
            self._save_unlocked(feature_list)
            return updated

    def save(self, feature_list: FeatureList) -> None:
        """显式写回整份清单(测试/自举用), 与 _commit 同走原子改写。"""
        with persistence._FileLock(self._lock_path):
            self._save_unlocked(feature_list)

    def _save_unlocked(self, feature_list: FeatureList) -> None:
        """无锁写回: 归一化 + 校验 + 原子落盘 + 重建渲染视图(调用方必须已持锁)。"""
        normalized = [self._reconcile_item(item) for item in feature_list.items]
        self._validate(normalized)
        newline = persistence.detect_newline(self._features_path)
        text = self._to_yaml(feature_list.schema, feature_list.milestone, normalized)
        persistence.atomic_write(text, self._features_path, newline)
        self._write_view(feature_list.schema, feature_list.milestone, normalized)

    def _validate(self, items: list[FeatureItem]) -> None:
        """结构校验: 重复 id / 空 behavior / 空 gate / 非法 state / e2e 缺 timeout。

        Args:
            items: 待写回的功能项列表。

        Raises:
            ValidationError: 任一校验不通过, 拒绝写回。
        """
        seen: set[str] = set()
        for item in items:
            if item.id in seen:
                raise ValidationError(f"重复的功能项 id: {item.id}")
            seen.add(item.id)
            if not item.behavior:
                raise ValidationError(f"{item.id} 的 behavior 不能为空")
            if not item.gate:
                raise ValidationError(f"{item.id} 的 gate 不能为空")
            if item.state not in VALID_STATES:
                raise ValidationError(f"{item.id} 的 state 非法: {item.state!r}")
            if item.e2e:
                if not item.timeout:
                    raise ValidationError(f"{item.id} 配了 e2e 但 timeout 缺失, 必须 >= {E2E_MIN_TIMEOUT}s")
                if item.timeout < E2E_MIN_TIMEOUT:
                    raise ValidationError(
                        f"{item.id} 的 timeout 必须 >= {E2E_MIN_TIMEOUT}s (e2e 慢, 当前 {item.timeout}s)")

    def _to_yaml(self, schema: int, milestone: str, items: list[FeatureItem]) -> str:
        """序列化整份清单为 YAML 文本(sort_keys=False 保持字段声明顺序)。

        Args:
            schema: 清单格式版本号。
            milestone: 当前迭代点位。
            items: 功能项列表。

        Returns:
            str: YAML 文本。
        """
        payload = {
            "schema": schema,
            "milestone": milestone,
            "items": [self._item_to_dict(item) for item in items],
        }
        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)

    def _item_to_dict(self, item: FeatureItem) -> dict:
        """功能项转 dict, 固定字段顺序供 YAML 落盘。

        Args:
            item: 功能项。

        Returns:
            dict: 按 features.yaml schema 排列的字段映射。
        """
        return {
            "id": item.id,
            "behavior": item.behavior,
            "behavior_hash": item.behavior_hash,
            "gate": item.gate,
            "e2e": item.e2e,
            "precheck": item.precheck,
            "e2e_passed": item.e2e_passed,
            "finished_time": item.finished_time,
            "state": item.state,
            "note": item.note,
            "timeout": item.timeout,
            "last_verify": item.last_verify,
        }

    # ---- 渲染视图 ----
    def render_markdown(self, feature_list: FeatureList) -> str:
        """渲染 features.md 表格: ID/行为/门禁/端到端/状态/完成时间/叙述。"""
        lines = [
            "# 功能清单",
            "",
            "> 由 harness 自动生成, 勿手改; 改状态请用 CLI 命令",
            "",
            "| ID | 行为 | 门禁验证 | 端到端验证 | 状态 | 完成时间 | 叙述 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in feature_list.items:
            lines.append(self._row(item))
        lines.append("")
        return "\n".join(lines)

    def _row(self, item: FeatureItem) -> str:
        """功能项渲染为 features.md 表格行, 空字段回退为 "-"。

        Args:
            item: 功能项。

        Returns:
            str: 单行表格行。
        """
        return render.table_row([
            render.cell(item.id),
            render.cell(item.behavior),
            render.cell(item.gate, code=True),
            render.cell(item.e2e, code=True),
            render.cell(item.state),
            render.cell(item.finished_time),
            render.note_cell(item.note),
        ])

    def _write_view(self, schema: int, milestone: str, items: list[FeatureItem]) -> None:
        """渲染并落盘 features.md 视图(每次写回后重建, 由 harness 生成不手改)。

        Args:
            schema: 清单格式版本号。
            milestone: 当前迭代点位。
            items: 功能项列表。
        """
        view = self.render_markdown(FeatureList(schema=schema, milestone=milestone, items=items))
        self._view_path.write_text(view, encoding="utf-8")
