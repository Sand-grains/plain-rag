"""功能项单元数据模型: features.yaml 的 items 结构定义与行为哈希, 与 I/O 解耦。

FeatureItem/FeatureList 被 registry(解析校验/原子改写)、reporter(报告)、
tracker(健康度)、scheduler(候选)、verifier(验证) 与各测试共享。
独立成模块, 避免"模型定义"与"文件操作"耦合在同一文件。
"""
import hashlib
from dataclasses import dataclass

from harness.core.states import NOT_STARTED


def behavior_hash(behavior: str) -> str:
    """行为描述的 sha256 前缀哈希, 用于检测行为文本是否被改动。

    Args:
        behavior: 功能项行为描述文本。

    Returns:
        str: 12 位十六进制 sha256 前缀。
    """
    return hashlib.sha256(behavior.encode("utf-8")).hexdigest()[:12]


@dataclass
class FeatureItem:
    """清单中一行功能项: 行为 + 验证命令 + 状态三原子。

    behavior_hash 由 __post_init__ 按 behavior 自动填充, 解析时若与存量不符, 说明行为描述被改, registry 会回退状态到 not_started。
    last_verify 存最近一次验证的紧凑证据(退出码/测试数/coverage/耗时), 供报告器展示与可疑项标记, 完整命令输出在 verify 时打到终端。
    """
    id: str  # 功能项唯一编号。全局按顺序编号(F01, F02...), 跨 milestone 追加不重置
    behavior: str  # 行为描述。文本变更会被 behavior_hash 检出并回退状态
    gate: str  # 门禁验证命令, 必须能用退出码判定成败
    state: str = NOT_STARTED  # 当前状态(仅 VALID_STATES 六个枚举, 且只能经 CLI 变更状态)
    e2e: str | None = None  # 端到端验证命令, 声明时必须配套 timeout
    precheck: str | None = None  # 前置自检命令(未满足则不判回归)
    e2e_passed: bool = False  # 端到端通过旗标, 仅 e2e 模式通过置 True, 转 regressed 清空
    finished_time: str | None = None  # 完成时间(精确到分), active -> passed 落值, 巡检不刷新
    note: str | None = None  # 叙述, reporter照抄
    timeout: float | None = None  # e2e 超时秒数(强制下限 600s)
    metric_gate: dict | None = None  # 可选指标门声明, 指标门不过则整体 passed=False: {command, thresholds:{name:{min/max}} (指标的绝对值是否跌穿底限 recall ≥ 0.5), delta:{name:{drop}}} (防缓慢劣化: 较基线 drop ≤ 0.05)
    baseline_run_id: str | None = None  # harness 自动维护的最近一次指标门通过的 run 指针(metrics.jsonl 中的 run_id) (基线, 仅通过时更新)
    behavior_hash: str = ""  # behavior 的 sha256 12位前缀, 检测行为描述是否被改
    last_verify: dict | None = None  # 最近验证证据(退出码/测试数/覆盖率/耗时/模式), 供可疑项判定

    def __post_init__(self) -> None:
        """行为哈希未显式给定时按 behavior 自动填充(存量项解析时保留原哈希供归一比对)。"""
        if not self.behavior_hash:
            self.behavior_hash = behavior_hash(self.behavior)


@dataclass
class FeatureList:
    """features.yaml 根结构"""
    schema: int # 清单格式版本号
    milestone: str # 当前迭代点位标识; 进入下一个计划点位时, 手动更新features.yaml(有且仅有这一处)
    items: list[FeatureItem] # 功能项列表 (真正的实体)
