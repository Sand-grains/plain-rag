"""追踪器: 状态分布 + 健康度(防虚荣), 供报告器与 status 命令展示。

健康度不用"完成数量"(这是一种虚荣指标), 而是三个可被博弈口径侵蚀的角度:
    - 状态分布: 各态计数
    - 回归率 = regressed / (passed + regressed)   (分母是曾通过验证的项)
    - 验证通过率 = passed / (passed + regressed + active) (active 计入分母, 因为半成品不能算通过)
分母 = 所有非 abandoned 项(跨 milestone); 分母为 0 时指标 = 0。
可疑通过项: passed 但 last_verify 缺失, 或测试数 < N / 覆盖率 < M, 进健康度供人工抽查;
e2e 证据(无 junit/coverage 属正常)跳过该判定。
端到端未验: passed 且声明了 e2e 但 e2e_passed=False(用户豁免/漏跑), 独立列表不与可疑混淆。
"""
from dataclasses import dataclass, field

from harness.core.models import FeatureItem, FeatureList
from harness.core.states import ABANDONED, ACTIVE, PASSED, REGRESSED, VALID_STATES


@dataclass(frozen=True)
class HealthMetrics:
    """健康度快照: 状态分布 + 回归率 + 通过率 + 可疑通过项 + 端到端未验。"""
    distribution: dict[str, int]   # 状态分布: 各态(六态)到项数的映射
    regression_rate: float         # 回归率 = regressed / (passed + regressed), 分母是曾通过验证的项
    pass_rate: float               # 验证通过率 = passed / (passed + regressed + active), active 计入分母防虚荣
    suspicious_ids: list[str]      # 可疑通过项 id: passed 但证据缺失/测试数不足/覆盖率不足, 供人工抽查
    e2e_unverified_ids: list[str] = field(default_factory=list)  # 端到端未验 id: passed 且声明 e2e 但 e2e_passed=False


def _is_suspicious(item: FeatureItem, min_tests: int, min_coverage: float) -> bool:
    """判定一项 FeatureItem 是否为"可疑通过"
    passed 但证据缺失/测试数不足/覆盖率不足视为可疑通过。

    e2e 证据(仅有退出码判定, 无 junit/coverage)属正常, 直接放行而不标可疑。

    Args:
        item: 待判定的功能项。
        min_tests: 测试数阈值, 低于它标记可疑。
        min_coverage: 覆盖率阈值, 缺失或低于它标记可疑。

    Returns:
        bool: True=可疑(进健康度供人工抽查), False=正常。
    """
    evidence = item.last_verify
    if not evidence:
        return True  # passed 却没有验证证据, 高度可疑
    if evidence.get("mode") == "e2e":
        return False  # e2e 证据无 junit/coverage 属正常, 只按退出码判定
    if (evidence.get("tests") or 0) < min_tests:
        return True
    coverage = evidence.get("coverage")
    if coverage is None or coverage < min_coverage:
        return True
    return False


def judge_health(feature_list: FeatureList, min_tests: int = 5,
                   min_coverage: float = 65.0) -> HealthMetrics:
    """统计状态分布与健康度指标。

    Args:
        feature_list: 清单快照。
        min_tests: 可疑判定阈值, 测试数低于它标记可疑。
        min_coverage: 可疑判定阈值, 覆盖率低于它(含缺失)标记可疑。

    Returns:
        HealthMetrics: 状态分布/回归率/通过率/可疑通过项/端到端未验 id 列表。
    """
    distribution = {state: 0 for state in VALID_STATES}
    for item in feature_list.items:
        distribution[item.state] += 1

    ever_passed = distribution[PASSED] + distribution[REGRESSED]
    regression_rate = distribution[REGRESSED] / ever_passed if ever_passed else 0.0

    denominator = distribution[PASSED] + distribution[REGRESSED] + distribution[ACTIVE]
    pass_rate = distribution[PASSED] / denominator if denominator else 0.0

    suspicious_ids = [
        item.id for item in feature_list.items
        if item.state == PASSED and _is_suspicious(item, min_tests, min_coverage)
    ]
    e2e_unverified_ids = [
        item.id for item in feature_list.items
        if item.state == PASSED and item.e2e and not item.e2e_passed
    ]
    return HealthMetrics(
        distribution=distribution,
        regression_rate=regression_rate,
        pass_rate=pass_rate,
        suspicious_ids=suspicious_ids,
        e2e_unverified_ids=e2e_unverified_ids,
    )
