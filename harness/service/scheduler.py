"""调度器: 从清单挑下一个 not_started 项, 只出候选, 用户拍板。

按 id 序选第一个 not_started, 附 behavior 供候选展示。 (只读, 不改任何状态)
调度器不做自动决策。
"""
from harness.core.models import FeatureItem, FeatureList
from harness.core.states import NOT_STARTED
from harness.core.utils import sorted_items


def next_candidate(feature_list: FeatureList) -> FeatureItem | None:
    """按 id 序返回第一个 not_started 功能项, 无候选时返回 None。

    Args:
        feature_list: 清单快照(registry.load() 的结果)。

    Returns:
        FeatureItem | None: 下一个候选功能项, 或 None。
    """
    for item in sorted_items(feature_list.items):
        if item.state == NOT_STARTED:
            return item
    return None
