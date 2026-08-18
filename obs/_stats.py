"""obs 通用纯函数：从 eval/utils 复制, 供 obs/monitor 消费, 与 eval 解耦。

本模块是 eval.utils 中 avg_of/percentile/p95/format_time 四个纯函数的复制品,
迁入 obs 是为了让 obs 包自成一体(不依赖 eval 命名空间), 同时保持 eval/utils
与其现有消费方不动。

注意: 复制意味着两处各持一份实现, 若 eval.utils 的这四个函数需要改动,
须同步更新本模块(纯函数, 改动极低频)。
"""


def avg_of(values: list[float]) -> float | None:
    """非 None 值均值；全为 None 时返回 None。

    Args:
        values: 数值列表（可含 None）。

    Returns:
        float | None：非 None 值的均值；全 None 时返回 None。
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def percentile(values: list[float], percentile: float) -> float:
    """计算数值列表的第 percentile 百分位；空列表返回 0.0。

    Args:
        values: 数值列表。
        percentile: 百分位（0-100，如 50/75/95）。

    Returns:
        float：对应百分位值；空列表时返回 0.0。
    """
    if not values:
        return 0.0
    import numpy
    return float(numpy.percentile(values, percentile))


def p95(values: list[float]) -> float:
    """第 95 百分位；空列表返回 0.0。

    Args:
        values: 数值列表。

    Returns:
        float：P95 值；空列表时返回 0.0。
    """
    return percentile(values, 95)


def format_time(seconds: float) -> str:
    """秒数格式化为 HH:MM:SS。

    Args:
        seconds: 经过的秒数。

    Returns:
        str：HH:MM:SS 格式。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
