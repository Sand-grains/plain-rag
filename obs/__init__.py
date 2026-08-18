"""obs 统一可观测层：指标采集 + 终端面板 + 日志 + trace + 归因。

MonitorMetrics/MonitorPanel 自 eval/monitor 迁入, 逻辑与类名不变, MonitorMetrics 唯一真源约定保留。

re-export 模块级单例与面板类，供 runner / judge 延迟导入使用。
"""

from obs.monitor_metrics import get_metrics, reset_metrics
from obs.monitor_panel import MonitorPanel, get_panel, set_panel
