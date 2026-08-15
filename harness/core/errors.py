"""harness 异常基类: 统一错误类型供 CLI 判定退出码。

所有可预期错误(非法转移/清单校验失败/文件锁超时/功能项不存在)都挂在
HarnessError 下, cli 捕获基类统一按失败处理; 具体子类用于区分告警文案。
"""
class HarnessError(Exception):
    """harness 全部可预期异常的基类。"""
