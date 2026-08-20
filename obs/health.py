"""obs 健康检查: 探针纯函数 + TTL 缓存 + CLI退出码, 补运行层健康可见性。
memory 模式下只探 cache
external 模式下探 PgSQL/ES/Milvus/Redis 四个基础设施
返回规范化结果{ok, latency_ms, detail} + 异常隔离

核心特性:
    - 每探针返回 {ok, latency_ms, detail}; 内部 try/except 隔离(失败不抛, detail 带异常类名)
    - 默认探针集按 STORAGE_BACKEND 推导: memory 只探缓存(经 infra/cache), external 探四件套(PgSQL/ES/Milvus/Redis)
    - 探针结果 TTL 缓存(HEALTH_TTL_SECONDS), 高频调用不打真探针; CLI 强制 refresh 读到最新状态
    - 重连接库(psycopg2/elasticsearch/pymilvus/redis)全部函数内懒加载, 模块 import 零副作用
    - 不依赖 eval/runner 与业务模块, 纯 stdlib + infra 层连接参数

与上层的关系: scripts/infra_check.py 是独立连通性 CLI, 本期不改其对外行为(条例二);
本模块是 obs 侧的独立健康探针, 供 `uv run python -m obs.health` CLI 或后续监控钩子消费。
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable

# 探针结果缓存 TTL(秒): 短 TTL 内重复调用直接回缓存, 防监控循环打爆服务连接
HEALTH_TTL_SECONDS = 30.0

# ---- 探针实现(每探针返回 {ok, latency_ms, detail}) ----

def check_cache() -> dict:
    """探测实际运行的缓存后端: RedisBackend 是否降级为 NoopBackend

    Returns:
        dict: {ok, latency_ms, detail}; NoopBackend(redis 降级)判失败。
    """
    def _probe() -> str:
        from infra.cache import get_cache
        from infra.cache.noop_backend import NoopBackend
        cache_backend = get_cache()
        if isinstance(cache_backend, NoopBackend):
            raise RuntimeError("缓存后端不可用(已降级 NoopBackend)")
        return "缓存后端可用"
    return _safe_probe(_probe)


def check_pgsql() -> dict:
    """PgSQL 连通性探针: SELECT 1 通过即健康。

    Returns:
        dict: {ok, latency_ms, detail}。
    """
    def _probe() -> str:
        import psycopg2
        from infra.config import POSTGRES_CONNECTION_URI
        connection = psycopg2.connect(POSTGRES_CONNECTION_URI)
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
        finally:
            connection.close()
        return "SELECT 1 通过"
    return _safe_probe(_probe)


def check_es() -> dict:
    """Elasticsearch 连通性探针: es.info() 取版本号即健康。

    Returns:
        dict: {ok, latency_ms, detail}。
    """
    def _probe() -> str:
        from elasticsearch import Elasticsearch
        from infra.config import ES_CONNECTION_URI
        es_client = Elasticsearch(ES_CONNECTION_URI)
        info = es_client.info()
        return f"Elasticsearch v{info['version']['number']}"
    return _safe_probe(_probe)


def check_milvus() -> dict:
    """Milvus 连通性探针: get_server_version() 即健康。

    Returns:
        dict: {ok, latency_ms, detail}。
    """
    def _probe() -> str:
        from pymilvus import MilvusClient
        from infra.config import MILVUS_CONNECTION_URI
        client = MilvusClient(uri=MILVUS_CONNECTION_URI)
        return f"Milvus v{client.get_server_version()}"
    return _safe_probe(_probe)


def check_redis() -> dict:
    """Redis 直连探针(external 四件套一员): ping 即健康。

    Returns:
        dict: {ok, latency_ms, detail}。
    """
    def _probe() -> str:
        import redis
        from infra.config import REDIS_CONNECTION_URL
        client = redis.from_url(REDIS_CONNECTION_URL)
        try:
            client.ping()
        finally:
            client.close()
        return "Redis ping 通过"
    return _safe_probe(_probe)


def _safe_probe(probe_fn: Callable[[], str]) -> dict:
    """包装探针执行: 记耗时, 捕获异常转 ok=False(detail 带异常类名), 不向外抛。

    Args:
        probe_fn: 无参探针函数, 成功返回 detail 文本。

    Returns:
        dict: {ok, latency_ms, detail}。
    """
    start = time.monotonic()
    try:
        detail = probe_fn()
        latency_ms = (time.monotonic() - start) * 1000
        return {"ok": True, "latency_ms": round(latency_ms, 2), "detail": detail}
    except Exception as error:
        latency_ms = (time.monotonic() - start) * 1000
        return {"ok": False, "latency_ms": round(latency_ms, 2), "detail": f"{type(error).__name__}: {error}"}


# ---- 探针注册表与默认集 ----

_PROBE_REGISTRY: dict[str, Callable[[], dict]] = {
    "cache": check_cache,
    "pgsql": check_pgsql,
    "es": check_es,
    "milvus": check_milvus,
    "redis": check_redis,
}

# service -> (探测时刻 monotonic, 结果); TTL 内命中直接复用
_HEALTH_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()


def _default_services() -> list[str]:
    """按 STORAGE_BACKEND 推导默认探针集: memory 只探缓存, external 探四件套。

    内存模式不依赖外三件(PgSQL/ES/Milvus), 全量探了必失败; 缓存经 infra/cache
    是两种模式共用的唯一基础设施, memory 下探 cache 即完整健康信号。
    """
    from config import STORAGE_BACKEND
    if STORAGE_BACKEND == "external":
        return ["pgsql", "es", "milvus", "redis"]
    return ["cache"]


def run_checks(services: list[str] | None = None, refresh: bool = False) -> dict[str, dict]:
    """执行一组健康探针, 结果按 service 带 TTL 缓存。

    Args:
        services: 探针名列表; None 时按 STORAGE_BACKEND 推导默认集。
        refresh: True 时绕过缓存强制真探针(CLI 用, 读到最新状态)。

    Returns:
        dict[str, dict]: service -> {ok, latency_ms, detail}; 未知名返回 ok=False。
    """
    names = list(services) if services is not None else _default_services()
    results: dict[str, dict] = {}
    now = time.monotonic()
    for name in names:
        probe_fn = _PROBE_REGISTRY.get(name)
        if probe_fn is None:
            results[name] = {"ok": False, "latency_ms": 0.0, "detail": f"未知探针: {name}"}
            continue
        with _CACHE_LOCK:
            cached = _HEALTH_CACHE.get(name)
            if not refresh and cached is not None and now - cached[0] < HEALTH_TTL_SECONDS:
                results[name] = cached[1]
                continue
        result = probe_fn()
        with _CACHE_LOCK:
            _HEALTH_CACHE[name] = (now, result)
        results[name] = result
    return results


# ---- CLI ----

def main() -> None:
    """CLI 入口: 全探针通过返 0, 任一失败返 1(强制 refresh, 结果逐行打到 stdout)。

    Raises:
        SystemExit: 全部通过退 0, 任一失败退 1。
    """
    from obs.logging_setup import make_stdio_encoding_safe
    make_stdio_encoding_safe()  # GBK 控制台重定向下 detail 可能含缺码字符, replace 防崩溃保退出码
    results = run_checks(refresh=True)
    failed = 0
    for name in sorted(results):
        result = results[name]
        status = "OK" if result["ok"] else "FAIL"
        print(f"[{status}] {name}  {result['detail']}  ({result['latency_ms']:.0f}ms)")
        if not result["ok"]:
            failed += 1
    print()
    if failed:
        print(f"{failed}/{len(results)} service(s) failed.")
        sys.exit(1)
    print("All services healthy.")


if __name__ == "__main__":
    main()
