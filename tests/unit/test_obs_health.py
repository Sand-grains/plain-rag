"""unit: obs/health 健康检查。

验证探针 ok/fail 判定与 latency 记录、默认探针集按 STORAGE_BACKEND 的模式语义、
TTL 缓存命中与 refresh 绕过、CLI 退出码 0/1。重连接库全部 monkeypatch, 不触真服务。
"""
import pytest

import config
import infra.cache
from obs import health
from infra.cache.noop_backend import NoopBackend


@pytest.fixture(autouse=True)
def _clear_health_cache():
    """每测试清空探针结果缓存, 防跨测试 TTL 命中污染。"""
    health._HEALTH_CACHE.clear()
    yield
    health._HEALTH_CACHE.clear()


# ---- 探针用的假连接对象 ----

class _FakePgsqlCursor:
    def __init__(self):
        self.sql = None

    def execute(self, sql):
        self.sql = sql

    def close(self):
        pass


class _FakePgsqlConnection:
    def __init__(self):
        self.cursor_created = _FakePgsqlCursor()

    def cursor(self):
        return self.cursor_created

    def close(self):
        pass


class _FakeRedisClient:
    def __init__(self, ok=True):
        self._ok = ok
        self.closed = False

    def ping(self):
        if not self._ok:
            raise ConnectionError("redis down")

    def close(self):
        self.closed = True


class _FakeES:
    def __init__(self, version="8.15.0"):
        self._version = version

    def info(self):
        return {"version": {"number": self._version}}


class _FakeMilvus:
    def __init__(self, version="2.6.0"):
        self._version = version

    def get_server_version(self):
        return self._version


class _FakeBackend:
    pass


def _ok_probe() -> dict:
    return {"ok": True, "latency_ms": 1.0, "detail": "fake ok"}


class TestProbes:
    def test_check_pgsql_ok(self, monkeypatch):
        import psycopg2
        monkeypatch.setattr(psycopg2, "connect", lambda url: _FakePgsqlConnection())
        result = health.check_pgsql()
        assert result["ok"] is True
        assert result["latency_ms"] >= 0
        assert "SELECT 1" in result["detail"]

    def test_check_pgsql_fail(self, monkeypatch):
        import psycopg2

        def _boom(url):
            raise psycopg2.OperationalError("connection refused")

        monkeypatch.setattr(psycopg2, "connect", _boom)
        result = health.check_pgsql()
        assert result["ok"] is False
        assert "OperationalError" in result["detail"]

    def test_check_es_ok(self, monkeypatch):
        import elasticsearch
        monkeypatch.setattr(elasticsearch, "Elasticsearch", lambda url: _FakeES("8.15.0"))
        result = health.check_es()
        assert result["ok"] is True
        assert "Elasticsearch v8.15.0" in result["detail"]

    def test_check_es_fail(self, monkeypatch):
        import elasticsearch

        def _boom(url):
            raise ConnectionError("es down")

        monkeypatch.setattr(elasticsearch, "Elasticsearch", _boom)
        result = health.check_es()
        assert result["ok"] is False
        assert "ConnectionError" in result["detail"]

    def test_check_milvus_ok(self, monkeypatch):
        import pymilvus
        monkeypatch.setattr(pymilvus, "MilvusClient", lambda uri: _FakeMilvus("2.6.0"))
        result = health.check_milvus()
        assert result["ok"] is True
        assert "Milvus v2.6.0" in result["detail"]

    def test_check_milvus_fail(self, monkeypatch):
        import pymilvus

        def _boom(uri):
            raise ConnectionError("milvus down")

        monkeypatch.setattr(pymilvus, "MilvusClient", _boom)
        result = health.check_milvus()
        assert result["ok"] is False
        assert "ConnectionError" in result["detail"]

    def test_check_redis_ok(self, monkeypatch):
        import redis
        monkeypatch.setattr(redis, "from_url", lambda url: _FakeRedisClient(ok=True))
        result = health.check_redis()
        assert result["ok"] is True
        assert "ping" in result["detail"]

    def test_check_redis_fail(self, monkeypatch):
        import redis
        monkeypatch.setattr(redis, "from_url", lambda url: _FakeRedisClient(ok=False))
        result = health.check_redis()
        assert result["ok"] is False
        assert "ConnectionError" in result["detail"]

    def test_check_cache_ok(self, monkeypatch):
        monkeypatch.setattr(infra.cache, "get_cache", lambda: _FakeBackend())
        result = health.check_cache()
        assert result["ok"] is True
        assert "缓存后端可用" in result["detail"]

    def test_check_cache_noop_degraded(self, monkeypatch):
        monkeypatch.setattr(infra.cache, "get_cache", lambda: NoopBackend())
        result = health.check_cache()
        assert result["ok"] is False
        assert "NoopBackend" in result["detail"]


class TestRunChecks:
    def test_default_services_memory(self, monkeypatch):
        monkeypatch.setattr(config, "STORAGE_BACKEND", "memory")
        monkeypatch.setattr(health, "_PROBE_REGISTRY", {"cache": _ok_probe})
        result = health.run_checks()
        assert list(result) == ["cache"]  # memory 只探缓存, 不探外三件

    def test_default_services_external(self, monkeypatch):
        monkeypatch.setattr(config, "STORAGE_BACKEND", "external")
        monkeypatch.setattr(health, "_PROBE_REGISTRY",
                            {name: _ok_probe for name in ("pgsql", "es", "milvus", "redis")})
        result = health.run_checks()
        assert sorted(result) == ["es", "milvus", "pgsql", "redis"]

    def test_result_structure(self, monkeypatch):
        monkeypatch.setattr(health, "_PROBE_REGISTRY", {"svc": _ok_probe})
        result = health.run_checks(services=["svc"], refresh=True)
        assert set(result["svc"]) == {"ok", "latency_ms", "detail"}

    def test_unknown_service(self):
        result = health.run_checks(services=["nope"])
        assert result["nope"]["ok"] is False
        assert "未知探针" in result["nope"]["detail"]

    def test_cache_hit_within_ttl(self, monkeypatch):
        calls = []

        def _counting_probe():
            calls.append(1)
            return {"ok": True, "latency_ms": 1.0, "detail": "fake"}

        monkeypatch.setattr(health, "_PROBE_REGISTRY", {"svc": _counting_probe})
        monkeypatch.setattr(health, "HEALTH_TTL_SECONDS", 60.0)
        first = health.run_checks(services=["svc"])
        second = health.run_checks(services=["svc"])
        assert len(calls) == 1  # TTL 内第二次命中缓存, 不打真探针
        assert first == second

    def test_refresh_bypasses_cache(self, monkeypatch):
        calls = []

        def _counting_probe():
            calls.append(1)
            return {"ok": True, "latency_ms": 1.0, "detail": "fake"}

        monkeypatch.setattr(health, "_PROBE_REGISTRY", {"svc": _counting_probe})
        health.run_checks(services=["svc"])
        health.run_checks(services=["svc"], refresh=True)
        assert len(calls) == 2  # refresh 强制真探针


class TestCli:
    def test_main_all_ok_clean_return(self, monkeypatch, capsys):
        monkeypatch.setattr(health, "run_checks", lambda **kwargs: {
            "cache": {"ok": True, "latency_ms": 2.0, "detail": "缓存后端可用"},
        })
        health.main()  # 全过: 不抛 SystemExit, 进程自然退 0(与 infra_check 成功路径同模式)
        out = capsys.readouterr().out
        assert "[OK] cache" in out
        assert "All services healthy." in out

    def test_main_fail_exit_one(self, monkeypatch, capsys):
        monkeypatch.setattr(health, "run_checks", lambda **kwargs: {
            "pgsql": {"ok": False, "latency_ms": 5.0, "detail": "ConnectionError: down"},
            "redis": {"ok": True, "latency_ms": 1.0, "detail": "Redis ping 通过"},
        })
        with pytest.raises(SystemExit) as exc:
            health.main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "[FAIL] pgsql" in out
        assert "1/2 service(s) failed." in out
