"""Redis 缓存后端：redis-py 封装，支持 SCAN 批量删除。

核心特性：
    - from_url 连接 + key_prefix 命名空间隔离
    - set 使用 setex 原子设置值 + TTL
    - delete_by_pattern 使用 SCAN 批量删除（非 KEYS，生产安全）
    - exists 基于 EXISTS 命令，O(1)

用法示例::

    from infra.cache.redis_backend import RedisBackend
    backend = RedisBackend("redis://localhost:6379/0", key_prefix="plain_rag")
    backend.set("key", "value", ttl_seconds=3600)

公共接口：
    - RedisBackend: Redis 缓存后端（一个实例持有一个连接池）
"""

import redis
from infra.cache.backend import CacheBackend


class RedisBackend(CacheBackend):
    """Redis 缓存后端。一个实例持有一个连接池。"""

    def __init__(self, redis_url: str, key_prefix: str = "plain_rag"):
        self._client = redis.Redis.from_url(redis_url)
        self._prefix = key_prefix

    def _full_key(self, key: str) -> str:
        """拼接命名空间前缀，避免多服务共用 Redis 时键冲突。

        Args:
            key: 原始缓存键。

        Returns:
            str：形如 "{key_prefix}:{key}" 的完整 Redis 键。
        """
        return f"{self._prefix}:{key}"

    def get(self, key: str) -> str | None:
        """读取缓存值。

        Args:
            key: 原始缓存键（自动加前缀）。

        Returns:
            str | None：缓存值；未命中时返回 None。
        """
        value = self._client.get(self._full_key(key))
        return value.decode("utf-8") if value else None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """原子写入缓存值并设置 TTL（setex）。

        Args:
            key: 原始缓存键。
            value: 缓存值。
            ttl_seconds: 存活时间（秒）。
        """
        self._client.setex(self._full_key(key), ttl_seconds, value)

    def delete_by_pattern(self, pattern: str) -> int:
        """按模式批量删除缓存键（SCAN 迭代，生产安全）。

        Args:
            pattern: 键匹配模式（如 "judge:*"）。

        Returns:
            int：实际删除的键数量。
        """
        full_pattern = f"{self._prefix}:{pattern}"
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=full_pattern, count=100)
            if keys:
                deleted += self._client.delete(*keys)
            if cursor == 0:
                break
        return deleted

    def is_exists(self, key: str) -> bool:
        """判断缓存键是否存在（EXISTS 命令，O(1)）。

        Args:
            key: 原始缓存键。

        Returns:
            bool：存在返回 True，否则 False。
        """
        return bool(self._client.exists(self._full_key(key)))
