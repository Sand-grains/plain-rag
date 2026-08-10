"""Infra 层配置：Redis / Milvus / Elasticsearch / PostgreSQL 连接参数。

核心特性：
    - 全部从 .env 读取，带默认值，被 infra/{cache,db,search,vector} 客户端消费
    - from config import _PROJECT_ROOT 触发 load_dotenv()，保证下方 os.getenv 能读到 .env
    - REDIS_DEFAULT_TTL 默认 259200（72h），匹配当前 1-3 天一个优化迭代的开发节奏
    - 横切配置（STORAGE_BACKEND / TOP_K / LLM_* 等）留在根 config.py，不随连接参数下沉

用法示例::

    from infra.config import REDIS_CONNECTION_URL, MILVUS_CONNECTION_URI, POSTGRES_CONNECTION_URI
"""

import os
from config import _PROJECT_ROOT  # 触发 load_dotenv()，保证下方 os.getenv 读到 .env（即使仅作副作用）

# Redis 缓存配置
REDIS_CONNECTION_URL = os.getenv("REDIS_CONNECTION_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "plain_rag")
REDIS_DEFAULT_TTL = int(os.getenv("REDIS_DEFAULT_TTL", "259200"))  # 72h

# Milvus 稠密向量检索配置
MILVUS_CONNECTION_URI = os.getenv("MILVUS_CONNECTION_URI", "http://localhost:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_child_chunks")
MILVUS_HNSW_EF = int(os.getenv("MILVUS_HNSW_EF", "128"))

# ElasticSearch 全文检索配置
ES_CONNECTION_URI = os.getenv("ES_CONNECTION_URI", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_parents_chunks")

# PostgreSQL chunk 元数据配置
POSTGRES_CONNECTION_URI = os.getenv("POSTGRES_CONNECTION_URI", "postgresql://postgres@localhost:5432/postgres")
PG_PENDING_CLEANUP_MINUTES = int(os.getenv("PG_PENDING_CLEANUP_MINUTES", "30"))
