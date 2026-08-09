"""项目根配置：LLM API、Embedding 模型、chunk 参数、eval 并发与重试、Generator 配置指纹。

核心特性：
    - 所有路径基于 __file__ 推导 _PROJECT_ROOT，不依赖 CWD
    - .env 通过 load_dotenv 加载，常量通过 os.getenv 读取并带默认值
    - GENERATOR_PROMPT_TEMPLATE 从 retrieval/generator.py 迁入（避免循环导入且语义为配置常量）
    - GENERATOR_CONFIG_HASH 模块级一次计算，sha256(model + temperature + max_tokens + top_p + generator_prompt_template) 捕获 Generator 全部配置变更

用法示例::

    from config import LLM_MODEL_ID, GENERATOR_CONFIG_HASH, TOP_K, _PROJECT_ROOT
    data_path = _PROJECT_ROOT / "data"

公共接口（常量）：
    - LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL: DeepSeek API 配置
    - EVAL_LLM_MODEL_ID: 评估专用低成本模型
    - GENERATOR_PROMPT_TEMPLATE: Generator 中文 RAG prompt 模板
    - GENERATOR_TEMPERATURE / GENERATOR_MAX_TOKENS / GENERATOR_TOP_P: Generator 配置（eval 场景默认 temperature=0）
    - GENERATOR_CONFIG_HASH: Generator 配置指纹（12 位 hex）
    - MONITOR_PANEL_MODE: 终端面板模式（"ansi" | "plain"）
    - EMBEDDING_MODEL_PATH / CHILD_CHUNK_SIZE / CHILD_OVERLAP / TOP_K: 检索管线参数
    - _PROJECT_ROOT: 项目根目录 Path（CWD 无关）
    - EVAL_THREADPOOL_WORKERS / JUDGE_MAX_RETRY / JUDGE_BASE_DELAY / JUDGE_DEADLINE: Eval 并发与重试参数
    - COST_INPUT_1K_PRICE / COST_OUTPUT_1K_PRICE: 成本估算单价
    - STORAGE_BACKEND: 存储后端 "memory" | "external"
    - RERANKER_* / MERGE_*: 精排（CrossEncoder 重排 + auto-merge 后处理）配置
    - 基础设施连接参数（MILVUS / ES / PostgreSQL / Redis）已下沉至 infra/config.py
"""

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # 基于 config.py 自身位置定位 .env, 不依赖 CWD

# LLM API 配置，从 .env 读取
LLM_API_KEY = os.getenv("LLM_API_KEY")        # API 密钥
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")      # 模型 ID，如 deepseek-v4-pro
LLM_BASE_URL = os.getenv("LLM_BASE_URL")      # API 地址，如 https://api.deepseek.com

EVAL_LLM_MODEL_ID = os.getenv("EVAL_LLM_MODEL_ID", "deepseek-v4-flash")  # 评估专用低成本模型

# === Generator 配置常量（eval 场景专用）===
GENERATOR_TEMPERATURE = float(os.getenv("GENERATOR_TEMPERATURE", "0"))
GENERATOR_MAX_TOKENS = int(os.getenv("GENERATOR_MAX_TOKENS", "0")) or None  # 0 表示不截断
GENERATOR_TOP_P = float(os.getenv("GENERATOR_TOP_P", "1.0"))

# === Generator prompt template（从 retrieval/generator.py 迁入, 本质是配置常量）===
GENERATOR_PROMPT_TEMPLATE = """
## Role
你是一个专业的知识库问答助手, 你的任务是严格根据提供的【参考文档】回答用户的问题

## Rules(关键)
1.必须**仅依赖**下方的【参考文档】进行回答, 不要使用你内部的训练知识
2.如果【参考文档】中没有包含回答问题所需的信息, 请直接回答: "知识库中未找到相关信息". **严禁编造**
3.回答需要简洁, 逻辑清晰, 准确, 有条理, 分点描述
4.引用来源时标注[来源X], 在回答的末尾注明引用的文档名称

## Context(检索到的片段)
以下是参考文档片段:
<context>
{context_str}
</context>

## User Question
用户问题是:
{query_str}

## 回答
请开始回答:"""

# === Generator 配置指纹（模块级常量, 一次计算, 整个 eval run 不变）===
_generator_fingerprint = "|".join([
    LLM_MODEL_ID,
    str(GENERATOR_TEMPERATURE),
    str(GENERATOR_MAX_TOKENS),
    str(GENERATOR_TOP_P),
    hashlib.sha256(GENERATOR_PROMPT_TEMPLATE.encode()).hexdigest()[:16],
])
GENERATOR_CONFIG_HASH = hashlib.sha256(_generator_fingerprint.encode()).hexdigest()[:12]

# === 监控模式 ===
MONITOR_PANEL_MODE = os.getenv("MONITOR_PANEL_MODE", "ansi")  # "ansi" 或 "plain"

# Embedding 模型路径，首次运行自动从 HuggingFace 下载到本地缓存
EMBEDDING_MODEL_PATH = "D:\Model\BGE-M3"  # BGE-M3, 1024 维, 本地路径

# chunk配置层
# CHUNK_SIZE = 500  CHUNK_OVERLAP = 100  # 滑动窗口
CHILD_CHUNK_SIZE = 300    # 子块默认 chunk_size（字符数）
CHILD_OVERLAP = 50        # 子块默认 overlap（字符数）
TOP_K = 5                 # 检索时返回相似度最高的 Top-K 个 chunk

# RRF
RRF_K = 60                # RRF 融合 k 值（从 retriever.py 迁入）

# ========== Reranker==========
RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "1") == "1"   # A/B 基线开关
RERANKER_MODEL_PATH: str = os.getenv("RERANKER_MODEL_PATH")          # 本地Reranker模型离线路径
RERANKER_TOP_K: int = 40                                             # 送入 reranker 的候选数（语料无关常数）
RERANKER_CACHE_TTL: int = 259200                                     # Redis 缓存 TTL(秒, 72h)
RERANKER_ENRICHMENT: bool = os.getenv("RERANKER_ENRICHMENT", "0") == "1"   # 给 CrossEncoder 的输入 pair 追加结构上下文的A/B实验开关; 重排器不仅看到 chunk还可看到它在文档树里的位置（基线后 A/B）
RERANKER_BATCH_SIZE: int = int(os.getenv("RERANKER_BATCH_SIZE", "32"))    # CrossEncoder predict 批次
RERANKER_FP16: bool = os.getenv("RERANKER_FP16", "1") == "1"              # GPU 下 FP16(无 CUDA 自动回退 FP32)
RERANKER_MAX_LENGTH: int = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))  # CrossEncoder max_length（实测父块 p99=1000/max=1887 tok，1024 覆盖 99.1%；512 截断 13.6%）
RERANKER_POOL_K: int = int(os.getenv("RERANKER_POOL_K", "40" if RERANKER_ENABLED else "0"))  # 召回的全量候选数（向量库 + BM25 各取多少条）  解耦 rerank 开关: 控制组 RERANKER_ENABLED=0 RERANKER_POOL_K=40
RERANKER_AGENT_ENABLED: bool = os.getenv("RERANKER_AGENT_ENABLED", "0") == "1"   # agent 交互路径开关（默认关: eval 走 RERANKER_ENABLED）

# ========== auto-merge==========
MERGE_BOOST: float = float(os.getenv("MERGE_BOOST", "0.0"))   # 默认关闭
MERGE_MIN_CHILD_HITS: int = 2    # 触发所需的最小非相邻子块命中数
# 不加 score 下限 —— dense top-40 命中分数全 ≥0.47（达标父块 min-kept-hit p5=0.505）, 下限无约束力（0.5 仅滤 2.7%）

# Splitter 配置
PARENT_CHUNK_SIZE = 1200   # 父块默认 chunk_size（flat_parent_child，父=4×子）
PARENT_OVERLAP = 0         # 父块 overlap 始终为 0（父块是最终返回文本，不需重叠防止语义断裂）
PARENT_MAX_CHARS = 8000    # 父级安全上限（≈3000+ token；prompt 预算 + BM25 长度归一化）
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]   # 递归切分分隔符栈（优先级从高到低）
HEADING_SPLIT_RULES = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

# DocQualityReport 阈值
EMBEDDING_MODEL_TOKEN_CONSTRAINT = 8192    # BGE-M3 Embedding模型 token 硬上限（诊断告警，不参与路由）
HEADING_DENSITY_THRESHOLD = 2000          # 每 N 字符内至少一个标题
HEADING_DENSITY_MIN_OK = 3                # 豁免: has_h1 + 总标题数 ≥ 此值则 density_ok
CHUNK_TOO_FRAGMENTED_THRESHOLD = 200       # section token 中位数低于此值视为文本过碎
TEXT_RATIO_WARN_THRESHOLD = 0.3            # 纯文本占比低于此值警告

# 向量缓存目录
VECTOR_CACHE_DIR = str(Path(__file__).resolve().parent / ".vector_cache")

# 项目根（CWD 无关）
_PROJECT_ROOT = Path(__file__).resolve().parent

# Eval 并发加速
EVAL_THREADPOOL_WORKERS = int(os.getenv("EVAL_THREADPOOL_WORKERS", "5"))  # 外层线程池 worker 数（query 级并发）

# Judge 重试
JUDGE_MAX_RETRY = int(os.getenv("JUDGE_MAX_RETRY", "3"))        # 最大重试次数
JUDGE_BASE_DELAY = float(os.getenv("JUDGE_BASE_DELAY", "1.0"))      # 指数退避初始延迟（秒）
JUDGE_DEADLINE = float(os.getenv("JUDGE_DEADLINE", "120.0"))        # 单次 LLM 调用超时（秒）

# 存储后端开关(memory/external)
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "memory")

# 成本监控配置
COST_INPUT_1K_PRICE = float(os.getenv("COST_INPUT_1K_PRICE", "0.0003"))
COST_OUTPUT_1K_PRICE = float(os.getenv("COST_OUTPUT_1K_PRICE", "0.0012"))
