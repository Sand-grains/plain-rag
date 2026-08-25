"""项目根配置：各模块横切配置
注: infra (Redis/Milvus/ES/PgSQL) 的连接参数分离到 infra/config.py 中独立配置。

核心特性：
    - 所有路径基于 __file__ 推导 _PROJECT_ROOT，不依赖 CWD
    - .env 通过 load_dotenv 加载，常量通过 os.getenv 读取并带默认值
    - GENERATOR_PROMPT_TEMPLATE 从 retrieval/generator.py 迁入（避免循环导入且语义为配置常量）
    - GENERATOR_CONFIG_HASH 模块级一次计算，sha256(model + temperature + max_tokens + top_p + generator_prompt_template) 捕获 Generator 全部配置变更

用法示例::

    from config import LLM_MODEL_ID, GENERATOR_CONFIG_HASH, TOP_K, _PROJECT_ROOT
    data_path = _PROJECT_ROOT / "data"

公共接口（按数据流顺序分节）：
    - 项目/横切: _PROJECT_ROOT / MONITOR_PANEL_MODE / STORAGE_BACKEND / VECTOR_CACHE_DIR
    - LLM 与生成: LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL / EVAL_LLM_MODEL_ID / GENERATOR_* / GENERATOR_CONFIG_HASH
    - 数据摄入(008): precheck(PRECHECK_ENABLED / PDF_* / HTML_* / DOCX_* / PPTX_*) / loaders(*_LOADER_ENABLED) /
      cleaner(CLEAN_PLAIN_TEXT / CLEAN_NEW_FORMAT) / parse_backends(*_ENABLED 门控)
    - Embedding: EMBEDDING_MODEL_PATH
    - 分块/切分: CHILD_CHUNK_SIZE / CHILD_OVERLAP / PARENT_* / SEPARATORS / HEADING_SPLIT_RULES /
      DocQualityReport 阈值 / TABLE_ATOMIC_MAX_CHARS / PROTECT_TABLES
    - 检索: TOP_K / RRF_K / RERANKER_* / MERGE_*
    - Eval: EVAL_THREADPOOL_WORKERS / JUDGE_* / COST_*
    - 观测: OBS_*（指标库 metrics_sink）
"""

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # 基于 config.py 自身位置定位 .env, 不依赖 CWD

# === 项目根路径与横切 ===
# 项目根（CWD 无关）
_PROJECT_ROOT = Path(__file__).resolve().parent

# 监控模式
MONITOR_PANEL_MODE = os.getenv("MONITOR_PANEL_MODE", "ansi")  # "ansi" 或 "plain"

# 存储后端开关(memory/external)
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "memory")

# 向量缓存目录
VECTOR_CACHE_DIR = str(_PROJECT_ROOT / ".vector_cache")

# === LLM 与生成 ===
# LLM API 配置，从 .env 读取（默认空串：无 .env 的 CI 环境也能 import，GENERATOR_CONFIG_HASH 只需确定性）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")        # API 密钥
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "")      # 模型 ID，如 deepseek-v4-pro
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")      # API 地址，如 https://api.deepseek.com

EVAL_LLM_MODEL_ID = os.getenv("EVAL_LLM_MODEL_ID", "deepseek-v4-flash")  # 评估专用低成本模型

# Generator 配置常量（eval 场景专用）
GENERATOR_TEMPERATURE = float(os.getenv("GENERATOR_TEMPERATURE", "0"))
GENERATOR_MAX_TOKENS = int(os.getenv("GENERATOR_MAX_TOKENS", "0")) or None  # 0 表示不截断
GENERATOR_TOP_P = float(os.getenv("GENERATOR_TOP_P", "1.0"))

# Generator prompt template
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

# Generator 配置指纹（模块级常量, 一次计算, 整个 eval run 不变）
_generator_fingerprint = "|".join([
    LLM_MODEL_ID,
    str(GENERATOR_TEMPERATURE),
    str(GENERATOR_MAX_TOKENS),
    str(GENERATOR_TOP_P),
    hashlib.sha256(GENERATOR_PROMPT_TEMPLATE.encode()).hexdigest()[:16],
])
GENERATOR_CONFIG_HASH = hashlib.sha256(_generator_fingerprint.encode()).hexdigest()[:12]

# === 数据摄入===
# precheck 预检开关
PRECHECK_ENABLED = os.getenv("PRECHECK_ENABLED", "1") == "1"          # 预检总开关
PDF_PRECHECK_ENABLED = os.getenv("PDF_PRECHECK_ENABLED", "1") == "1"  # 各格式预检开关(默认开)
HTML_PRECHECK_ENABLED = os.getenv("HTML_PRECHECK_ENABLED", "1") == "1"
DOCX_PRECHECK_ENABLED = os.getenv("DOCX_PRECHECK_ENABLED", "1") == "1"
PPTX_PRECHECK_ENABLED = os.getenv("PPTX_PRECHECK_ENABLED", "1") == "1"

# precheck 采样阈值
PDF_SAMPLE_PAGES = int(os.getenv("PDF_SAMPLE_PAGES", "5"))        # 采样页数(首/中/尾分布, 小文档全采)
PDF_TEXT_THRESHOLD = int(os.getenv("PDF_TEXT_THRESHOLD", "150"))   # 文本页最小字符数
PDF_IMAGE_AREA_RATIO = float(os.getenv("PDF_IMAGE_AREA_RATIO", "0.5"))  # 文本页"图不主导版面"的图占比上限(F4)
PDF_TEXT_PAGE_RATIO = float(os.getenv("PDF_TEXT_PAGE_RATIO", "0.9"))    # WHOLE_TEXT_PIPELINE 的文本页占比阈值
PDF_MULTI_COLUMN_GAP = float(os.getenv("PDF_MULTI_COLUMN_GAP", "40"))   # 词 x 坐标聚列判定间距(pt)

HTML_MIN_TEXT = int(os.getenv("HTML_MIN_TEXT", "100"))             # HTML 正文最小字符数
HTML_LOW_TEXT_TAG_RATIO = float(os.getenv("HTML_LOW_TEXT_TAG_RATIO", "10"))  # text_tag_ratio(正文字符/标签数)低于此值视为低质量

DOCX_TEXT_THRESHOLD = int(os.getenv("DOCX_TEXT_THRESHOLD", "150"))  # DOCX WHOLE_TEXT_PIPELINE 最小总字符数

PPTX_SLIDE_TEXT_THRESHOLD = int(os.getenv("PPTX_SLIDE_TEXT_THRESHOLD", "50"))  # 文本 slide 最小字符数

# loaders 解析开关(默认开)
PDF_LOADER_ENABLED = os.getenv("PDF_LOADER_ENABLED", "1") == "1"
HTML_LOADER_ENABLED = os.getenv("HTML_LOADER_ENABLED", "1") == "1"
DOCX_LOADER_ENABLED = os.getenv("DOCX_LOADER_ENABLED", "1") == "1"
PPTX_LOADER_ENABLED = os.getenv("PPTX_LOADER_ENABLED", "1") == "1"

# cleaner 清洗开关
CLEAN_PLAIN_TEXT = os.getenv("CLEAN_PLAIN_TEXT", "0") == "1"   # .md/.txt 是否过 cleaner, 默认关(保 private_v6 字节级不变)
CLEAN_NEW_FORMAT = os.getenv("CLEAN_NEW_FORMAT", "1") == "1"   # 新格式归一化 MD 是否过 cleaner, 默认开

# parse_backends 重型后端门控(v2 逐个接入)
DOCLING_ENABLED = os.getenv("DOCLING_ENABLED", "0") == "1"
MINERU_ENABLED = os.getenv("MINERU_ENABLED", "0") == "1"
MARKER_ENABLED = os.getenv("MARKER_ENABLED", "0") == "1"
LLAMAPARSE_ENABLED = os.getenv("LLAMAPARSE_ENABLED", "0") == "1"
VLM_ENABLED = os.getenv("VLM_ENABLED", "0") == "1"
COLPALI_ENABLED = os.getenv("COLPALI_ENABLED", "0") == "1"

# === Embedding ===
# Embedding 模型路径，首次运行自动从 HuggingFace 下载到本地缓存
EMBEDDING_MODEL_PATH = "D:\Model\BGE-M3"  # BGE-M3, 1024 维, 本地路径

# === 分块/切分 ===
# chunk 配置层
CHILD_CHUNK_SIZE = 300    # 子块默认 chunk_size（字符数）
CHILD_OVERLAP = 50        # 子块默认 overlap（字符数）

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

# splitter 表格保护(008 条例四, 仅新格式归一化 MD 生效)
TABLE_ATOMIC_MAX_CHARS = int(EMBEDDING_MODEL_TOKEN_CONSTRAINT * 2 * 0.7)  # 表格保护阈值: 8192 token × len//2 反推 × 安全余量 0.7 ≈ 11468
PROTECT_TABLES = os.getenv("PROTECT_TABLES", "1") == "1"           # 表格保护总开关(仅新格式归一化 MD 生效)

# === 检索 ===
TOP_K = 5                 # 检索时返回相似度最高的 Top-K 个 chunk

# RRF
RRF_K = 60                # RRF 融合 k 值（从 retriever.py 迁入）

# Reranker
RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "1") == "1"   # A/B 基线开关
RERANKER_MODEL_PATH: str = os.getenv("RERANKER_MODEL_PATH")          # 本地Reranker模型离线路径
RERANKER_TOP_K: int = 40                                             # 送入 reranker 的候选数（语料无关常数）
RERANKER_CACHE_TTL: int = 259200                                     # Redis 缓存 TTL(秒, 72h)
RERANKER_ENRICHMENT: bool = os.getenv("RERANKER_ENRICHMENT", "0") == "1"   # 给 CrossEncoder 的输入 pair 追加结构上下文的A/B实验开关; 重排器不仅看到 chunk还可看到它在文档树里的位置（基线后 A/B）
RERANKER_BATCH_SIZE: int = int(os.getenv("RERANKER_BATCH_SIZE", "32"))    # CrossEncoder predict 批次
RERANKER_FP16: bool = os.getenv("RERANKER_FP16", "1") == "1"              # GPU 下 FP16(无 CUDA 自动回退 FP32)
RERANKER_MAX_LENGTH: int = int(os.getenv("RERANKER_MAX_LENGTH", "1024"))  # CrossEncoder max_length（实测父块 p99=1000/max=1887 tok，1024 覆盖 99.1%；512 截断 13.6%）
RERANKER_POOL_K: int = int(os.getenv("RERANKER_POOL_K", "40" if RERANKER_ENABLED else "0"))  # 召回的全量候选数（向量库 + BM25 各取多少条）  解耦 rerank 开关: 控制组 RERANKER_ENABLED=0 RERANKER_POOL_K=40
RERANKER_AGENT_ENABLED: bool = os.getenv("RERANKER_AGENT_ENABLED", "0") == "1"   # agent 交互路径开关（默认关: 仅 eval 走 RERANKER_ENABLED）

# auto-merge
MERGE_BOOST: float = float(os.getenv("MERGE_BOOST", "0.0"))   # 默认关闭
MERGE_MIN_CHILD_HITS: int = 2    # 触发所需的最小非相邻子块命中数
# 不加 score 下限 —— dense top-40 命中分数全 ≥0.47（达标父块 min-kept-hit p5=0.505）, 下限无约束力（0.5 仅滤 2.7%）

# === Eval ===
# Eval 并发加速
EVAL_THREADPOOL_WORKERS = int(os.getenv("EVAL_THREADPOOL_WORKERS", "5"))  # 外层线程池 worker 数（query 级并发）

# Judge 重试
JUDGE_MAX_RETRY = int(os.getenv("JUDGE_MAX_RETRY", "3"))        # 最大重试次数
JUDGE_BASE_DELAY = float(os.getenv("JUDGE_BASE_DELAY", "1.0"))      # 指数退避初始延迟（秒）
JUDGE_DEADLINE = float(os.getenv("JUDGE_DEADLINE", "120.0"))        # 单次 LLM 调用超时（秒）

# 成本监控配置
COST_INPUT_1K_PRICE = float(os.getenv("COST_INPUT_1K_PRICE", "0.0003"))
COST_OUTPUT_1K_PRICE = float(os.getenv("COST_OUTPUT_1K_PRICE", "0.0012"))

# === 观测 ===
OBS_ENABLED = os.getenv("OBS_ENABLED", "1") == "1"    # 观测总开关(日志落盘/trace)
OBS_LOG_DIR = str(_PROJECT_ROOT / "logs")  # 日志目录(__file__ 推导, CWD 无关)
OBS_SANITIZE = os.getenv("OBS_SANITIZE", "0") == "1"  # trace.jsonl 落盘脱敏开关, 默认关

# 观测的跨 run 指标库(metrics_sink)配置
OBS_METRICS_DIR = str(_PROJECT_ROOT / "eval" / "results" / "metrics")  # 指标库目录(timeline 同址, 不受 logs prune 管)
OBS_METRICS_MAX_RUNS = int(os.getenv("OBS_METRICS_MAX_RUNS", "500"))   # 指标库最大 run 数, record 后超限截尾(500 run 压力可控)
