# CLAUDE.md

slight_rag —— 从零构建的企业级 RAG 核心，含文档索引、双路检索（稠密+稀疏）、LLM 生成、Agent 编排、多层测评、外部基础设施（PgSQL+ES+Milvus）。

## 环境

- **Python 3.11**（`.python-version` 锁定）
- **包管理**：`uv`（`uv sync` / `uv run` / `uv add`），不混用 conda
- **LLM**：DeepSeek API（OpenAI 兼容），`.env` 配置 `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL`
- **Embedding**：BGE-M3，本地路径 `D:\Model\BGE-M3`，HF 镜像 `hf-mirror.com`
- **缓存**：Redis（`REDIS_URL` 在 `.env`），无 Redis 时自动降级 NoopBackend
- **基础设施**（external 模式）：PostgreSQL 16+（chunk 元数据） + Elasticsearch 8.x + IK 分词器（BM25 稀疏检索） + Milvus 2.6 standalone（HNSW IP 稠密检索），三者均为 Windows 服务/Docker 手动启动，脚本 `scripts/infra_start.bat` / `scripts/infra_stop.bat`
- **依赖**：`elasticsearch`、`psycopg2-binary`、`pymilvus`（>=3.0）

## 项目结构

```
config.py              # 根配置（LLM + Generator + Eval + STORAGE_BACKEND 开关）；基础设施连接参数见 infra/config.py
preprocess/            # Markdown 结构诊断（DocQualityReport + diagnose → Router 路由输入）
indexing/              # 加载 + 分块 + 索引存储（loader / router / splitter/ 三算子 / chunk.py / chunk_ingest_ex.py / index_store.py）
retrieval/             # 双路检索 + LLM 生成（retriever / generator / embedding）
eval/                  # 测评系统（benchmark + Layer1 检索 + Layer2 Judge + 缓存 + 终端面板）
  core/                #   核心：monitor_metrics / llm_as_judge / judge_cache / live_panel / retrieval_layer
  results/             #   输出（gitignored）：timeline/<ts>/ + check/history.jsonl
infra/                 # 基础设施客户端
  config.py            #   连接参数（Redis / Milvus / ES / PgSQL），横切开关仍在根 config.py
  db/                  #   PgSQLClient：chunk 元数据 + status 状态机
  search/              #   ESClient：BM25 全文检索（IK 分词器）
  vector/              #   MilvusClientWrapper：HNSW IP 稠密向量检索
  cache/               #   Redis + Noop 降级
agent/                 # Agent 编排（hello-agents 框架）
scripts/               # 运维脚本（infra_start.bat / infra_stop.bat / infra_check.py）
tests/                 # 分层测试体系（详见"测试体系"章节）：unit/ + integration/ + regression/ + _fakes.py + conftest.py + report.md
guide/                 # 设计文档（gitignored）
data/                  # 语料（gitignored）
```

## 关键约定

### 代码规范：docstring 与命名（重点参照 indexing/splitter/recursive_splitter.py）
新写/修改代码严格遵循以下约定：
- docstring: Google 式中文。模块开头有标准多行模块 docstring（职责 + 关键机制 + 与上层的关系）；除 `__init__` 外的每个方法都有 docstring
- docstring 结构：首行一句话总结 → 正文（机制、边界、调用方陷阱）→ `Args:` → `Returns:`。`Args` 描述参数的类型语义（如"全局起始偏移"）；`Returns` 写明阶段边界（如 `_split_text` 注明"未做 overlap/孤儿合并，由 split() 后处理完成"）。
- 变量命名: 完整语义化单词，命名在6个单词24个字符以内就绝对不缩写。。不用 `seg/sep/good/cur/n/i` 这类缩写，用 `segment/separator/good_pieces/current_chunk/length/cursor_index`；循环变量也全名化（`index`、`separator_candidate`、`code_range_start`）。
- 类型注解: 原生泛型 + `|` 联合**。用 `list[...]`/`tuple[...]`/`dict[...]`（不用 `typing.List/Tuple/Dict`），可空用 `X | None`（PEP 604，不用 `Optional[X]`）；方法签名必须带完整返回类型注解。
- 注释用中文，解释"为什么/语义"而非复述代码（如 `cursor = 0  # 切分游标`）。
- 长文件分区: 用 `# ---- 分区名 ----` 标记逻辑分区（如 `# ---- 核心切分逻辑 ----` / `# ---- 后处理 ----`）。

### 路径：`__file__` 推导，不依赖 CWD

`config.py` 中 `_PROJECT_ROOT = Path(__file__).resolve().parent`。所有 I/O 路径以此为基准推导，确保 PyCharm 任意工作目录 + 终端 `python -m eval.runner` 均正常。

### 测评温度：eval 场景 temperature=0

Generator 和 Judge 在 eval 下均 `temperature=0`（确定性输出）。这是内容寻址缓存生效的前提——任何非确定性都会使缓存键失效。
- Generator：`config.py` 中 `GENERATOR_TEMPERATURE` 默认 0，`generator.generate()` 默认取此值
- Judge：`_call_llm()` / `run_judge()` 默认 `temperature=0.0`

### MonitorMetrics 是唯一真源

LivePanel、`render_final()`、`reporter.py` 全部从同一个 `MonitorMetrics` 实例读取数据。不存在两条独立计算路径。`layer1_results: list` 和 `layer2_results: list` 使用裸 `list` 类型注释（避免对 retrieval_layer 的 import 依赖）。

### 线程安全：浅拷贝后遍历

CPython GIL 下 `list.append()` 原子，但迭代不安全。daemon 线程的所有聚合方法（`layer1_means()` / `layer2_means()` / `stage_percentiles()`）内部先 `list(self.xxx)` 做浅拷贝再遍历。

### LivePanel.stop() 幂等

正常路径和 `finally` 块都可能调用 `stop()`。实现：`if not self._running: return`。第二次调用直接返回。

### push_alert：全局单例，非参数透传

`get_panel()` 模块级单例。深层调用点（`_call_llm` / `_judge_with_retry`）直接 import 全局单例，避免 6 层函数签名各加一个 `panel` 参数。

### 缓存键：内容寻址，五重失效

```
Generator: generator:{query_id}:{context_hash}:{GENERATOR_CONFIG_HASH}
Judge:     judge:{query_id}:{context_hash}:{GENERATOR_CONFIG_HASH}:{prompt_version}:{model_id}
```

`GENERATOR_CONFIG_HASH` = sha256(model_id | temperature | max_tokens | top_p | sha256(prompt_template)[:16])[:12]，模块级常量，一次计算。变更 → 重启进程生效。

五重失效：context_hash / GENERATOR_CONFIG_HASH / prompt_version / model_id / TTL 72h。

### 双线程池架构

- **外层池**（`_get_outer_pool()`，max_workers=5）：query 级并发，`as_completed` 收集
- **内层池**（临时 `ThreadPoolExecutor(max_workers=2)`）：单 query 内 faithfulness + quality 并行
- 双池隔离，杜绝死锁

### 终端双模式：ANSI / plain

`.env` 中 `LIVE_PANEL_MODE=ansi|plain`。PyCharm embedded terminal 可能不完全支持 ANSI 转义码——切 `plain` 降级，用分隔线追加面板块。

### STORAGE_BACKEND：双模式存储后端

`.env` 中 `STORAGE_BACKEND=memory|external`，**默认 `memory`**。这是 Phase 1 基础设施迁移的核心开关：

- **memory**（v5 原行为）：numpy 内存数组 + pickle 持久化 + jieba 分词 BM25。零外部依赖，适合快速开发。
- **external**：PgSQL（chunk 元数据）+ Elasticsearch（IK 分词 BM25）+ Milvus（HNSW IP 稠密向量）。可扩展，适合大规模语料。

`IndexStore` 是统一门面（`indexing/index_store.py`），对外接口不变，内部分 `_xxx_memory()` / `_xxx_external()` 两条路径。`agent_pipeline.py` 和 `eval/runner.py` 均按 `STORAGE_BACKEND` 分支。

### 外部模式写入顺序与状态机

三库写入有严格的先后顺序，保证最终一致性：

```
rollback_doc(doc_id)           # 1. 清理旧数据（幂等）
  → PgSQL insert (suspending)  # 2. 先写 PgSQL，标记 suspending
  → ES insert                  # 3. ES 索引
  → Milvus insert              # 4. Milvus 写入向量
  → PgSQL update (indexed)     # 5. 全部成功后标记 indexed
```

任何步骤失败 → 残留 `suspending` 行由 `cleanup_suspending(ttl=30min)` 定时回滚。`IndexStore._init_external()` 启动时自动调用一次。

### ES 分词：IK 而非 jieba

ES 8.14.3 无官方 jieba 插件，使用社区标准 IK 分词器。索引用 `ik_max_word`（最大切分），搜索用 `ik_smart`（粗粒度）。memory 模式仍用 Python jieba。两种分词器粒度差异约 43%，导致 external 模式 MRR 偏差 ~0.03（可接受范围）。

### history.jsonl 向前兼容

新版本追加字段时用 `.get()` 读取，旧行缺字段返回 `None` → 面板/报告显示 `—`。新字段不破旧数据。

## 常用命令

```bash
# 基础设施健康检查
uv run python scripts/infra_check.py

# 仅 Layer 1 检索评估（无 LLM，免费）
uv run python -m eval.runner --mode retrieval

# 完整评估（Layer 1 + Layer 2，调 Judge LLM，计费）
uv run python -m eval.runner --mode full

# 对比两次运行
uv run python -m eval.runner --compare <run_id_1> <run_id_2>

# 指定 benchmark 文件
uv run python -m eval.runner --mode full --benchmark benchmark/public.json

# Judge 校准
uv run python -m eval.core.llm_as_judge.judge_calibrate

# benchmark 标注
uv run python annotate_helper.py

# 测试（详见"测试体系"章节）
uv run pytest -q
```

## 测试体系

分层测试金字塔（pytest），`testpaths=tests`，默认 `addopts = -m "not external and not slow"`。

```
tests/
  conftest.py       # sys.path 注入 + 共享 fixture（memory_store / mini_corpus / semantic_fakes）+ report.md 渲染钩子
  _fakes.py         # Fake 层：FakeEmbedding/Reranker/Cache/Generator + install_all_fakes 一键替换（含 G3 接口自检）
  unit/             # 纯函数/模块级：metrics、splitter 全家、retriever 助手、judge_cache、_select_best 等
  integration/      # 模块间链路：memory 全链路、检索+生成、评测 gate 五类 diagnosis、external
  regression/       # 回归迁移：splitter 6 输入、needle 判定、calibrate boost 循环
  report.md         # 每次 pytest 自动生成：分层统计 + 模块覆盖 + 失败明细（含 known_issues.md）
  known_issues.md   # 测试期已知问题记录（手动维护，report 失败明细持久嵌入）
```

### 运行

```bash
uv run pytest -q                                     # 全量（默认排除 external/slow）
uv run pytest tests/unit -q                           # 单层
uv run pytest tests/regression/test_needle.py -q      # 单文件
# external：需三服务 + 环境变量（双门控，见下）
SLIGHT_RAG_EXTERNAL=1 uv run pytest tests/integration/test_external_backend.py -m external -q
# slow：needle 完整流程，需 data/synthetic/manifest.json（先 synthetic_docs_generate.py）
uv run pytest -m slow
```

### 关键机制（写测试/改代码必须知道）

- **autouse install_all_fakes**：unit/integration/regression 三层 conftest 都 autouse 装 Fake 层，隔离 BGE-M3 / CrossEncoder / Redis / LLM 重依赖。`STORAGE_BACKEND` 被强制打成 `memory`（`.env` 是 external，性命攸关——不 patch 会真连三库）。
- **semantic_fakes(groups)**：断言"query 命中预期 chunk"用。用法 `semantic_fakes({"rag": ["RAG"], "code": ["Python"]})`——组关键词子串匹配归组、组间正交 one-hot 向量。
- **模块级 import 陷阱**：`from retrieval.embedding import embed` 在测试模块顶部会绑定真实 embed（collection 时未装 fake）。必须函数内 import（参照 integration/conftest 的 ingest_docs 写法）。
- **全局可变配置隔离**：覆盖 `config` 项（如 MERGE_BOOST）必须 `monkeypatch.setattr(config, "X", v)`，禁止直接赋值。
- **external 双门控**：`@pytest.mark.external` + `skipif(not SLIGHT_RAG_EXTERNAL)` + 三端口探活 skip 而非 fail。默认 `-m "not external"` 排除；要跑需服务在线 + 环境变量 + `-m external`。
- **slow 门控**：needle 完整流程标 slow + `skipif(manifest 缺失)`，opt-in。
- **report.md**：每次 pytest 后自动渲染；`--collect-only` 有守卫不覆盖；模块覆盖表取自 pytest-cov `_cov` 插件；失败明细持久嵌入 `known_issues.md`。

### 铁律：改代码必测（G-MAINT）

**后续对项目的任何改动都必须配套测试并跑通**：
- 改核心逻辑（splitter / retriever / eval / index_store / 状态机等）→ 连带更新受影响测试，跑对应层后再全量 `uv run pytest -q` 收尾
- 新功能 → 按测试金字塔补用例：纯函数进 unit/，链路进 integration/，脚本迁移进 regression/
- 改分块策略后旧 benchmark 失效 → 必须重标（见"不要做的事"）
- splitter 断言允许主动重标：改断言 = 承认行为变更，记录到 `known_issues.md`

## 测评管线数据流

```
_evaluate_one(query)
  ├── retriever.retrieve()           → chunks
  ├── [Generator 缓存 查/写]         → answer（miss 时才调 generate）
  ├── generator.generate()           → answer（temperature=GENERATOR_TEMPERATURE）
  └── run_judge(chunks, answer)      → JudgeResult
        ├── [Judge 缓存 查/写]
        ├── judge_faithfulness()     ┐
        └── judge_quality()          ┘ 并行 (inner pool, max_workers=2)
                                       → verdict = execute_verdict(scores)
```

阶段延迟：`end_to_end = retrieve + generate + max(faithfulness, quality)`。overhead_ms 是残差（线程调度 + JSON 解析），在最终报告中以脚注展示。

## 不要做的事

- 不要改 `run_retrieval_mode()` 加 daemon 线程——检索模式无 LLM 调用，同步执行快，`render_final()` 足够
- 不要在测评场景下用 `temperature > 0`——会破坏缓存可复现性
- 不要改动分块策略（Splitter 配置 / Router 路由）后旧 benchmark 继续用——父块 id 全量重排，`expected_parent_ids` 全部失效，必须重新标注（含 `expected_child_ids`）
- 不要在 `config.py` 和 `retrieval/generator.py` 之间创建循环导入——`PROMPT_TEMPLATE` 已迁至 `config.py`
- 不要引入 asyncio / tqdm / 事件总线 / Web Dashboard——记入 `add_up.md` 作为未来备选
- 不要在 `STORAGE_BACKEND=external` 时忘记先跑 `scripts/infra_check.py`——三服务缺一不可启动索引或检索
- 分块是父子层级（Phase 2 已完成）：父块存 PgSQL/ES、子块存 Milvus（含 parent_id），检索/benchmark 单元是父块——改分块策略必须重标 benchmark
- 不要移除空文档跳过逻辑——splitter 对空文本返回零分块，agent_pipeline / needle_test / ingest_doc / _batch_add_memory 均有 guard
- 不要改核心逻辑后不跑测试——测试是行为契约，任何改动必须配套测试并全绿（见"测试体系/铁律"）
