# CLAUDE.md

## 这个项目是什么
plain-rag 是一个从零构建, 覆盖索引到测评全链路的 RAG 引擎。覆盖文档索引、混合检索、LLM 生成、Agent 编排、双层eval测评。


## 快速开始
uv sync (同步依赖)
uv run python -m harness status (治理入口: 功能清单状态分布/健康度)
uv run pytest -q (分层测试)
uv run python -m eval.runner --mode retrieval (最小测评)
uv run python agent_pipeline.py (agent交互问答)

(完整测评, 计费)
uv run python scripts/infra_check.py
uv run python -m eval.runner --mode full


## 技术栈与环境说明
- 语言/工具: Python 3.11(`.python-version` 锁定)+ `uv`(不混用 conda);`pytest` + `pytest-cov`
- LLM:DeepSeek(OpenAI 兼容,`openai` SDK),`.env` 配 `LLM_API_KEY/LLM_MODEL_ID/LLM_BASE_URL`; Generator 与 Judge
共用,`eval` 一律 `temperature=0`
- Embedding: 本地 BGE-M3(`sentence-transformers`, 1024 维), 路径为 `D:\Model\BGE-M3` (HF 镜像 `hf-mirror.com`)
- 精排: CrossEncoder(`sentence-transformers`)
- 检索: memory模式: `rank-bm25`+`jieba`、numpy 数组;
      external模式: ES 8.x+IK (sparse)、Milvus 2.6 HNSW IP (dense)
- 元数据/状态: PostgreSQL 16+ (存储 chunk 元数据 + indexed/suspending 状态机)
- 缓存:Redis, 无 Redis 则自动降级 NoopBackend
- Agent/统计:`hello-agents`框架 (SimpleAgent/ToolRegistry); `scipy`(Wilcoxon 符号秩检验)
- external 三服务: PgSQL+ES+Milvus, Windows 服务 + Docker 手动启动,`scripts/infra_start.bat`/`infra_stop.bat`;依赖
`elasticsearch`/`psycopg2-binary`/`pymilvus>=3.0`
- 未实现: `tree-sitter`(仅声明而未 import, 作为预留)


## 全局硬约束
以下内容必须严格遵守
1. 每次只做一个功能点, 每一个功能点都必须配套端到端测试, 验证通过(全绿)后才可开始下一个 (WIP=1)
2. 禁止在实现功能A时"顺便"重构功能B
3. 新增核心逻辑或原有核心逻辑的变动必须配套测试并全绿
4. 新增/修改代码必须遵循 docs/inner/code-style.md 中说明的约束
5. 生成文档需严格遵守`docs/inner/doc-style.md`中的规范
6. eval 场景一律 temperature=0
7. I/O 路径 __file__ 推导,不依赖 CWD
8. run_retrieval_mode() 不加 daemon 线程
9. 禁止 config.py ↔ generator.py 循环导入
10. external 前必须 infra_check
11. 不移除空文档跳过逻辑
12. 外部模式三库写入顺序不可乱
13. 改分块策略必须重标 benchmark


## 必要说明
eval/results/、data/、tests/report.md 是 gitignored 输出, 默认不去里面找实现
docs/inner/plan/read和docs/inner/adr/read 是用户阅读的, 默认别去里面
mermaid 图渲染: 图类型用 graph(flowchart 兼容旧渲染器); 节点文本含特殊字符(/ . + 全角标点)必须引号包裹, 如 A["text"]; 箭头一律用 ASCII ->, 不用 Unicode →; 生成文档后自查渲染, 不渲染即为失败

## 专题文档路由
- `docs/inner/code-style.md` — 写新代码/改代码前必读(docstring/命名/注解)
- `docs/inner/doc-style.md` — 生成/修改文档必读规范
- `docs/inner/adr/main` — 项目架构决策记录
- `docs/inner/plan/main` — 项目迭代过程子计划(点位)
- `PLAN.md` — 项目大方向计划
- `PROGRESS.md` — 项目当前进度, 所处点位(harness 自动生成, 每次新开会话必读)
- `docs/inner/instruction-set.md` — 项目运行所有指令合集
- `docs/inner/eval.md` — 改 eval/agent 缓存或评测逻辑时读(文档暂不存在, 实际以代码为主)
- `docs/inner/eval-pipeline.md` — 想理解eval端到端数据流时读(文档暂不存在, 实际以代码为主)
- `docs/inner/testing-standards.md` — 写测试、跑分层测试时读(文档暂不存在, 实际以代码为主)


## 协作风格
用户每次新开的会话, 交付每一个任务时, 你都要阅读 PROGRESS.md , 确认当前项目进度, 并获取PROGRESS.md中的相关计划文档, 再根据计划文档路由获取必要上下文(你所需要的信息都已包含在项目仓库里)。

【harness 治理】
新功能点实现必须登记为 harness/features.yaml 的一项(行为+验证命令), 经 start→verify 收口; 未登记项不算进入治理, 也就不能进入实现。
功能项实现状态(not_started/active/blocked/passing/regressed/abandoned) 只由 harness 命令决定, agent 只能提交 verify 请求, 不得自行宣布完成。
完成与否以程序命令判定为准(严禁擅自决定): 
- `uv run python -m harness status` (状态分布/健康度)
- `uv run python -m harness verify <ID>` (跑门禁判通过)
- `uv run python -m harness next` (出候选, 由用户拍板)
功能实现清单唯一权威是 harness/features.yaml, 勿改其中 state 字段;
完整命令集见 docs/inner/instruction-set.md 第10点"harness治理层"

【spec】
风格主要是为了明确因果链, 明确边界, 和禁止什么
1. 用户如果需要新增功能, 先与用户详细讨论, 明确用户需求, 了解用户为何要添加该功能, 想要实现怎样的效果.
直到你最终明确因果链后, 将你的理解回馈给用户, 最后再讨论落地方案
2. 用户如果需要维护某些功能或删除某些功能, 先与用户讨论清楚用户希望删除或维护后实现什么目的
直到你最终明确因果链后, 将你的理解回馈给用户。最后评估实现该目的会牵扯到哪些模块, 逻辑。再给出具体的落地方案。
3. 用户如果明确指明出现的非系统性bug或问题, 先给用户看详细的复现路径. 确认复现后, 分析可能的模块功能点以及相关文档, 然后进行深层分析, 
如果分析期复现路径期间遇到与当前项目决策点有分歧, 及时反馈给用户。 与用户讨论完明确因果链后, 最后再输出具体的方案;
4. 如果需要生成设计文档, 必须遵循`docs/inner/doc-style.md`中的约束
5. spec的风格主要是明确因果链, 明确边界, 以及禁止什么
6. harness的落地原则: 不包含纯产品意图(功能意图), 探索型功能, 文案, UI, 一次性设计
7. 对于新功能, 如果最终落地且自测通过, 需要反馈给用户是否需要spec
8. 对于维护或bug修复, 根据spec和harness落地原则, 自行评估是否需要spec