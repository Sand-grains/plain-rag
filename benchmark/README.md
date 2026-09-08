# benchmark —— 标注与评测机制

## 一、三个标注集的定位（相互独立）

三个标注集是**独立定位**，彼此不串联、无先后依赖，各自独立标注与生产。仅 `private_builtin.json` 是**主测评集**，评测以其为基准。

- `private_builtin.json` —— 人工种子集 / **主测评集**
  - `query` 与 `reference_facts` 由 LLM 生成；`reference_facts` 约束为"在源文档中复现，或同语义改写"。
  - `expected_parent_ids`（父块引用）、`relevance`（3/2/1 分级）、`difficulty`（single_chunk/multi_chunk）、`question_type`（factual/comparison/conditional/multi_hop）由**人工**逐条标注。
  - 无解、低质量（找不到支撑）的条目在人工阶段被删除。
- `private_crawler.json` —— 独立集，经下述 LLM 辅助标注方法标注。
- `public.json` —— 独立公开集，作辅助对照。

## 二、人工标注方式（用于 private_builtin.json）

- 交互式逐条标注：键盘 `Enter` 标注 / `e` 跳过 / `m` 结束本条，支持从第 N 条断点续标。
- 约束：运行前必须校验索引状态；索引过期需先为文档重建并持久化，否则 chunk_id 与条目不匹配。

## 三、LLM 辅助标注方法（方法级，可应用于任一独立集）

该方法与数据集解耦，可应用于任意待标注的独立集（如 private_crawler.json）。对每条 query 依序执行：

1. **提议**：提议者读该条来源文档的全文父块，输出候选标注——`expected_parent_ids`（≤4 块）、每块 `relevance` 与 `evidence_quote`（摘一句能独立回答的原文）、`difficulty`/`question_type`/`confidence`；若无解或问偏，输出 `mark_for_delete=true` 与 `hint_location`。输出 chunk_id 先做存在性校验，非法时发纠正提示词重试一次，仍错则进复核。
2. **两轮对拍**：校验者第一轮只看被提块 + 块级负样本，第二轮读全文；程序比较两轮 grounded 集合，`ROUND1 ⊆ ROUND2` 才算一致；块级负样本（同文档内与 query 无关的块）必须被拒，否则视为"全标倾向"进复核；第二轮命中超过 4 块时，由第三者只读候选块做充分性判定，非 sufficient 进复核。
3. **程序化校验**：
   - 引用存在性：`evidence_quote` 归一化后必须是对应 chunk 内容的子串；
   - 语义校验：query 与度量对象（multi_chunk 按序拼接各块证据、single_chunk 用所在 chunk 全文）做本地 BGE-M3 余弦 + 词面重叠校验（硬门 0.85、弱地板 0.4、词面 floor 0.1）；
   - 一致性：`difficulty` 由块数确定性派生并覆盖 LLM 值；`multi_hop` 与 `single_chunk` 矛盾、或块数超 4，进复核。
4. **删除核对**：无解/问偏条目按 `hint_location` 读原文复核，确认文档中确无支撑才进删除清单。
5. **置信度分流**：`confidence` 低于阈值（默认 0.8）的条目进复核。
6. **写回**：通过全部校验的条目由各块 `evidence_quote` 拼接生成 `reference_facts`，合并回条目，剔除删除项并重排；写回前校验索引可覆盖（否则报错）。
7. **落盘与续跑**：每次运行按时间戳分包落盘 passed / review / delete / obs 四类产物（obs 含两轮对拍、余弦、归因明细）；支持 checkpoint 断点续跑；`--limit` 分批时只累计、不写回不重排，保留原 query_id。

## 四、方法验证（不构成数据集串联）

- preflight：先做清洗副本 → dry-run → gate1 → 停在人工检查点，确认后才执行全量。
- gate1 用**人工标注的 builtin** 作为 gold，验证 **LLM 标注方法**的输出正确性（自动采纳条目相对 builtin 的有效块命中率，目标 ≥0.85）。
- 这里的 builtin 只充当"验证方法用的 gold"：**是方法验证，不是数据集之间的先后串链**——三个集仍是独立定位，只是共用同一套 LLM 标注方法。
- 过程指标另含 review 占比健康阈值、无解探针（默认 10%，验证删除路径）。

## 五、独立与闭环

- 三个集各自独立、平行定位；`private_builtin.json` 为主测评集。
- "标注 → 校验 → 写回"的闭环在每个集内部自洽，而非三个集首尾相接；LLM 辅助标注是方法级的，独立应用到具体某个集。
