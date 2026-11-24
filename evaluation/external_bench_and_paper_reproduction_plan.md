# CameraMaster 外部 Benchmark 与论文评估复刻计划

## 1. 评测边界

MovieBench 处理结果是 CameraMaster 的数据源，不是三个外部 benchmark 的原生输入格式。American Beauty 的重构层级为 **1 个全片 L4、16 个 L3 局部故事弧、598 个 L2 事件、749 个 L1 镜头证据单元**；另有 143 条旧 L3 scene occurrence 作为来源上下文。正式 L3 文件中的 arc 不是 L4 全片故事，旧字段 `source_l4_story_arc_id` 只是来源 ID。

总流程应为：

```text
MovieBench L1/L2/L3/L4
  -> 统一 sample/arc manifest
  -> benchmark-specific adapter
  -> 官方或协议复刻评估器
  -> per-sample raw result
  -> 总报告（分体系，不做未经校准的总平均）
```

主评测只包含：CameraMaster Track A、Track B、ConStory-Bench、DirectorBench、LoCoT2V-Bench。`fl2va_event_keyframe_only` 与 `ref2va_semantic_assets` 的盲评比较不属于主评测。

## 2. MovieBench 到外部 benchmark 的适配

### 2.1 统一中间 manifest

每个 movie、L3 arc 或评测序列先生成一条 manifest，至少包含：

```json
{
  "sample_id": "movie_arc_01",
  "movie_id": "0001_American_Beauty",
  "l4_global_id": "movie_global_story",
  "l3_story_arc_ids": ["movie_arc_01"],
  "ordered_l2_event_ids": ["event_01", "event_02"],
  "source_l1_shot_ids": [],
  "source_storyline": "...",
  "generated_segment_mapping": [],
  "node_descriptions": [],
  "characters": [],
  "locations": [],
  "items": [],
  "video_path": "...",
  "script_path": "...",
  "storyboard_path": "...",
  "duration_sec": null,
  "model_config": {},
  "prompt_version": "..."
}
```

manifest 还必须记录 `sample_granularity`（`movie` / `l3_arc` / `l2_event` / `l1_shot`）、每个生成片段对应的来源 ID、片段时长、顺序、缺失/合并/拆分情况、引用映射状态和输入 hash。不能把层级名直接当成生成片段长度：一个 5 秒生成视频是实验定义的 **generated segment**，而不是天然等于一个 L1、L2 或 L3。

对当前 5 秒生成设置，建议以 **L2 event 为事件语义来源**，L1 shot 提供动作/构图证据，L3 arc 提供局部叙事上下文，L4 提供全片故事及长期状态约束。但 L2 event 的原始 duration 不一定是 5 秒，所以生成前必须冻结映射规则：一个 L2 对应一个生成片段、多个短 L2 合并，或较长 L2 按有证据的 shot/action boundary 拆分；输出中记录实际来源 ID 和映射。若具体任务要求 shot-level 再用 L1 生成，必须单独标记该粒度并重新计算预算。整片/长视频评测按经审计的全片时间顺序组织生成片段；只有实际时长和连续性满足对应 benchmark 条件时才运行，不通过把多个无关 L3 arcs 简单拼接来凑时长。

重构报告当前整体 `validation.status=incomplete`：虽然 L1/L2 index 覆盖完整，部分旧 arc 的 L1 shot 和 scene occurrence 引用未解析。涉及这些引用的样本须标记 mapping incompleteness，并排除或单独报告，不能静默猜配。

### 2.2 ConStory-Bench

**评测对象：文本，不是视频。**

需要从全片 L4、L3 局部 arc、L2 事件以及 CameraMaster 实际生成的 storyboard/script 中导出：

- `story_id`：明确选择 `movie_id + l3_story_arc_id` 或全片 `movie_id`，不同粒度分开评测；
- `generated_story`：模型生成的长故事/剧情扩写文本；
- 可选 `source_story`：仅用于分析，不发送给不允许访问源文本的 judge；
- prompt、模型、judge 配置。

ConStory 的输入是被测系统生成的长故事文本，而非源 MovieBench 标签。不能把参考 `movie_bible.json`、L3 `full_story_S` 或 `narrative_nodes.json` 误当成 CameraMaster 的生成输出。只有真实生成的 story/script 才作为 `generated_story`；若将生成的 storyboard 节点串联，需标明这是 storyboard-text setting。运行官方 ConStory Checker 后报告：

- CED（越低越好）；
- GRR（仅在多个系统共享相同 story_id 时比较，越低越好）；
- 5 类、19 子类错误统计；
- judge 原始 CSV、解析失败和 prompt 版本。

必要工作：实现 `moviebench_to_constory.py`，从 arc/storyboard 生成 parquet，并做 story_id 对齐检查。

### 2.3 DirectorBench

**评测对象：约一分钟的视频及其 prompt/script/storyboard。**

适配方法：

1. 先确定评测样本粒度：若某个连贯序列本身约一分钟，按该序列评；若不是，只有在 L3 arc 的时间范围经审计为互补、顺序可确定且不重叠/不重复时，才可按全片 L4 chronology 组织多个 arc 的生成片段。L4 storyline 本身不能证明其下 16 个 L3 arcs 是一条无缝、互斥的视频时间分区；
2. 对 5 秒片段设置，优先以 L2 事件定义内容，并按冻结的合并/拆分规则形成生成片段；再依经验证的 L4/L3 顺序构成适配视频；保存 L2/L1->输出片段映射；
3. 用对应层级的 storyline 作为 `user_prompt`：arc 评测用 L3 `short_storyline_L`，全片评测用 L4 `global_storyline.story`；
4. 将实际生成节点的事件、镜头描述和顺序导出为 `script_text` 或 `storyboard`，不要将参考真值当作系统生成 script；
5. 音频不存在时显式记录 `audio=unavailable`，不把音频分数当作 0；
6. 记录实际时长。明显短于约一分钟的样本只能作为 smoke/diagnostic，不应与标准一分钟结果直接混合。L3 arc 数量或 L2 event 数量本身不能替代时长检查。

保留 Script、Video、Audio、Stability、Cross-modal 的子指标、confidence、skipped dimension、bottleneck 和 tool trace。总分必须固定 `UserProfile`，并记录实际使用的权重。

必要工作：实现 `moviebench_to_directorbench.py` 和 `run_directorbench_batch.py`，包括拼接、输入 JSONL、失败样本登记和结果归一化。

### 2.4 LoCoT2V-Bench

**评测对象：长时、多场景视频；需要 prompt metadata。** 评测序列应从 L4 全片故事与其有序 L2/L1 子单元构造；不能把单个 L3 arc 默认视为长视频。

不能只生成 `{sample_id: video_path}`，还要为每个 sample 构造与 `prompt.json` 对齐的 metadata：

- `prompt` / `purified_prompt`；
- `character_infos` 与 `sam_prompt`；
- `background_infos`；
- `scene_infos`；
- `fga_questions`；
- `herd_expectations` / `herd_questions`。

MovieBench 中可映射：L4/L3 characters -> character_infos，L3/L1 locations/evidence -> background_infos，按全片顺序的 L2 events（并关联 source L1）-> scene_infos，事件和约束 -> FGA/HERD 问题。`purified_prompt` 必须将本地角色名替换为可观察外观描述。标注是评测元数据，须由人工审核，不能将 LLM 自生成问题直接当 benchmark 真值。

运行并分别保存：Overall/Fine-Grained Alignment、Character/Background Consistency、Warping Error、Perceptual Quality、Dynamic Degree、Motion Smoothness、DEVIL、HERD、Dynamic Quality。四个 conda 环境和 checkpoint 必须记录。

必要工作：实现 `moviebench_to_locot2v.py`、问题生成与人工审核；长视频不足时只运行适用的指标，并在报告中标记 `not_applicable`，不能用 5 秒片段伪装成长视频结果。

## 3. 两篇未开源 benchmark 的协议复刻

这里实现的是“评估协议复刻版”，不是官方 benchmark 的等价实现。必须在论文中标记 `protocol reproduction`，不能写成官方分数。

### 3.1 ViMax 评估复刻

#### A. 视频一致性：CC/IC/GC

适用前提：一个 sample 必须有多个有序生成 shots，并有可追溯的 scene grouping。MovieBench 可由生成片段对应的 L2/L1 IDs 回连到 L3 arc 或 source scene occurrence；不要把 L3 arc 本身当成 shot。

1. 将每个真实生成的独立视频片段视为一个 shot/clip；优先使用 L2 对应的 5 秒生成片段，保存其 L1 evidence 和 L3 scene/arc 分组；
2. 用 ViCLIP 提取 video-level embedding；
3. 同 scene clip pairs 求 cosine，得到 IC；
4. 跨 scene clip pairs 求 cosine，得到 CC；
5. 所有 clip pairs 求 cosine，得到 GC；
6. 分别报告短/长/总体，另按 Character Persistence、Background Persistence、Multi-Person Interaction 标签分组。

需要保存 pair-level CSV、scene grouping、embedding checkpoint、采样帧/片段策略。不得把 CC/IC/GC 与 Track A 的 KER/FDC 相加。

#### B. storyboard 规划 judge

对 CameraMaster 生成的 storyboard 使用与论文相同的 1–5 维度协议：CBC、NC、PPR、VSU、STQ、NF。固定 judge 模型、temperature、JSON schema、重试和证据字段；NF 才允许访问 source narrative，其余维度只看 storyboard batch 与规定上下文。

输出每维分数、理由、缺失/不可评说明和六维平均；平均分仅作为该规划实验内部摘要，不与视频指标合并。

#### C. 不复刻的部分

ViMax 与 baseline 的成对人工胜率、原始 35-story split、ViMax 内部 best-of-k VLM 筛选均不能从 MovieBench 数据自动声称复现。若实施，必须单独建立 pair protocol 并标记为自建比较。

### 3.2 A²RD 评估复刻

#### A. LVbench-C cyclic consistency

适用前提：样本需要实体/地点消失后重现，并有足够历史跨度。A²RD 的 `segment` 指评测中独立的生成视频片段，不是数据层级标签。以当前设置每个生成片段约 5 秒时，segment 数按实际生成片段计数；推荐从 L4 的全片顺序选取 L2 事件作为候选，使用 L1 镜头证据追踪实体状态，并保留 L3 局部上下文。对每个角色、物体、环境：

1. 记录历史相关 scene description 与参考帧；
2. 提供当前 scene description、当前帧和历史参考给 MLLM；
3. 按论文 prompt 判断差异是 expected 还是 unexplained；
4. 仅明显、无故事解释的漂移记为 0，否则记为 1；
5. 分别计算 character/object/environment consistency ratio，并按 window size 分析。

当前 16 个 L3 arcs 的 L3 数量范围为 1–12，但这**不能**用来判断 A²RD 是否满足跨度条件。要检查的是经时间/映射审计后的生成 segment 序列：目标实体在参考出现后，至少经历 10 个 segment 不出现，再次出现。若每段 5 秒，最低结构约为 12 个生成片段（首次出现 + 至少 10 个中间片段 + 再现），约 60 秒；这是最低结构门槛，不等于官方 3/5/10 分钟 benchmark 设置。不能因 16 个 L3 arcs 都隶属同一 L4 就默认可拼接：只有它们的来源时间范围、顺序、互补性和叙事连续性经审计后，才可共同构成长序列；否则标记 `cyclic_protocol_not_applicable` 或仅报告局部诊断。

#### B. 常规视频指标

可在 MovieBench arc-level 视频上复刻：ViCLIP semantic alignment、Gemini narrative coherence、DINO/检测区域的 inter-shot character/environment consistency，以及 VBench motion smoothness。每项独立保存，明确 checkpoint 和采样策略。

#### C. 人评协议

如果要复刻论文人评，使用匿名视频、随机顺序和 1–5 Likert 的六维：Character/Object/Environment Consistency、Transition Smoothness、Narrative Coherence、Reference Consistency。Reference Consistency 在没有参考图时为 N/A；它不属于自动 Track A/B 指标。

## 4. 必须新增的工程工作

1. `schemas/unified_eval_manifest.json`：稳定 sample/arc/node/shot ID、路径、prompt、版本和 hash；
2. `adapters/moviebench_to_constory.py`；
3. `adapters/moviebench_to_directorbench.py`；
4. `adapters/moviebench_to_locot2v.py`；
5. `scripts/run_external_benchmarks.py`：只编排，不修改官方 benchmark 内部实现；
6. `metrics/vimax_protocol_replica.py`：ViCLIP CC/IC/GC 与 storyboard judge；
7. `metrics/a2rd_protocol_replica.py`：cyclic judge 和可适用的常规指标；
8. `scripts/validate_benchmark_inputs.py`：检查长度、字段、sample 对齐、缺失音频和 checkpoint；
9. 总报告增加 `official`、`protocol_reproduction`、`not_applicable` 三种来源/状态标签。

## 5. 推荐执行顺序

1. 先用 `0001_American_Beauty` 和 `3075_SITTER` 生成 unified manifest；在 American Beauty manifest 中显式记录 L4=1、L3=16、L2=598、L1=749 以及 143 source occurrences；
2. 先跑 DirectorBench smoke，因为它只需要视频、prompt 和可选 script；
3. 再构造 LoCoT2V metadata，人工审核 FGA/HERD 问题后运行；
4. 用 CameraMaster 生成的 storyboard/script 跑 ConStory；
5. 实现 ViMax CC/IC/GC；
6. 在 L4 全片顺序上检查实体重现及至少 10 个 generated segment 的间隔，再决定是否运行 A²RD cyclic judge；
7. 最后由 `run_evaluation_matrix.py` 汇总 Track A、Track B、三个外部 benchmark 和两个协议复刻报告，但不计算统一总分。
