# CameraMaster 评测代码建设计划与注意事项

## 1. 结论先行

CameraMaster 后续评测应采用“**自有 CineWorldSim-Bench 为主、三个外部 benchmark 为辅**”的结构：

- **Track A：剧情锚点给定**，回答“既定剧情有没有被正确视觉化”。
- **Track B：故事梗概给定**，回答“有没有在约束内合理扩展剧情”。
- **外部 benchmark** 用于补充可比性：ConStory-Bench 评文本一致性，DirectorBench 评一分钟视频综合质量，LoCoT2V-Bench 评长视频对齐、一致性和动态质量。

不能把四套体系的分数直接平均。它们的输入、样本粒度、分数方向和判定目标不同。

## 2. 评测代码总体架构

```text
原始 MovieBench 资产
  -> 数据审计与稳定 ID
  -> L1 shot evidence / L2 event / L3 local story arc / L4 global movie story
  -> Track A/B 标注与参考资产
  -> 按评测任务选择 L1/L2 生成单元，并以 L3/L4 组织上下文
  -> 生成结果 manifest
  -> 盲态故事/事件重建
  -> 结构化对齐与指标计算
  -> 人工校准
  -> 外部 benchmark 适配器
  -> 分轨、分指标、可追溯报告
```

当前 American Beauty 重构数据的层级是：**1 个全片 L4 global movie story、16 个 L3 局部故事弧、598 个 L2 事件、749 个 L1 镜头证据单元**。另有 143 条旧 L3 scene occurrence，属于生成/重构上下文，不是新 L3 arc 清单。L4 是全片唯一的整体故事线；L3 是局部 arc，不是全片故事。

每个样本和结果必须能回溯到：`movie_id -> l4_global_id -> l3_story_arc_id -> l2_event_id -> l1_shot_id -> benchmark_item_id -> run_id -> prompt/reference/model -> metric result`。L1/L2/L3/L4 是语义与证据层级，不自动等同于视频时长或生成 segment。5 秒生成任务应在实验 manifest 中明确映射到 L2 event 或 L1 shot；不得默认把一个 L3 arc 压成一个 5 秒视频。

**粒度决策：** 不重写或删除 L1-L3；新增 evaluation view。L1 保留为 shot evidence，L2 作为主要事件内容来源，L3 作为局部叙事上下文，L4 作为全片故事上下文。由相邻 L2 按有证据的动作/shot 边界合并或拆分为约 5 秒 generated segments，再组织为 8–16 segment 的短序列、约 1 分钟序列或 12+/24/40 segment 的长程序列。具体方案见 `cameramaster_data_granularity_adjustment.md`。

由于 598 个 L2 直接乘 5 秒会产生约 2,990 秒的生成预算，正式实验前增加一层 `macro_event` 聚合：以 143 条 source scene occurrences 作为初始候选边界，沿 occurrence 聚合，并按超过 30 秒事件间隔或 90 秒总跨度切分。首版试跑得到 161 个候选，按每个候选生成 5 秒计算约 805 秒（13.4 分钟）；这仍需人工边界审核和阈值敏感性分析。原始 L1/L2 保留用于证据追踪。

## 3. 实施优先级

### P0：先冻结评测契约

建立统一 schema、manifest、配置和输出目录。必须固定或记录：

- dataset split/version、story/movie ID、模型和 checkpoint；
- image/video backend、seed、分辨率、采样/帧预算；
- `causal` 或 `full_story` 上下文模式；
- reference strategy、shot/node budget、judge model、temperature、retry；
- prompt 版本、输入哈希、运行日志、失败/fallback/recovery 状态。

建议输出：`run_manifest.json`、`intermediate/`、`prompts/`、`references/`、`metrics/`、`failures.jsonl`。

### P1：MovieBench 数据审计和层级构建

先核验电影目录、有效 JSON、无效文件、镜头数、序列数及 `160movies` 与实际目录差异。对已重构样本，必须同时报告单元集覆盖和跨层引用完整性；当前 American Beauty 的 L1/L2 index 覆盖完整，但 `hierarchy_restructure_report.json` 的整体 validation 为 `incomplete`，因为部分旧 arc 的 L1 shot/scene occurrence 引用无法解析。不得把“L1/L2 文件集 complete”扩写成“全层级引用完整”。

如需从原始数据重建层级：

1. 将 L1 作为镜头级观察证据，将 L2 作为主要事件级生成/评测候选；使用 L3 arc 组织局部叙事上下文，以唯一 L4 global story 组织全片顺序和长期状态。
2. 不把旧 143 条 `source_context/l3_scene_occurrences.json` 当作新 L3 arc，也不把 `source_l4_story_arc_id` 误读为新层级中的 L4。
3. 沿用已处理的 L1-L4 时，保留原有边界；若重新构建任何层级，必须版本化并重新做映射和覆盖审计。

必须分离 `observed_facts`、`inferred_details`、`uncertainties`，不能将单次 LLM 描述直接当作真值。

### P2：双 Track benchmark 标注

当前 `track_a.json` 与 `track_b.json` 各有 16 个样本，沿用旧 arc 样本并保留旧 source IDs；它们没有因 L4 重构而自动成为经过新层级语义审核的样本。正式计算指标前，应建立 `sample_id -> l3_story_arc_id -> source_l2_ids -> source_l1_ids` 映射，检查 `withheld_event_ids`、`source_trace` 与输入 prompt，重做泄漏审查，并对无法解析的映射标记排除/待审，不得猜配。

Track A 标注：

- `essential_events` / KER 候选；
- FDC 细粒度可观察单元；
- anchor 必须保留的语义和顺序；
- source evidence、可观察性、置信度、审核状态。

Track B 标注：

- 1-3 句 short storyline；
- 人物、初始情境、核心冲突、明确结局边界等 hard constraints；
- 允许扩展的维度；
- 有效扩展判据：新颖、可观察、因果相关、非重复、与约束兼容。

Track B 不应把原片后续剧情当成唯一正确答案。

### P3：防泄漏评测器

评测分两阶段：

1. 盲态重建：只给生成视频/图像和该 Track/benchmark 允许提供的 prompt，重建事件、角色、地点、物体、状态变化及顺序。重建输入不得读取隐藏 Track 标签、目标答案或未授权 L4/L3 全文。
2. 对齐判定：将重建结果与 Track A/B 的独立标注对齐，计算指标。

重建阶段禁止访问源故事、隐藏 anchor 真值、参考标签、内部 world state 和人工答案。保存完整 evaluator input/output，做泄漏测试并失败关闭。

## 4. CameraMaster 主指标

### 4.1 Track A

| 指标 | 核心问题 | 建议定义 |
|---|---|---|
| KER/EF | 必要事件是否发生 | 被判定为满足的 essential events / 可评 essential events |
| FDC | 细节是否保留 | 满足的独立可观察细节单元 / 细节单元总数 |
| NSP | 当前事件中的状态是否正确 | 角色、地点、物体状态在事件时点的正确率 |
| LNSC | 远距离状态是否保持 | 跨非相邻事件的状态一致率，按角色/地点/物体分项 |
| Anchor fulfillment | 锚点语义是否遵守 | anchor 必需语义和顺序的满足率 |
| Order/Causal | 事件顺序是否正确 | before/after 与有证据 causal edge 的准确率；没有因果证据时不能强判 causes |

主表至少报告事件、角色、地点、物体和因果/顺序分项，不只报总平均。

### 4.2 Track B

| 指标 | 核心问题 | 建议定义 |
|---|---|---|
| Hard-constraint compatibility | 是否违反故事前提 | 满足的硬约束 / 适用硬约束 |
| Valid expansion count | 增加了多少有效内容 | 通过独立审核的新增事件/细节数 |
| EVR | 扩展是否有效 | 有效新增单元 / 全部新增候选单元 |
| Causal linkage | 新增内容是否推动故事 | 有明确前因后果或目标推进的新增单元比例 |
| Novelty/non-redundancy | 是否只是重复 | 去重后新增内容比例 |
| Visual observability | 是否能拍出来 | 可由画面或声音观察到的新增单元比例 |
| State consistency | 新内容是否保持人物/世界状态 | 新增内容与既有状态兼容比例 |

Track B 必须同时报告“扩展数量”和“质量调整后的扩展比例”，避免用更多但无效的剧情获得虚假收益。

## 5. 外部 benchmark 接入策略

### ConStory-Bench

适用于长篇故事、剧本或 storyboard 文本，不适用于最终视频。被测文本必须是 CameraMaster 实际生成的产物，而不是参考 L4 `movie_bible` 或 L3 `full_story_S` 标签。样本粒度需明确为全片 L4 或单个 L3 arc，不混算。运行 ConStory-Checker 后报告 CED、GRR 和五类错误分项。必须记录 judge 模型、prompt、temperature、原始 CSV 和解析失败。

### DirectorBench

适用于约一分钟视频。生成片段以 L2 内容为主，但须冻结 L2 到 5 秒视频的合并/拆分规则；L1 是证据、L3 是局部上下文、L4 是全片上下文，不可混为视频节点。只有来源时间范围和顺序经核验后，才将多个 L3 arcs 组织成长视频；固定 `UserProfile`、配置和参考/reference-free 模式。保留 script、video、audio、stability、cross-modal 的子指标、confidence、skipped dimension、bottleneck 和 tool trace。不要只报 profile-dependent overall score。

### LoCoT2V-Bench

适用于长时、多场景 text-to-video。全片长序列应以唯一 L4 的全片故事为组织依据，并沿经审计的 L2/L1 时间序列构造；单个 L3 arc 不默认视为长视频。至少接入 Overall Alignment、Fine-Grained Alignment、Character Consistency、Background Consistency、Perceptual Quality、Warping Error、HERD 和 Dynamic Quality。使用 `purified_prompt`，保证 `prompt.json` 与 sample ID 对齐，记录四个 conda 环境和所有 checkpoint。对不满足时长或问题元数据条件的指标标为 `not_applicable`，不能用短片结果冒充长视频 benchmark 结果。

### ViStoryBench

虽然不在本次三个本地仓库中，但它是计划中的外部主要对照。官方 `Shots[]` 必须固定，不得被 CineWorldSim 节点选择器替换、合并、删除或重排；支持 `anchor_only`、`anchor_plus_sim`、`causal`、`full_story` 四种明确模式，并直接调用官方指标。

## 6. 实验矩阵

### 6.1 主 baseline

- B0 Direct prompting
- B1 Storyboard-only
- B2 Visual-memory baseline
- B3 Agentic narrative without state anchoring
- Ours CineWorldSim full

所有方法固定故事、图像/视频后端、分辨率、seed、输出预算和评测协议。未实现或未复现的 baseline 不能写入正式结果表。

### 6.2 核心消融

优先运行：

1. No world-state tracker
2. No narrative node selector
3. No reference gating

随后补充：No local world update、No path planner、Event-keyframe-only、No character refs、No world/item refs、No auxiliary shots、No fallback recovery。

### 6.3 执行阶段

| 阶段 | 范围 | 目的 |
|---|---|---|
| Smoke | 5-10 个样本 | 跑通数据、生成、盲态评测和报告全链路 |
| Pilot | 20-50 部代表性电影/事件 | 检查标注质量、成本、指标区分度 |
| Main | 审核通过的完整候选集 | 得到论文主结果 |
| External | ViStoryBench、ConStory、DirectorBench、LoCoT2V | 外部可比性和补充分析 |

先用 `0001_American_Beauty` 和 `3075_SITTER` 贯通工程流程；它们只是调试样本，不代表数据集分布。对 American Beauty，必须区分一条全片 L4、16 个 L3 局部 arc、598 个 L2 事件、749 个 L1 镜头，以及 143 条来源场景 occurrence。

## 7. 评测代码模块划分

建议按以下边界实现：

```text
schemas/
  evaluation_types.py
  *schema.json
evaluation/
  run_context.py
  blind_reconstruction.py
  metrics/track_a.py
  metrics/track_b.py
  agreement.py
  reporting.py
adapters/
  vistorybench_adapter.py
  constory_adapter.py
  directorbench_adapter.py
  locot2v_adapter.py
scripts/
  audit_moviebench_assets.py
  refine_l2_micro_events.py
  build_l3_scene_segments.py
  build_l4_story_arcs_and_movie_bible.py
  build_track_a_annotations.py
  build_track_b_annotations.py
  build_reference_asset_manifest.py
  check_split_leakage.py
  run_evaluation_matrix.py
```

## 8. 关键注意事项

1. **真值边界**：时间连续不是语义事件；LLM 生成描述不是自动真值；因果只能在有证据时标注。
2. **防泄漏**：按 movie/story arc 切分；盲态重建不接触源故事和隐藏标签；Track A 锚点不能写入全部待恢复细节。
3. **开放式公平性**：Track B 允许多种合理扩展，不使用唯一参考续写做 exact match。
4. **指标隔离**：KER、FDC、NSP、LNSC、EVR 各司其职，避免同一错误重复计数。
5. **外部 benchmark 方向**：CED/GRR、LoCoT2V Warping、Dynamic Degree、Motion Smoothness 的方向和含义不同，报告时显式标注 `higher/lower is better`。
6. **条件分母**：音频缺失、不可观察事件、无适用角色/物体时应记为 N/A 或排除分母，不能默认记 0。
7. **LLM judge 稳定性**：固定模型、prompt、temperature、采样次数、JSON parser 和重试策略；先做人工校准再大规模调用。
8. **成本控制**：先 smoke/pilot，启用缓存、断点续跑、批处理、速率限制和成本日志，不直接处理全部 56,929 个序列。
9. **参考图记录**：记录每张图的来源、选择理由和许可；不能用最终 `image_count` 反推 reference strategy。
10. **失败恢复**：区分 simulation、node selection、path planning、reference selection、rendering 和 evaluator failure，分别统计恢复率、成本和下游一致性。
11. **结果可追溯**：所有论文数字必须可从 raw output、manifest、metric JSON 和版本信息重建。
12. **主张范围**：若状态消融没有显著差异，只能收缩为工程整合或条件性收益，不能宣称全面领先。

## 9. 验收标准

- 两部调试电影能够完成 L1-L4、Track A/B 标注、生成结果登记、盲态重建、指标计算和报告输出。
- 泄漏测试能够证明评测器无法读取源故事和隐藏真值。
- 至少一个 baseline、Full 和三个核心消融在同一配置下可比较。
- 每个指标都有明确分母、方向、N/A 规则和 per-sample 输出。
- 外部 benchmark smoke test 能保留官方输出语义和 checkpoint/environment 信息。
- 人工校准报告事件、状态、顺序和扩展判定的一致性及分歧原因。
- 不在没有主结果、消融和人工复核的情况下宣称方法全面优于 baseline。

## 10. 推荐近期顺序

1. 先实现 schema、manifest、配置和审计脚本，并纳入全片 L4/局部 L3/事件 L2/镜头 L1 的显式 ID 字段。
2. 用两部电影核验现有 L1-L4 处理结果及映射质量；不要默认重跑已经完成的层级构建。American Beauty 当前计数为 L4=1、L3=16、L2=598、L1=749，另有 143 条 source scene occurrences；全层级引用 validation 不完整，需把缺失映射列入数据审计。
3. 实现 evaluation-view builder，先产出 `CM-Event-5s`、`CM-Story-8to16`、`CM-Long-Cycle` 三类派生 manifest，再运行各 benchmark。
3. 实现盲态重建和 Track A/B 指标，完成小规模人工校准。
4. 接入 ViStoryBench 官方评测，再接入三个本地 benchmark。
5. 运行 Direct、Storyboard-only、Full 及三个核心消融。
6. 最后扩展样本、多 seed、完整 split 和成本/失败分析。
