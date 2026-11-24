# CineWorldSim 当前数据处理与输出说明：American Beauty 样本

## 1. 文档范围

本文依据目录 `outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4/` 中实际生成的文件和 JSON schema 编写。该目录是原始数据的旁路重构输出，不覆盖原始 MovieBench 数据。

本文同时以当前实现脚本 `scripts/restructure_cineworldsim_hierarchy.py`、层级定义 `docs/CineWorldSim_Hierarchy_Redefinition.md` 和处理评估规划 `docs/moviebench_cineworldsim_processing_and_evaluation_plan.md` 为准，说明该样本当前的数据加工范围。更早的 `docs/CineWorldSim_Bench_Data_Processing_Specification.md` 是规划文档，其中部分阶段脚本/字段属于计划设计，且其旧版 L4 定义与当前层级约定不一致；实际解释和使用数据时，应以当前脚本、当前产物和层级重定义文档为准。

本次目录统计：1 个 L4 全局故事、16 个 L3 局部故事弧、598 个 L2 事件、749 个 L1 镜头证据单元，以及 143 条来源 L3 场景 occurrence。`l1_shots/index.json` 和 `l2_events/index.json` 的覆盖检查均显示 `complete: true`。重构报告的整体 `validation.status` 则为 `incomplete`，原因是原 story arc 的部分 L1 shot / scene occurrence 引用无法映射；详见第 8 节。

## 1.1 当前层级定义

```text
L4：单部电影唯一的全局总故事线与全片设定
└── L3：全片总故事线中的局部故事弧
    └── L2：事件级叙事单元
        └── L1：镜头级视觉证据
```

L4 描述电影总体剧情、主要冲突、人物发展、反复出现的地点/物体和受证据支持的全局约束。L3 是局部 arc，不是另一个电影级总故事。L2 是事件记录，不保证每个事件天然适合固定 5 秒的视频生成；具体生成单元仍需按时长、可视动作和叙事完整性筛选。L1 是最底层的镜头观察和视觉证据。

## 1.2 当前处理链路与实现边界

当前样本的重构脚本**不是**从 MovieBench 原始标注开始重建全部 L1-L4 的端到端脚本。它读取已经处理好的样本目录：

```text
outputs/moviebench_cineworldsim/0001_American_Beauty/
  l4_story/movie_bible.json
  l4_story/story_arcs.json
  l3_scenes/final.json
  l2_events/units/
  l1_shots/units/
  track_a.json                 # 若存在
  track_b.json                 # 若存在
```

当前脚本执行两部分工作：

1. **层级重组**：将现有 Movie Bible 作为唯一 L4 文件；将旧 `story_arcs.json` 中的 16 个 arc 转为 L3，并生成稳定的 `l3_story_arc_id`；保留旧 arc ID 和来源 scene occurrence ID；原样复制 L1、L2、Track A/B；把 143 条旧 L3 scene occurrence 放入 `source_context/`。
2. **可选 L4 再生成**：只有传入 `--generate-l4` 才调用 OpenAI-compatible LLM。模型以源 Movie Bible 为先验，并结合压缩的 L3 场景与 L2 事件证据生成新的全片 L4；不会将所有原始 L1 JSON 直接拼进 prompt，也不会在该步骤重新生成 L1、L2、L3 或 Track A/B。

现行脚本与本次样本的处理状态：

| 处理环节 | 当前状态 | 说明 |
|---|---|---|
| 原始 MovieBench 清洗、镜头标注及连续片段处理 | 上游已有输入，本脚本不执行 | 本文不能据此认定 MovieBench 全集原始处理已完成。 |
| L1 镜头证据 | 已存在并复制 | 当前样本 749 条；本脚本不重写其正文。 |
| L2 事件 | 已存在并复制 | 当前样本 598 条；本脚本不重新分段/生成事件。 |
| 来源 L3 场景 occurrence | 已存在并保留 | 当前样本 143 条，作为 L4 输入证据和 `source_context` 保存。 |
| L3 局部故事弧 | 由旧 arc 重标/映射 | 当前样本 16 条；脚本不从 143 个 scene occurrence 重新推导 arc。 |
| L4 全局总故事 | 本次选择调用 LLM 生成 | 写入 `l4_global/movie_bible.json`，并写请求/响应审计；具体状态以报告为准。 |
| Track A/B | 原样复制 | 当前各 16 个样本；不是在此次重构中重新标注或重新验证语义。 |
| 视觉资产、关键帧、视频生成与正式评测 | 不由本脚本处理 | 需由相应资产/视频/评测流程另行执行。 |

### L4 生成的数据流

```text
源 Movie Bible（全局先验）
  + 压缩后的 143 条 L3 场景
  + 压缩后的 598 条 L2 事件及来源 L1 shot IDs
  -> OPENAI scope 下配置的模型
  -> 新 L4 movie_bible.json
  -> l4_generation_request.json / l4_generation_response.json 审计
```

压缩后的 L3 证据保留场景 ID、有限 L2 ID 样本及事件数量、时间、地点、人物、场景目的/叙事、事件推进和状态转变。压缩后的 L2 证据保留事件 ID、来源 shot IDs、标题、描述、动作和状态变化。请求文件包含 prompt 本文和 hash，不包含 API key。

### 可复现实验命令

以下是该样本对应的 LLM-L4 重构命令示例。输出路径必须不存在；脚本拒绝覆盖已有目录。

```bash
python -u scripts/restructure_cineworldsim_hierarchy.py \
  --input-root outputs/moviebench_cineworldsim/0001_American_Beauty \
  --output-root outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4 \
  --allow-unresolved-references \
  --generate-l4 \
  --llm-scope OPENAI \
  --llm-retries 3 \
  --llm-timeout 180
```

`OPENAI` scope 从项目 `.env`/配置读取 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。可用 `--llm-model` 显式覆盖模型名。默认 LLM 调用或 JSON 校验失败时脚本终止；只有显式加 `--fallback-existing-bible` 才会回退至源 Movie Bible，并在输出和报告中标注 `source_fallback`。`--allow-unresolved-references` 只允许保留无法解析的旧引用并记录缺口，不会让引用自动修复，也不改变 API 失败策略。

### 与较早数据处理规范的关系

`CineWorldSim_Bench_Data_Processing_Specification.md` 仍可作为更广泛的数据集构建目标参考，但不是当前重构脚本的执行说明。该文档提出的清洗、完整 L2/L3 重建、候选筛选、视觉参考资产、Benchmark 标注和评测器等步骤，不能仅凭本次输出目录的存在就视为已完成。它将 L4 描述为多场景故事弧/全片描述，与当前明确的层级定义有冲突；当前采用唯一 L4 全片故事、L3 局部 arc 的定义。若沿用旧规范中的脚本名称、字段或阶段状态，应先与实际仓库代码及输出核对。

## 2. 目录结构和职责

```text
0001_American_Beauty_llm_l4/
  hierarchy_restructure_report.json
  track_a.json
  track_b.json
  l1_shots/
    index.json
    units/<shot-id>.json                 # 749 个 L1 单元
  l2_events/
    index.json
    units/<sequence-id>.json             # 400 个序列文件，含 598 个事件
  l3_arcs/
    story_arcs.json                      # 16 个 L3 局部故事弧
  l4_global/
    movie_bible.json                     # 正式 L4 文件
    l4_generation_request.json           # 发送给 LLM 的请求审计
    l4_generation_response.json          # LLM 原始结构化响应审计
  source_context/
    l3_scene_occurrences.json            # 143 条来源场景 occurrence
```

### 各目录职责

| 路径 | 职责 |
|---|---|
| `l1_shots/` | 镜头级视觉证据，是细粒度事实和时间信息的来源。 |
| `l2_events/` | 由镜头组织出的事件单元，保留事件动作、因果、状态变化及镜头来源。 |
| `l3_arcs/` | 将原有局部故事弧重标为 L3 arc；L3 是局部叙事组织层，不是全片总故事。 |
| `l4_global/` | 全片唯一 L4：电影级 synopsis、global storyline、人物/地点/物体及证据追踪。另存生成请求和响应以便审计。 |
| `source_context/` | 保存用于理解/生成 L4 的旧 L3 场景 occurrence 记录，作为来源上下文，不等同于新 L3 arc。 |
| `track_a.json` / `track_b.json` | 保留原有两类评测样本及其来源信息，未自动重写成新层级语义。 |
| `hierarchy_restructure_report.json` | 汇总层级、数量、映射、引用校验和 L4 生成状态。 |

## 3. 通用 JSON 包装结构

L1、L2 序列单元、L3 arc、L4 `movie_bible.json` 和来源场景文件采用处理单元包装结构，常见字段如下：

| 字段 | 含义 |
|---|---|
| `unit_id` | 文件/处理单元标识。 |
| `stage` | 数据阶段标识，例如 `l1`、`l2`、`l3_story_arcs` 或 `l4_movie_bible`。 |
| `status` | 单元处理状态，通常为 `done`。 |
| `result` | 数据正文。各阶段字段见后文。 |
| `meta` | 处理元数据；常见内容包括状态、模型、尝试次数和 prompt 哈希。具体文件可能有所不同。 |

注意：`l4_generation_response.json` 是审计响应格式，采用 `status`、`model`、`result` 外层，而非上述完整单元包装。`track_a.json` 和 `track_b.json` 则采用 Track 数据集包装格式。

## 4. L4 全局故事

### 正式文件

`l4_global/movie_bible.json` 是下游读取 L4 的正式文件。实际 L4 内容位于 `result` 内；同目录的 `l4_generation_response.json` 保存 API 响应审计。两者的 `result` 字段应对应。

### `result` 字段

| 字段 | 含义 |
|---|---|
| `movie_id` | 电影样本 ID。 |
| `movie_synopsis` | 结构化的全片梗概，包括短梗概、按顺序的剧情梗概、主要冲突和重大状态转变。 |
| `movie_synopsis.short_synopsis` | 全片的简短概括。 |
| `movie_synopsis.chronological_synopsis` | 按剧情顺序排列的阶段/片段摘要；每项包含 `status`、`story`、`source_l3_ids`。 |
| `movie_synopsis.main_conflicts` | 全片主要冲突；每项包含 `conflict`、`status`、`source_l3_ids`。 |
| `movie_synopsis.major_state_transitions` | 重要人物、关系或局势变化；每项包含 `transition`、`status`、`source_l3_ids`。 |
| `global_storyline` | 整部电影的一条连贯总故事线。当前实际值是对象，包含 `status`、`story` 和 `source_l3_ids`。它不是任何单个 L3 局部 arc。 |
| `global_character_profiles` | 全片人物档案列表。每项含 `character`、`observed_profile`（证据直接支持的描述）、`inferred_arc`（人物发展推断）、`uncertainties`（不确定点）和 `source_l3_ids`。 |
| `recurring_locations` | 重复地点列表。每项含 `location`、`narrative_function`（地点在全片叙事中的作用）和 `source_l3_ids`。 |
| `recurring_objects` | 重复或重要物体列表。每项含 `object`、`observed_use`（证据中观察到的用途）、`constraint`（与物体相关的叙事/证据约束）和 `source_l3_ids`。 |
| `global_constraints_supported_by_evidence` | 有证据支持的全局约束列表。每项含 `constraint`、`evidence` 和 `source_l3_ids`。这些约束用于限定哪些全局叙事判断受到数据支持。 |
| `unresolved_questions` | 现有材料无法确定的问题列表。每项含 `question`、`status` 和 `source_l3_ids`。 |
| `evidence_trace` | L4 全局层面的来源索引，含 `source_l3_ids`、`source_l2_ids`、`source_l1_ids` 三个 ID 列表。它提供溯源线索，不等于人工事实审核结论。 |
| `hierarchy_level` | 层级标签；当前为 `L4_global_movie_story`。 |
| `generation_method` | L4 生成方式；当前为 `llm_compaction`。 |
| `generation_metadata` | 生成记录：`scope`、`model`、`generated_at`、来源 Movie Bible/L3/L2 路径，以及参与生成的 L3/L2 记录数量。 |

### L4 外层 `meta`

`movie_bible.json.meta` 保存处理状态、模型、尝试次数和 prompt 哈希。密钥不会写入该文件。`generation_metadata.model` 与外层 `meta.model` 都记录模型信息，但前者属于 L4 正文的生成来源记录，后者属于处理单元元数据。

## 5. L3 局部故事弧

文件：`l3_arcs/story_arcs.json`。正文位于 `result.story_arcs`，当前有 16 条。

每条 arc 的主要字段：

| 字段 | 含义 |
|---|---|
| `l3_story_arc_id` | 重构后的 L3 arc ID。 |
| `source_l4_story_arc_id` | 来源数据中的旧 arc ID；保留其来源命名，不表示它仍属于新定义的 L4。 |
| `arc_title` | 局部故事弧标题。 |
| `full_story_S` / `short_storyline_L` | Track A/B 使用的长故事与短故事文本。 |
| `time_range` | arc 时间范围，包含 timecode、毫秒和秒数表示。 |
| `characters`、`locations`、`key_objects` | arc 涉及的人物、地点、关键物体。 |
| `l2_event_ids`、`l1_shot_ids` | arc 关联的事件和镜头 ID。 |
| `arc_structure` | 局部弧线结构：`initial_state`、`trigger_or_inciting_change`、`development`、`turning_points`、`current_or_final_state`、`unresolved_elements`。 |
| `essential_events` | 核心事件；每项关联 `source_l2_ids` 和 `source_scene_occurrence_ids`。 |
| `fine_grained_units` | 更细的叙事单元；每项含 `unit`、`source_l3_id`、`source_l2_ids`。 |
| `cross_scene_state_transitions` | 跨场景状态变化；每项含 `transition`、`status` 和 `source_l3_ids`。 |
| `evidence_trace` | arc 级 L3/L2/L1 来源 ID。 |
| `quality` | arc 质量/检查信息，包括连贯性、证据覆盖、unsupported claims 和 leakage check。 |
| `source_scene_occurrence_ids` | 指向 `source_context/l3_scene_occurrences.json` 的旧场景 occurrence ID。 |

## 6. L2 事件和 L1 镜头

### L2

`l2_events/index.json` 汇总事件覆盖；`l2_events/units/<sequence-id>.json` 按来源序列存储记录。当前有 400 个序列文件、598 个事件。每个序列文件的 `result` 包含：

| 字段 | 含义 |
|---|---|
| `movie_id` | 电影 ID。 |
| `source_sequence_id` | 输入镜头序列 ID。 |
| `segmentation_decision` | 分段决策摘要：输入镜头数、输出事件数、是否拆分、拆分原因和置信度。 |
| `l2_events` | 该序列内的事件数组。 |
| `coverage_check` | 输入 shot 分配完整性：是否每个输入 shot 恰好分配一次，以及缺失/重复 ID。 |
| `id_normalization` | 事件 ID 规范化方法及结果。 |

每个 `l2_events[]` 事件的主要字段：

| 字段 | 含义 |
|---|---|
| `l2_event_id` | 事件唯一 ID。 |
| `source_shot_ids`、`shot_order` | 来源镜头及其在序列中的顺序。 |
| `time_range` | 事件起止 timecode、毫秒和持续秒数。 |
| `event_title`、`event_description` | 事件标题和描述。 |
| `main_characters`、`setting`、`locations`、`items` | 参与人物及场景/地点/物体信息。 |
| `actions`、`interactions` | 动作和人物/对象交互。 |
| `temporal_progression` | 事件内部的时间推进。 |
| `causal_preconditions`、`causal_effects` | 事件发生前提及其造成的结果。 |
| `state_changes` | 事件前后状态变化。 |
| `camera_and_visual_style` | 镜头和视觉风格描述。 |
| `shot_to_event_mapping` | 镜头到事件的映射；每项含 `shot_id`、`role`。 |
| `evidence_trace` | 来源 L1 ID 和来源字段。 |
| `quality` | 边界置信度、覆盖检查及 unsupported claims。 |

### L1

`l1_shots/index.json` 汇总镜头单元覆盖；`l1_shots/units/<shot-id>.json` 每个文件对应一条镜头标注。当前有 749 个 shot 单元。其 `result` 主要包含：

| 字段 | 含义 |
|---|---|
| `l1_shot_id`、`movie_id`、`source_file` | 镜头、电影和来源文件标识。 |
| `time_range`、`order_index` | 镜头时间范围和顺序。 |
| `characters`、`location_mentions`、`item_mentions` | 镜头中涉及的人物、地点和物体。 |
| `actions`、`interactions` | 可观察动作和交互。 |
| `background_state` | 镜头背景/环境状态。 |
| `camera`、`style_elements` | 摄影机信息和视觉风格元素。 |
| `observable_facts`、`interpretive_facts` | 直接可观察事实与解释性判断分开保存。 |
| `source_evidence` | 支持标注的来源证据。 |
| `annotation_quality` | 标注质量/置信度等信息。 |

### Index 文件

- `l1_shots/index.json`：`stage`、`unit_count`、`done_count`、`coverage`。当前 749/749 完成，缺失、重复和意外 ID 列表为空。
- `l2_events/index.json`：`stage`、`sequence_count`、`event_count`、`coverage`。当前 400 个序列、598 个事件，覆盖检查完整。

## 7. 来源场景、Track 和 L4 审计

### 来源 L3 场景 occurrence

`source_context/l3_scene_occurrences.json` 保留生成 L4 时使用的 143 条场景记录。`result` 包含 `movie_id`、`l3_scenes`、`coverage_check` 和 `consolidation`。场景项含：

- 身份和关联：`l3_scene_id`、`l2_event_ids`。
- 时间与语境：`time_range`、`scene_setting`、`active_characters`、`scene_purpose`、`scene_narrative`。
- 叙事变化：`event_progression`、`narrative_state_transition`、`causal_links`、`location_continuity`。
- 溯源与质量：`evidence_trace`、`quality`。

`coverage_check` 汇总完整性、缺失/重复/意外 ID；`consolidation` 记录场景整合方式及使用的事件窗口。该文件是旧场景 occurrence 来源数据，不是 L3 arc 清单。

### Track A / Track B

两个文件的共同外层字段为 `schema_version`、`track`、`movie_id`、`samples`，各有 16 条样本。

`track_a.json` 样本字段：

- `sample_id`、`movie_id`、`full_story_S`：样本身份和完整故事文本。
- `storyboard_anchors_A`：故事板锚点；每项含 `shot_id`、`surface_action`、`camera`（运动、景别/视角、确定性）和 `source_l1_id`。
- `essential_events_E_source`：核心事件及其 L3/L2 来源 ID。
- `fine_grained_units_FDC`：细粒度单元及其 L3/L2 来源。
- `source_trace`：样本的 L3/L2/L1 来源集合。

`track_b.json` 样本字段：

- `sample_id`、`movie_id`、`short_storyline_L`：样本身份和短故事线。
- `known_characters`、`known_facts`：生成/评测前可用的人物与事实。
- `hard_constraints`：硬约束，分人物、前提/场景、核心冲突和结果边界。
- `withheld_event_ids`：刻意保留/不提供给输入的事件 ID。
- `short_storyline_source_event_ids`：短故事线来源事件。
- `reference_story`：参考故事文本。
- `allowed_expansion_space`：允许扩展的内容范围。
- `leakage_check`：泄漏检查及泄漏事件 ID。

Track 文件由旧 arc 样本迁移/保留而来。存在新 ID 映射不代表样本语义已全部按新层级重新设计；用于新评测前仍需核查输入与目标之间的信息泄漏。

### L4 请求/响应审计

`l4_generation_request.json` 保存 `scope`、`model`、来源路径、输入计数、prompt 哈希及实际 `system_prompt` / `user_prompt`。文件不应包含 API key。

`l4_generation_response.json` 保存响应状态、模型和 LLM 返回的 `result`。其 `result` 与正式 `movie_bible.json.result` 对应，便于追踪模型原始结构化输出。

## 8. 重构报告与重要校验说明

`hierarchy_restructure_report.json` 的主要字段：

| 字段 | 含义 |
|---|---|
| `movie_id` | 样本 ID。 |
| `source_root`、`output_root` | 原始输入和旁路输出路径。 |
| `hierarchy` | L4/L3/L2/L1 的层级定义。 |
| `counts` | 各层数量。当前是 L4=1、L3=16、L2=598、L1=749。 |
| `track_sample_arc_mapping` | 每个来源 arc ID 到重构 L3 arc ID 的映射。 |
| `validation` | 引用解析校验状态和无法解析的 ID 列表。 |
| `copied_unchanged` | 原样复制而未改写的路径。 |
| `source_scene_occurrences` | 来源场景文件位置。 |
| `l4_generation` | L4 生成方法、状态、是否 fallback、scope/model 和输入场景/事件数量。 |

本次 `l4_generation` 记录 `method=llm_compaction`、`status=done`、`fallback_used=false`，输入为 143 个 L3 场景和 598 个 L2 事件。当前 L4 `evidence_trace` 实际包含 37 个 L3 ID、25 个 L2 ID、0 个 L1 ID；尽管部分 L2 事件带有 L1 来源，L4 总级追踪并未列出 L1 IDs，因此不能据此声称每条 L4 主张都可直接追溯到镜头级。

`validation.status=incomplete`，`unresolved_l2_event_ids` 为空，但有 31 个无法解析的旧 `l1_shot_ids` 和 25 个无法解析的 `scene_occurrence_ids`。所以应分别理解：L1/L2 文件集自身的 index 覆盖完整；旧 arc 中的一部分来源引用仍未解析。不能把整个重构数据称作引用完全完整。脚本对 LLM 回复的自动校验也只检查 JSON 对象、movie ID、非空 synopsis/storyline 和 evidence trace 字段类型，不等同于人工审核或完整事实校验。

## 9. 建议读取顺序

1. 先看 `hierarchy_restructure_report.json`，确认版本、计数、生成状态和校验缺口。
2. 读取 `l4_global/movie_bible.json.result` 获取全片 L4 总体故事与人物/地点/物体全局信息。
3. 读取 `l3_arcs/story_arcs.json.result.story_arcs` 获取局部弧线。
4. 根据 L3/L4 中的来源 ID 查 L2 事件，再沿 `source_shot_ids` 查 L1 镜头证据。
5. 需要核查生成过程时，查看 `l4_generation_request.json` 和 `l4_generation_response.json`。
6. 评测 Track A/B 时，额外审查 `withheld_event_ids`、`source_trace` 和 `leakage_check`。
