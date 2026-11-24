# CameraMaster 数据粒度与评测视图调整结论

## 1. 结论

不要直接删除或重写当前 L1-L3 数据。当前重构数据应继续作为可追溯的 source hierarchy：

```text
L4 global movie story: 1 / movie
L3 local story arcs: 16 / American Beauty
L2 events: 598
L1 shot evidence units: 749
Macro-event candidates: 143 source scene occurrences -> audited macro-events
```

需要新增的是 **evaluation view**，而不是再定义一个与 L1-L3 冲突的新真值层级：

```text
L4 global story
  -> Evaluation sequence/window
      -> Macro-event（连续事件聚合单元）
          -> Generated segment（约 5 s，可含一个或多个 macro-event phase）
              -> L2 event bundle
                  -> L1 evidence references
```

核心原因是：L1-L3 是数据和证据层级，ViMax、A²RD、DirectorBench、LoCoT2V 使用的是不同的实验单元。一个统一的“新 L1/L2/L3”无法同时满足所有 benchmark。

## 2. 现有数据是否真的过细

对 American Beauty 的已处理数据统计：

| 层级 | 数量 | 作用 | 是否适合直接作为生成样本 |
|---|---:|---|---|
| L4 | 1 | 全片 global story、人物/地点/物体全局约束 | 否，作为上下文 |
| L3 | 16 | 局部故事弧和状态组织 | 否，作为序列上下文 |
| L2 | 598 | 事件、动作、因果、状态变化 | 是，作为内容来源 |
| L1 | 749 | shot-level 可观察事实和时间证据 | 否，作为证据/参考 |

L2 事件持续时间约为：最小 0.081 s，中位数 2.963 s，均值 4.507 s，最大 41.114 s。L1 镜头持续时间中位数约 2.203 s，均值约 3.548 s。因而 L2 的统计粒度已经接近 5 s 生成预算；不应因为总数 598 就把 L2 全部粗暴合并。

真正需要控制的是：

1. 全片数据不应每次实验全部生成；
2. 评测样本应按 benchmark 的目标长度切成有序窗口；
3. L1/L2 的细节应作为 segment metadata，而不是全部变成独立样本；
4. L3 arc 不能默认当作一个视频 segment，也不能因同属 L4 就直接拼接。
5. 598 个 L2 应按连续性聚合成较少的 macro-events，再从 macro-events 组织 benchmark sequence。

### 推荐聚合规模

`source_context/l3_scene_occurrences.json` 当前有 143 条来源场景 occurrence。它们不是新 L3 arc，但可以作为 macro-event 聚合的初始边界：每条平均约包含 4.2 个 L2 事件。首版脚本沿 occurrence 聚合，并在 occurrence 内部按超过 30 秒的事件间隔或超过 90 秒的总跨度切分。对当前 American Beauty 得到 **161 个 macro-event 候选**；这不是人工确认后的最终真值。

这意味着：

- 598 个 L2 -> 当前试跑 161 个 macro-events，平均每个包含约 3.7 个 L2；
- 每个 macro-event 通常对应一个连续动作或局部场景变化；
- 若每个 macro-event 固定生成 5 秒，总生成预算约为 13.4 分钟，而不是 50 分钟；
- 仍保留足够数量，能够组织 8–16 segment 的 ViMax/DirectorBench 序列和 24/40 segment 的长视频序列。

该首版并不因人物/地点字符串差异或 `state_changes` 字段自动切分，因为这些字段存在异构标注和合法状态演变；这些差异保留在 macro-event 中供人工审核。30 秒 gap 和 90 秒 max span 是可配置的候选规则，需在 pilot 中对 10/20/30 秒 gap 与 30/60/90 秒 max span 做敏感性比较。若一个宏事件含过多独立动作阶段，应人工拆分；若两个相邻候选实际属于同一动作链，也可审核后合并。

## 3. 建议的数据模型

### 3.1 Source hierarchy：保留不动

每个 source record 保留：

- `l4_global_id`、`l3_story_arc_id`、`l2_event_id`、`l1_shot_id`；
- 原始 time range；
- `observed_facts`、`inferred_details`、`uncertainties`；
- evidence trace、quality、mapping status。

L1 不再作为主评测样本，而是作为 evidence index。L3 不再被解释成全片 story 或固定时长 video node。

### 3.2 Macro-event：新增聚合层

每个 macro-event 应满足：

1. 时间上连续，或只有可解释的短间隔；
2. 主要人物集合基本稳定，或人物变化本身是明确转折；
3. 地点/空间布局保持，或存在明确转场边界；
4. 共享一个局部目标、动作阶段或因果链；
5. 不把两个独立冲突、独立地点或不同时间段强行合并。

建议字段：

```json
{
  "macro_event_id": "movie_ME_001",
  "source_l2_ids": [],
  "source_l1_ids": [],
  "source_scene_occurrence_ids": [],
  "l3_story_arc_id": "...",
  "time_range": {},
  "characters": [],
  "locations": [],
  "dominant_goal": "...",
  "actions": [],
  "state_before": {},
  "state_after": {},
  "boundary_reason": "shared_goal_and_continuous_space",
  "quality": {"review_status": "pending"}
}
```

### 3.3 Evaluation sequence/window：新增

一个 evaluation sequence 是从唯一 L4 的全片时间线上截取的、连续且有证据的窗口。它必须记录：

```json
{
  "eval_sequence_id": "movie_seq_001",
  "l4_global_id": "movie_global_story",
  "source_l3_arc_ids": ["arc_01", "arc_02"],
  "ordered_macro_event_ids": ["movie_ME_001", "movie_ME_002"],
  "ordered_l2_event_ids": ["event_01", "event_02"],
  "ordered_l1_shot_ids": [],
  "generation_segment_ids": [],
  "time_range": {},
  "continuity_check": {
    "chronological": true,
    "non_overlapping": true,
    "mapping_complete": true
  },
  "benchmark_targets": {
    "vimax": "8-16 shots",
    "a2rd_short": "8 segments/about 1 min",
    "directorbench": "about 1 min",
    "locot2v": "long multi-scene sequence"
  }
}
```

不满足时间顺序、连续性或映射完整性的 arc 只能作为局部诊断，不得跨 arc 静默拼接。

### 3.3 Generated segment：实验生成单元

生成片段不是新的真值层级，而是实验协议中的输出单元。建议默认：

- 以一个 L2 event 作为主要语义来源；
- 若 L2 过短且同一动作连续，可合并相邻 L2；
- 若 L2 过长或包含明显阶段变化，按有证据的 L1 shot/action boundary 拆分；
- 每个生成片段目标约 5 s，但保存实际时长和 `source_l2_ids/source_l1_ids`；
- 一个 segment 可以对应多个 L2，也可以对应一个 L2 的一个阶段。

推荐字段：`segment_id`、`source_l2_ids`、`source_l1_ids`、`source_l3_arc_id`、`prompt`、`duration_sec`、`entity_state_before/after`、`video_path`。

## 4. 各 benchmark 的推荐粒度

### Track A/B

- Track A：以 evaluation sequence 为样本，segment 级输出后在 sequence 级计算 KER/FDC/NSP/LNSC/Order。
- Track B：以 L3 局部 arc 或更长 evaluation sequence 为输入上下文；新增内容在 segment/event bundle 级审核。
- L1 只用于核查可观察性和证据，不作为独立主表样本。

### ViMax protocol reproduction

ViMax 原生故事是 2–4 scenes、8–16 shots。因此使用 8–16 个有序 generated segments 作为一个 ViMax-style sequence；L3 arc 只提供 scene grouping，不能直接等于 shot。

### A²RD protocol reproduction

A²RD 的 segment 是独立生成 clip，可跨一个或多个 shot/scene。使用全片 L4 chronology 下的 generated segments；不要按 L3 数量判断跨度。cyclic consistency 需要目标实体再次出现前至少 10 个 segment 不出现，即最低约 12 个 5 s segments。

### DirectorBench

使用约 1 分钟的 evaluation sequence，通常由约 8–12 个 5 s segments 组成。短于该范围的结果仅作为 smoke/diagnostic，不能与标准一分钟结果混合。

### LoCoT2V-Bench

需要更长的多场景 sequence；其 prompt metadata 以全片 L4/L3 上下文、L2 event 和 L1 evidence 共同生成。若实际序列太短，只运行适用的指标并标记 `not_applicable`。

### ConStory-Bench

输入是生成文本，不是视频 segment。可按 L4 full story 或 L3 local arc 分别构造文本样本，但不能把 source `movie_bible` 或参考 `full_story_S` 当成生成结果。

## 5. 推荐的实验数据集版本

不要只有一个“全量数据集”；应发布三个派生 split：

### `CM-Event-5s`

- 以 L2 event bundle 为核心；
- 目标 5 s segment；
- 用于 Track A/B 的事件、状态和动作评测；
- 保留每个 segment 的 L1 evidence。

### `CM-Story-8to16`

- 每个 sequence 选择 8–16 个有序 generated segments；
- 用于 ViMax-style consistency 和约一分钟 DirectorBench；
- sequence 必须有明确 scene/L3 grouping。

### `CM-Long-Cycle`

- 沿唯一 L4 chronology 选择 12 个以上、最好 24/40 个 generated segments；
- 只保留有角色/物体/环境消失—重现证据的序列；
- 用于 A²RD cyclic protocol reproduction 和长视频指标。

这三个 split 是同一 source hierarchy 的不同评测视图，不是三套互相独立的真值。

## 6. 不建议的做法

1. 删除 L1/L2，只保留 L3；会丢失 Track A 的细节、状态和可观察性证据。
2. 把 598 个 L2 全部直接当成 598 个必须生成的视频；成本和长视频拼接会失控。
3. 把每个 L3 arc 压成一个 5 s 视频；会破坏 arc 内部事件结构。
4. 因为多个 arc 共享一个 L4 就直接拼接；必须先验证时间范围、顺序、互补性和映射完整性。
5. 用 L1 数量或 L3 数量代替实际 generated segment 数量；A²RD、ViMax 和 DirectorBench 的计数单位都是实验输出片段/shot。

## 7. 最终执行建议

第一阶段不要重做 L1-L3 标注。首版聚合脚本为 `build_macro_events.py`：

1. 读取 L4 movie bible、L3 arcs、L2 event units、L1 shot units；
2. 按 L4 chronology 建立事件图；
3. 以 source scene occurrence 为候选边界，按可配置时间 gap/span 合并/拆分 L2，形成 macro-event candidates；
4. 生成 8–16 segment、约 1 min 和 12+/24/40 segment 的 evaluation sequences；
5. 执行 mapping、时间顺序、重叠、实体 recurrence 和泄漏校验；
6. 输出 benchmark-specific manifests；
7. 再运行 Track A/B、三个外部 benchmark 和两个 protocol reproduction。

最终判断：**598 个 L2 对逐项 5 秒生成而言确实过细。保留 source hierarchy，新增 161 个初版 macro-event candidates，并按 benchmark 构建 segment/sequence 视图，是比重写 L1-L3 更稳妥、也更符合 ViMax、A²RD、DirectorBench 和 LoCoT2V 评测方式的方案。macro-event 在完成边界审核前属于实验候选，不应当作新的 ground truth。**
