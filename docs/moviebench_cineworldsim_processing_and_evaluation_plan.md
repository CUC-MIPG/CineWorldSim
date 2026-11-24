# MovieBench 接入新版 CineWorldSim 的处理与评估方案

本文基于 [CineWorldSim 研究方案总纲](CineWorldSim_idea.md) 和当前 MovieBench 目录、脚本及已有结果制定。目标不是把 MovieBench 直接改造成 ViStoryBench，而是将其适配为可以验证新版 CineWorldSim 两条路径的真实电影叙事数据：

1. Track A：固定故事/固定镜头锚点下的叙事保真。
2. Track B：短故事线约束下的开放式叙事拓展。

当前结论严格区分“已经存在的处理结果”和“正式实验前仍需补充的处理”。

## 1. 当前 MovieBench 能直接提供什么

当前数据根目录：

```text
/data1/AgentG/hhb/CameraMaster/MovieBench
```

现有镜头标注主要包含：

- `Characters`
- `Style Elements`
- `Plot`
- `Background Description`
- `Camera Motion`
- 文件名中的起止时间码

已有连续片段索引记录：

- 166 个本地电影目录；
- 56,929 个连续序列；
- 107,667 个镜头；
- 连续判定阈值为相邻时间差绝对值小于 1000 ms；
- 3 个无效 JSON 被跳过。

已有实际 LLM 事件描述仅覆盖 `0001_American_Beauty` 的 5 个连续序列，两组 GPT 输出各记录 `written=5, failed=0`。这只能作为流程样例，不能作为全量事件数据。

## 2. 直接使用前必须解决的限制

### 2.1 当前数据缺少 CineWorldSim 的完整状态结构

MovieBench 镜头标注是逐镜头描述，不等于 CineWorldSim 所需的：

```text
人物目标、人物关系、事件状态、地点状态、物体状态、因果关系、约束、历史信息
```

因此需要新增叙事状态中间层，不能直接把 `Plot` 拼接成 `world_summary`。

### 2.2 当前连续分组不等于语义事件

1000 ms 时间连续性只能作为候选生成规则。一个连续时间段可能跨越：

- 场景切换；
- 人物切换；
- 空间切换；
- 象征性插入镜头；
- 多条并行动作线。

因此要将时间连续序列作为候选单元，再通过规则和 LLM 进行语义事件边界复核。

### 2.3 当前本地未确认存在源视频和可用视觉资产

当前已确认的是 JSON 标注和由标注生成的中间 JSON。新版 pipeline 需要：

- 角色初始化图；
- location 远/中/近/特写图；
- item 初始化图；
- 事件关键帧；
- 生成视频对应的源视频或参考视频，用于评估。

如果源视频文件不在当前工作区，需要先建立源视频路径映射，不能把文本标注当作视觉参考图。

### 2.4 必须防止评测信息泄漏

如果使用源电影视频帧生成角色/location/item 资产，同时又用同一源视频作为评测参照，必须明确这是：

- 参考资产辅助生成实验；或
- 无源视觉资产实验。

不能将使用源视频帧提取的参考图结果与纯文本输入结果直接混为同一主表。

## 3. 必须新增的数据处理层

建议新增统一适配目录：

```text
MovieBench/CineWorldSim_Adapter/
├── manifest.json
├── movies/
│   └── <movie_id>/
│       ├── movie_manifest.json
│       ├── source_shots.json
│       ├── semantic_events.json
│       ├── narrative_state_sequence.json
│       ├── track_a.json
│       ├── track_b.json
│       ├── assets_manifest.json
│       └── annotations/
└── splits/
    ├── train.json
    ├── validation.json
    └── test.json
```

### 3.1 建立数据清单和可复现 split

需要记录：

- 电影 ID、电影名称、来源目录；
- 镜头数量、连续序列数量、候选事件数量；
- 是否存在源视频；
- 是否存在可用角色/地点/物体资产；
- 处理脚本版本和参数；
- API/模型版本；
- train/validation/test 划分。

划分必须按电影进行，不能把同一电影的不同片段拆到训练和测试两边，否则人物、场景、风格和故事上下文会泄漏。

建议首先建立三种可复现实验子集：

| 子集 | 用途 |
|---|---|
| `smoke` | 1 部电影、少量序列，验证全链路 |
| `pilot` | 多部电影、每部固定数量事件，调试标注和评估 |
| `full` | 全部满足资产和标注条件的测试电影 |

### 3.2 镜头元数据标准化

保留现有 `clip_meta`，并统一增加：

```json
{
  "movie_id": "...",
  "shot_id": "...",
  "order_index": 1,
  "start_ms": 51926,
  "end_ms": 54129,
  "duration_sec": 2.203,
  "characters": [],
  "location_mentions": [],
  "item_mentions": [],
  "source_annotation_path": "..."
}
```

其中 `location_mentions` 和 `item_mentions` 不能只靠字符串规则直接作为最终标签，应保留抽取方法和人工复核状态。

### 3.3 语义事件边界复核

处理顺序应为：

```text
时间连续候选序列
  -> 读取每个镜头的 Plot/Background/Characters/Camera
  -> 判断是否仍属于同一事件线程
  -> 检查是否发生空间、主要角色或目标切换
  -> 输出 semantic_events.json
```

每个语义事件至少需要：

```json
{
  "event_id": "movie_event_0001",
  "source_shot_ids": ["..."],
  "start_ms": 0,
  "end_ms": 0,
  "characters": [],
  "location": "",
  "items": [],
  "event_summary": "",
  "actions": [],
  "interactions": [],
  "causal_preconditions": [],
  "causal_effects": [],
  "state_changes": [],
  "boundary_confidence": 0.0,
  "annotation_source": "rule|llm|human|hybrid"
}
```

LLM 生成的 `event_description` 可以保留，但不能替代结构化事件字段。事件文本主要用于生成；结构化字段用于状态更新、节点选择和评估。

### 3.4 构建叙事状态序列

需要从事件序列编译出：

```text
S_t^nar = 事件进展、人物目标、关系、约束、历史事实
W_t^vis = 人物外观、地点、物体及其已发生的视觉状态变化
```

每个事件至少保存：

- 事件前状态；
- 当前事件；
- 接受的角色行动；
- 事件后状态；
- 新增事实；
- 被保留的事实；
- 被否决或不确定的候选事实；
- 来源镜头和时间范围。

MovieBench 原始标注通常不直接提供人物目标和因果关系，因此这些字段需要：

1. 从标注规则抽取；
2. LLM 辅助结构化；
3. 人工抽样复核；
4. 标记为 `observed`、`inferred` 或 `uncertain`。

不能把 LLM 推断的心理动机写成电影标注中的确定事实。

### 3.5 构建 Track A：锚点约束数据

MovieBench 的镜头标注天然适合构造 Track A，但需要明确锚点层级：

- `shot_anchor`：单个原始镜头的 camera、action、plot、background；
- `event_anchor`：多个相关镜头组成的事件；
- `story_context`：当前事件前后有限范围内的故事状态。

Track A 的输入建议为：

```text
完整事件/故事上下文 + 原始镜头锚点 + 允许的参考资产
```

输出必须保留：

- 原始镜头顺序；
- 原始镜头核心动作；
- 原始角色数量；
- 原始地点和关键物体；
- 原始 camera motion；
- 允许补充但不能改变的上下文。

需要生成 `track_a.json`，记录每个锚点的源字段、生成提示、参考图、输出视频和评估标签。

### 3.6 构建 Track B：短故事线和约束

MovieBench 不是现成的开放式短故事线数据，因此需要从原始事件序列构造输入：

```text
完整事件序列 -> 截取前置信息/目标/冲突/结局 -> 压缩为短故事线
```

每个 Track B 样本必须同时保存：

- `short_storyline`：提供给系统的短故事线；
- `known_characters`；
- `known_facts`；
- `hard_constraints`：不能改变的角色、地点、核心冲突、结局或事实；
- `withheld_events`：从输入中隐藏、但仅用于后续分析的原始事件；
- `reference_story`：评估用，不暴露给生成阶段；
- `allowed_expansion_space`：哪些新增内容可被视为合理拓展。

这里的原始完整事件序列只能作为评估参考或约束构建依据。若把 `withheld_events` 直接放入 prompt，就不再是开放式拓展实验。

## 4. 新版 pipeline 在 MovieBench 上的适配方式

推荐流程：

```text
MovieBench 镜头标注
  -> 时间和字段标准化
  -> 连续候选分组
  -> 语义事件边界复核
  -> 结构化事件和状态序列
  -> 角色/location/item 资产索引
  -> Track A 或 Track B 输入构建
  -> 节点选择：一人一事件，互动事件允许多人
  -> world_state 更新判断
  -> 事件关键帧生成
  -> 事件视频生成
  -> 辅助空镜/状态变化镜头插入
  -> 拼接、旁白/对白时间戳和字幕 JSON
  -> 无泄漏叙事评估
```

### 4.1 节点构建适配

新版节点规则需要在 MovieBench 中落地为：

- 默认一个角色在一个事件中对应一个节点；
- 同一事件中两个或多个角色在同一地点发生明确互动时，允许合并为多人互动节点；
- 仅仅在同一电影、同一时间段或同一场景出现，不足以合并节点；
- 节点必须记录 `event_id`、`shot_ids`、角色、地点、物体、互动关系和合并理由。

需要额外防止把原始连续序列中的多个并行镜头错误合成一个多人事件。

### 4.2 world_state 适配

对每个事件的 location 和 item 进行变化判断：

- 只有事件文本明确描述颜色、形状、外观、纹理或可观察物理状态变化时，才更新 world state；
- 没有物理变化时保存 `no_update`，不能因为事件提到了地点或物体就生成新状态图；
- 更新图不能出现人物；
- 更新图只能表达事件明确描述的变化，不得引入事件未提及的新元素；
- location 更新需要再选择远景、中景、近景或特写中的一个视角；
- 新状态图应以上一轮状态图或原始 location 图为参考。

MovieBench 当前没有结构化 location/item 状态标签，因此需要新增 `world_state_annotations.json`，并区分 `mentioned`、`physically_changed`、`visual_update_required` 和 `selected_view`。

### 4.3 资产与参考图处理

如果具备源视频，应为每个电影建立：

- 角色初始化图；
- location 初始化图，远/中/近/特写四种视角；
- item 初始化图；
- 事件关键帧；
- 资产来源时间戳和是否来自源视频。

如果不具备源视频，则只能运行文本或外部生成资产条件，并必须在实验表中单独标记，不能声称使用了 MovieBench 原始视觉资产。

视频生成的参考图策略应单独做开关：

1. 仅事件关键帧；
2. 人物初始化图 + 被提及的 item/location 图；
3. 事件关键帧 + 人物/item/location 图。

每个节点保存最终实际发送给 VideoAgent 的参考图清单，防止只根据命令行参数推断参考图模式。

## 5. 需要新增的评估标注

### 5.1 Track A 标注

每个评估样本至少需要：

- 必要事件列表 `required_events`；
- 细粒度内容单元 `fine_grained_units`；
- 角色行动和互动单元；
- 地点/物体状态变化单元；
- 事件顺序约束；
- 不允许发生的锚点违背项 `anchor_violations`。

细粒度单元必须可观察和可判定，例如：

- 某角色拿起某物体；
- 某角色与另一角色发生明确交互；
- 某物体从桌面移动到手中；
- 某地点从明亮变为昏暗；
- 某动作发生在另一个动作之后。

不能只标注抽象主题，如“气氛紧张”，并将其直接当作事件召回单位。

### 5.2 Track B 标注

每个样本需要：

- 短故事线硬约束；
- 可接受事实集合；
- 原始完整事件作为参考但不作为唯一答案；
- 可能的合理拓展类型；
- 重复内容判定规则；
- 违反人物、地点、核心冲突和结局的规则。

MovieBench 原始后续事件不能简单等同于唯一正确的开放式答案，因为同一短故事线可能存在多个合理发展方向。

## 6. 评估指标与实验协议

### 6.1 Track A 主指标

需要实现：

#### Event Recall

```text
ER = 已表达的必要事件数 / 必要事件总数
```

按事件单元而不是按视频时长计算，并记录事件匹配证据。

#### Story Similarity

先只给评估器生成视频，让其独立重建故事，再与源故事比较：

- 主要事件；
- 事件顺序；
- 人物行为；
- 关键状态变化。

不能把源故事直接提供给第一阶段评估器，也不能只用文本 embedding 相似度代替完整故事比较。

#### Fine-grained Detail Coverage

```text
FDC = 已表达细粒度单元数 / 细粒度单元总数
```

额外记录锚点违背率，包括角色数量、核心动作、地点、物体和顺序违背。

### 6.2 Track B 主指标

需要实现：

- 有效拓展量/覆盖度；
- Expansion Validity Rate；
- 故事约束兼容性；
- 重复拓展率；
- 因果连贯性。

其中：

```text
EVR = 有效新增叙事单元数 / 识别出的新增叙事单元总数
```

如果没有识别出新增单元，应单独标记 `no_expansion`，不能直接记为 100%。

### 6.3 外部 benchmark 指标

MovieBench 适配不替代方案中已有的外部评估：

- ViMax-Bench：CC、IC、GC、CSC、SF、AQ；
- VBench-Long：按官方协议的语义、叙事连贯性和跨片段指标；
- LVBench-C：人物、物体、环境的长距离状态一致性。

这些指标与 MovieBench 的 ER、FDC、EVR 不应混成一个总分。MovieBench 主要承担叙事内容、事件结构和状态变化评估。

## 7. 必须运行的对照和消融

### 7.1 Track A 对照

至少运行：

1. `Storyboard-only`：只使用 MovieBench 原始镜头标注；
2. `Summary/Outline`：先压缩为摘要再生成；
3. `CineWorldSim-anchor`：加入状态序列和历史信息；
4. `CineWorldSim-anchor-no-world-state`：移除显式视觉世界状态。

保持相同：

- 生成模型；
- 视频时长、分辨率、帧率；
- 镜头/事件数量；
- 参考图条件；
- 提示预算和重试预算。

### 7.2 Track B 对照

至少运行：

1. `Direct Prompting`：短故事线直接生成；
2. `Summary/Outline Planning`：短故事线先生成摘要/大纲；
3. `Agentic Planning Baseline`：通用智能体规划；
4. `CineWorldSim-open`：状态推演 + 开放式事件和分镜构建。

若当前系统没有独立的故事约束模块，不应虚构“移除约束模块”的消融；应先把约束处理实现为可开关组件，或将该项延后。

### 7.3 参考图消融

在 MovieBench 视觉资产可用后，单独比较：

- 仅事件关键帧；
- 仅人物/location/item；
- 两者同时使用；
- 无视觉参考图。

这组实验不能与叙事状态消融混在同一个变量中。

## 8. 质量控制与可靠性

正式实验前必须完成：

- 电影级 train/validation/test split；
- 事件边界人工抽样复核；
- 角色、location、item 抽取准确性抽样复核；
- world state 更新判断一致性复核；
- 关键事件和细粒度单元标注一致性；
- 自动评估与人工评估的一致性；
- 生成失败、fallback、缺帧、缺音频、拼接失败统计；
- API 调用模型、参数、成本、重试、失败日志保存。

建议至少报告：

- 事件边界 F1 或人工一致率；
- 细粒度单元标注一致率；
- 评估器与人工复核一致率；
- 每个电影和每个事件的成功率，而非只报告总体成功数。

## 9. 推荐执行顺序

### 阶段 0：数据和资产审计

1. 确认本地是否有 166 个电影对应的源视频。
2. 确认视频文件、时间码和 JSON 标注的映射。
3. 确认许可和是否允许抽帧、生成参考资产及公开结果。
4. 建立电影级 split 和 manifest。

### 阶段 1：小样本结构化适配

1. 选 1 部电影和 5 个连续序列。
2. 生成结构化 `semantic_events.json`。
3. 生成 `narrative_state_sequence.json`。
4. 人工检查事件边界、人物、地点、物体和状态变化。
5. 验证新版节点选择与 world_state 逻辑。

### 阶段 2：Track A pilot

1. 运行 Storyboard-only 和 CineWorldSim-anchor。
2. 生成事件关键帧和视频。
3. 记录每个节点实际 prompt、参考图、视频参数和失败信息。
4. 实现 ER、SS、FDC 和锚点违背率。
5. 对自动评估结果做人工复核。

### 阶段 3：Track B pilot

1. 从完整事件序列构造短故事线和 withheld events。
2. 运行 Direct Prompting、Agentic Planning 和 CineWorldSim-open。
3. 对新增事件做价值、连贯、重复和兼容性判断。
4. 报告有效拓展量、EVR 和约束兼容性。

### 阶段 4：扩大规模和外部 benchmark

1. 扩展到多电影 test split。
2. 固定模型、预算、输出规格和参考条件。
3. 执行三组核心消融。
4. 再接入 ViMax-Bench、VBench-Long 和 LVBench-C。

## 10. 当前最重要的结论

MovieBench 当前最适合先作为 CineWorldSim 的**叙事事件与长距离状态评测数据源**，而不是立即作为完整端到端视频生成 benchmark。正式接入新版架构前至少必须补齐：

1. 电影级 split 和数据 manifest；
2. 语义事件边界与结构化事件字段；
3. 叙事状态序列；
4. Track A/Track B 输入协议；
5. 角色、地点、物体视觉资产或明确的无资产实验设置；
6. 必要事件、细粒度单元、故事约束和有效拓展标注；
7. ER、SS、FDC、有效拓展量、EVR 和约束兼容性评估器；
8. 基线、消融、失败日志和人工可靠性协议。

在这些内容完成前，可以做 smoke/pilot 测试，但不能把现有 5 个 GPT 事件描述样例或连续片段索引直接称为 CineWorldSim 在 MovieBench 上的完整实验结果。
