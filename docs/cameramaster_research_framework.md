# CineWorldSim 研究框架

> 文档用途：统一记录 CineWorldSim 的研究背景、研究目标、方法、baseline、工程 pipeline、实验设计和预期效果。
>
> 状态说明：本文中的“已实现”表示代码中已有对应模块；“已验证”表示已有真实运行证据；“待实验”表示尚未形成论文级结论，不应表述为已证明结果。

## 1. 研究定位

CineWorldSim 面向长叙事故事到视频生成（long-horizon story-to-video generation）。核心定位不是提出新的视频生成 backbone，而是提出一个**世界状态感知的叙事规划与事件级视觉生成框架**：先让故事世界在多个 step 中演化，再从显式状态中选择可拍摄事件，最后使用角色、空间、物体和事件关键帧将事件转换为图像或视频。

核心论点是：

> 显式、持续更新的叙事世界状态可以作为故事文本与视觉生成模型之间的中间表示，从而改善长故事中的人物一致性、地点和物体状态一致性、事件顺序以及失败可诊断性。

CineWorldSim 不主张：

- 新的视频生成模型架构；
- 通用物理世界模拟器；
- 对任意长篇小说都能自动生成高质量电影；
- 已经证明优于所有现有 story-to-video 方法。

## 2. 研究背景与问题

### 2.1 背景

文本到视频模型已经能够根据单个事件或短 prompt 生成具有较好视觉质量的片段，但长叙事视频的难点不只是生成更长的视频。长故事需要同时维护：

1. 人物身份、外观、角色关系和当前行动；
2. 地点的空间属性以及跨事件的变化；
3. 关键物体的存在、位置、状态和变化；
4. 事件之间的时间、因果和任务依赖；
5. 镜头之间的视觉衔接和叙事节奏。

直接把完整故事或长段落交给视频模型，通常缺少一个可以持续读写的“故事世界账本”。模型容易忘记前文状态、引入无关人物、改变物体外观、跳过关键事件，或者生成视觉上连续但叙事上不正确的片段。

### 2.2 现有方法的主要缺口

当前相关方法大致覆盖长视频建模、多智能体故事规划、storyboard 生成、视觉记忆和世界模型等方向，但仍存在以下研究缺口：

- **故事状态与视觉渲染脱节**：故事推演结果通常只是文本，不能稳定约束地点、物体和人物的视觉状态。
- **事件粒度不稳定**：整段故事、step、角色行动和镜头之间缺少统一的可拍摄事件表示。
- **多人物与无关人物控制不足**：多角色故事容易把同一 step 的所有角色错误地放入同一视频，或引入行人、人群和不相关人物。
- **世界变化无法局部更新**：地点和物体发生变化时，系统容易重新生成完整场景，导致背景和未变化元素漂移。
- **错误不可定位**：最终视频失败时，很难判断问题来自叙事推演、节点筛选、路径排序、参考图选择还是视频生成。

### 2.3 问题定义

给定故事文本 `Y`、初始角色集合 `C`、地点集合 `L` 和关键物体集合 `O`，系统需要生成一组有序的视频事件：

```text
Y, C, L, O -> {S_t} -> {N_i} -> {P_i} -> {V_i} -> V_final
```

其中：

- `S_t`：第 `t` 个叙事 step 的世界状态；
- `N_i`：可拍摄叙事节点；
- `P_i`：节点对应的视觉提示词、参考图和分镜计划；
- `V_i`：事件视频或辅助镜头；
- `V_final`：按叙事路径拼接后的故事视频。

每个节点必须能够追溯到角色、事件、地点、物体、前置状态和后置状态。

## 3. 研究目标与研究问题

### 3.1 总体目标

构建一个可解释、可恢复、可评测的长叙事视频生成框架，使故事世界状态成为视觉生成的约束源，而不是仅作为 prompt 历史缓存。

### 3.2 研究问题

**RQ1：世界状态是否能提升长故事中的连续性？**

比较有无显式 world state tracker 时，人物身份、地点属性、物体状态和事件顺序是否更稳定。

**RQ2：事件级节点和路径规划是否比直接按故事文本切分更适合作为视频生成中间表示？**

比较直接 prompt、storyboard-only、按 step 生成和 CineWorldSim narrative nodes 在事件覆盖、冗余、因果顺序和镜头可执行性上的差异。

**RQ3：受控参考图选择是否能提高视觉忠实度并减少无关内容？**

比较只使用事件关键帧、使用角色/地点/物体资产、无条件使用多张参考图等策略，测量角色、空间、物体和事件对齐程度。

**RQ4：显式状态和结构化中间结果能否改善失败诊断与恢复？**

统计失败发生在 simulation、node selection、path planning、reference selection 或 rendering 哪一层，并验证 fallback 节点重跑和下游恢复机制。

## 4. 核心方法

CineWorldSim 的方法由八个部分组成：

```text
World Initialization
  -> Narrative Simulation
  -> World-state Tracking
  -> Event Keyframe Anchoring
  -> Narrative Node Selection
  -> Causal Path Orchestration
  -> VideoAgent Shot Planning
  -> Video Rendering and Post-production
```

### 4.1 世界与角色初始化

`WorldModelAgent` 从故事中抽取角色、地点、空间关系和关键物体，形成初始世界蓝图。初始化阶段生成：

- 角色肖像图；
- 地点的远景、中景、近景和特写图；
- 关键物体的初始化图；
- 角色目标、角色关系和初始环境配置。

初始化图的用途是建立后续生成的视觉身份锚点。角色和物体图强调主体占比，减少不必要背景；地点图覆盖不同空间尺度，供世界状态更新和事件关键帧使用。

当前代码涉及：

- `narrative/simulation_scaffold.py`
- `narrative/world_renderer.py`
- `narrative/event_keyframe_renderer.py`
- `agents/generation_backends.py`

### 4.2 多角色叙事推演

`StateAnalysisMachine` 以 step 为单位推进故事。每个 step 中，每个角色基于自己的目标、关系、环境和上一轮状态提出行动，再由模型判断行动是否符合当前叙事状态。

节点粒度与 step 粒度必须区分：

- 一个 step 是一次全局叙事推进；
- 默认一个候选叙事节点是一个角色在一个 step 中发生的一件事；
- 只有多个角色在同一 step、同一地点并存在明确互动事件时，才允许合并为一个多人节点；
- 15 个 step、3 个主角通常最多形成约 45 条角色级候选事件，但最终节点数量由事件重要性和合并规则决定，不强制为 45。

该约束避免把同一 step 的多个独立行动错误地压缩为一个视频。

### 4.3 世界状态跟踪与局部更新

世界状态由地点、物体、角色状态、关系、环境条件和故事进度组成。每个 step 产生一个 delta，写入新的世界快照。

当前目标逻辑是：只有当当前 step 的角色事件明确描述地点或物体的物理变化，如颜色、形状、外观或纹理变化时，才更新对应的 world state 图像。更新提示词必须要求：

- 不出现人物；
- 除事件中明确变化的内容外，不引入其他元素；
- 只描述更新后的地点或物体状态；
- location 更新时由 LLM 判断应更新远景、中景、近景或特写中的哪一个视图；
- 优先使用上一轮更新图，若没有则使用初始世界图作为参考。

该机制的研究目的不是增加图像数量，而是减少无依据的视觉漂移。

### 4.4 事件关键帧

完成当前 step 必要的 world state 更新后，再生成事件关键帧。关键帧的输入包括：

- 当前事件文本；
- 事件所在 location；
- 事件中明确提及的 item；
- 相关角色参考图；
- 当前有效的 location/world 图像。

参考图选择遵循“事件需要什么才提供什么”的原则，不以最大化参考图数量为目标。事件关键帧的 prompt 还必须抑制无关行人、人群和非主角人物。

### 4.5 叙事节点选择

`NarrativeNodeSelector` 将角色 step 日志转换为可拍摄节点。选择器的职责包括：

- 保留重要事件；
- 维持角色和 step 的基本粒度；
- 识别同地、同事件、明确互动的多人节点；
- 删除重复、低信息量或不可视化的节点；
- 给节点提供事件摘要、参与角色、地点、物体、前后状态和参考资产。

节点不是简单的文本 chunk，而是连接叙事状态和视觉生成的结构化中间表示。

### 4.6 路径编排

`PathOrchestratorAgent` 和 `PathOrchestratorMCTSAgent` 根据事件依赖、时间顺序、地点连续性、角色状态和视觉衔接，对选中的节点进行排序。

路径编排不能破坏节点语义，也不能把本应独立的视频强行合并。允许的合并仅限于节点选择阶段已经判定为同一多人互动事件的节点。

### 4.7 VideoAgent 与参考图策略

VideoAgent 将节点转换为视频 prompt，并自行规划 shot 数量、每个 shot 的时间范围和视频帧数。当前 H3 视频长度支持两种主要配置：

- 124 帧，对应约 5 秒；
- 243 帧，对应约 10 秒。

事件视频支持三类参考策略：

1. 只使用事件关键帧；
2. 不使用世界关键帧，但使用角色初始化图、事件提及的 item 图和 location 图；
3. 允许同时使用事件关键帧、角色、item 和 location 图。

研究实验中应明确记录使用的策略，而不能仅根据最终 `image_count` 推测方法设置。`event_keyframe_only` 模式下，多人互动节点可能仍然有多张事件关键帧，这是“关键帧唯一来源”，不是“请求只能有一张图片”。

### 4.8 辅助镜头与后期处理

辅助镜头不替代事件视频，主要包括：

- 首次进入某个 location 前的空镜头；
- world state 或 item 状态发生变化前后的过渡镜头；
- 用于建立空间、展示变化或连接事件的短镜头。

最后阶段负责事件视频、辅助镜头、旁白和对话的时间轴对齐，并生成可用于字幕制作的同名 JSON 文件。完整运行还应输出拼接视频、失败记录、manifest 和可追溯的中间结果。

## 5. 当前 Pipeline 框架

### 5.1 总体流程

```text
Story text
  -> Character and world extraction
  -> Initial portraits / locations / items
  -> Step-by-step multi-agent simulation
  -> World delta judgment and state snapshots
  -> Local world-state image updates
  -> Event keyframes
  -> Narrative node selection
  -> LLM path ordering / MCTS path ordering
  -> VideoAgent prompt and shot planning
  -> H3 event videos and auxiliary videos
  -> Timestamped dialogue/narration JSON
  -> Concatenation and final video
```

### 5.2 主要执行脚本

| 脚本 | 职责 | 状态 |
| --- | --- | --- |
| `scripts/run_full_narrative_world_video_pipeline.py` | 完整故事到视频入口 | 已实现 |
| `scripts/run_narrative_world_path_pipeline.py` | 世界阶段、节点和路径编排 | 已实现 |
| `scripts/run_dual_backend_narrative_world_pipeline.py` | Banana/Flux 后端调度 | 已实现 |
| `scripts/run_keyframe_only_event_video_pipeline.py` | 固定事件关键帧唯一参考的视频运行 | 已实现 |
| `scripts/run_orchestrated_event_video_generation.py` | 从已有节点和路径单独重跑视频阶段 | 已实现 |
| `scripts/rerun_fallback_narrative_nodes.py` | 重跑失败或 fallback 节点并恢复下游推演 | 已实现 |
| `scripts/run_vistorybench_pipeline.py` | ViStoryBench 固定 shot 的图像生成 | 已实现 |
| `utils/export_vistorybench_outputs.py` | ViStoryBench 输出检查和导出 | 已实现 |

### 5.3 主要中间结果

```text
pipeline_result.json
story_bundle.json
step_logs.json
narrative_checkpoint.json
world_state_step*.json
world_deltas/*.json
narrative_artifacts/portraits/
narrative_artifacts/world_model/
narrative_artifacts/world_updates/
narrative_artifacts/event_keyframes/
path_orchestration/narrative_nodes.json
path_orchestration/path_llm_direct/narrative_path_plan.json
event_video_pipeline_*/
final_orchestrated_events.mp4
字幕或时间轴 JSON
```

每一个最终视频片段都应能够回溯到 story、step、角色、节点、路径位置、prompt、参考图和生成请求。

### 5.4 ViStoryBench 适配流程

ViStoryBench 的 `Shots[]` 是固定评测锚点，不能被 CineWorldSim 自由节点选择器替换、合并、删除或重排。适配流程为：

```text
ViStoryBench story.json
  -> fixed benchmark shot anchor
  -> optional causal CineWorldSim context
  -> enriched shot prompt
  -> one PNG per benchmark shot
  -> official benchmark evaluation
```

当前模式：

- `vistorybench_anchor_only`：使用 benchmark shot 要求和角色参考图，不加入 simulation context；
- `vistorybench_anchor_plus_sim`：在不改变 benchmark shot 语义的前提下加入 CineWorldSim 连续性上下文；
- `causal`：只使用当前 shot 之前的信息，作为主实验设置；
- `full_story`：允许完整故事上下文，作为信息泄漏风险和上下文范围消融。

已知边界：当前 ViStoryBench runner 已接入事件关键帧补充，但“完整 location/item/world asset + event keyframe”组合仍需单独完成和验证。

## 6. Baseline 设计

Baseline 的目标不是简单增加方法数量，而是隔离 CineWorldSim 的具体贡献。所有对照应尽可能固定故事、图像/视频后端、分辨率、随机种子、输出预算和评测协议。

### 6.1 主 baseline

| ID | 方法 | 去除或保留内容 | 验证问题 |
| --- | --- | --- | --- |
| B0 | Direct prompting | 故事直接转换为一个或多个 prompt，无显式状态 | 世界状态是否必要 |
| B1 | Storyboard-only | 有 storyboard/事件切分，但无持续 world state | 仅有 storyboard 是否足够 |
| B2 | Visual-memory baseline | 保留角色/地点参考图，但不使用因果状态更新 | 视觉记忆与状态记忆的差异 |
| B3 | Agentic narrative baseline | 有多智能体故事规划，但无 CineWorldSim 状态锚定 | 多智能体本身是否解释收益 |
| B4 | ViStoryBench image baselines | 对齐 benchmark 中的 story visualization 方法或官方结果 | 外部可比性 |
| B5 | Story-video baselines | 可复现的 storyboard/agentic video 方法 | 多镜头视频层面的可比性 |
| Ours | CineWorldSim full | world state、节点、路径和受控参考图全部启用 | 完整系统效果 |

需要明确标注：上述 B2-B5 中部分方法目前属于实验规划或待复现对象，不能写成已完成实现。

### 6.2 CineWorldSim 消融

| 变体 | 删除内容 | 主要观察指标 |
| --- | --- | --- |
| Full | 无 | 完整性能 |
| No world-state tracker | 不累计 `S_t` 和 world delta | world/location/item consistency |
| No local world update | 不更新变化的 location/item 图 | 世界状态视觉忠实度 |
| No narrative node selector | 按原始 step 或文本 chunk 生成 | 事件覆盖、冗余、节点数量 |
| No path planner | chronological order | 因果顺序和镜头衔接 |
| No reference gating | 无条件堆叠角色、地点、物体图 | image count、无关元素、身份一致性 |
| Event-keyframe-only | 只提供事件关键帧 | 关键帧锚定效果 |
| No character refs | 不使用角色初始化图 | 人物一致性变化 |
| No world/item refs | 不使用世界和物体资产 | 场景与物体状态变化 |
| No auxiliary shots | 不生成空镜和世界变化镜头 | 空间建立、状态变化可读性 |
| No fallback recovery | 失败节点不重跑 | 可恢复性和最终成功率 |

## 7. 实验设计

### 7.1 实验分层

实验分为两条互补路径：

1. **故事可视化实验**：输出每个事件/benchmark shot 的图像，验证叙事表示、角色一致性、场景和物体状态。
2. **多镜头视频实验**：输出事件视频及最终拼接视频，验证事件实现、跨镜头状态连续性、时间因果和视频质量。

两条路径使用相同故事集合、相同角色和世界定义，并尽量共享生成后端和随机种子。

### 7.2 主实验 A：世界状态的有效性

对比：

```text
Direct prompting
Storyboard-only
Visual-memory baseline
CineWorldSim full
```

固定：故事、角色、图像后端、视频后端、输出数量、分辨率、seed 和评测 prompt。重点测量：

- 事件是否被正确表达；
- 人物身份、服装和角色数量是否正确；
- location 是否符合当前状态；
- item 是否按前后状态出现、变化或消失；
- 事件顺序是否符合因果路径；
- 是否引入行人、人群和无关人物。

### 7.3 主实验 B：事件节点和路径编排

比较以下节点组织方式：

- 原始故事直接分段；
- 每个 step 一个视频；
- 每个角色 step 一个节点；
- 角色 step 加多人互动合并规则；
- CineWorldSim 节点加 LLM path；
- CineWorldSim 节点加 MCTS path。

主要验证：

- 是否减少独立行动被错误合并；
- 是否覆盖关键叙事事件；
- 是否减少重复片段；
- 是否保持前置条件与后置结果的顺序；
- 是否改善多角色事件的可读性。

### 7.4 主实验 C：参考图策略

至少运行：

1. `event_keyframe_only`；
2. 角色初始化图 + 事件提及的 location/item 图；
3. 事件关键帧 + 角色/location/item 图；
4. 无条件最大参考图数量，作为负面对照。

参考图数量必须记录在 manifest 中，并记录每张参考图的来源和选择理由。不能把“参考图越多”作为默认优化目标。

### 7.5 主实验 D：ViStoryBench

优先级：

1. `FLUX2 Klein + vistorybench_anchor_only`；
2. `FLUX2 Klein + vistorybench_anchor_plus_sim`；
3. `causal` 与 `full_story`；
4. Banana 与 FLUX2 Klein；
5. 有无 event keyframe；
6. 有无 location/item/world continuity；
7. lite split smoke test；
8. full split 正式评测。

已真实验证的事实是：已有 `FLUX2 Klein + anchor_only` 运行生成 23/23 张真实 PNG，尺寸为 `1024x576`，调用日志位于对应 run 目录的 `xi_api_calls.jsonl`。官方 `bench_run.py` 指标、完整 `anchor_plus_sim` 和多故事统计仍属于待完成实验。

### 7.6 主实验 E：故障恢复

在 narrative simulation 和 rendering 阶段注入或收集失败，比较：

- 不恢复，直接终止；
- 只重跑失败节点；
- 重跑失败节点并恢复下游 step；
- 保留已成功资产，仅重跑缺失生成任务。

指标包括恢复成功率、额外 API 调用次数、恢复时间、下游结果一致性和最终视频完整率。

## 8. 评价指标

### 8.1 自动指标

ViStoryBench 侧可使用其支持的指标，例如：

- `CIDS`：角色身份和角色数量相关一致性；
- `CSD`：风格或视觉一致性；
- `PA`：prompt alignment；
- `OCCM`：对象/内容一致性相关指标；
- `Aesthetic`、`Diversity` 等通用图像指标。

多镜头视频侧可结合已有视频质量指标，但不能只报告低层视觉质量。建议增加：

- `EF`：Event Fidelity，视频是否表达目标事件；
- `CSC`：Character-State Consistency，人物身份和状态；
- `WSC`：World-State Consistency，地点和环境状态；
- `OSP`：Object-State Persistence，物体状态持续与变化；
- `TCO`：Temporal/Causal Ordering，时间和因果顺序；
- `NUC`：Narrative Unit Coverage，叙事单元覆盖率；
- `UR`：Unrelated-person Rate，无关人物出现率；
- `RR`：Redundancy Rate，重复事件比例。

### 8.2 人工评测

对采样事件或完整视频进行 pairwise 或 Likert 评估，问题应具体化：

1. 是否表达了目标事件？
2. 正确角色是否出现，无关人物是否被抑制？
3. 地点是否符合当前 world state？
4. 关键物体的状态是否正确？
5. 事件顺序是否合理？
6. 哪个输出更好地保持了故事世界的连续性？

## 9. 实验控制与可复现性

每个 run 必须固定或记录：

- story id、story 文件和角色列表；
- 方法名、模式名和上下文模式；
- 图像/视频 backend 及模型名称；
- API URL 的非敏感标识，不在文档或日志中暴露密钥；
- seed、分辨率、采样步数和超时；
- 节点上下限、是否启用 LLM scoring、path mode；
- 参考图策略、每张参考图的来源和选择理由；
- event shot 数、视频时长和帧数；
- 失败、fallback、重试和恢复记录。

实验输出应至少包含 `run_manifest.json`、中间节点、prompt、参考图清单、请求日志、指标 JSON 和失败报告。所有论文表格数字都应能追溯到原始 run。

## 10. 当前实现与验证状态

### 10.1 已实现

- 多角色 step-by-step narrative simulation；
- 角色级叙事节点和多人互动节点约束；
- world state、world delta 和事件关键帧产物；
- Banana 和本地 FLUX.2 Klein 图像后端接入；
- 事件关键帧唯一参考的视频生成模式；
- 辅助空镜和世界变化镜头入口；
- H3 shot 时间和帧数规划；
- 失败节点 fallback 重跑和下游恢复脚本；
- ViStoryBench 固定 shot 锚点模式；
- PNG 输出和 API 调用日志。

### 10.2 已真实验证

- FLUX.2 Klein 服务 `/health` 和 `/generate` 接口可用；
- ViStoryBench `anchor_only` 已有 story `01` 的 23/23 张真实 PNG 输出；
- 生成图像尺寸为 `1024x576`；
- 对应 run 中存在 `xi_api_calls.jsonl`，记录工具调用开始和完成事件；
- CineWorldSim 完整视频样例曾完成事件视频和辅助视频生成及最终拼接。

### 10.3 尚未完成或不能提前宣称

- ViStoryBench 官方 `bench_run.py` 指标尚未形成完整结果表；
- `anchor_plus_sim` 尚未完成包含所有 location/item/world asset 的统一正式实验；
- 规划中的所有 baseline 尚未全部复现；
- Banana 在不同 API 状态下存在 HTTP 500、503、404 和 TLS 错误，不能把历史 Banana 运行视为稳定基准；
- 尚未通过完整多故事、多 seed 统计证明 CineWorldSim 优于其他方法；
- state-faithfulness 指标和人评协议仍需实现、校准和报告。

## 11. 预期效果与可证伪假设

### 11.1 预期效果

如果核心假设成立，CineWorldSim 应当相对于直接 prompt 和无状态 storyboard 方法表现出：

- 更高的角色一致性和正确角色覆盖率；
- 更低的无关人物、人群和行人出现率；
- 更高的地点和物体状态一致性；
- 更少的事件遗漏和重复；
- 更稳定的时间和因果顺序；
- 更清晰的空间建立和 world state 变化表达；
- 更高的失败可诊断性和恢复成功率。

### 11.2 需要接受的可能结果

CineWorldSim 可能不会在单帧审美质量或短视频运动质量上全面优于直接 prompt。合理的研究结论应集中在长程状态、事件覆盖、因果组织和可诊断性，而不是声称所有视觉指标都更高。

若去除 world state tracker、节点选择或参考图 gating 后，状态忠实度没有显著下降，则需要收缩研究主张，重新判断该框架是否主要是工程整合而不是有明确方法贡献。

## 12. 建议的论文证据链

```text
Claim 1: 显式世界状态能改善长故事连续性。
Evidence: Full vs No-world-state ablation + CSC/WSC/OSP/TCO.

Claim 2: Narrative Nodes 比直接文本切分更适合作为视频中间表示。
Evidence: event coverage, redundancy, causal ordering and human evaluation.

Claim 3: 受控参考图选择比无条件堆叠参考图更稳定。
Evidence: event-keyframe-only / gated-assets / all-assets comparison.

Claim 4: 结构化中间结果使失败可定位并可恢复。
Evidence: failure taxonomy, retry cost, downstream recovery and final completion.
```

## 13. 下一步实验顺序

1. 固定一套可复现的故事 manifest、模型、seed、分辨率和输出预算。
2. 完成 `anchor_only` 的 ViStoryBench lite 指标 smoke test。
3. 完成 `anchor_plus_sim`，补齐 location/item/world asset 选择记录。
4. 在同一图像后端下运行 Direct、Storyboard-only 和 CineWorldSim full。
5. 跑 No-world-state、No-node-selector、No-reference-gating 三个核心消融。
6. 对完整视频运行 state-faithfulness 指标和小规模人评。
7. 再扩展到 Banana、不同故事、full split 和多 seed。
8. 最后根据结果决定论文主张范围，而不是预先假设 CineWorldSim 在所有指标上领先。

## 14. 关键代码与文档索引

### 核心代码

- `narrative/simulation_scaffold.py`
- `narrative/world_renderer.py`
- `narrative/world_state_tracker.py`
- `narrative/world_delta_judge.py`
- `narrative/event_keyframe_renderer.py`
- `agents/narrative_node_selector.py`
- `agents/path_orchestrator_agent.py`
- `agents/path_orchestrator_mcts_agent.py`
- `agents/video_agent.py`
- `scripts/run_full_narrative_world_video_pipeline.py`
- `scripts/run_narrative_world_path_pipeline.py`
- `scripts/run_dual_backend_narrative_world_pipeline.py`

### ViStoryBench 适配

- `agents/vistorybench_anchor_selector.py`
- `agents/vistorybench_prompt_adapter.py`
- `scripts/run_vistorybench_pipeline.py`
- `utils/export_vistorybench_outputs.py`
- `utils/vistorybench_image.py`

### 相关文档

- `docs/current_architecture.md`
- `docs/current_pipeline_architecture_and_visualization.md`
- `docs/current_pipeline_execution_and_output_reference.md`
- `docs/cameramaster_iclr_boundary_and_main_axis.md`
- `iclr/experiment_implementation_protocol.md`
- `/data1/AgentG/hhb/vistorybench-main/docs/cameramaster_vistorybench_modification.md`

