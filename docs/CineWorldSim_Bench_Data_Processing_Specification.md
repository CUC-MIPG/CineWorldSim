# CineWorldSim-Bench 数据构建与四级叙事聚合处理规范

## 0. 文档定位与背景

本文档基于工作区 `/data1/AgentG/hhb/CameraMaster/MovieBench` 的现有数据状态与《CineWorldSim 研究方案总纲》的方法及评测要求制定。

针对当前数据集仅有单镜头标注（L1）与初步时间连续片段（L2）、缺乏全片级总故事描述的问题，本规范采用**自底向上（Bottom-up）的四级叙事单元聚合方案**，从底层分镜逐步构建“微事件 -> 场景叙述段落 -> 多场景故事弧/全片描述”，并在此基础上派生出支撑 **Track A（叙事保真与锚点约束视觉化）** 与 **Track B（开放式叙事拓展）** 的完整评测数据集。

---

## 1. 四级叙事单元（L1–L4）划分标准与映射关系

为确保自底向上聚合时不产生幻觉、自顶向下评测时能精准定位信息损失，全流程统一采用以下四级叙事结构：

| 层级 | 单元名称 | 英文标识 | 典型规模 | 边界划分准则 | 对应 CineWorldSim 模型与评测角色 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **L1** | **镜头单元** | `Shot Unit` | 1 个镜头 (2~6s) | 原始单镜头 JSON，画面连续不中断 | **既定分镜锚点 $A_i$ 的原子粒度**；提供细粒度内容覆盖率（FDC）的底层真值（表情、局部动作、运镜、物体状态）。 |
| **L2** | **微事件 / 动作拍** | `Micro-Event / Beat` | 2~8 个连续镜头 (10~30s) | 物理时间连续（相邻间隔 $<1000\text{ ms}$）且同一空间内核心动作未中断 | **系统核心推演步长 $e_t$ 与编译生成单元 $U_i$**；提供事件召回率（ER）中必要事件 $E_{\mathrm{source}}$ 的基本颗粒度。 |
| **L3** | **场景叙事段落** | `Scene Segment` | 1~5 个 L2 微事件 (30~120s) | **时空与局部戏剧目标一致**（同一地点 `setting`、连续时间段、围绕同一局部冲突展开的大叙述段落） | **跨镜头/场景内叙事整合层**；用于弥合因漏标镜头造成的微事件断层，记录场景级状态转移 $Z_{\mathrm{in}} \to Z_{\mathrm{out}}$。 |
| **L4** | **多场景故事弧 / 全片描述** | `Story Arc / Macro-Story` | 2~5 个关联 L3 场景（故事弧）或全片 L3 集合 | **跨场景因果链与核心冲突闭环**（人物目标跨场景延续，状态发生跨场景传递） | **Benchmark 评测任务源输入**：多场景故事弧构成完整故事 $S$（Track A 输入）及压缩后的简短故事线 $L$（Track B 输入）；全片汇总提供全局人物设定与世界观上下文。 |

---

## 2. 端到端数据处理链路总览

```text
[阶段 1: 清洗与 L2 语义切分]
MovieBench 原始镜头 JSON (L1, 107,667 镜头)
  ├─> scripts/process_moviebench_clip_metadata.py (解析时间码與 clip_meta)
  ├─> scripts/build_moviebench_continuous_dataset.py (间隔 <1000ms 物理分组)
  └─> scripts/refine_l2_micro_events.py (基于 llm_scene_segment 拆分过长或跨景序列，生成标准 L2 单元)

[阶段 2: L2 微事件描述生成]
标准 L2 序列
  └─> scripts/build_moviebench_event_descriptions_from_continuous_shots.py (批量生成 L2 事件描述与 shot-to-event 映射)

[阶段 3: L3 场景段落与 L4 故事弧/全片聚合]
单片全量 L2 事件描述
  ├─> scripts/build_l3_scene_segments.py (合并同场景相邻 L2，生成 L3 大叙述段落与状态进出变化)
  └─> scripts/build_l4_story_arcs_and_movie_bible.py (聚合全片人物/设定档案，切分跨场景 L4 故事弧 S)

[阶段 4: CineWorldSim-Bench 双轨评测集构建]
精选 L4 故事弧候选池 (100~300 个高叙事密度样本) + 关联 L3/L2/L1 数据
  ├─> Track A 标注生成：完整故事 S + 分镜锚点 {A_i} + 必要事件集 E_source + 细粒度单元 FDC_units
  ├─> Track B 标注生成：简短故事线 L + 故事硬约束集 Story_Constraints
  └─> 视觉参考资产提取：绑定角色/场景/道具参考图 R_i
```

---

## 3. 分阶段处理规范与操作细则

### 阶段一：基础数据清洗与 L2 微事件边界精修

#### 3.1 异常数据清理与目录对齐
1. **核验无效文件**：定位并排查 `continuous_dataset_index.json` 中跳过的 3 个无效 JSON，记录文件路径至 `outputs/moviebench_analysis/invalid_files_manifest.json` 并在后续流程中显式排除。
2. **固定电影清单**：对比标称 `160movies` 与实际 166 个子目录，排查是否存在重复、空目录或分卷目录，输出唯一确定的有效电影 ID 列表。

#### 3.2 物理连续序列的语义二次切分（L1 -> Refined L2）
物理间隔 `< 1000 ms` 的序列可能包含硬切转场（前一个镜头在室内，下一帧 0ms 直接切到室外新场景）或超长对话序列（>15 个镜头）。需结合 `scripts/llm_scene_segment_moviebench.py` 执行二次精修：
- **输入**：`Continuous_Clip_Groups_11.15_V2_160movies/`
- **切分触发条件**：
  1. 相邻镜头的 `Background Description` 发生空间切换；
  2. 序列总镜头数 $> 10$ 且内部动作已发生阶段性转折。
- **过滤条件**：剔除纯空镜序列（所有镜头 `Characters` 均为空且无实质 `Plot` 推进）或总镜头数 $< 3$ 的碎片序列。

---

### 阶段二：L2 微事件描述生成（L1 -> L2）

使用现有脚本 `scripts/build_moviebench_event_descriptions_from_continuous_shots.py` 扩展运行范围（优先在精选的 20~50 部代表性电影或全量合格 L2 序列上批量执行）。

#### L2 标准 JSON Schema（承接现有输出并规范化）
```json
{
  "l2_event_id": "0001_American_Beauty_seq_0012",
  "movie_id": "0001_American_Beauty",
  "time_range_ms": [117944, 135210],
  "shot_count": 5,
  "event_title": "Lester and Burnham Family Morning Routine",
  "event_description": "基于连续分镜整合的完整微事件自然语言叙述...",
  "main_characters": ["Lester Burnham", "Carolyn Burnham"],
  "setting": "Burnham family house - Dining room and driveway",
  "temporal_progression": [
    "Shot 1-2: Carolyn hurries through the dining room...",
    "Shot 3-5: Lester watches silently with a coffee mug..."
  ],
  "shot_to_event_mapping": [
    {
      "shot_id": "0001_American_Beauty_00.01.57.944-00.02.01.546",
      "role_in_event": "Establishes Carolyn's anxious demeanor and the dining room setting."
    }
  ]
}
```

---

### 阶段三：L3 场景叙述段落与 L4 全片/故事弧聚合（核心新增环节）

#### 3.1 L2 聚合为 L3 场景叙述段落（`scripts/build_l3_scene_segments.py`）
将同一部电影内的所有 L2 微事件按 `time_range_ms` 升序排列，通过滑动窗口与 LLM 判定相邻 L2 序列 $(e_k, e_{k+1})$ 是否属于同一 **L3 场景叙述段落**。

**聚合判定准则：**
1. **时间邻近性**：$e_{k+1}.\text{start\_ms} - e_k.\text{end\_ms} \le 60,000\text{ ms}$（允许中间存在少量未标注的过渡镜头）；
2. **空间一致性**：`setting` 属于同一地点或紧密相连的子空间（如厨房与餐厅）；
3. **戏剧连贯性**：出场人物存在交集，且围绕同一局部事件或对话冲突展开。

**L3 场景段落 JSON Schema（作为大叙述段落）：**
```json
{
  "l3_scene_id": "0001_American_Beauty_scene_004",
  "movie_id": "0001_American_Beauty",
  "l2_event_ids": [
    "0001_American_Beauty_seq_0012",
    "0001_American_Beauty_seq_0013"
  ],
  "time_range_ms": [117944, 182400],
  "scene_setting": "Burnham Residence - Interior & Driveway",
  "active_characters": ["Lester Burnham", "Carolyn Burnham", "Jane Burnham"],
  "scene_narrative": "（大叙述段落）完整描述该场景的起因、人物互动过程、情绪变化与场景结局，严格基于下属 L2 事件整合，不凭空编造未出现的事实...",
  "narrative_state_transition": {
    "entry_state": "一家人早晨准备各自出门，气氛冷漠疏离",
    "key_actions_and_conflicts": [
      "Carolyn 急促催促家人，展现出控制欲与焦虑",
      "Jane 对父母表现出明显的抵触情绪",
      "Lester 麻木顺从地跟在妻女身后走向车库"
    ],
    "exit_state": "三人同乘一辆车离开家，家庭内部的压抑关系被确立",
    "key_objects_and_states": [
      "Carolyn 的玫瑰剪刀/公文包（随身携带）",
      "Lester 的咖啡杯"
    ]
  }
}
```

#### 3.2 L3 聚合为 L4 全片设定档案与多场景故事弧（`scripts/build_l4_story_arcs_and_movie_bible.py`）
由于单部电影可能存在较长的时间跳跃，L4 聚合分为两个产物：

1. **全片级设定与故事总览（Movie Bible & Macro-Synopsis）**
   - 汇总全片所有 L3 的 `scene_narrative` 与状态变化，生成：
     - `movie_synopsis`：全片主线剧情概述（分段/分幕组织，对时间跨度大于 5 分钟的断层显式标记为不同篇章）；
     - `global_character_profiles`：全片核心人物身份、性格特征、核心目标及人物关系演化图谱；
     - `recurring_locations_and_objects`：跨场景复现的关键地点与道具状态说明。
2. **多场景评测故事弧（Multi-Scene Story Arc $S$）**
   - 从全片 L3 序列中提取包含 **2~4 个具有因果承接关系或跨场景人物状态延续的 L3 场景组合**（总计包含约 3~6 个 L2 微事件、10~25 个 L1 镜头）；
   - 结合 `global_character_profiles` 中的必要人物关系背景，将这组 L3 的 `scene_narrative` 融合成一段连贯、自包含的 **完整源故事 $S$（300~600 词）**。

---

### 阶段四：CineWorldSim-Bench 双轨评测数据生成（`scripts/build_cineworldsim_benchmark_annotations.py`）

从阶段三生成的 L4 多场景故事弧中，按“角色交互密度、跨场景状态转移明确性、镜头数量（10~25 镜头）”筛选出 **100~300 个高质量故事弧** 作为最终 Benchmark 样本，并生成双轨输入与评测真值：

#### 4.1 Track A（叙事保真与锚点约束视觉化）数据结构
- **系统输入**：
  1. `full_story_S`：阶段三生成的 L4 完整故事文本 $S$。
  2. `storyboard_anchors_A`：按顺序提取底层 L1 镜头的简略分镜锚点 $\{A_1, A_2, \dots, A_n\}$。**注意：生成 $A_i$ 时必须隐去全局动机、前因后果及微小的情绪/道具上下文，仅保留当前镜头的基础画面构图、运镜（`Camera Motion`）与表层动作**，从而检验模型能否利用 $S$ 和状态推演 $Z_t$ 补足细节。
  3. `visual_references_R`：绑定的角色与场景参考图路径。
- **评测真值（Ground Truth）**：
  1. `essential_events_E_source`（用于计算 **ER**）：从 L2/L3 中拆解出的原子级必要事件列表，每项包含 `event_id`、`description`、`source_l2_id`、`is_observable_in_video: true`。
  2. `fine_grained_units_FDC`（用于计算诊断指标 **FDC**）：从底层 L1 的 `Characters`、`Plot`、`Background Description` 中提取的细粒度可观察单元（分为四类：`character_emotion_motivation`、`subtle_interaction`、`causal_transition_cue`、`object_environment_state`），这些细节明确存在于完整故事 $S$ 中，但未显式写在简略分镜锚点 $A_i$ 里。

#### 4.2 Track B（开放式叙事拓展）数据结构
- **系统输入**：
  1. `short_storyline_L`：将完整故事 $S$ 压缩为 1~3 句话的简短故事概要，仅交代主要人物、初始情境、核心冲突及大致走向，抹除中间具体事件与分镜规划。
  2. `visual_references_R`：与 Track A 相同的角色与初始场景参考图。
- **评测真值与约束集（Story Constraints，用于计算 VNE / EVR / Groundedness）**：
  ```json
  {
    "storyline_L": "简要故事线文本...",
    "hard_constraints": {
      "character_constraints": [
        "Lester 与 Carolyn 处于婚姻冷战状态，不可突然表现为亲密恩爱"
      ],
      "premise_and_setting_constraints": [
        "故事发生在现代美式郊区家庭与通勤场景中"
      ],
      "core_conflict_constraints": [
        "围绕家庭成员间的疏离感与表面维持体面的矛盾展开"
      ],
      "outcome_boundary": "场景结束时家庭压抑感持续，未发生彻底和解"
    },
    "valid_expansion_dimensions": [
      "符合人物性格的具体行动与生活细节",
      "体现疏离或紧张关系的人物微互动与视线交流",
      "合理的跨空间转换与因果衔接动作",
      "与核心氛围一致的环境或道具交互状态变化"
    ]
  }
  ```

---

## 4. 两阶段防泄漏评测流水线规范（`scripts/evaluate_cineworldsim_bench.py`）

为严格符合《研究方案总纲》第 2.5 与 3.4 节要求，评测脚本必须物理隔离两个阶段的上下文：

1. **阶段一：盲态视频故事重建（Blind Video-to-Story Reconstruction）**
   - **输入隔离**：仅读取待测模型生成的视频文件 $V$（按统一帧率采样），**严禁在 Prompt 中传入源故事 $S$、故事线 $L$、分镜锚点 $A_i$ 或真值列表**。
   - **重建输出**：调用统一多模态大模型（VLM）输出结构化重建报告 `reconstructed_report.json`，包含：按时间顺序观察到的事件列表、人物动作与表情变化、人物互动关系、场景与关键物体状态变化、整体重建故事文本 $S_{\mathrm{rec}}$。
2. **阶段二：结构化对齐与指标计算（Metric Alignment & Scoring）**
   - **Track A 计算**：
     - **ER（Event Recall）**：逐项比对 `essential_events_E_source` 是否在 `reconstructed_report.json` 中得到表达：
       $$ER = \frac{\sum_{e \in E_{\mathrm{source}}} \mathbb{1}[e \text{ is expressed in } V]}{|E_{\mathrm{source}}|}$$
     - **SS（Story Similarity）**：从主要事件覆盖、事件顺序一致性、人物行为匹配度和状态变化一致性四个子维度，对比 $S_{\mathrm{rec}}$ 与源故事 $S$。
     - **FDC（Fine-grained Detail Coverage）**：逐项检验 `fine_grained_units_FDC` 中的细粒度单元是否在重建报告中被捕获。
   - **Track B 计算**：
     - **新增单元识别（$N_{\mathrm{exp}}$）**：对比 `reconstructed_report.json` 与简短故事线 $L$，剥离出 $L$ 已经显式给出的基础事实，将视频新展开的具体事件、动作、互动与状态推进切分为独立的新增叙事单元集合（总数为 $N_{\mathrm{exp}}$）。
     - **有效性逐项核验（$N_{\mathrm{valid}}$ 与 EVR）**：对每个新增单元逐一检查四项标准：① 具有实质叙事推进价值；② 前后逻辑连贯；③ 非画面或动作的无意义重复；④ 不违背 `hard_constraints` 中的任何条款。四项全通过则计入 $N_{\mathrm{valid}}$，计算有效拓展量 $N_{\mathrm{valid}}$ 与拓展有效率 $EVR = N_{\mathrm{valid}} / N_{\mathrm{exp}}$。

---

## 5. 推荐执行排期与脚本开发清单

| 优先级 | 步骤名称 | 涉及脚本（现有 / 待新增） | 阶段验收产物 |
| :---: | :--- | :--- | :--- |
| **P0** | 单片全链路贯通测试（以 `0001_American_Beauty` 和 `3075_SITTER` 为例） | 现有：`build_moviebench_event_descriptions...py`<br>新增：`scripts/build_l3_scene_segments.py` | 2 部电影完整的 L2 -> L3 场景段落 JSON 与边界切分质量检查报告 |
| **P1** | L4 全片档案与多场景故事弧构建 | 新增：`scripts/build_l4_story_arcs_and_movie_bible.py` | 2 部测试电影的 `movie_bible.json` 及提取出的 5~10 个候选 L4 故事弧 $S$ |
| **P2** | 双轨标注生成验证（Track A & Track B） | 新增：`scripts/build_cineworldsim_benchmark_annotations.py` | 5~10 个完整包含 $\{S, A_i, E_{\mathrm{source}}, \text{FDC}, L, \text{Constraints}\}$ 的样板评测题 |
| **P3** | 批量筛选与全集构建（扩展至 166 部电影精选集） | 新增：`scripts/select_cineworldsim_candidates.py` | 100~300 个高质量多场景故事弧的完整 CineWorldSim-Bench 数据集及参考图库 $R_i$ |
| **P4** | 双盲两阶段评测器开发与人工对齐验证 | 新增：`scripts/evaluate_cineworldsim_bench.py` | 自动评测流水线代码 + 10%~20% 样本的人工复核一致性报告 |
