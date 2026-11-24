# ViMax 与 A²RD：Benchmark 评估体系整理

本文整理两篇论文中用于实验对比的 benchmark、指标实现、LLM/VLM 评测协议及复现注意事项。重点区分自动指标、人工偏好/打分，以及生成过程中用于候选筛选的 judge，避免把它们当作同一套指标。

## 1. ViMax: Agentic Video Generation

论文：`2606.07649v2.pdf`，重点参见 Section 3、Appendix A.2-A.4 及 Figures 12-15。

### 1.1 ViMax-Bench 的任务与数据

- 35 个多镜头故事，涵盖 Medium 与 Long 两类叙事长度；每个故事含 2-4 个场景、8-16 个 shots。
- 每个 shot 提供首帧描述和运动 prompt，目标是 hard-cut 多镜头视频生成，不是单镜头连续视频。
- 三种一致性挑战：
  - **Character Persistence**：跨环境、灯光、构图变化维持角色身份。
  - **Background Persistence**：跨视点变化维持室内空间几何与布局。
  - **Multi-Person Interaction**：多人共同动作时仍能区分不同角色。
- 论文将故事分成 Medium/Long，Table 5 给出分布：A/B/C 类型合计分别为 12/23/35 个。

### 1.2 视频生成的自动指标：CC、IC、GC

ViMax-Bench Table 1 的三个分数都由 **ViCLIP shot embedding 的 cosine similarity** 自动计算；论文说明无需专家标注，分数越高表示跨 shot 特征越稳定。

1. 将同一故事中每个生成 shot 编码成一个 video-level ViCLIP feature。
2. 对同一故事内的 shot 两两计算 cosine similarity。
3. 按 shot pair 关系聚合：
   - **IC (Intra-scene Consistency)**：同一 scene 内的 shot pairs 平均。
   - **CC (Cross-scene Consistency)**：不同 scene 之间的 shot pairs 平均。
   - **GC (Global Consistency)**：故事内所有 shot pairs 平均。
4. 分别对 Medium、Long、Overall 报告结果。

Character Persistence、Background Persistence、Multi-Person Interaction 是 benchmark 的挑战类型标签，不是额外的人工评分项。

### 1.3 ViMax-Bench 的成对人评

人评用于补充 CC/IC/GC 无法覆盖的语义执行和整体观感：

- 覆盖全部 35 个故事、438 个生成 shots。
- ViMax 分别与五个 baseline 对比，得到 35 × 5 = 175 个唯一比较 pair。
- 两段视频匿名并排展示，左右位置随机；15 位有 AI 相关背景的评审参与。
- 每个唯一 pair 由 3 位评审独立判断，共 525 次 pairwise judgment。
- 三个标准分别选出更优视频：
  - **CSC (Cross-Scene Consistency)**：角色、物体、衣着、身份特征、视觉风格、动作/行为在 shots 和场景间的一致性。
  - **SF (Semantic Following)**：主体、动作、交互、环境和故事进展是否符合输入。
  - **AQ (Aesthetic Quality)**：画面质量、构图、视觉吸引力与电影感。
- Table 2 报告 ViMax 相对各 baseline 的胜率；Avg 是三个标准胜率的算术平均。

### 1.4 Novel-to-Video Planning 的 GPT-5.4 LLM Judge

这是**规划/storyboard 质量评估**，与 ViMax-Bench 成品视频评价分开。输入是小说和结构化 storyboard，不生成视频。

- 评测集：从 NarrativeQA 的 Project Gutenberg 部分随机抽取 50 部小说。
- 所有被比较方法的 LLM backbone 固定为 GPT-5.4；MovieAgent、AniMaker 也在关闭视频合成后以 storyboard 参评。
- Judge 按 **1-5 分**评五个 storyboard 维度，并分别给证据化理由：
  - **CBC (Character Behavior Consistency)**：角色动机、情绪、动作、身份与互动逻辑是否跨镜头一致。
  - **NC (Narrative Coherence)**：故事是否形成逻辑连接的开端、发展和推进。
  - **PPR (Plot Pacing and Rhythm)**：事件进展与镜头分配是否有有效节奏，平衡铺垫、行动和回报。
  - **VSU (Visual Specificity and Shot Usability)**：主体、动作、环境、摄影机信息是否具体到可执行。
  - **STQ (Scene Transition and Continuity Quality)**：相邻镜头/场景的空间、时间、动作过渡是否顺畅且合乎逻辑。
- 另用 **NF (Narrative Faithfulness)** 评 storyboard 是否忠实于小说的场景、人物、环境和关键情节，是否加入无依据内容。NF judge 可访问小说元信息、全文和当前 storyboard batch。
- 报告 Avg = 五个 storyboard 维度加 NF 六项的算术平均。
- 为统一比较单元：一般 baseline 只评其前 20 个 sub-storyboards；ViMax 前 100 个切为五组，每组 20 个独立评分后取平均。
- Figure 12-14 的 prompt 要求机器可读 JSON、1-5 整数分、按维度给出证据和 strengths/weaknesses/suggestions；不可评时需说明并保守打分。

### 1.5 不应混淆的 ViMax 候选筛选 judge

ViMax 生成流程中的 VLM best-of-k 是质量控制，不是 Table 1 的 benchmark 指标。图像候选按角色一致性、空间一致性、目标描述准确性选优；论文报告视频候选也会综合视觉保真、叙事一致性、shot specification adherence 选优。若复现实验比较系统能力，需将这一候选选择策略作为方法实现的一部分说明，不能把筛选分数直接当最终评测分。

### 1.6 ViMax baseline 公平性说明

Appendix A.3 说明，ViMax-Bench 比较中 MovieAgent 和 AniMaker 保留原生规划/协同流程，但使用 Gemini-3-Pro-Image 与 Veo 3.1 作为图像/视频 backbone，并在方法名标出 backbone。Nova-to-Video planning 比较则统一 GPT-5.4 LLM backbone。复现表格时应保留这两种设置的区分。

## 2. A²RD: Agentic Autoregressive Diffusion for Long Video Consistency

论文：`2605.06924v1.pdf`，重点参见 Section 4、5.2、5.3、6.3-6.4、Appendix C.1-C.2、E.5。

### 2.1 LVbench-C 的任务与数据

LVbench-C 专门测试实体或地点跨长视频出现、消失、重现时的一致性，并允许重现时发生故事解释得通的状态演变：

- **Evolving Character States**：角色衣着、外貌或身体状态变化。
- **Evolving Object States**：物体位置、朝向或损坏/状态变化。
- **Evolving Environment States**：地点细节逐步揭示或环境随事件演变。
- 3 分钟和 5 分钟设置各 50 个场景脚本；10 分钟设置 25 个。脚本由多个简洁 scene descriptions 组成，按分项合计为 125 个。
- 构造约束包括：有因果且自然的故事；相关实体/地点消失至少 10 个 segments 后再现；状态变化要有现实活动和可观察线索。作者还做人工审核、去重和 MLLM 自检。

**与 MovieBench 层级的区分：** A²RD 的 `segment` 是 benchmark 故事中一个有序的视频/场景片段，不是 MovieBench 的 L1/L2/L3/L4 标签。CameraMaster 若每个生成片段为 5 秒，应以实际生成片段计 segment；可用 L2 定义事件内容、L1 提供证据、L3 提供局部上下文、唯一 L4 提供全片顺序。目标实体在首次出现后需至少 10 个 segment 不出现再重现，因此最低结构约 12 个生成片段（若每段 5 秒则约 60 秒）。这只是循环重现条件的结构下限，不代表复现了官方 3/5/10 分钟设置；多个 L3 arcs 也只有在时间与叙事关系经核验后才能组成同一长序列。

### 2.2 LVbench-C 核心指标：MLLM cyclic consistency judge

Section 6.3 / Appendix E.5 对角色、物体、环境分别判定。对每个待评 scene 和维度 `d ∈ {char, obj, env}`，向 MLLM 提供：

- 被比较的参考图像及图像标签；
- 过去相关 scene descriptions；
- 当前 scene description；
- 当前场景生成画面。

Judge 执行三步：

1. **识别视觉差异**：按维度比较参考画面与当前画面。
2. **解释差异**：判断变化是否由故事合理解释，还是 unexplained。
3. **严格判违规**：只将明显、无解释的差异标为 `0`；否则记为 `1`。

各维度 prompt 的判定边界：

| 维度 | 重点比对 | 判为违规的明显变化 | 明确忽略的变化 |
|---|---|---|---|
| Character | 身份、脸、头发、体型、整体外观 | 换成不同的人；脸/体型明显不同；物种变化 | 轻微衣物污渍、配饰、光照、姿势 |
| Object | 关键物体身份、形状、颜色、大小 | 完全不同物件；形状明显错误；颜色剧变 | 轻微位置变化、磨损、光照、部分遮挡 |
| Environment | 同一物理空间的建筑结构和空间布局 | 地点、建筑或布局明显不同 | 摄影机角度、剧情驱动事件、轻微光照变化 |

每份 prompt 仅输出 JSON，包含差异说明、expected/unexplained 分类、对应一致性二值字段和简短理由。环境字段命名为 `background_consistent`，角色/物体分别为 `character_consistent` / `object_consistent`。

对维度 `d`，论文定义的 ratio 为全部 scene 二值 judge 值的平均：

`Consistency_d = (1 / |S|) * Σ_i MLLM_const_vfy(S_i, F_i, F_ref, S_ref, d)`

即 ratio × 100% 可转成百分数。它是“明显且无故事解释的状态漂移率”的反向指标，不是逐帧像素一致率，也不等同于视觉 embedding 相似度。Section 6.3 的 5 分钟实验还按 scene window size（1-40）展示分数随长程跨度变化；Section 6.4 用同一 judge 评估 10 分钟场景。

作者人工核查了一部分 judge 输出，称对明显错误的判断约 80% agreement；同时承认 subtle inconsistencies 可能被漏检，因为 prompt 有意只标明显违规。

### 2.3 A²RD 的常规自动视频指标

这些是 Section 5.2 的一般视频评估，不应与 LVbench-C 的 cyclic judge 混为一谈：

- **Semantic Alignment**：ViCLIP 对视频与 scene description 编码后计算 cosine similarity。
- **Narrative Coherence**：Gemini 3 Pro 根据原故事与生成视频评 0-1；采用 Self-Consistency，多次评估/聚合以稳定判断，关注故事、实体与环境进展、因果逻辑，并惩罚重复或不连贯。
- **Inter-shot Character / Environment Consistency**：先由 MLLM 将需要维持同角色/环境状态的场景分组；YOLOv8 从各 scene keyframe 分割角色/背景区域，提取 DINOv3 embedding；对组内跨场景 pairs 求 cosine similarity，再跨组平均。
- **Inter-shot Motion Consistency**：为连续场景边界抽取前一片段最后 10 帧及后一片段最初 10 帧，计算 VBench `motion_smoothness`。
- **Intra-shot Subject / Environment Consistency**：直接采用 VBench 对应指标。

论文 Section 5.2 称总共评估“nine metrics”，但其列举和主结果表实际只展示上述七项（语义 1、叙事 1、跨镜头 3、镜头内 2）。Appendix B.1 另列六项 general video quality VBench 指标（Subject Consistency、Background Consistency、Aesthetic Quality、Imaging Quality、Temporal Flickering、Motion Smoothness），但没有明确解释“nine”的计数口径。复现或转述时宜按表格实际列出的名字报告，并注明这一处不一致，不自行补猜缺少的两项。

### 2.4 A²RD 人工评测

Section 5.3 / Appendix C.2 的人评对象是 VBench-Long 视频，不是 LVbench-C cyclic judge：

- 7 位评审；视频与方法匿名、随机顺序；每个样本至少 2 人评分，按评审平均。
- 六个维度均为 1-5 Likert：Character Consistency、Object Consistency、Environment Consistency、Transition Smoothness、Narrative Coherence、Reference Consistency（没有参考图时 N/A）。
- 锚点为 1 = Very poor、3 = Acceptable/OK to Watch、5 = Excellent。

## 3. 两套体系的关键差异

| 方面 | ViMax | A²RD |
|---|---|---|
| 主要 benchmark | 35 个多镜头故事的 ViMax-Bench；另有 50 部小说的 planning 评测 | LVbench-C 长时长、循环出现/消失/重现的角色、物体与环境；另评公开视频 benchmark |
| 视觉一致性自动指标 | ViCLIP shot embedding 两两相似度，聚合 CC/IC/GC | 常规指标含 ViCLIP、DINOv3 + YOLOv8；LVbench-C 核心另用严格阈值 MLLM 二值判定 |
| LLM 评测对象 | GPT-5.4 对 storyboard 规划质量及小说忠实度打 1-5 分 | MLLM 对 LVbench-C 当前 scene 相对历史参考的差异作 expected/unexplained 和二值判定；Gemini 3 Pro 单独评 narrative coherence |
| 人评形式 | ViMax 与 baseline 成对偏好，输出胜率 | 六个维度各自 1-5 Likert 评分 |
| 适合回答的问题 | 多镜头整体视觉稳定吗？计划是否具体、连贯、忠于原作？ | 隔很久重现的实体/地点是否保留身份与状态？明显漂移是否随生成长度累积？ |

## 4. 对照实验复现建议

1. **先固定被测产物与输入信息。** 每个系统使用相同 story/script、scene/shot 数量、参考图像约束和输出时长；记录系统是否能访问历史帧、是否有 best-of-N 筛选。
2. **不要混合规划评测和成片评测。** Storyboard 的 CBC/NC/PPR/VSU/STQ/NF 应在文本结构产物上打分；视频 CC/IC/GC 或 cyclic consistency 应在生成视频/帧上评估。
3. **复刻 judge 时锁定模型与 prompt 版本。** 保存完整 system/user prompt、图片采样策略、温度、重复评估次数、JSON parser 规则和失败重试策略。不同模型的 0/1 judge 阈值不应默认可互换。
4. **为 LLM judge 做人工校准。** 抽取不同类型和不同分数段样本双人标注，报告 agreement；对 A²RD 式严格阈值同时统计明显错误和细微错误，明确其漏检边界。
5. **避免只报总分。** 分别报角色/物体/环境、跨场景/场景内、短/长时长；提供样本数、pair 数、平均及方差/置信区间。ViMax 的 CC/IC/GC 与 A²RD 的 binary consistency ratio 是不同定义，不应横向直接比较绝对值。
6. **透明报告已知歧义。** ViMax baselines 使用适配 backbone 的设置要明示；A²RD 的“nine metrics”与实际列出的七项不一致，且 LVbench-C 正文一处写 120 个场景而分项数量（50 + 50 + 25）及 Table 1 是 125，需结合实际 release 代码/数据确认。

## 5. 来源定位

- **ViMax**, `2606.07649v2.pdf`：Section 3.1-3.3；Appendix A.2（PDF 第 11 页附近）；A.3-A.4；Figures 12-15（PDF 第 18-21 页）。
- **A²RD**, `2605.06924v1.pdf`：Section 4、5.2-5.3、6.3-6.4；Appendix B.1、C.1-C.2、E.5（PDF 第 57-59 页）。
- DirectorBench 仓库中的 `DirectorBench-master/README.md` 属于本地评估工具说明，不是上述两篇论文的 benchmark 协议，因此未混入本文的论文指标对照表。
