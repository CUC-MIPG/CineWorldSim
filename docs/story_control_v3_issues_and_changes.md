# CameraMaster Track B：StoryControl v3 问题与修改报告

更新日期：2026-11-18（北京时间）  
案例：`artist_extreme_weather_typeA.json`  
范围：CameraMaster 的 Track B 叙事推演、节点覆盖、分镜与实验入口。ViMax 基线方法本身没有在本轮修改。

## 1. 版本与证据范围

- v2 案例输出：`/data1/AgentG/hhb/CameraMaster/outputs/vimax_track_ab_4030_storycontrol_v2/artist_extreme_weather_typeA/track_B/`。
- v3 案例输出：`/data1/AgentG/hhb/CameraMaster/outputs/vimax_track_ab_4030_storycontrol_v3/artist_extreme_weather_typeA/track_B/`。
- 关键审计数据：各输出目录中的 `narrative_simulation/all_banana/narrative_checkpoint.json`，以及完成后生成的 `story_control.json`、`story_coverage_report.json`、`narrative_nodes.json` 和 `video/event_video_summary.json`。
- 本文中的 v3 运行状态是 **2026-11-18 的一次运行中快照**，不是最终 40 步结论。该次主进程仍在运行。

## 2. v2 暴露的具体问题

| 问题 | 案例证据 | 后果 |
|---|---|---|
| 主角收到“可行动作清单”，但没有必须说明阻碍、策略及选择理由的决策结构 | 第 1 步观察同时列出继续画山脊、擦汗、检查颜料等选项；主角直接继续画 | 环境风险可以被主线目标压过，无法解释为何此刻先画而不处理颜料或身体状态。 |
| 角色初始公开环境和系统提示含完整故事 | 第 1 步记录的底层环境包含沙漠、海岸风暴、暴风雪与最终笔触 | 即使可见观察只描述局部，角色侧仍有未来剧情泄露通道。 |
| 终局动作发生过早 | 第 2 步在海岸风雨中落下“最终一笔”，状态机把画作标为完成 | 第 3 步进入暴风雪时画已经完成，无法再形成暴风雪中的最终创作。 |
| 节点覆盖把历史叙述当成本步动作 | B3 于第 3 步、B4 于第 6 步被标为覆盖；这两步实际都是撤笔和保护画布 | 形式上 B1–B4 全覆盖，语义上 B3、B4 没有发生所要求的动作。 |
| 固定 40 步候选导致终局后的重复状态 | v2 第 6 步标记终局，第 7–40 步大多是遮挡雪、观察能见度 | 候选数足够，但有效剧情密度低。`--max-nodes 30` 是选取上限，不保证 30 个独立剧情事件。 |

特别说明：v2 的 B1（沙漠中继续创作）有本步行动证据；B2（风暴海岸）有环境与创作证据，但提前完成画作破坏了后续顺序；B3、B4 属于错误覆盖。因此，v2 的“全部覆盖”不能作为故事完整性的有效证明。

## 3. v3 已实施的修改

### 3.1 开场信息与角色可见信息

`narrative/story_control.py` 通过 `opening_public_state()` 只从首个硬节点生成开场环境、角色身份及当前个人目标。`narrative/simulation_scaffold.py` 在主角第一次行动前，将世界公开环境、角色环境、角色目标及角色系统提示替换为该开场信息。完整故事仍供隐藏 StoryControl 和最终完成评估使用。

每轮世界模型输出可观察环境、变化、风险、`condition_details`（条件程度及实际影响）、历史事件，并在内部保留 `available_affordances`。发送给主角的观察会剔除 `available_affordances`。例如应描述颜料“已变稠到画笔可能留下断续笔触，精细山脊绘制会受影响”，由主角自己决定是否先调色。

### 3.2 主角自主判断与世界状态匹配

角色动作输出增加 `obstacles` 和 `strategy`：主角自行说明哪些当前条件妨碍个人目标、影响程度、解决策略、选择的一步动作及其理由。世界模型的 `match_action()` 在内部把该动作与自身 affordance、当前条件和故事先后约束比较；内部清单不是穷举，允许新的可行办法。

匹配结果及 `permitted_state_changes` 交给状态机进行动作接受和世界更新；审计日志写入 `action_alignment`、`action_accepted`、`actual_step_events`。下一轮观察基于更新后的世界状态生成。未通过可行性检查的动作不会被状态机接受。

### 3.3 覆盖证据与顺序约束

节点覆盖改为引用本步**被接受**的具体动作和本步事件，不再把角色理由、内心独白、历史事件摘要当成本步证据。每步最多新增一个硬节点，前置节点必须在更早步骤覆盖。对本案例的“最终笔触”，新增过早落笔拦截，并要求终局覆盖时本步实际包含落笔动作；单纯撤笔或回忆不能覆盖 B4。

StoryControl checkpoint 版本升至 `3`，旧 v2 checkpoint 不可混用；须用新的输出根目录运行。

### 3.4 实验脚本

- `scripts/run_vimax_track_ab_pipeline.py` 修复了 Track A/B 参数传递错误；Track B 的 StoryControl 和完整覆盖开关会按命令行生效。
- 视频汇总要求 `success == total`、`failed == 0` 且拼接成功，防止部分视频被联合脚本标为完成。
- `scripts/run_joint_cameramaster_vimax_case.sh` 默认 v3、40 步、最多 30 节点；阶段完成标记按 v3 与运行参数隔离，并在启动前拒绝旧版 checkpoint。
- `scripts/run_artist_extreme_weather_comparison_resume.sh` 指向 v3 案例目录和新的比较输出目录，当前只评估 Track B。
- ViMax 原始 pipeline 没有因这些修改而改变；联合脚本仍分别使用 `mifage` 与 `Vimax` 环境。

## 4. 验证结果与目前发现的 v3 问题

代码同步到服务器后，四个相关 Python 入口完成编译检查，两份 Bash 实验脚本完成语法检查。针对性离线验证确认：过早最终落笔会被拒绝；历史“最终一笔已完成”不能凭撤笔动作覆盖 B4。**尚未完成完整在线质量验收。**

在 2026-11-18 读取的 v3 运行快照中：checkpoint 版本为 3，已完成 22/40 步，22 步均有内部动作匹配记录，动作均被接受，但 `covered_beat_ids=[]`，B1–B4 全部未覆盖；当时尚无 `narrative_nodes.json` 和最终视频汇总。继续读取的第 23 步仍停在沙漠中的风险应对。

这揭示了新的推进问题：第 1 步观察已经描述画家正在沙漠中作画，但本步主角选择停笔找阴影；覆盖判定只接受本步实际动作，因而 B1 没有覆盖。随后主角多次擦汗、坐下、找阴影或呼救。状态更新谨慎地写成“动作已尝试，但视线变清晰、降温或恢复未确认”，世界模型继续分发同一阻碍，主角再次选择准备或缓解动作，形成停滞循环。此时不能宣称 v3 已修复完整故事推进。

另外，角色模型有时返回对象形式的 `obstacles`，当前记录层将对象转为字符串；可读性尚可，但不利于稳定的结构化分析。

## 5. 后续修复与验收标准

### 2026-11-18 最终运行补记

该次 v3 运行最终达到 40/40 步，`covered_beat_ids=[]`，B1–B4 全部缺失。第 36–40 步仍在沙漠中反复尝试转动画架、寻找阴影和擦汗；第 40 步的擦汗动作虽被接受，结果却说未确认汗水被擦掉或视线改善。`story_coverage_failure.json` 明确列出四个缺失节点，视频生成按设计被阻止；没有产生可用的 v3 Track B 视频或正式评估成绩。

针对这次失败，另起 StoryControl v4：允许首轮已公开、明确可见的开场活动成为 B1 证据；确认已接受的直接身体动作的短时效果；向角色反馈重复策略的实际结果；增加有因果依据的世界外部变化环节。v4 的代码与局部检查已完成，**完整 40 步质量仍待新目录重跑验证**。v3 目录不能直接用于 v4 恢复。

本节是**待办**，不是已经实现的功能。

1. 将缓解动作的可观察结果写成可检查的状态转移。例如擦汗后明确记录“汗已离开眼部但高温仍在”，避免所有效果长期处于“未确认”。如果没有效果，也应给出新证据，使角色改变策略。
2. 为重复阻碍设停滞检测与世界反馈：连续多轮同一风险、同类准备动作且状态不变时，世界模型必须提供真实的新观察或说明当前方案无效；不直接命令角色下一步行动。
3. 将 `obstacles` 标准化为结构化的观察、影响和主角自行判断的严重程度，保持角色决策与世界内部匹配可以审计。
4. 对 B1–B4 逐项检查本步行动证据、画作未完成/完成状态和场景顺序；对 B3、B4 严格核实暴风雪阶段的创作与最终笔触。
5. 完成 v3 的 40 步运行后，检查选中节点数、缺失节点、重复动作比例、视频任务成功率和拼接结果；质量合格后再运行独立目录中的 Track B 定量评估。旧 v2 指标不可直接覆盖为 v3 成绩。

## 6. 复现实验

```bash
cd /data1/AgentG/hhb/CameraMaster

/data1/AgentG/anaconda3/envs/mifage/bin/python -u scripts/run_vimax_track_ab_pipeline.py \
  --benchmark-root /data1/AgentG/hhb/ViMax/vimax_benchmark \
  --case artist_extreme_weather_typeA.json \
  --track B \
  --output-root outputs/vimax_track_ab_4030_storycontrol_v3 \
  --backend all_banana \
  --max-steps 40 \
  --max-nodes 30 \
  --max-reference-images 4 \
  --track-b-story-control \
  --track-b-require-full-coverage \
  --strict-online \
  --retries 3 \
  --log-level INFO
```

同目录重跑会读取 v3 叙事 checkpoint。正在运行的进程不应同时再启动第二份相同输出目录的命令；观察进度可只读检查 checkpoint。若更改叙事规则或 checkpoint 格式，应使用新的输出目录和版本号。
