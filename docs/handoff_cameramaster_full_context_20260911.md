# CineWorldSim Full Context Handoff

更新时间：2026-09-11（Asia/Shanghai）

这是 CineWorldSim 的项目级交接文档。它区分已实现、已验证和目标/计划，避免新会话把设计目标误认为运行事实。

## 1. 项目目标

CineWorldSim 是事件级、多智能体、世界状态感知的电影视频生成系统。

```text
Story -> Initialization -> Simulation -> World State -> Narrative Nodes -> Path -> H3 Video -> Auxiliary Shots -> Concat
```
## 12. 时长、帧数、视频后端和可视化

固定帧数映射位于 narrative/pipeline_schemas.py：

```text
5 seconds  -> 124 frames
10 seconds -> 243 frames
```

## 9. NarrativeNodeSelector 和 PathOrchestrator

节点选择入口是 scripts/run_narrative_node_selector.py，核心代码是 agents/narrative_node_selector.py。

流程：从 step_logs 构建 role candidates；按同 step、同 location、直接交互条件形成 event candidates；使用规则或 LLM 评分；输出 selected_nodes 和 rejected_nodes。

评分关注 dramatic conflict、goal progress、emotional turning point、information reveal、visual potential、causal importance 和 redundancy penalty。

PathOrchestrator 代码：

```text
agents/path_orchestrator_agent.py
agents/path_orchestrator_mcts_agent.py
```

路径阶段只排序，不重新拆分或合并节点。单角色 node 保持一个单角色视频事件；selector 已合法形成的多人 node 保持多人事件。

主要产物：

```text
path_orchestration/narrative_nodes.json
path_orchestration/path_candidates.json
path_orchestration/path_llm_direct/narrative_path_plan.json
path_orchestration/path_llm_direct/path_plan.json
path_orchestration/path_mcts/narrative_path_plan_mcts.json
path_orchestration/path_mcts/path_plan_mcts.json
path_orchestration/path_mcts/mcts_trace.jsonl
```

path-mode both 会生成 LLM 和 MCTS 两条路径；完整视频入口通常使用 LLM path，具体以 run manifest 为准。

## 10. VideoAgent、story_description、dialogue 和 H3 skill

代码：

```text
agents/video_agent.py
agents/event_story_tool.py
agents/event_dialogue_tool.py
agents/video_shot_planning_tool.py
agents/h3_prompt_adapter.py
```

处理链路：

```text
video_description
  -> EventStoryTool / LLM
  -> story_description
  -> EventDialogueTool / LLM
  -> dialogue
  -> reference selection
  -> shot/frame planning
  -> H3 skill prompt
  -> MiniMax-H3 vLLM
```

video_description 保留用于追溯原始节点，可能包含 action、target、reason 或推理式表达。

story_description 只给视频 prompt 使用，应按时间顺序写成可拍摄剧情：谁做了什么、对谁说了什么、对方如何回应、动作如何推进和结束；不能只是“角色选择某动作，因为某理由”。

dialogue 规则：没有对白写 none；不能让每个角色机械说一句；角色可以在一个事件中多次说话；对话按剧情时间顺序；对话服务于冲突、信息和动作；无法识别角色时使用 Character1、Character2 等默认名。

H3 声音字段：

```text
dialogue:
bgm:
overall_soundscape:
non_diegetic_music:
```

Skill policy 在 configs/agent_skill_policies.yaml。VideoAgent 允许 h3-prompt-writing、cameraatist-event-video-prompt、cameraatist-event-multishot-prompt。

MINIMAX_H3_PROMPT_STYLE=official_ref 时优先 h3-prompt-writing；compact 模式优先 CineWorldSim 内置 prompt。skill 后通常有 subject_definitions、summary、retention_analysis、detailed_description、overall_soundscape 和 non_diegetic_music。

多镜头写在 detailed_description 的 [Shot 1]、[Shot 2] At ... 中；event-shot-count=4 是一次视频请求内部的四段镜头组织，不是四次视频调用。
当前只允许这两个组合。

典型 MiniMax-H3 vLLM 环境：

```text
VIDEO_GEN_PROVIDER=minimax_h3
MINIMAX_H3_PROMPT_STYLE=official_ref
MINIMAX_H3_NUM_FRAMES=243
MINIMAX_H3_VLLM_ENABLED=true
MINIMAX_H3_VLLM_API_URL=http://127.0.0.1:9001/v1/videos/sync
MINIMAX_H3_VLLM_TIMEOUT_SEC=3600
MINIMAX_H3_CUDA_VISIBLE_DEVICES=2,3
```

历史上 vLLM 出现过超长请求、async output timeout、Future InvalidStateError、HTTP 500/504。这和 LLM API fallback、Banana SSL 故障是不同链路。

典型视频目录：

```text
event_video_pipeline_minimax_h3_vllm_multishot/
  event_video_*.mp4
  event_video_tasks.json
  event_video_summary.json
  final_orchestrated_events.mp4
```

可视化脚本：

```text
scripts/generate_narrative_world_pipeline_html.py
scripts/generate_full_pipeline_stage_visualization.py
```

常见可视化产物：

```text
visualizations/combined_overview.html
visualizations/full_stage_pipeline_visualization.html
```

展示故事、角色、世界、step、world state、delta、keyframe、nodes、path、H3 prompt、视频任务和 concat 状态。
## 13. MovieBench 数据集上下文

时间连续性脚本：

```text
scripts/process_moviebench_clip_metadata.py
```

当前规则：

```text
abs(current.start_ms - previous.end_ms) < 1000
```

当前 clip 的 start_ms 与上一个按时间排序 clip 的 end_ms 相差不到 1 秒才连续。若不连续，is_continuous_with_prev=false、prev_file_name=null，同时保留 prev_chronological_file_name 用于审计时间顺序。

连续视频副本脚本：

```text
scripts/build_moviebench_continuous_dataset.py
```

每个 clip 保留 Characters、Style Elements、Plot、Background Description、Camera Motion、clip_start_tc、clip_end_tc、duration_sec，并按 sequence JSON 聚合连续片段。

连续 shots 合并事件脚本：

```text
scripts/build_moviebench_event_descriptions_from_continuous_shots.py
```

目标是按时间顺序把同一连续片段所有 shot 合并为一个具体总事件，同时保留每个 shot 的人物、动作、背景、摄影和风格，使总事件能够重新拆解回多个 shot。role_in_event 表示某 shot 在总事件中的作用，是 LLM 输出，不是原始字段。
## 14. 实验协议和论文上下文

实验文档：

```text
iclr/experiment_implementation_protocol.md
```

两条互补主线：

```text
Narrative representation / story visualization
  story manifest -> narrative units -> image sequence -> metrics

Storyboard-anchored multi-shot video
  story manifest -> narrative units + path -> videos -> state/video metrics
```

协议规划的 baseline 包括 Direct prompting、Storyboard-only、BookWorld-style agents、故事可视化 baseline、故事视频 baseline、STAGE 和 CineWorldSim full。

协议规划的指标包括 narrative quality、story visualization consistency、shot quality、cross-shot consistency、event fidelity、world-state consistency、character-state consistency、object-state persistence、temporal/causal ordering 和 Narrative Unit Coverage。

协议中的建议 runner 和 metric 脚本不代表全部已经实现，执行前必须检查实际文件。
## 15. 关键历史运行结果

四 genre manifest：

```text
outputs/four_genre_demos_15nodes_20260816/four_genre_demo_manifest.json
```

记录 status=done，包含 suspense、adventure、action、urban 四个 demo，并记录 pipeline、path、video summary 和 visualization 路径。这是早期 15-node 运行。

两个三主角 adventure demo：

```text
outputs/two_adventure_new_pipeline_20260828/two_demo_runner_manifest.json
```

关键设置：all_banana、path-mode=both、max-steps=15、node-min/max=15、node-llm-scoring、multi-shot-events、event-shot-count=4、max-reference-images=4、no-use-previous-tail-frame、无 previous audio、dialogue-backfill、insert-auxiliary-videos、video-reference-mode=semantic_assets。

manifest 已记录：

```text
demo1: 21 success / 21 total, concat_ok=true
demo2: 16 success / 16 total, concat_ok=true
```

总数包含主事件和辅助镜头，不能等同于 selected node 数。

Keyframe-only manifest：

```text
outputs/keyframe_only_event_video_20260908/01_demo1_shared_space/keyframe_only_event_video_pipeline_manifest.json
```

已验证：selected_node_count=25、event_task_count=25、auxiliary_task_count=1、video_success=26、video_failed=0、video_skipped=0、concat_ok=true、final_video_exists=true、event_reference_mode_ok=true。

这证明 keyframe-only 路径曾成功生成 25 个主事件视频和 1 个辅助视频，但不是 45 个 selected nodes。
## 16. gpt55 pre-video 结果

目录：

```text
outputs/gpt55_pre_video_demo1/all_banana
```

检查报告：

```text
/tmp/cameramaster_final_gpt55_check/pre_video_health_report.json
/tmp/cameramaster_final_gpt55_check/pre_video_recovery_plan.json
```

报告为 status=healthy，结构上确认 15 个 simulation steps、16 个 world states、15 个 world deltas、45 个 event keyframes、3 个 portraits、location 四视图、item 图、node selection、LLM path 和 MCTS path。

限制：该目录在结构化 fallback 记录接入前生成。healthy 只能说明当前 JSON/资产结构完整，不能反证 2026-09-10 日志中的 step 13 没有 503/fallback。

对以下目录检查 mp4、webm、mov 时没有发现视频文件：

```text
outputs/gpt55_pre_video_demo1
outputs/gpt55_pre_video_demo1_repaired_step13
outputs/gpt55_pre_video_demo1_repaired_step13_retry
```

因此 gpt55 demo1 当前只能确认 pre-video 产物，不能声称已经生成最终视频。
## 9. NarrativeNodeSelector 和 PathOrchestrator

节点选择入口是 scripts/run_narrative_node_selector.py，核心代码是 agents/narrative_node_selector.py。

流程是：从 step_logs 构建 role candidates；按同 step、同 location、直接交互条件形成 event candidates；使用规则或 LLM 评分；输出 selected_nodes 和 rejected_nodes。

评分关注 dramatic conflict、goal progress、emotional turning point、information reveal、visual potential、causal importance 和 redundancy penalty。

产物：

```text
path_orchestration/narrative_nodes.json
```

PathOrchestrator 代码：

```text
agents/path_orchestrator_agent.py
agents/path_orchestrator_mcts_agent.py
```

路径阶段只排序，不重新拆分或合并节点。单角色 node 保持一个单角色视频事件；selector 已合法形成的多人 node 保持多人事件。

主要产物：

```text
path_orchestration/path_candidates.json
path_orchestration/path_llm_direct/narrative_path_plan.json
path_orchestration/path_llm_direct/path_plan.json
path_orchestration/path_mcts/narrative_path_plan_mcts.json
path_orchestration/path_mcts/path_plan_mcts.json
path_orchestration/path_mcts/mcts_trace.jsonl
```

path-mode both 会生成 LLM 和 MCTS 两条路径；完整视频入口通常使用 LLM path，具体以 run manifest 为准。
## 10. VideoAgent、story_description、dialogue 和 H3 skill

VideoAgent 及工具：

```text
agents/video_agent.py
agents/event_story_tool.py
agents/event_dialogue_tool.py
agents/video_shot_planning_tool.py
agents/h3_prompt_adapter.py
```

处理链路：

```text
video_description
  -> EventStoryTool / LLM
  -> story_description
  -> EventDialogueTool / LLM
  -> dialogue
  -> reference selection
  -> shot/frame planning
  -> H3 skill prompt
  -> MiniMax-H3 vLLM
```

video_description 保留用于追溯原始节点，可能是 action、target、reason 或推理式表达。

story_description 只给视频 prompt 使用，应是按时间顺序展开的可拍摄剧情：谁做了什么、对谁说了什么、对方如何回应、动作如何推进和结束；不能只是“角色选择某动作，因为某理由”。

dialogue 规则：没有对白写 none；不让每个角色机械说一句；角色可以在一个事件中多次说话；对话按剧情时间顺序；对话服务于冲突、信息和动作；角色无法识别时使用 Character1、Character2 等默认名。

H3 声音字段：

```text
dialogue:
bgm:
overall_soundscape:
non_diegetic_music:
```

Skill policy：

```text
configs/agent_skill_policies.yaml
```

VideoAgent 允许 h3-prompt-writing、cameraatist-event-video-prompt、cameraatist-event-multishot-prompt。Skill runtime 在 agents/skills/ 下。

MINIMAX_H3_PROMPT_STYLE=official_ref 时优先 h3-prompt-writing；compact 模式优先 CineWorldSim 内置 prompt。skill 前输入 story_description、dialogue、声音信息、资产映射和 shot plan；skill 后通常有 subject_definitions、summary、retention_analysis、detailed_description、overall_soundscape、non_diegetic_music。

多镜头在 detailed_description 中写 [Shot 1]、[Shot 2] At ...。event-shot-count=4 是一次视频请求内部的四段镜头组织，不是四次视频调用。
## 11. 视频参考图、尾帧和辅助镜头

VideoAgent 支持三种 reference mode：

```text
event_keyframe_only
semantic_assets
all_allowed
```

event_keyframe_only 只使用事件关键帧；semantic_assets 使用角色初始图、location 图和事件提及的 item 图；all_allowed 允许关键帧和语义资产共同进入候选集合。

参考图选择规则：

- 单角色事件默认最多一个角色参考。
- 多人物图只有事件确实需要多人直接交互时才允许。
- location 图按事件 location 匹配。
- item 图只有事件文本实际提及时才选择。
- 不为了凑满 max_reference_images 而强行添加图像。
- 去重并记录 reference_selection、excluded_reference_paths 和选择原因。

上一事件尾帧默认关闭：

```text
use_previous_tail_frame=false
```

只有显式开启 --use-previous-tail-frame，VideoAgent 才让 LLM 比较前后剧情、人物关系、location、空间关系和物品状态。场景切换、时间跳跃、角色变化、空间不连续或物品状态冲突时拒绝使用；LLM 失败时默认不使用。

辅助镜头包括 location 首次出现前的 establishing shot 和 world state 更新前后的 world_transition shot。辅助 task 通常为 5 秒、124 帧，并参加最终 concat，所以 task 数可能大于 selected node 数。

相关代码：

```text
agents/video_agent.py
narrative/asset_reference_selector.py
```
## 8. WorldDelta、WorldState 和事件关键帧

WorldDeltaJudge 位于 narrative/world_delta_judge.py，WorldStateTracker 位于 narrative/world_state_tracker.py。

只有事件文本明确造成 location 或 item 的可观察物理变化时，才更新世界图像。可更新变化包括颜色、形状、外观、纹理、结构、损坏、开闭配置、淹水和燃烧等。

发现、注意、思考、情绪变化、拥有、移动或叙事重要性变化本身不能触发图像刷新，除非同时有可观察的物理变化。

location 更新时由 LLM 选择 wide、medium、close 或 extreme_close。更新图只能描述事件明确造成的变化，不出现人物、人群、行人和无关元素，并参考上一轮同视图或初始化图。item 更新使用上一轮 item 图或初始 item 图。

正确顺序：

```text
当前 step 推演
  -> 必要的 world state 更新
  -> EventKeyframeRenderer 生成事件关键帧
```

事件关键帧参考当前事件文本、角色初始图、location 图、事件提及的 item 图和必要的 world update 图。

单角色事件只使用必要的单角色参考；只有合法多人交互事件才允许多个相关人物参考。无关人物、人群和行人由 prompt 约束禁止。

主要产物：

```text
world_state_step0.json ... world_state_step15.json
world_deltas/world_delta_step1.json ... world_delta_step15.json
narrative_artifacts/world_updates/locations/*
narrative_artifacts/world_updates/items/*
narrative_artifacts/event_keyframes/step_001/* ... step_015/*
```

---

## 31. 文件末尾恢复入口

本文件前部保留了此前多次交接过程中形成的历史章节，存在重复编号和重复内容。为避免破坏历史记录，本次没有删除这些内容。新会话应以第 21 至 30 节为当前权威的恢复指引，再按需阅读第 1 至 20 节；本节是文件物理末尾的最终入口。

当前最安全的动作顺序是：

```text
1. 不假设旧进程仍在运行
2. 对目标 pipeline 执行 check
3. 查看 health report 和 recovery plan
4. 区分 LLM、Banana、MiniMax-H3 三条服务链路
5. 先修复 pre-video，再单独决定是否重跑视频
6. 保留新输出目录、日志和 manifest
7. 用实际文件和 JSON 状态确认完成，不用历史日志推断当前完成
```

本次交接操作只修改了本 handoff 文档；没有修改代码、配置、输入故事、恢复脚本、视频脚本或输出产物。
+## 9. NarrativeNodeSelector 和 PathOrchestrator

节点选择入口是 scripts/run_narrative_node_selector.py，核心代码是 agents/narrative_node_selector.py。

流程：从 step_logs 构建 role candidates；按同 step、同 location、直接交互条件形成 event candidates；使用规则或 LLM 评分；输出 selected_nodes 和 rejected_nodes。

评分关注 dramatic conflict、goal progress、emotional turning point、information reveal、visual potential、causal importance 和 redundancy penalty。

PathOrchestrator 代码：

```text
agents/path_orchestrator_agent.py
agents/path_orchestrator_mcts_agent.py
```

路径阶段只排序，不重新拆分或合并节点。单角色 node 保持一个单角色视频事件；selector 已合法形成的多人 node 保持多人事件。

主要产物：

```text
path_orchestration/narrative_nodes.json
path_orchestration/path_candidates.json
path_orchestration/path_llm_direct/narrative_path_plan.json
path_orchestration/path_llm_direct/path_plan.json
path_orchestration/path_mcts/narrative_path_plan_mcts.json
path_orchestration/path_mcts/path_plan_mcts.json
path_orchestration/path_mcts/mcts_trace.jsonl
```

path-mode both 会生成 LLM 和 MCTS 两条路径；完整视频入口通常使用 LLM path，具体以 run manifest 为准。

## 10. VideoAgent、story_description、dialogue 和 H3 skill

代码：

```text
agents/video_agent.py
agents/event_story_tool.py
agents/event_dialogue_tool.py
agents/video_shot_planning_tool.py
agents/h3_prompt_adapter.py
```

处理链路：

```text
video_description -> EventStoryTool / LLM -> story_description
story_description -> EventDialogueTool / LLM -> dialogue
dialogue + references + shot plan -> H3 skill prompt -> MiniMax-H3 vLLM
```

video_description 保留用于追溯原始节点，可能是 action、target、reason 或推理式表达。

story_description 只给视频 prompt 使用，应按时间顺序写成可拍摄剧情：谁做了什么、对谁说了什么、对方如何回应、动作如何推进和结束；不能只是角色选择某动作及其理由。

dialogue 规则：没有对白写 none；不能让每个角色机械说一句；角色可以在一个事件中多次说话；对话按剧情时间顺序；对话服务于冲突、信息和动作；无法识别角色时使用 Character1、Character2 等默认名。

H3 声音字段：

```text
dialogue:
bgm:
overall_soundscape:
non_diegetic_music:
```

Skill policy 在 configs/agent_skill_policies.yaml。VideoAgent 允许 h3-prompt-writing、cameraatist-event-video-prompt、cameraatist-event-multishot-prompt。

MINIMAX_H3_PROMPT_STYLE=official_ref 时优先 h3-prompt-writing；compact 模式优先 CineWorldSim 内置 prompt。skill 后通常有 subject_definitions、summary、retention_analysis、detailed_description、overall_soundscape 和 non_diegetic_music。

多镜头写在 detailed_description 的 [Shot 1]、[Shot 2] At ... 中；event-shot-count=4 是一次视频请求内部的四段镜头组织，不是四次视频调用。

## 11. 视频参考图、尾帧和辅助镜头

VideoAgent 支持三种 reference mode：

```text
event_keyframe_only
semantic_assets
all_allowed
```

event_keyframe_only 只使用事件关键帧；semantic_assets 使用角色初始图、location 图和事件提及的 item 图；all_allowed 允许关键帧和语义资产共同进入候选集合。

选择规则：单角色事件默认最多一个角色参考；多人图只有事件确实需要多人直接交互时才允许；location 图按事件地点匹配；item 图只有事件文本实际提及时才选；不为了凑满 max_reference_images 而添加图像；去重并记录 reference_selection 和 excluded_reference_paths。

上一事件尾帧默认关闭：

```text
use_previous_tail_frame=false
```

只有显式开启 --use-previous-tail-frame，VideoAgent 才让 LLM 比较前后剧情、人物关系、location、空间关系和物品状态。场景切换、时间跳跃、角色变化、空间不连续或物品状态冲突时拒绝使用；LLM 失败时默认不使用。

辅助镜头包括 location 首次出现前的 establishing shot 和 world state 更新前后的 world_transition shot。辅助视频通常 5 秒、124 帧，也参与最终 concat，因此 task 数可能大于 selected node 数。

## 12. 时长、帧数、视频后端和可视化

固定映射位于 narrative/pipeline_schemas.py：

```text
5 seconds  -> 124 frames
10 seconds -> 243 frames
```

典型 MiniMax-H3 vLLM 参数：

```text
VIDEO_GEN_PROVIDER=minimax_h3
MINIMAX_H3_PROMPT_STYLE=official_ref
MINIMAX_H3_NUM_FRAMES=243
MINIMAX_H3_VLLM_ENABLED=true
MINIMAX_H3_VLLM_API_URL=http://127.0.0.1:9001/v1/videos/sync
MINIMAX_H3_VLLM_TIMEOUT_SEC=3600
MINIMAX_H3_CUDA_VISIBLE_DEVICES=2,3
```

历史上 vLLM 出现过超长请求、async output timeout、Future InvalidStateError、HTTP 500/504。这和 LLM API fallback、Banana SSL 故障是不同链路。

典型视频目录：

```text
event_video_pipeline_minimax_h3_vllm_multishot/
  event_video_*.mp4
  event_video_tasks.json
  event_video_summary.json
  final_orchestrated_events.mp4
```

可视化入口：scripts/generate_narrative_world_pipeline_html.py 和 scripts/generate_full_pipeline_stage_visualization.py；常见产物是 combined_overview.html 和 full_stage_pipeline_visualization.html。
+## 13. MovieBench 数据集上下文

时间连续性脚本：scripts/process_moviebench_clip_metadata.py。

当前规则：

```text
abs(current.start_ms - previous.end_ms) < 1000
```

当前 clip 的 start_ms 与上一个按时间排序 clip 的 end_ms 相差不到 1 秒才连续。若不连续，is_continuous_with_prev=false、prev_file_name=null，同时保留 prev_chronological_file_name 用于审计时间顺序。

连续视频副本脚本：scripts/build_moviebench_continuous_dataset.py。每个 clip 保留 Characters、Style Elements、Plot、Background Description、Camera Motion、clip_start_tc、clip_end_tc 和 duration_sec。

连续 shots 合并事件脚本：scripts/build_moviebench_event_descriptions_from_continuous_shots.py。目标是按时间顺序把同一连续片段的所有 shot 合并为一个具体总事件，同时保留每个 shot 的人物、动作、背景、摄影和风格，使总事件可以重新拆解回多个 shot。role_in_event 表示 shot 在总事件中的作用，是 LLM 输出，不是原始字段。

## 14. 实验协议

实验文档：iclr/experiment_implementation_protocol.md。

两条主线：

```text
Narrative representation / story visualization
  story manifest -> narrative units -> image sequence -> metrics

Storyboard-anchored multi-shot video
  story manifest -> narrative units + path -> videos -> state/video metrics
```

协议规划的 baseline 包括 Direct prompting、Storyboard-only、BookWorld-style agents、故事可视化 baseline、故事视频 baseline、STAGE 和 CineWorldSim full。

协议规划的指标包括 narrative quality、story visualization consistency、shot quality、cross-shot consistency、event fidelity、world-state consistency、character-state consistency、object-state persistence、temporal/causal ordering 和 Narrative Unit Coverage。

协议中的建议 runner 和 metric 脚本不代表全部已经实现，执行前必须检查实际文件。

## 15. 关键历史运行结果

四 genre manifest：outputs/four_genre_demos_15nodes_20260816/four_genre_demo_manifest.json。记录 status=done，包含 suspense、adventure、action、urban 四个 demo。这是早期 15-node 运行。

两个三主角 adventure manifest：outputs/two_adventure_new_pipeline_20260828/two_demo_runner_manifest.json。

关键设置：all_banana、path-mode=both、max-steps=15、node-min/max=15、node-llm-scoring、multi-shot-events、event-shot-count=4、max-reference-images=4、no-use-previous-tail-frame、无 previous audio、dialogue-backfill、insert-auxiliary-videos、video-reference-mode=semantic_assets。

已记录结果：

```text
demo1: 21 success / 21 total, concat_ok=true
demo2: 16 success / 16 total, concat_ok=true
```

总数包含主事件视频和辅助镜头，不能等同于 selected node 数。

Keyframe-only manifest：outputs/keyframe_only_event_video_20260908/01_demo1_shared_space/keyframe_only_event_video_pipeline_manifest.json。

已验证：selected_node_count=25、event_task_count=25、auxiliary_task_count=1、video_success=26、video_failed=0、video_skipped=0、concat_ok=true、final_video_exists=true、event_reference_mode_ok=true。

这证明 keyframe-only 路径曾成功生成 25 个主事件视频和 1 个辅助视频，但不是 45 个 selected nodes。

## 16. gpt55 pre-video 结果

目录：outputs/gpt55_pre_video_demo1/all_banana。

报告：/tmp/cameramaster_final_gpt55_check/pre_video_health_report.json 和 /tmp/cameramaster_final_gpt55_check/pre_video_recovery_plan.json。

报告为 status=healthy，结构上确认 15 steps、16 world states、15 world deltas、45 event keyframes、3 portraits、location 四视图、item 图、node selection、LLM path 和 MCTS path。

限制：该目录在结构化 fallback 记录接入前生成。healthy 只能说明当前 JSON/资产结构完整，不能反证 2026-09-10 step 13 日志中的 503/fallback 没有发生。

+## 17. Fallback 记录和恢复机制

已完成相关代码：narrative/simulation_scaffold.py、scripts/recover_pre_video_pipeline.py、scripts/rerun_fallback_narrative_nodes.py 和 docs/pre_video_recovery.md。

ModelConfig 增加 call_history。每次 LLM 调用记录 call_id、call_label、status、fallback_used 和 error。角色及 step 记录 model_calls、fallback_used 和 fallback_reasons。

通用检查器支持 check -> plan -> repair，阶段顺序为 world_initialization、narrative_simulation、world_delta、event_keyframe、node_selection、path_llm、path_mcts。

窄范围节点恢复使用 STEP:ACTOR:SOURCE，例如 13:Mara_Venn:evaluation。恢复必须写新目录，不能覆盖源 pipeline。恢复工具只负责 pre-video，不负责视频生成和最终 concat。

## 18. step 13 恢复失败事实

原始日志出现 Error code: 503 - Service temporarily unavailable，以及 Temporary failure in name resolution。

已执行过 step 13 到 15 的 fallback 节点恢复。结果是相关 LLM 调用仍然 fallback，重建 world delta 时因 httpx.ConnectError DNS 错误失败。

失败位置包括 narrative/world_delta_judge.py 和 agents/llm_tool.py。失败目录只有 partial checkpoint、partial logs、部分 world states/deltas 和 portraits，没有完整 pipeline_result.json，不能用于 node selection、path 或视频。

retry 目录 outputs/gpt55_pre_video_demo1_repaired_step13_retry/all_banana 有 checkpoint、partial logs、states 和 deltas，但没有完整 pipeline manifest；completed_steps=15 不能证明恢复成功。

## 19. 当前卡点

最深卡点是 LLM endpoint 稳定性，表现为 503、DNS name resolution failure 和 requests SSL error。这会影响角色 action、evaluation、story anchor、world delta、story_description、dialogue、node scoring、path planning 和 MCTS scoring。

另外存在两条独立链路：Nano Banana 图像请求 SSL/网络问题，以及 MiniMax-H3 vLLM 超时、500/504。不要把三类故障混为一个问题。

当前 CineWorldSim 的 .git 是空目录，不是有效 Git 仓库。git rev-parse、git status 和 git log 都返回 fatal: not a git repository，因此无法提供可靠 branch、commit 或 modified/untracked 状态；这不等于工作区干净。

## 20. 已执行验证

```bash
pytest -q tests/test_pre_video_recovery.py tests/test_rerun_fallback_narrative_nodes.py
```

结果：14 passed。

```bash
pytest -q tests
```

结果：135 passed。

```bash
python -m py_compile narrative/simulation_scaffold.py scripts/recover_pre_video_pipeline.py scripts/rerun_fallback_narrative_nodes.py
```

结果：通过。本次只读进程检查没有发现 CineWorldSim pipeline、fallback recovery、two-demo runner 或视频生成任务。

项目不直接把故事交给视频模型，而是先构建人物、空间和物体世界，在 step-by-step 推演中维护状态，再筛选和影视化事件。

核心研究主张：

- Simulation-based Cinematic Generation：先构建世界和排练，再进行拍摄。
- Bridging Open Simulation and Structured Narrative：从开放推演图中筛选收敛的叙事事件。
- State-Anchored Multi-shot Decomposition：多个镜头来自同一时空状态，降低人物、地点和物体漂移。
- Simulation/Rendering Decoupling：推演和渲染解耦，便于检查、重跑、比较和消融。

## 2. 当前完整架构

```text
Story
  -> OrchestratorAgent
  -> WorldModelAgent
  -> Initial character/location/item assets
  -> StateAnalysisMachine
  -> WorldDeltaJudge + WorldStateTracker
  -> EventKeyframeRenderer
  -> NarrativeNodeSelector
  -> LLM Path / MCTS Path
  -> VideoAgent
  -> Auxiliary shots + concat + visualization
```

## 3. 关键代码和入口

核心推演和世界：

```text
narrative/simulation_scaffold.py
narrative/initial_asset_renderer.py
narrative/world_renderer.py
narrative/world_delta_judge.py
narrative/world_state_tracker.py
narrative/event_keyframe_renderer.py
narrative/asset_reference_selector.py
narrative/visual_prompt_constraints.py
narrative/pipeline_schemas.py
```

---

## 21. 新会话优先阅读区：当前任务

当前任务不是单独修复某一个视频，而是维护和验证 CineWorldSim 的完整叙事视频生成框架：

```text
故事文本
  -> 人物、地点、物体初始化
  -> 多智能体按 step 的时空推演
  -> world state / world delta 更新
  -> 事件关键帧
  -> 叙事节点筛选和路径编排
  -> story_description / dialogue
  -> 参考图选择
  -> H3 prompt
  -> MiniMax-H3 视频
  -> 辅助镜头
  -> 拼接和可视化
```

研究主线是“多智能体叙事推演 + 影视世界建模 + 状态锚定的视频生成”，核心不是把原始故事直接交给视频模型，而是先建立可追踪的世界状态，再从推演结果中选取并拍摄事件。

新会话首先应区分以下三类对象：

- `step`：模拟时间节点，包含一个或多个角色的同步动作。
- `narrative node`：从角色-step 候选中选出的可影视化事件。
- `video task`：一个主事件视频或一个辅助镜头任务，数量不一定等于 selected nodes。

## 22. 已完成、已验证和仅属于设计目标的边界

### 已完成或已有实现

- 世界模型、人物配置、角色关系和初始环境的结构化输出。
- 人物 portrait、location 四视图、item 初始化资产的生成逻辑及产物约定。
- 15 step 的多角色同步推演、world state、world delta 和事件关键帧链路。
- 单角色单 step 的默认节点粒度，以及同 step、同 location、直接交互时的多人事件合并规则。
- LLM path 和 MCTS path 的路径编排。
- `video_description` 保留、`story_description` 生成，以及 dialogue backfill 工具链。
- H3 skill 接入、三种视频参考图模式、尾帧默认关闭和辅助镜头任务设计。
- `event_keyframe_only` 视频路径，以及主事件视频和辅助视频的 concat 产物。
- pre-video fallback 检查、计划和新目录恢复工具。
- MovieBench 的连续 clip 判断和连续 shot 总事件构建脚本。
- 两个三主角 adventure demo 的历史完整视频运行记录。

### 已验证事实

- pre-video 专项测试为 `14 passed`。
- 全量测试为 `135 passed`。
- 相关 Python 文件的 `py_compile` 已通过。
- 两个 adventure demo 的历史 manifest 记录：demo1 为 `21/21` 成功且 `concat_ok=true`；demo2 为 `16/16` 成功且 `concat_ok=true`。
- keyframe-only 历史运行记录：25 个主事件视频、1 个辅助视频，全部成功且 `concat_ok=true`。

### 不能直接当作已完成的内容

- `node-min-nodes=15` 不等于系统一定有 15 个或 45 个最终节点。
- 三角色、15 step 的 45 是角色-step 候选上限，不是强制 selected node 数。
- 辅助镜头会使 video task 数量大于 selected node 数。
- pre-video `healthy` 不等于视频已经生成，也不等于 LLM 调用没有发生 fallback。
- `completed_steps=15` 不等于所有角色、world delta、关键帧和路径都无 fallback。
- 规划文档中列出的实验 baseline 和 metric，不等于相应脚本已经全部实现。

## 23. 当前实际状态和最深卡点

只读检查时没有发现仍在运行的 CineWorldSim、恢复脚本或视频生成进程。不要根据旧日志推断当前进程仍在工作；重新判断时应再次执行进程和产物检查。

最深卡点是外部 LLM endpoint 的稳定性，已出现过：

```text
503 Service temporarily unavailable
Temporary failure in name resolution
requests.exceptions.SSLError
httpx.ConnectError
```

受影响的阶段可能包括角色 action、角色 evaluation、story anchor、world delta、`story_description`、dialogue、节点评分、LLM path 和 MCTS path。

必须分别排查三条链路：

1. LLM API：503、DNS、SSL 或连接失败，可能触发 fallback。
2. Nano Banana：图像初始化、world update 或事件关键帧的 API/SSL 问题。
3. MiniMax-H3 vLLM：视频请求超时、500、504、async output timeout 或服务端 Future 状态错误。

其中一条链路失败，不能自动推断另外两条也失败。

截至本次只读检查，gpt55 demo1 可确认 pre-video 结构曾经健康，但历史检查未发现对应视频文件；step 13 的恢复曾被 503 和 DNS 错误阻断，恢复目录不能直接视为可用于视频生成的完整快照。

## 24. 通用检查和恢复指引

新任务开始时，优先使用只读检查：

```bash
cd /data1/AgentG/hhb/CameraMaster

python scripts/recover_pre_video_pipeline.py \
  --pipeline-dir <pipeline_dir> \
  --mode check \
  --report-dir <report_dir>
```

检查以下内容：

- initialization：人物、location 四视图、item 和 world model。
- simulation：step 数、checkpoint、角色日志、`fallback_used` 和 `model_calls`。
- world delta：每个 step 的 delta 和对应更新图。
- event keyframe：每个角色事件是否有关键帧及状态。
- node selection：候选节点、selected/rejected 节点和评分记录。
- path：LLM path、MCTS path 及其 trace。

若需要生成修复计划：

```bash
python scripts/recover_pre_video_pipeline.py \
  --pipeline-dir <pipeline_dir> \
  --mode plan \
  --report-dir <report_dir>
```

若确认 endpoint 已恢复，再在新的空目录执行修复：

```bash
python scripts/recover_pre_video_pipeline.py \
  --pipeline-dir <source_pipeline_dir> \
  --mode repair \
  --output-dir <new_repaired_pipeline_dir> \
  --backend-mode all_banana \
  --strict-online \
  --retries 5
```

该工具只覆盖 VideoAgent 之前的阶段，不重跑视频，不负责最终 concat。已知单节点可使用旧工具，但未知失败范围时应优先使用通用检查器，而不是手工猜测 step 或 actor。

## 25. 下一步执行计划

### A. 先确认外部服务

1. 确认 `.env` 中实际使用的 LLM base URL、密钥变量和模型配置，不在日志或 handoff 中记录密钥。
2. 确认 endpoint 对当前模型返回正常响应，区分 503、DNS、SSL 和鉴权错误。
3. 确认 Nano Banana 服务可用，并确认 `.env` 中的模型名称为 `nano-banana-2` 的事实是否仍与当前环境一致。
4. 确认 MiniMax-H3 vLLM 服务健康，再执行视频阶段。

### B. 再检查已有 pipeline

1. 对目标 pipeline 执行 `check`。
2. 读取 `pre_video_health_report.json` 和 `pre_video_recovery_plan.json`。
3. 根据 `earliest_failed_stage` 决定是复用已有资产、重跑后续阶段，还是从 world initialization 重建。
4. 检查所有 fallback 标记，不能只看 `completed_steps`。

### C. 最后执行视频生成

1. 确认节点、路径、关键帧和 prompt 文件完整。
2. 确认视频参考模式：事件视频若要求只用关键帧，使用 `event_keyframe_only`；不能把 `max-reference-images` 当作强制填满数量。
3. 确认尾帧策略；当前默认不使用上一个视频尾帧。
4. 运行 VideoAgent，实时保存 task 状态和 stderr/stdout。
5. 生成完成后单独检查 `video_success`、`video_failed`、`video_skipped`、最终 mp4 是否存在，以及 `concat_ok`。

## 26. 高成本重复问题和注意事项

- 不要重复完整重跑：先检查最早失败阶段，初始化失败会影响全部下游，path-only 失败不必重建图像。
- 不要覆盖源 pipeline：恢复工具设计为写入新目录，源目录用于审计和对比。
- 不要把日志中的 stderr 标签当成真实错误：双 demo runner 可能把子进程全部输出统一写成 `[demo][stderr]`，必须看该行内容和子进程 return code。
- 不要把模型 fallback 当作成功调用：fallback 可能仍产生结构化 JSON，但不代表内容经过目标模型生成。
- 不要把节点数、事件视频数和任务数混用：辅助镜头、多人合并和失败重试都会改变数量。
- 不要强制添加参考图：参考图应由事件语义决定；未提及的 item、location 或人物不应为了达到上限而引入。
- 不要默认使用尾帧：除非明确开启并通过前后事件连贯性判断，否则保持关闭。
- 不要把事件关键帧和 world update 图混用：world update 只表达明确物理变化，事件关键帧表达事件中的人物、动作和必要物体。
- 不要把无对白写成虚构对白：没有自然对白时严格写 `dialogue=none`。
- 不要把 shot 数当成 video task 数：一个 H3 视频请求内部可以有多个 shot，辅助镜头则是独立 task。
- 不要把历史成功 manifest 当作当前服务可用性的证明：网络、模型和输出目录都可能已经变化。
- 不要把空 `.git` 目录解释为干净工作区：它不是有效 Git 仓库，不能提供可靠的变更审计。

## 27. 关键路径索引

### 入口脚本

```text
scripts/run_full_narrative_world_video_pipeline.py
scripts/run_narrative_world_path_pipeline.py
scripts/run_dual_backend_narrative_world_pipeline.py
scripts/run_two_adventure_demo_pipelines.py
scripts/resume_two_adventure_demo_pipelines.py
scripts/run_keyframe_only_event_video_pipeline.py
scripts/run_orchestrated_event_video_generation.py
scripts/concat_existing_event_videos.py
scripts/recover_pre_video_pipeline.py
scripts/rerun_fallback_narrative_nodes.py
```

### Agent 和工具

```text
agents/orchestrator.py
agents/video_agent.py
agents/event_story_tool.py
agents/event_dialogue_tool.py
agents/video_shot_planning_tool.py
agents/h3_prompt_adapter.py
agents/narrative_node_selector.py
agents/path_orchestrator_agent.py
agents/path_orchestrator_mcts_agent.py
agents/generation_backends.py
agents/llm_tool.py
agents/skills/
```

### Narrative 模块

```text
narrative/simulation_scaffold.py
narrative/initial_asset_renderer.py
narrative/world_renderer.py
narrative/world_delta_judge.py
narrative/world_state_tracker.py
narrative/event_keyframe_renderer.py
narrative/asset_reference_selector.py
narrative/visual_prompt_constraints.py
narrative/pipeline_schemas.py
```

### 配置和实验文档

```text
.env
configs/agent_skill_policies.yaml
configs/reference_assets.example.json
docs/current_architecture.md
docs/current_pipeline_architecture_and_visualization.md
docs/pre_video_recovery.md
iclr/experiment_implementation_protocol.md
```

### 输入和历史输出

```text
inputs/demo1_adventure_same_space.txt
inputs/demo2_adventure_three_spaces.txt
outputs/four_genre_demos_15nodes_20260816/
outputs/two_adventure_new_pipeline_20260828/
outputs/keyframe_only_event_video_20260908/
outputs/gpt55_pre_video_demo1/
outputs/gpt55_pre_video_demo1_repaired_step13/
outputs/gpt55_pre_video_demo1_repaired_step13_retry/
```

### MovieBench 脚本

```text
scripts/process_moviebench_clip_metadata.py
scripts/build_moviebench_continuous_dataset.py
scripts/build_moviebench_event_descriptions_from_continuous_shots.py
```

## 28. 新会话恢复顺序

新会话不要从聊天记忆推测当前状态，按以下顺序恢复：

1. 阅读本文件的第 21 至 30 节，再按需回看第 1 至 20 节的历史细节。
2. 阅读 `docs/current_architecture.md` 和 `docs/current_pipeline_architecture_and_visualization.md`，确认代码结构和产品描述是否一致。
3. 阅读 `docs/pre_video_recovery.md`，确定检查、计划和修复边界。
4. 阅读 `iclr/experiment_implementation_protocol.md`，区分已实现脚本和协议规划。
5. 对目标输出目录执行只读 manifest、文件存在性、进程和日志检查。
6. 只有用户明确要求代码修改时才编辑代码；若只要求确认状态，保持只读。
7. 若服务可用，先完成 pre-video 检查，再决定是否视频生成；若服务不可用，记录失败类型，不伪造完成状态。
8. 每次恢复和重跑都保留新输出目录、命令、日志和 manifest，便于后续比较。

## 29. 已执行关键命令记录

以下命令在此前工作中已执行并记录过，结果以第 15、16、18 和 20 节为准；新会话重新执行前应先确认路径和环境没有变化：

```bash
pytest -q tests/test_pre_video_recovery.py tests/test_rerun_fallback_narrative_nodes.py
pytest -q tests
python -m py_compile narrative/simulation_scaffold.py scripts/recover_pre_video_pipeline.py scripts/rerun_fallback_narrative_nodes.py
python scripts/run_two_adventure_demo_pipelines.py
python scripts/recover_pre_video_pipeline.py --pipeline-dir <pipeline_dir> --mode check --report-dir <report_dir>
```

历史运行还使用过以下类型的环境设置：

```bash
MINIMAX_H3_PROMPT_STYLE=official_ref
MINIMAX_H3_NUM_FRAMES=243
MINIMAX_H3_VLLM_ENABLED=true
MINIMAX_H3_VLLM_API_URL=http://127.0.0.1:9001/v1/videos/sync
MINIMAX_H3_VLLM_TIMEOUT_SEC=3600
MINIMAX_H3_CUDA_VISIBLE_DEVICES=2,3
```

命令中的 `<pipeline_dir>`、输出目录、模型、服务状态和运行结果必须以当前文件系统为准，不能机械复制历史命令。

## 30. 本次交接核验结论

- 本次操作目标是补全项目级 handoff 文档。
- 本次没有修改代码、配置、输入故事或输出视频目录。
- 本次只读检查发现：无正在运行的相关进程；`.git` 不是有效 Git 仓库；关键历史 manifest、脚本和文档路径存在于项目目录中。
- 由于 `.git` 为空目录且不是有效仓库，无法给出可靠的 git branch、commit、tracked/untracked 或 diff 结论。
- 历史结果、当前结构健康和当前服务可用性是三种不同证据，后续必须分别核验。
- 继续工作时，优先以本节和第 21 至 29 节作为恢复入口，以实际日志、JSON manifest 和文件存在性作为最终证据。

节点、路径、视频：

```text
agents/narrative_node_selector.py
agents/path_orchestrator_agent.py
agents/path_orchestrator_mcts_agent.py
agents/video_agent.py
agents/event_story_tool.py
agents/event_dialogue_tool.py
agents/video_shot_planning_tool.py
agents/h3_prompt_adapter.py
```

主要运行入口：

```text
scripts/run_dual_backend_narrative_world_pipeline.py
scripts/run_narrative_world_path_pipeline.py
scripts/run_full_narrative_world_video_pipeline.py
scripts/run_orchestrated_event_video_generation.py
scripts/run_keyframe_only_event_video_pipeline.py
scripts/run_two_adventure_demo_pipelines.py
scripts/resume_two_adventure_demo_pipelines.py
```

失败检查和恢复：

```text
scripts/recover_pre_video_pipeline.py
scripts/rerun_fallback_narrative_nodes.py
docs/pre_video_recovery.md
```

## 4. 两个三主角 demo

输入文件：

```text
inputs/demo1_adventure_same_space.txt
inputs/demo2_adventure_three_spaces.txt
```

Demo 1 角色为 Mara Venn、Ilya Trent、Nora Hale。三人主要在同一座废弃灯塔档案室及淹水通道中行动，用于测试同一空间的多人交互事件。

Demo 2 角色为 Leona、Darius、Mira。三人先分布在三个不同空间，通过灯光、无线电和镜面信号关联，最终在 Sunken Atrium 汇合，用于测试跨空间推演和最终多人关联事件。

## 5. 角色、世界和初始化资产

OrchestratorAgent 位于 narrative/simulation_scaffold.py，负责角色解析和 final_goal、role_profile、known_relations、current_environment 的生成。

WorldModelAgent 抽取 locations、items、空间关系和 world blueprint。产物写入 pipeline_result.json 的 character_configs 和 world_model。

初始化资产在 narrative/initial_asset_renderer.py 中生成：

- 人物 portrait：人物占画面最大比例，身份、面部和服装清晰，最小背景，不出现额外人物。
- Location：每个地点生成 wide、medium、close、extreme_close 四种视图；默认没有人物、人群或无关元素。
- Item：物体占画面最大比例，孤立呈现，不出现手、人物或无关物体。

主要产物：

```text
narrative_artifacts/initial_assets/initial_assets.json
narrative_artifacts/portraits/*
narrative_artifacts/world_model/locations/*
narrative_artifacts/world_model/items/*
```

图像后端映射：all_flux -> flux_local，all_banana -> nano_banana。当前 .env 中记录的 Banana 模型名称是 nano-banana-2；本文不记录任何密钥。
## 6. Step-by-step 叙事推演

StateAnalysisMachine 位于 narrative/simulation_scaffold.py。每个 step 中，各角色根据自身目标、关系和当前环境输出动作；系统记录 action、target、location、reason 和角色文本，更新共享环境、story anchor、故事阶段和完成度，再广播给下一步。

step 是模拟时间节点，不等于最终视频节点。

主要产物：

```text
step_logs.json
step_logs.partial.json
narrative_checkpoint.json
world_state_step0.json
world_state_step1.json ... world_state_step15.json
world_deltas/world_delta_step1.json ... world_delta_step15.json
```

## 7. 节点粒度、15 和 45

当前概念关系：

```text
step            模拟时间节点
role candidate  一个角色在一个 step 中的动作
narrative node  经过筛选后值得影视化的事件
video task      一个主事件视频或辅助镜头视频
```

当前默认粒度是：

```text
one character + one step + one action = one candidate node
```

三角色、15 step 理论上最多有 45 个 role candidates，但不保证最终选择 45 个 nodes。

旧实验常设置 node-min-nodes=15、node-max-nodes=15，这是固定 15 个 selected nodes 的实验预算，不代表只有 15 个候选。

当前允许：固定 15 个节点、最多 45 个节点、少于 45 个节点，以及合法多人交互合并节点。

只有同时满足以下条件才允许合并多个角色：同一个 step、同一个具体 location、存在直接交互或互相因果、合并后读起来是一个共同事件。不能因为角色同属一个故事、同一个 step 或同一个大场景就合并。

相关代码：

```text
narrative/visual_prompt_constraints.py
agents/narrative_node_selector.py
```
## 8. WorldDelta、WorldState 和事件关键帧

WorldDeltaJudge 位于 narrative/world_delta_judge.py，WorldStateTracker 位于 narrative/world_state_tracker.py。

只有事件文本明确造成 location 或 item 的可观察物理变化时，才更新世界图像。允许的变化包括颜色、形状、外观、纹理、结构、损坏、开闭配置、淹水和燃烧等。

发现、注意、思考、情绪变化、拥有、移动或叙事重要性变化本身不能触发图像刷新，除非同时有可观察的物理变化。

Location 更新时由 LLM 选择 wide、medium、close 或 extreme_close；更新图只描述明确变化，不出现人物、人群、行人和无关元素，并优先参考上一轮同视图或初始化图。Item 更新使用上一轮 item 图或初始 item 图。

正确顺序：

```text
当前 step 推演
  -> 必要的 world state 更新
  -> EventKeyframeRenderer 生成事件关键帧
```

事件关键帧参考当前事件文本、角色初始图、location 图、事件提及的 item 图和必要的 world update 图。

主要产物：

```text
world_state_step0.json ... world_state_step15.json
world_deltas/world_delta_step1.json ... world_delta_step15.json
narrative_artifacts/world_updates/locations/*
narrative_artifacts/world_updates/items/*
narrative_artifacts/event_keyframes/step_001/* ... step_015/*
```

---

## 32. EOF 恢复入口

本文件前部保留了此前多次交接过程中形成的历史章节，存在重复编号和重复内容。为避免破坏历史记录，本次没有删除这些内容。新会话应以第 21 至 30 节为当前权威的恢复指引，再按需阅读第 1 至 20 节；本节位于文件物理末尾。

当前最安全的动作顺序是：

```text
1. 不假设旧进程仍在运行
2. 对目标 pipeline 执行 check
3. 查看 health report 和 recovery plan
4. 区分 LLM、Banana、MiniMax-H3 三条服务链路
5. 先修复 pre-video，再单独决定是否重跑视频
6. 保留新输出目录、日志和 manifest
7. 用实际文件和 JSON 状态确认完成，不用历史日志推断当前完成
```

本次交接操作只修改了本 handoff 文档；没有修改代码、配置、输入故事、恢复脚本、视频脚本或输出产物。
