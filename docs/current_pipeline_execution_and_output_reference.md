# CineWorldSim 当前 Pipeline 执行与输出说明

本文档描述当前 CineWorldSim 的叙事世界建模、节点编排、事件视频生成、字幕时间轴输出流程，并以已经完成的运行结果为例：

```text
/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana
```

该运行的事实状态：15 个叙事 step、45 条角色 step log、16 个 world state、15 个 world delta、45 个事件关键帧记录、31 个叙事节点、31 个事件视频、19 个辅助视频。视频阶段共 50 个任务，全部成功，最终拼接成功。

## 1. Pipeline 目标与顺序

当前 pipeline 将故事文本转换为可追溯的视频序列。主流程如下：

```text
故事文本 + 角色名
  -> 世界空间、物品、角色初始化资产
  -> step-by-step 叙事推演与 world state 更新
  -> 事件关键帧生成
  -> 叙事节点筛选与路径编排
  -> story_description / dialogue backfill
  -> VideoAgent 分镜时长、帧数和 shot 时间规划
  -> MiniMax H3 Ref2VA 视频生成
  -> 辅助空镜/世界变化镜头插入
  -> 视频拼接
  -> 视频同名字幕时间轴 JSON
```

事件节点默认按“一个角色在一个 step 中的一件事”拆分为一个视频。只有角色处于同一地点、同一事件并发生明确互动时，节点选择器才可以合并为多人事件节点。多人节点可因此有多张事件关键帧参考图，但 `event_keyframe_only` 模式不会引入角色初始化图、地点图、物品图或上一视频尾帧。

## 2. 执行脚本

### 2.1 完整从头运行入口

标准一体化入口是：

```text
scripts/run_full_narrative_world_video_pipeline.py
```

它依次调用：

1. `scripts/run_narrative_world_path_pipeline.py` 中的世界阶段。
2. `NarrativeNodeSelector`，从角色 step 记录筛选叙事节点。
3. `PathOrchestrator`，产出按叙事因果、视觉衔接和节奏排序的路径。
4. `agents.video_agent.VideoAgent.process_orchestrated_events`，生成事件视频、辅助视频并拼接。

可复用的完整命令示例：

```bash
cd /data1/AgentG/hhb/CameraMaster

PYTHONUNBUFFERED=1 \
MINIMAX_H3_PROMPT_STYLE=official_ref \
MINIMAX_H3_VLLM_ENABLED=true \
MINIMAX_H3_VLLM_API_URL=http://127.0.0.1:9001/v1/videos/sync \
MINIMAX_H3_CUDA_VISIBLE_DEVICES=2,3 \
conda run -n mifage python -u scripts/run_full_narrative_world_video_pipeline.py \
  --story-file inputs/demo1_adventure_same_space.txt \
  --characters 'Mara Venn,Ilya Trent,Nora Hale' \
  --output-dir outputs/new_full_run \
  --backend-mode all_banana \
  --path-mode both \
  --max-steps 15 \
  --stop-on-complete \
  --enable-narrative-node-selector \
  --node-selection-mode top_k \
  --node-min-nodes 1 \
  --node-max-nodes 45 \
  --node-llm-scoring \
  --strict-online \
  --multi-shot-events \
  --event-shot-count 4 \
  --max-reference-images 4 \
  --video-reference-mode event_keyframe_only \
  --no-use-previous-tail-frame \
  --no-use-previous-audio \
  --no-upload-previous-audio \
  --dialogue-backfill \
  --insert-auxiliary-videos \
  --log-level INFO
```

关键参数：

| 参数 | 当前含义 |
| --- | --- |
| `--backend-mode all_banana` | 世界资产和事件关键帧通过 Banana 图像后端生成。 |
| `--max-steps 15` | 最多推演 15 个全局 step。三角色通常会形成最多约 45 条角色候选事件，但多人互动节点可合并。 |
| `--node-min-nodes 1 --node-max-nodes 45` | 允许节点选择器根据剧情保留 1 至 45 个节点，不强制输出 45 个。 |
| `--path-mode both` | 同时运行 LLM 和 MCTS 路径编排；视频默认消费 LLM 路径。 |
| `--video-reference-mode event_keyframe_only` | 事件视频只使用节点关联的事件关键帧。多人互动节点可使用多张事件关键帧。 |
| `--multi-shot-events` | 每个事件生成一条 H3 视频，但 prompt 中可含多个带时间码的 shot，不会生成每 shot 独立 MP4。 |
| `--event-shot-count 4` | 仅在显式总时长覆盖或 fallback 规划时作为 shot 数参考；LLM shot planner 可自主决定实际 shot 数。 |
| `--no-use-previous-tail-frame` | 默认不以此前视频尾帧作为本视频参考。 |
| `--dialogue-backfill` | 用 LLM 为每个事件生成 `Character1`、`Character2` 等槽位形式的对话 JSON，并回填 H3 prompt。 |
| `--insert-auxiliary-videos` | 在首次进入地点时插入空镜头，并在 world state 更新之间插入世界变化镜头。 |

### 2.2 固定为“事件关键帧唯一参考”的 wrapper

脚本：

```text
scripts/run_keyframe_only_event_video_pipeline.py
```

它会调用完整入口，并固定：

```text
--backend-mode all_banana
--video-reference-mode event_keyframe_only
--multi-shot-events
--insert-auxiliary-videos
```

它同时将 vLLM、CUDA、日志文件和最终 manifest 统一封装。适合新的完整测试运行。它的默认视频输出目录名称是：

```text
<output-dir>/all_banana/event_video_pipeline_minimax_h3_vllm_multishot/
```

### 2.3 仅重跑视频阶段

脚本：

```text
scripts/run_orchestrated_event_video_generation.py
```

输入是已存在的：

```text
<pipeline-run-dir>/path_orchestration/narrative_nodes.json
<pipeline-run-dir>/path_orchestration/path_llm_direct/narrative_path_plan.json
```

典型命令：

```bash
cd /data1/AgentG/hhb/CameraMaster

PYTHONUNBUFFERED=1 \
MINIMAX_H3_PROMPT_STYLE=official_ref \
MINIMAX_H3_VLLM_ENABLED=true \
MINIMAX_H3_VLLM_API_URL=http://127.0.0.1:9001/v1/videos/sync \
MINIMAX_H3_CUDA_VISIBLE_DEVICES=2,3 \
conda run -n mifage python -u scripts/run_orchestrated_event_video_generation.py \
  --pipeline-run-dir outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana \
  --output-dir outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/event_video_pipeline_minimax_h3_vllm_event_keyframe_only \
  --video-reference-mode event_keyframe_only \
  --multi-shot-events \
  --insert-auxiliary-videos \
  --no-use-previous-tail-frame \
  --no-use-previous-audio \
  --no-upload-previous-audio \
  --dialogue-backfill \
  --no-dry-run \
  --strict-online \
  --log-level INFO
```

### 2.4 fallback 节点恢复入口

脚本：

```text
scripts/rerun_fallback_narrative_nodes.py
```

当推演阶段发生 LLM fallback 或失败时，它读取源 `pipeline_result.json` 和 checkpoint，从指定 step/角色重新推演，并可使用 `--rerun-downstream` 使后续 step 与修复结果保持一致。

本例的历史恢复命令是：

```bash
conda run -n mifage python -u scripts/rerun_fallback_narrative_nodes.py \
  --pipeline-dir outputs/gpt55_pre_video_demo1/all_banana \
  --node 13:Mara_Venn:evaluation \
  --node 13:Ilya_Trent:action \
  --rerun-downstream \
  --output-dir outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana \
  --strict-online \
  --retries 5 \
  --log-level INFO
```

该命令恢复了 step 13 至 15。恢复完成后，需要重新运行路径编排和视频阶段，不能直接复用旧路径规划结果。

## 3. 当前样例运行的实际执行链

当前样例不是从一条 wrapper 命令直接产出。实际链路为：

```text
源运行 outputs/gpt55_pre_video_demo1/all_banana
  -> rerun_fallback_narrative_nodes.py
  -> outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana
  -> run_narrative_world_path_pipeline.py（existing pipeline result 模式）
  -> path_orchestration/
  -> run_orchestrated_event_video_generation.py
  -> event_video_pipeline_minimax_h3_vllm_event_keyframe_only/
  -> final_orchestrated_events.mp4
```

视频日志记录的实际视频配置包括：

```text
provider: minimax_h3
task: ref2va
CUDA devices: 2,3
parallel mode: balanced
device map: balanced_low_0
text encoder: offload
VAE: tile + offload
reference resize: 512x288
multi-shot events: true
event duration: LLM planner decided, only 5 s / 124 frames or 10 s / 243 frames
previous tail frame: disabled
previous audio: disabled
```

## 4. 输出目录结构

以下为当前完整样例的目录树。`<run-root>` 指：

```text
/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana
```

```text
<run-root>/
├── fallback_rerun_manifest.json
├── pipeline_result.json
├── story_bundle.json
├── narrative_checkpoint.json
├── step_logs.json
├── step_logs.partial.json
├── world_state_step0.json ... world_state_step15.json
├── world_deltas/
├── narrative_artifacts/
│   ├── portraits/
│   ├── world_model/
│   │   ├── locations/
│   │   └── items/
│   └── event_keyframes/
├── path_orchestration/
│   ├── pipeline_manifest.json
│   ├── narrative_nodes.json
│   └── path_llm_direct/
│       └── narrative_path_plan.json
├── visualizations/
├── path_orchestration.log
├── video_generation.log
└── event_video_pipeline_minimax_h3_vllm_event_keyframe_only/
    ├── event_video_###_NN_###.mp4
    ├── event_video_###_NN_###.timeline.json
    ├── event_video_###_NN_###.minimax_h3_vllm.json
    ├── event_video_###_NN_###.minimax_h3_vllm.headers.txt
    ├── aux_video_###_{establishing|world_transition}_NN_###.mp4
    ├── aux_video_###_{establishing|world_transition}_NN_###.timeline.json
    ├── aux_video_###_*.minimax_h3_vllm.json
    ├── aux_video_###_*.minimax_h3_vllm.headers.txt
    ├── dialogue_artifacts/
    ├── story_artifacts/
    ├── shot_plan_artifacts/
    ├── continuity_refs/
    ├── event_video_tasks.json
    ├── event_video_summary.json
    └── final_orchestrated_events.mp4
```

## 5. 根目录与叙事阶段输出

| 文件或目录 | 内容与职责 |
| --- | --- |
| `fallback_rerun_manifest.json` | 只在恢复运行中存在。记录源运行目录、被要求重跑的节点、重跑 step 范围、是否重跑 downstream，以及恢复后的主要产物路径。 |
| `story_bundle.json` | 输入故事、角色与世界设定的结构化 bundle，供叙事推演与世界建模使用。 |
| `narrative_checkpoint.json` | 最近完成的叙事 step checkpoint，用于恢复推演。 |
| `step_logs.json` | 完整的角色级推演日志。每个角色 step 记录 action、evaluation、acceptance、reason、world context 等。 |
| `step_logs.partial.json` | 推演过程中的中间日志快照；用于异常中断后的诊断和恢复。 |
| `world_state_step<N>.json` | 第 `N` 个 step 结束后的世界状态快照。`step0` 是初始状态，后续文件体现事件造成的状态演进。 |
| `world_deltas/` | 每个 step 的 world state 更新判定和变化描述。只对发生物理外观变化的 location/item 进行更新。 |
| `pipeline_result.json` | 世界和叙事阶段的汇总结果。包含 `world_model`、`world_states`、`world_deltas`、`step_logs`、`event_keyframes`、角色配置与恢复元数据。它是节点筛选和路径编排的输入。 |
| `path_orchestration.log` | 节点筛选和路径编排的运行日志。 |

## 6. 视觉资产输出

### 6.1 `narrative_artifacts/portraits/`

角色初始化肖像图。每个主角一张独立人物图，生成提示词要求主体人物在画面中占比最大，用于人物身份一致性参考。

### 6.2 `narrative_artifacts/world_model/locations/`

世界地点初始化图。每个 location 可含远景、 中景、近景和特写视角，用于建立空间布局、材料、光线和环境状态。发生 world state 更新时，更新应只替换被判定需要变化的 location 视角图。

### 6.3 `narrative_artifacts/world_model/items/`

可叙事物品的初始化图，例如罗盘、账本、信件、无线电、隐藏救援路线。物品图强调物体主体占比和可辨识形状。

### 6.4 `narrative_artifacts/event_keyframes/`

事件关键帧。每个角色事件关键帧由当前事件描述、相关角色、相关 location 与显式提及物品生成。事件视频在 `event_keyframe_only` 模式下只选取这里的关键帧作为 Ref2VA 图片参考。

当前恢复目录存在一个历史特性：本地 `event_keyframes/` 只包含重新生成的 step 13、14、15；step 1 至 12 的事件视频任务引用了源目录 `outputs/gpt55_pre_video_demo1/all_banana/narrative_artifacts/event_keyframes/` 中已存在的关键帧。因此，复制或迁移本次结果时，必须同时保留源目录中的被引用图片，或把这些引用重新物化到恢复目录。

## 7. 节点筛选与路径编排输出

### 7.1 `path_orchestration/narrative_nodes.json`

节点筛选结果。主要字段：

| 字段 | 含义 |
| --- | --- |
| `selected_nodes` | 供视频生成的叙事节点。 |
| `rejected_nodes` | 未入选的候选节点和淘汰原因。 |
| `source_pipeline_result_json` | 输入 `pipeline_result.json` 的路径。 |
| `selector_config` | 节点选择参数。 |
| `role_candidate_count` / `candidate_count` / `selected_count` | 候选和最终节点统计。 |

当前样例中 `selected_count=31`。节点 ID 为 `NN_001` 等，视频 rank 由路径排序决定，不一定与 NN 编号相同。

### 7.2 `path_orchestration/path_llm_direct/narrative_path_plan.json`

LLM 路径编排结果。主要字段：

| 字段 | 含义 |
| --- | --- |
| `ordered_node_ids` | 最终视频/拼接顺序的权威节点序列。 |
| `transition_notes` | 相邻节点的叙事和视觉衔接说明。 |
| `storyline_fit_summary` | 对整体故事目标的匹配说明。 |
| `quality_checks` | 对因果性、节奏与衔接的检查。 |
| `node_rank_map` | 节点 ID 到排序 rank 的映射。 |

当 `--path-mode both` 时还可产生 MCTS 版本路径文件；当前视频阶段默认使用 `path_llm_direct/narrative_path_plan.json`。

### 7.3 `path_orchestration/pipeline_manifest.json`

路径阶段 manifest。记录 backend、故事、节点选择是否启用、候选数量、各 JSON artifact 路径和运行状态。

## 8. 视频阶段输出

视频输出根目录为：

```text
<run-root>/event_video_pipeline_minimax_h3_vllm_event_keyframe_only/
```

### 8.1 事件视频

文件命名：

```text
event_video_<rank>_<node_id>.mp4
```

例如：

```text
event_video_005_NN_006.mp4
```

`rank=005` 表示它在 `ordered_node_ids` 中的第五个事件，`NN_006` 是被路径编排的节点 ID。当前结果共有 31 个事件视频。

### 8.2 辅助视频

文件命名：

```text
aux_video_<task-rank>_establishing_<node_id>.mp4
aux_video_<task-rank>_world_transition_<node_id>.mp4
```

两类职责：

| 类型 | 触发条件 | 内容 | 参考图 |
| --- | --- | --- | --- |
| `establishing` | 某 location 第一次出现在视频顺序中 | 无人物空镜头，展示空间布局与气氛 | location/world update 图。 |
| `world_transition` | 相邻事件之间存在可见 world state 变化 | 无人物镜头，展示 location/item 的物理变化 | 变化后的 location 或 item 图。 |

辅助视频是独立于 `event_keyframe_only` 事件视频策略的世界展示镜头，因此可以使用地点和世界更新图。

### 8.3 `event_video_tasks.json`

视频任务清单。每个任务是一个 event 或 auxiliary 视频，包含：

```text
rank, node_id, task_type, output_video_path,
prompt, reference_images, reference_mode, reference_selection,
characters, locations, items, story_description,
dialogue_backfill, story_backfill, event_shot_plan,
duration_sec, num_frames, continuity_reference
```

其中 `event_shot_plan` 是 VideoAgent 对一条完整事件视频的 shot 级规划。`shot_video_tasks` 是每 shot 的 prompt 规划记录，标记为 `planned_only=true`，不会各自生成 MP4。

### 8.4 `event_video_summary.json`

视频阶段最终汇总。它记录：

```text
total, success, failed, skipped, generated,
concat_ok, concat_output, task_json
```

当前样例：`total=50`、`success=50`、`failed=0`、`skipped=0`、`concat_ok=true`。

### 8.5 `dialogue_artifacts/`

每个事件一组文件：

```text
event_dialogue_<rank>_<node_id>.json
event_dialogue_<rank>_<node_id>.meta.json
```

内容：

| 文件 | 内容 |
| --- | --- |
| `.json` | `Character1`、`Character2` 等键到角色台词的映射。没有台词时值为 `none`。 |
| `.meta.json` | `status`、角色槽位到真实人物名的 `slot_mapping`、用于 prompt 的 `rendered_dialogue` 和错误信息。 |

这些 JSON 本身不携带时间戳。时间信息来自 shot plan，并在 `.timeline.json` 中与文本合并。

### 8.6 `story_artifacts/`

每个事件一组文件：

```text
event_story_<rank>_<node_id>.json
event_story_<rank>_<node_id>.meta.json
```

`.json` 包含该节点的 `story_description`。它是时间顺序的故事化剧情描述，用于替代推理式 `video_description` 进入 H3 prompt。它默认不被视为旁白，不进入字幕时间轴。

### 8.7 `shot_plan_artifacts/`

文件命名：

```text
shot_plan_<rank>_<node_id>.json
```

每个文件包含：

```text
duration_sec, num_frames, shots, status, error
```

每个 `shots[]` 项包含：

```text
shot_index, start_timecode, start_sec, duration_sec,
video_description, cinematography, dialogue, narration, bgm
```

有效视频总时长只有两个选项：5 秒对应 124 帧，10 秒对应 243 帧。shot 的 `start_timecode` 用于 H3 prompt 内多分镜时间规划和字幕时间轴。

### 8.8 视频请求元数据与响应头

每个 MP4 同名存在：

```text
event_video_<...>.minimax_h3_vllm.json
event_video_<...>.minimax_h3_vllm.headers.txt
```

作用：

| 文件 | 内容 |
| --- | --- |
| `.minimax_h3_vllm.json` | Ref2VA 请求元数据，包括 `task`、输出路径、实际 `reference_images`、参考音频、HTTP 状态和响应字节数。用于确认参考图选择是否符合预期。 |
| `.headers.txt` | vLLM HTTP 响应头原始记录。用于服务端诊断。 |

### 8.9 视频字幕时间轴 JSON

每个 event 或 auxiliary MP4 都有同名 sidecar 文件：

```text
event_video_<rank>_<node_id>.timeline.json
aux_video_<task-rank>_<type>_<node_id>.timeline.json
```

结构示例：

```json
{
  "schema_version": "1.0",
  "video_path": ".../event_video_005_NN_006.mp4",
  "video_filename": "event_video_005_NN_006.mp4",
  "rank": 5,
  "node_id": "NN_006",
  "task_type": "event",
  "timing_basis": "shot_interval",
  "duration_sec": 10.0,
  "segments": [
    {
      "index": 1,
      "type": "dialogue",
      "speaker": "Ilya Trent",
      "character_slot": "Character1",
      "text": "Hold the pulse steady; every depth mark matches the ledger.",
      "shot_index": 1,
      "start_sec": 0.0,
      "end_sec": 3.0,
      "start_timecode": "00:00:00.000",
      "end_timecode": "00:00:03.000",
      "source": "shot_plan"
    }
  ]
}
```

字幕时间是 shot 区间级精度，`timing_basis="shot_interval"`。它不伪造逐字时间码。`story_description` 不会被自动转成旁白；仅显式 `narration`、`voiceover`、`narration_text` 或 `voiceover_text` 字段才产生 `type="narration"` 段。

这些 sidecar JSON 可直接转换为 SRT、VTT 或供后续配音/字幕合成工具消费。

当前历史运行的视频是在时间轴功能加入前完成的，因此 `.timeline.json` 已事后补写在对应 MP4 旁边。后续新运行由 `VideoAgent` 自动写入，任务 JSON 与 summary 行会带 `timeline_path`。

### 8.10 `continuity_refs/`

当启用上一视频尾帧或音频连续性时保存抽取的尾帧 JPG 和音频 WAV。当前样例禁用了尾帧和音频参考，因此该目录不作为当前视频参考输入。

### 8.11 `final_orchestrated_events.mp4`

最终按 `event_video_summary.json.generated[]` 中 `status=done` 的任务顺序拼接的成片。该样例包含事件视频和辅助视频，共 50 个输入视频。

## 9. 最常用的排查入口

| 目标 | 优先文件 |
| --- | --- |
| 查看完整状态 | `event_video_summary.json` |
| 查看某个视频用了哪些参考图 | 对应 `.minimax_h3_vllm.json` 的 `reference_images` |
| 查看该视频剧情与台词 | `story_artifacts/`、`dialogue_artifacts/` 和 `shot_plan_artifacts/` 中相同 rank/node ID 文件 |
| 制作字幕 | 对应 MP4 的 `.timeline.json` |
| 查视频顺序 | `path_llm_direct/narrative_path_plan.json` 的 `ordered_node_ids` |
| 查节点由哪些角色 step 形成 | `narrative_nodes.json` 的 `selected_nodes` |
| 查推演或 fallback | `step_logs.json`、`fallback_rerun_manifest.json`、`narrative_checkpoint.json` |
| 查世界变化 | `world_state_step<N>.json`、`world_deltas/` |
| 查 H3/vLLM 服务请求 | `video_generation.log`、每个视频 `.minimax_h3_vllm.json` 和 `.headers.txt` |

## 10. 当前关键路径

| 名称 | 绝对路径 |
| --- | --- |
| 项目根目录 | `/data1/AgentG/hhb/CameraMaster` |
| 当前恢复后的 pipeline 根目录 | `/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana` |
| 当前视频输出目录 | `/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/event_video_pipeline_minimax_h3_vllm_event_keyframe_only` |
| 最终成片 | `/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/event_video_pipeline_minimax_h3_vllm_event_keyframe_only/final_orchestrated_events.mp4` |
| 节点列表 | `/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/path_orchestration/narrative_nodes.json` |
| LLM 路径计划 | `/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/path_orchestration/path_llm_direct/narrative_path_plan.json` |
| 视频任务清单 | `/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/event_video_pipeline_minimax_h3_vllm_event_keyframe_only/event_video_tasks.json` |
| 视频汇总 | `/data1/AgentG/hhb/CameraMaster/outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/event_video_pipeline_minimax_h3_vllm_event_keyframe_only/event_video_summary.json` |
