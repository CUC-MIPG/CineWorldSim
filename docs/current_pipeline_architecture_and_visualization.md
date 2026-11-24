# CineWorldSim Pipeline Idea And Flow

## Idea

CineWorldSim 当前 pipeline 的核心思想是：**先构建可持续演化的叙事影视世界，再从推演出的时空状态中选择可拍摄事件，最后把事件转译为故事视觉生成请求**。

它不是直接把一段故事文本丢给视频模型生成整段视频，而是把电影生成拆成三个相互解耦但连续衔接的层次：

1. **World / Character Initialization**
   - 系统先从故事中抽取角色、空间、关键物品和初始世界状态。
   - 每个主角会形成独立的角色配置，包括 `final_goal`、`role_profile`、`known_relations`、`current_environment`。
   - 世界侧会形成地点和物品蓝图，并生成初始世界空间图与关键物品图。
   - 这样后续视觉生成不是从空白文本开始，而是锚定在一个已经初始化的影视世界里。

2. **Simulation-based Narrative Evolution**
   - 角色不是一次性写死剧情，而是在 `StateAnalysisMachine` 中按 step 同步推演。
   - 每个 step 中，所有角色基于自己的目标、关系、环境和上一轮状态输出动作。
   - 世界状态同时更新，包括地点、物品、环境状态、角色行动和故事完成度。
   - 事件关键帧基于角色初始图、世界空间图和当前事件文本生成，用来把推演状态转成可视化锚点。

3. **State-Anchored Cinematic Generation**
   - 系统从推演出的时空状态图中筛选 `Narrative Nodes`，也就是真正适合进入电影表达的关键事件。
   - 节点默认粒度是“一个角色在一个 step 中发生的事”；只有当多个角色在同一 step、同一地点，并且存在直接交互时，才合并为一个多角色事件节点。
   - `PathOrchestratorAgent` 不重新拆分或合并节点，只负责把这些原子事件排列成可观看的剧情路径。
   - `VideoAgent` 把每个事件转换为 MiniMax-H3 prompt；当前多分镜不是多次视频调用，而是在同一次视频请求的 prompt 中组织 `[Shot 1] / [Shot 2] / ...`。

这套设计的学术和工程价值在于：它把**开放式多智能体推演**与**结构化电影表达**分离。底层可以持续生成丰富状态，上层只截取有戏剧价值的时空切片进行影视化表达。这样可以缓解文本直出视频常见的状态遗忘、人物不一致、空间跳变和叙事扁平化问题。

当前 pipeline 还引入了两类重要约束：

- **人物一致性约束**：只有当前事件确实需要多人物交互时，才允许使用多个人物参考图；否则 prompt 会明确要求避免无关行人、人群和非主角人物出现。
- **连续性参考约束**：上一事件尾帧默认不作为参考；只有在开启 `--use-previous-tail-frame` 后，`VideoAgent` 才会让 LLM 根据剧情连贯性、人物关系、场景一致性和物体状态判断是否使用上一事件尾帧。

## Pipeline Flow

### 1. Story Input

输入是一段完整故事文本，例如：

- `inputs/demo1_adventure_same_space.txt`
- `inputs/demo2_adventure_three_spaces.txt`

主要入口脚本：

```bash
scripts/run_full_narrative_world_video_pipeline.py
```

核心输入参数：

- `--story-file`：故事文本路径。
- `--characters`：可选角色列表；如果不显式传入，脚本使用默认角色名，实际角色配置仍会由故事和 orchestrator 生成。
- `--backend-mode all_banana`：使用 banana 图像生成后端。
- `--max-steps 15`：最多推演 15 个 step。
- `--path-mode both`：同时运行 LLM path 和 MCTS path。

主要产物：

```text
<output_dir>/
  full_pipeline_manifest.json
  all_banana/
    pipeline_result.json
```

### 2. OrchestratorAgent: Character And World Setup

脚本内部调用：

```text
narrative/simulation_scaffold.py
  OrchestratorAgent
```

执行逻辑：

1. 读取 story。
2. 解析角色和角色关系。
3. 为每个角色生成可推演配置：
   - `final_goal`
   - `role_profile`
   - `known_relations`
   - `current_environment`
4. 生成影视世界的初始描述。

主要产物写入：

```text
all_banana/pipeline_result.json
  character_configs
  world_model
```

### 3. WorldModelAgent: World Blueprint And Initial Visuals

脚本内部调用：

```text
narrative/simulation_scaffold.py
  WorldModelAgent
narrative/world_renderer.py
  WorldRenderer
```

执行逻辑：

1. 从故事中抽取地点、关键物品和空间关系。
2. 为地点生成初始世界空间图。
3. 为关键物品生成初始物品图。
4. prompt 中加入人物控制逻辑：世界图和物品图默认避免无关人物、人群和行人，除非资产本身必须包含人物。

主要产物：

```text
all_banana/narrative_artifacts/world_model/
  locations/*
  items/*
```

### 4. StateAnalysisMachine: Step-by-step Co-simulation

脚本内部调用：

```text
narrative/simulation_scaffold.py
  StateAnalysisMachine
```

执行逻辑：

1. 进入 step-by-step 推演。
2. 每个 step 中，所有角色同步输出动作。
3. 每个角色动作会包含当前行动、目标、原因、位置和可能的交互对象。
4. 系统同步维护故事完成度、阶段摘要和角色状态。

主要产物：

```text
all_banana/step_logs.json
all_banana/pipeline_result.json
  step_logs
```

### 5. WorldDeltaJudge And WorldStateTracker: World State Evolution

脚本内部调用：

```text
narrative/world_delta_judge.py
  WorldDeltaJudge
narrative/world_state_tracker.py
  WorldStateTracker helpers
```

执行逻辑：

1. 根据每个 step 的角色动作判断世界状态变化。
2. 更新地点、物品、环境状态。
3. 必要时生成新的世界状态演化图。
4. 将每轮变化写成 delta，并把累积状态写成 world state。

主要产物：

```text
all_banana/world_state_step0.json
all_banana/world_state_step1.json
...
all_banana/world_state_step15.json

all_banana/world_deltas/
  world_delta_step1.json
  ...
  world_delta_step15.json

all_banana/narrative_artifacts/world_updates/
```

### 6. EventKeyframeRenderer: Event Visual Anchors

脚本内部调用：

```text
narrative/event_keyframe_renderer.py
  EventKeyframeRenderer
```

执行逻辑：

1. 为每个 step、每个角色生成事件关键帧。
2. 事件关键帧会参考：
   - 角色初始图 / 角色形象信息
   - 当前世界空间图
   - 当前事件文本
   - 当前人物控制策略
3. 如果事件是单人事件，prompt 会压制无关人物、人群和行人。
4. 如果事件确实是多角色交互，才允许使用多角色参考图。

主要产物：

```text
all_banana/narrative_artifacts/event_keyframes/
  step_001/<Role>_event_keyframe.*
  ...
  step_015/<Role>_event_keyframe.*
```

### 7. NarrativeNodeSelector: Select Cinematic Events

脚本内部调用：

```text
agents/narrative_node_selector.py
  NarrativeNodeSelector
```

执行逻辑：

1. 先从 `step_logs` 构建角色级候选事件。
2. 默认候选粒度是：**一个角色在一个 step 中发生的事件**。
3. 只有满足以下条件时才合并为多角色节点：
   - 同一个 step。
   - 同一个地点。
   - 角色之间存在直接交互。
4. 使用规则评分或 LLM scoring 选择最适合视频化的节点。
5. 如果测试目标是 15 个节点，需要显式设置：
   - `--node-selection-mode top_k`
   - `--node-min-nodes 15`
   - `--node-max-nodes 15`

主要产物：

```text
all_banana/path_orchestration/narrative_nodes.json
```

### 8. PathOrchestratorAgent And MCTS Path: Order Events

脚本内部调用：

```text
agents/path_orchestrator_agent.py
  PathOrchestratorAgent
agents/path_orchestrator_mcts_agent.py
  PathOrchestratorMCTSAgent
```

执行逻辑：

1. 读取 `narrative_nodes.json`。
2. 将 selected nodes 转换成 path candidates。
3. `PathOrchestratorAgent` 使用 LLM 生成顺序式剧情路径。
4. `PathOrchestratorMCTSAgent` 使用 MCTS 生成另一套路径结果。
5. 路径阶段只排序，不改变节点粒度，不额外合并或拆分事件。

主要产物：

```text
all_banana/path_orchestration/path_candidates.json

all_banana/path_orchestration/path_llm_direct/
  path_plan.json
  narrative_path_plan.json

all_banana/path_orchestration/path_mcts/
  narrative_path_plan_mcts.json
  mcts_trace.jsonl
```

### 9. VideoAgent: Story Description, Dialogue, H3 Skill Prompt, Video

脚本内部调用：

```text
agents/video_agent.py
  VideoAgent
agents/h3_prompt_adapter.py
agents/skills/adapter.py
```

执行逻辑：

1. 按 `narrative_path_plan.json` 的顺序逐个处理事件。
2. 保留原始 `video_description`。
3. 额外通过 LLM 生成 `story_description`，把推理式事件描述转换成可拍摄的剧情动作描述。
4. 可选通过 LLM 自动生成 dialogue JSON，默认开启回填：
   - 无对话时写 `none`。
   - 有对话时按剧情时间顺序组织多轮角色台词。
   - 未识别角色时使用 `Character1`、`Character2` 这类默认名。
5. 使用 `story_description + dialogue + reference assets` 进入 H3 skill prompt 层。
6. 输出 MiniMax-H3 `official_ref` 风格 prompt。
7. 每个 event 生成一个视频文件。

当前视频语义：

- `MINIMAX_H3_NUM_FRAMES=243` 表示每个事件视频请求生成 243 帧。
- `--event-shot-count 4` 表示 prompt 内组织 4 个 shot 段落。
- `--event-duration-sec 10` 表示这个事件 prompt 的分镜时间总长度按 10 秒组织。
- 多分镜发生在 prompt 文本中，不会拆成 4 次 MiniMax 调用。

主要产物：

```text
all_banana/event_video_pipeline_minimax_h3_vllm_multishot/
  event_video_001_NN_001.mp4
  event_video_002_NN_002.mp4
  ...
  event_video_tasks.json
  event_video_summary.json
  final_orchestrated_events.mp4
```

### 10. PostProduction And Visualization

脚本内部调用：

```text
agents/video_agent.py
  concat videos
scripts/generate_full_pipeline_stage_visualization.py
```

执行逻辑：

1. 将所有事件视频按 path 顺序拼接。
2. 输出最终成片。
3. 可视化脚本读取 story、world model、step logs、world state、selected nodes、path plan、video tasks 和 video summary。
4. 生成完整 HTML 页面，展示从故事输入到最终视频的全过程。

主要产物：

```text
all_banana/event_video_pipeline_minimax_h3_vllm_multishot/final_orchestrated_events.mp4
all_banana/visualizations/full_stage_pipeline_visualization.html
```

### 11. Current Two-demo Full Test Command

当前两个 adventure demo 的完整测试入口是同一个 shell 指令顺序跑两次：

```bash
PYTHONUNBUFFERED=1 \
MINIMAX_H3_PROMPT_STYLE=official_ref \
MINIMAX_H3_NUM_FRAMES=243 \
MINIMAX_H3_VLLM_ENABLED=true \
MINIMAX_H3_VLLM_API_URL=http://127.0.0.1:9001/v1/videos/sync \
MINIMAX_H3_CUDA_VISIBLE_DEVICES=2,3 \
MINIMAX_H3_VLLM_TIMEOUT_SEC=3600 \
bash -lc '
set -euo pipefail

OUT_ROOT=/data1/AgentG/hhb/CameraMaster/outputs/four_genre_demos_15nodes_20260816
mkdir -p "$OUT_ROOT"

run_one() {
  local story_file="$1"
  local out_dir="$2"

  conda run -n mifage python scripts/run_full_narrative_world_video_pipeline.py \
    --story-file "$story_file" \
    --output-dir "$out_dir" \
    --backend-mode all_banana \
    --path-mode both \
    --max-steps 15 \
    --stop-on-complete \
    --enable-narrative-node-selector \
    --node-selection-mode top_k \
    --node-min-nodes 15 \
    --node-max-nodes 15 \
    --node-llm-scoring \
    --strict-online \
    --multi-shot-events \
    --event-shot-count 4 \
    --event-duration-sec 10 \
    --max-reference-images 4 \
    --use-previous-tail-frame \
    --no-use-previous-audio \
    --no-upload-previous-audio \
    --dialogue-backfill \
    --log-level INFO
}

run_one /data1/AgentG/hhb/CameraMaster/inputs/demo1_adventure_same_space.txt \
        "$OUT_ROOT/01_demo1_shared_space"

run_one /data1/AgentG/hhb/CameraMaster/inputs/demo2_adventure_three_spaces.txt \
        "$OUT_ROOT/02_demo2_three_spaces"
'
```
