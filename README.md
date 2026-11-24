# CineWorldSim Command Guide (RAG + Agent Workflow)

本仓库已经实现：

1. 多书籍 RAG（LangChain + LangGraph）
2. `MovieProducer` 多知识库 Agent
3. 多智能体生成工作流（Director / Shot / Story Supervisor / Video / Post-Production）
4. 信息池（Info Pool）：记录工作流中所有智能体输出

## Environment

```bash
conda activate mifage
# 或在下面所有命令里使用 conda run -n mifage ...
```

请先准备 `.env`（可由 `.env.example` 复制）并配置：

- `RAG_API_KEY / RAG_BASE_URL / RAG_MODEL_NAME`
- `DIRECTOR_API_KEY / DIRECTOR_BASE_URL / DIRECTOR_MODEL_NAME`
- `SHOT_API_KEY / SHOT_BASE_URL / SHOT_MODEL_NAME`
- `VIDEO_API_KEY / VIDEO_BASE_URL / VIDEO_MODEL_NAME`
- `SUPERVISOR_API_KEY / SUPERVISOR_BASE_URL / SUPERVISOR_MODEL_NAME`
- `OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL`（fallback）
- `IMAGE_GEN_PROVIDER=flux|nano_banana`
- `VIDEO_GEN_PROVIDER=magref|wan_r2v`
- `NANO_BANANA_*`、`WAN_R2V_*`、`MAGREF_API_URL`、`FLUX_API_URL` 等后端参数

视频超时说明：
- 主工作流（`VideoAgent`）默认不传视频请求超时，会等待单条视频生成完成。
- 若你在工具级脚本里显式使用超时变量：`MAGREF_TIMEOUT_SEC`、`WAN_R2V_TIMEOUT_SEC` 设为 `0`/`None` 可关闭超时。

## RAG Commands

### 1) 查看当前知识库书目

```bash
conda run -n mifage python scripts/rag_list_books.py
```

### 2) 构建索引（单本）

```bash
conda run -n mifage python scripts/rag_build.py \
  --book-id movie_narratology_theory_cases_cn \
  --rebuild
```

### 3) 构建索引（全部书籍）

```bash
conda run -n mifage python scripts/rag_build.py --all --rebuild
```

或：

```bash
conda run -n mifage python scripts/rag_build_all.py --rebuild
```

### 4) 单书 RAG 查询（抽取式）

```bash
conda run -n mifage python scripts/rag_query.py \
  --book-id movie_narratology_theory_cases_cn \
  --query "这本书如何界定当代叙事学与电影叙事理论的关系？"
```

### 5) 单书 RAG 查询（LLM 生成式）

```bash
conda run -n mifage python scripts/rag_query.py \
  --book-id movie_narratology_theory_cases_cn \
  --query "总结本书对电影叙事理论框架的核心观点" \
  --use-llm \
  --model gpt-4o
```

### 6) MovieProducer Agent：查看挂载的全部知识库

```bash
conda run -n mifage python scripts/movie_producer_query.py --list-kbs
```

### 7) MovieProducer Agent：多知识库联合查询

```bash
conda run -n mifage python scripts/movie_producer_query.py \
  --question "如何把叙事学原则转换成可执行的场景设计策略？" \
  --template scene_design \
  --constraints "面向短片，3场戏，预算受限" \
  --k-per-book 2 \
  --top-k-total 10 \
  --save-json outputs/movie_producer/query_scene_design.json
```

可选模板：

- `general_research`
- `story_development`
- `scene_design`
- `shot_design`

按指定书籍检索（可选）：

```bash
conda run -n mifage python scripts/movie_producer_query.py \
  --question "如何设计长镜头与主观视角结合的叙事策略？" \
  --template shot_design \
  --book-ids movie_narratology_theory_cases_cn,film_directing_shot_by_shot \
  --k-per-book 3 \
  --top-k-total 8
```

## Workflow Commands (Agents)

提示词模板库参数已统一为 `--prompt-library`，默认值为：

- `/data1/AgentG/hhb/CameraMaster/prompt_library/default`

工作流日志参数：

- `--log-level`（默认 `INFO`，可选 `DEBUG/INFO/WARNING/ERROR`）
- 运行时会在终端输出 stage/scene/shot 进度。

参考图输入参数已统一为 `--reference-assets-json`（单文件），格式示例见：

- `/data1/AgentG/hhb/CameraMaster/configs/reference_assets.example.json`

### 0) 固定运行入口（推荐）

3智能体固定入口（Director + Shot + Video）：

```bash
conda run -n mifage python scripts/run_three_agents_only.py \
  --storyline "一个落魄特技车手为寻找失踪搭档重返地下赛道。" \
  --prompt-library /data1/AgentG/hhb/CameraMaster/prompt_library/default \
  --output-dir outputs/three_agents_only_demo \
  --dry-run-video \
  --no-strict-online
```

```bash
conda run -n mifage python scripts/resume_three_agents_only.py \
  --output-dir outputs/three_agents_only_demo \
  --dry-run-video \
  --no-strict-online
```

5智能体全流程固定入口（Director + Shot + Story Supervisor + Video + Post-Production）：

```bash
conda run -n mifage python scripts/run_five_agents_full.py \
  --storyline "同一剧情，生成分镜、剧情监制规划与后期任务。" \
  --prompt-library /data1/AgentG/hhb/CameraMaster/prompt_library/default \
  --output-dir outputs/five_agents_full_demo \
  --dry-run-video \
  --no-strict-online
```

```bash
conda run -n mifage python scripts/resume_five_agents_full.py \
  --output-dir outputs/five_agents_full_demo \
  --dry-run-video \
  --no-strict-online
```

叙事世界 + 路径编排统一入口（Orchestrator + WorldModel + SAM + WorldState + EventKeyframe + PathOrchestrator）：

```bash
conda run -n mifage python scripts/run_narrative_world_path_pipeline.py \
  --storyline "一个记者进入废弃灯塔，发现破裂罗盘，并逼迫隐藏目击者说出档案室真相。" \
  --characters Journalist,HiddenWitness,Caretaker \
  --backend-mode all_banana \
  --max-steps 3 \
  --output-dir outputs/narrative_world_path_demo \
  --path-mode llm \
  --no-strict-online \
  --log-level INFO
```

该入口默认启用 Stage 2 `NarrativeNodeSelector`，路径编排消费的是筛选后的 `NN_001/NN_002/...`，不是原始 `step::role` 节点。关键输出：

- `path_orchestration/narrative_nodes.json`
- `path_orchestration/path_candidates.json`
- `path_orchestration/path_llm_direct/narrative_path_plan.json`
- `path_orchestration/pipeline_manifest.json`

单独测试 Stage 2：

```bash
conda run -n mifage python scripts/run_narrative_node_selector.py \
  --pipeline-result-json outputs/narrative_world_pipeline_all_banana_real_retry/all_banana/pipeline_result.json \
  --output-dir outputs/narrative_node_selector_demo \
  --selection-mode hybrid \
  --score-threshold 0.55 \
  --min-nodes 3 \
  --max-nodes 8 \
  --no-llm-scoring
```

打开在线评分后，选择器会复用仓库现有的 `OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL` 配置：

```bash
conda run -n mifage python scripts/run_narrative_node_selector.py \
  --pipeline-result-json outputs/narrative_world_pipeline_all_banana_real_retry/all_banana/pipeline_result.json \
  --output-dir outputs/narrative_node_selector_demo_llm \
  --selection-mode hybrid \
  --score-threshold 0.55 \
  --min-nodes 3 \
  --max-nodes 8 \
  --llm-scoring
```

如果需要回退到旧逻辑，让 PathOrchestrator 直接消费原始 `step::role` 节点：

```bash
conda run -n mifage python scripts/run_narrative_world_path_pipeline.py \
  --existing-pipeline-result-json outputs/narrative_world_pipeline_all_banana_real_retry/all_banana/pipeline_result.json \
  --storyline "一个记者进入废弃灯塔，发现破裂罗盘，并逼迫隐藏目击者说出档案室真相。" \
  --output-dir outputs/narrative_world_path_raw_nodes \
  --no-enable-narrative-node-selector \
  --path-mode llm \
  --no-strict-online
```

同时对比 FLUX 和 banana：

```bash
conda run -n mifage python scripts/run_narrative_world_path_pipeline.py \
  --storyline "一个记者进入废弃灯塔，发现破裂罗盘，并逼迫隐藏目击者说出档案室真相。" \
  --characters Journalist,HiddenWitness,Caretaker \
  --run-both \
  --max-steps 3 \
  --output-dir outputs/narrative_world_path_dual_demo \
  --path-mode both \
  --no-strict-online \
  --log-level INFO
```

如果已经有 `pipeline_result.json`，可以跳过前面的叙事世界生成，只做路径编排：

```bash
conda run -n mifage python scripts/run_narrative_world_path_pipeline.py \
  --existing-pipeline-result-json outputs/narrative_world_pipeline_all_banana_real_retry/all_banana/pipeline_result.json \
  --storyline "一个记者进入废弃灯塔，发现破裂罗盘，并逼迫隐藏目击者说出档案室真相。" \
  --output-dir outputs/narrative_world_path_from_existing \
  --path-mode llm \
  --no-strict-online \
  --log-level INFO
```

### 1) 可配置入口（高级）

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "一个落魄特技车手为寻找失踪搭档重返地下赛道。" \
  --output-dir outputs/three_agents_demo \
  --max-shots-per-scene 4 \
  --enable-story-supervisor \
  --character-image-policy auto \
  --scene-image-policy auto \
  --video-mode reference \
  --no-enable-post-production \
  --dry-run-video \
  --no-strict-online
```

运行后会自动生成信息池目录：

- `outputs/.../info_pool/events.jsonl`：按时间追加记录所有 agent 输出 + 流程起止事件
- `outputs/.../info_pool/latest_snapshot.json`：每个 agent / stage 的最新输出快照

### 2) 文本+角色图输入（JSON），自动补场景图

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "雨夜都市追踪任务，主角逐步逼近真相。" \
  --reference-assets-json /abs/path/to/reference_assets.json \
  --enable-story-supervisor \
  --scene-image-policy auto \
  --character-image-policy never \
  --video-mode reference \
  --no-enable-post-production \
  --dry-run-video \
  --output-dir outputs/three_agents_text_plus_char
```

### 3) 已有角色图+场景图，不再生成图像

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "同一剧情，直接使用已有美术资产出分镜与视频任务。" \
  --reference-assets-json /abs/path/to/reference_assets.json \
  --enable-story-supervisor \
  --character-image-policy never \
  --scene-image-policy never \
  --video-mode reference \
  --no-enable-post-production \
  --dry-run-video \
  --output-dir outputs/three_agents_full_assets
```

### 4) 关键帧模式（当前为伪代码占位）

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "通过关键帧方式生成镜头任务。" \
  --enable-story-supervisor \
  --video-mode keyframe \
  --no-enable-post-production \
  --character-image-policy auto \
  --scene-image-policy auto \
  --dry-run-video \
  --output-dir outputs/three_agents_keyframe_demo
```

### 5) 从 checkpoint 恢复

```bash
conda run -n mifage python scripts/resume_three_agents.py \
  --output-dir outputs/three_agents_demo \
  --enable-story-supervisor \
  --no-enable-post-production \
  --dry-run-video \
  --no-strict-online
```

### 6) 执行队列中的视频任务（读取 `video_info.json`）

```bash
conda run -n mifage python scripts/process_video_tasks.py \
  --task-json outputs/three_agents_demo/videos/video_info.json \
  --strict-online
```

### 7) 启用 Post-Production Agent（伪代码工具）

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "同一剧情，生成后期配音与配乐任务。" \
  --output-dir outputs/three_agents_with_post \
  --enable-post-production \
  --dry-run-video \
  --no-strict-online
```

后期产物目录：

- `outputs/three_agents_with_post/post_production/post_production_info.json`
- `outputs/three_agents_with_post/post_production/dubbing/*.todo.txt`
- `outputs/three_agents_with_post/post_production/music/*.todo.txt`
- `outputs/three_agents_with_post/post_production/narration/*.todo.txt`
- `outputs/three_agents_with_post/post_production/master/*.todo.txt`

### 8) 单独运行 Post-Production Agent

```bash
conda run -n mifage python scripts/run_post_production.py \
  --storyboard-json outputs/three_agents_case/storyboard.json \
  --story-supervisor-plan outputs/three_agents_case/story_supervisor/story_supervisor_plan.json \
  --output-dir outputs/three_agents_case/post_production \
  --final-video-path outputs/three_agents_case/final_concat.mp4
```

### 9) 单独运行剧情监制 Agent（Story Supervisor）

```bash
conda run -n mifage python scripts/run_story_supervisor.py \
  --storyboard-json outputs/three_agents_case/storyboard.json \
  --output-dir outputs/three_agents_case/story_supervisor \
  --shot-min-sec 1.5 \
  --shot-max-sec 6.0 \
  --scene-sec-per-shot 3.0 \
  --no-strict-online
```

`StorySupervisorAgent` 输出中会包含 `shot_plans`，逐镜头给出 `duration_sec` 与节奏说明，并自动约束总时长与场景时长一致。

`reference_assets.json` 支持结构（角色/场景都可一对多）：

```json
{
  "character": {
    "角色名1": ["/abs/path/to/character_refs1.png", "/abs/path/to/character_refs2.png"],
    "角色名2": ["/abs/path/to/character_refs3.png", "/abs/path/to/character_refs4.png"]
  },
  "scene": {
    "Scene1": ["/abs/path/to/scene_refs1.png", "/abs/path/to/scene_refs2.png"],
    "Scene2": ["/abs/path/to/scene_refs3.png", "/abs/path/to/scene_refs4.png"]
  }
}
```

路径可使用绝对路径；若使用相对路径，则相对于 `reference_assets.json` 文件所在目录解析。

## Tool Test Commands (Optional)

### 1) 测试 `generate_image_nano_banana` + `generate_video_wan_r2v`

```bash
conda run -n mifage python scripts/test_generation_tools.py \
  --image-output outputs/tool_unit_tests/nano_image_test.png \
  --video-output outputs/tool_unit_tests/wan_video_test.mp4
```

### 2) 单测 nano-banana I2I

```bash
conda run -n mifage python scripts/test_nano_i2i.py \
  --input-image /data1/AgentG/hhb/nano_banana/character_Cassidy.png \
  --output-image outputs/tool_unit_tests/nano_i2i_test.png
```

## Concrete End-to-End Example

下面是一套可直接执行的示例流程：

1. 先构建全量索引

```bash
conda run -n mifage python scripts/rag_build.py --all --rebuild
```

2. 用 `MovieProducer` 生成场景设计建议并保存 JSON

```bash
conda run -n mifage python scripts/movie_producer_query.py \
  --question "赛博雨夜短片如何设计三场戏的叙事推进和镜头策略？" \
  --template scene_design \
  --constraints "时长3分钟，低成本，2位角色" \
  --k-per-book 2 \
  --top-k-total 10 \
  --save-json outputs/movie_producer/case_scene_design.json
```

3. 运行智能体流程（先 dry-run 产出 storyboard 和 video task）

```bash
conda run -n mifage python scripts/run_three_agents.py \
  --storyline "赛博雨夜中，一名调查员追查失踪搭档并揭开监控城市阴谋。" \
  --output-dir outputs/three_agents_case \
  --max-shots-per-scene 4 \
  --enable-story-supervisor \
  --character-image-policy auto \
  --scene-image-policy auto \
  --video-mode reference \
  --no-enable-post-production \
  --dry-run-video \
  --no-strict-online
```

生成结果重点查看：

- `outputs/three_agents_case/storyboard.json`
- `outputs/three_agents_case/story_supervisor/story_supervisor_plan.json`
- `outputs/three_agents_case/videos/video_info.json`
- `outputs/movie_producer/case_scene_design.json`
- `outputs/three_agents_case/info_pool/events.jsonl`
- `outputs/three_agents_case/info_pool/latest_snapshot.json`
