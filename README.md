# CineWorldSim：RAG、叙事推演与对比实验

本文前半部分介绍 RAG 和传统多智能体流程；当前叙事世界推演、StoryControl v4、CameraMaster–ViMax 四组实验与统一评估的可执行命令见文末对应章节。更完整的恢复与指标说明见 `docs/run_and_evaluation_guide.md`。

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

## 叙事世界推演与视频 Pipeline

当前主流程是：故事文本与角色 → 世界、地点和角色初始化 → 世界模型按轮次分发当前可观察信息 → 角色选择动作 → 动作评估及世界状态更新 → 事件关键帧 → 叙事节点筛选与排序 → 分镜视频与拼接。`--max-steps` 是推演轮数上限，不表示必须运行到该轮；输出的节点数也可能少于上限。

只运行通用故事的世界推演、查看角色动作和 world state：

```bash
cd /data1/AgentG/hhb/CameraMaster
/data1/AgentG/anaconda3/envs/mifage/bin/python -u scripts/run_dual_backend_narrative_world_pipeline.py \
  --story-file inputs/demo1_adventure_same_space.txt \
  --characters 'Mara Venn,Ilya Trent,Nora Hale' \
  --backend-mode all_banana --max-steps 15 \
  --output-dir outputs/readme_narrative_world \
  --strict-online --retries 3 --log-level INFO
```

主要查看 `outputs/readme_narrative_world/all_banana/pipeline_result.json`、`narrative_checkpoint.json` 和 `world_state_step*.json`。通用入口默认**不启用** benchmark Track B 的 StoryControl v4。要从推演继续生成分镜视频，使用完整入口：

```bash
/data1/AgentG/anaconda3/envs/mifage/bin/python -u scripts/run_full_narrative_world_video_pipeline.py \
  --story-file inputs/demo1_adventure_same_space.txt \
  --characters 'Mara Venn,Ilya Trent,Nora Hale' \
  --output-dir outputs/readme_narrative_video \
  --backend-mode all_banana --path-mode llm --max-steps 15 \
  --enable-narrative-node-selector --node-selection-mode top_k \
  --node-min-nodes 1 --node-max-nodes 30 \
  --max-reference-images 4 --strict-online --retries 3 --log-level INFO
```

完整入口在 `outputs/readme_narrative_video/all_banana/` 下保存推演、路径编排与 `event_video_pipeline_minimax_h3_vllm_multishot/final_orchestrated_events.mp4`。生成会调用已配置的 LLM、图像和 MiniMax 视频服务；各服务变量见 `.env.example`。完整阶段说明见 `docs/current_pipeline_execution_and_output_reference.md`。

### Benchmark Track B 的 StoryControl 推演

与上面的通用故事入口不同，以下命令只给 CameraMaster `story_overview` 作为剧情内容；benchmark 的 shot 文本仅用于抽取角色名称。StoryControl v4 检查必需剧情事件，缺少证据时在 `story_coverage_failure.json` 列出缺项并阻止视频生成。

```bash
/data1/AgentG/anaconda3/envs/mifage/bin/python -u scripts/run_vimax_track_ab_pipeline.py \
  --benchmark-root /data1/AgentG/hhb/CameraMaster/ViMax/vimax_benchmark \
  --case artist_extreme_weather_typeA.json --track B \
  --output-root outputs/readme_artist_track_b_v4 \
  --backend all_banana --max-steps 40 --max-nodes 30 \
  --max-reference-images 4 --track-b-story-control \
  --track-b-require-full-coverage --strict-online \
  --retries 3 --log-level INFO
```

重点查看 `outputs/readme_artist_track_b_v4/artist_extreme_weather_typeA/track_B/narrative_simulation/all_banana/` 中的 `narrative_checkpoint.json`、`story_control.json` 和 `pipeline_result.json`，以及 `track_B/story_coverage_report.json`、`narrative_nodes.json`、`path_plan.json`、`video/event_video_summary.json`。`--max-nodes 30` 只限制最多选取 30 个节点，不强制凑满 30 个。

## ViMax 基线与统一对比实验

改造后的 ViMax 源码在根目录 `ViMax/`；来源与许可证见 `ViMax/INTEGRATION.md`。两种方法共用 `ViMax/vimax_benchmark/` 的 case。Track A 使用 benchmark 的逐 shot 描述，Track B 只用故事梗概；同一 Track 内比较两种方法。下面的联合脚本使用 `mifage` 运行 CameraMaster、`Vimax` 运行 ViMax。

### 1. 生成四组视频

```bash
cd /data1/AgentG/hhb/CameraMaster
export CM_ROOT="$PWD"
export CM_OUT="$CM_ROOT/outputs/readme_artist_comparison_v4"
export VIMAX_OUT="$CM_ROOT/ViMax/outputs/vimax_benchmark/readme_artist_comparison"
export CMP_OUT="$CM_ROOT/outputs/comparisons/readme_artist_comparison_v4"

bash scripts/run_joint_cameramaster_vimax_case.sh \
  --case artist_extreme_weather_typeA.json \
  --camera-track both --vimax-track both \
  --camera-output-root "$CM_OUT" --vimax-output-dir "$VIMAX_OUT" \
  --backend all_banana --max-steps 40 --max-nodes 30 \
  --max-reference-images 4 --retries 3 --log-level INFO
```

`--camera-track both` 必须显式给出：联合脚本默认只运行 CameraMaster Track B。若只做 Track B，可将 `--camera-track` 和 `--vimax-track` 都设为 `B`。ViMax 的独立入口是 `ViMax/scripts/run_vimax_benchmark_pipeline.py`，在线生成需传 `--run --resume`。联合脚本将成功阶段记录在 `$CM_OUT/.joint_run_state/`；中断后原样重跑同一命令以继续，换模型或改变实验条件时请使用新的输出目录。

### 2. 统一评估并生成对照表

四组视频全部完成后运行：

```bash
/data1/AgentG/anaconda3/envs/mifage/bin/python -u evaluation/run_cameramaster_vimax_comparison.py \
  --benchmark-case "$CM_ROOT/ViMax/vimax_benchmark/artist_extreme_weather_typeA.json" \
  --cameramaster-output-root "$CM_OUT/artist_extreme_weather_typeA" \
  --vimax-output-root "$VIMAX_OUT/artist_extreme_weather" \
  --output-dir "$CMP_OUT" --run-all --resume --timeout 900
```

只生成了 Track B 时，在评估命令中加入 `--tracks B`。结果位于 `$CMP_OUT/quantitative/comparison.csv`、`comparison.md`、各方法的 visual/planning/SSS JSON，以及 `$CMP_OUT/qualitative/index.html`。后者按 Track 和镜头序号并排展示视频及分镜文本；视频通过本机链接引用，不随 Git 提交。`--resume` 跳过已有的非空 JSON 子结果，要求复用同一 case、模型和输出目录；需要完全重算时选择新目录。

视觉评估提供 IC、CC、GC；规划 LLM 提供 CBC、NC、PPR、VSU、STQ；SSS_emb 对比代理故事文本与视频重建的 L4 故事 embedding。当前比较入口不计算 NF，表格中的 N/A 不代表 0 分。Track B 缺少源 scene 标签时不把 IC/CC 当作主结果。评估依赖 ViCLIP `/embed`、LLM 和 embedding 服务。

详细的环境设置、ViMax 独立执行、视频失败恢复、StoryControl 覆盖检查、指标解释及结果验收见 `docs/run_and_evaluation_guide.md`；架构与交付清单见 `docs/pipeline_experiment_inventory.md`。旧独立 ViMax 仓库的结果可通过 `VIMAX_ROOT`、`VIMAX_OUTPUT_DIR` 显式引用，避免和根目录 `ViMax/` 的新输出混用。
