# CameraMaster pipeline 与实验交付清单

本文件记录源码、实验入口和恢复边界。运行结果、模型权重、外部数据集与 `.env` 留在服务器本地，不进入 Git。

## 目录和调用链

| 位置 | 用途 |
| --- | --- |
| `scripts/run_vimax_track_ab_pipeline.py` | 将一个 `vimax_benchmark` case 映射到 CameraMaster Track A/B。 |
| `scripts/run_dual_backend_narrative_world_pipeline.py` | Track B 调用的叙事世界推演与资产生成入口。 |
| `narrative/story_control.py`, `narrative/simulation_scaffold.py` | StoryControl v4、世界状态、角色动作与叙事 checkpoint。 |
| `agents/narrative_node_selector.py`, `agents/path_orchestrator_agent.py` | 从推演证据选取节点并编排路径。 |
| `agents/video_agent.py`, `agents/video_timeline.py` | 事件视频、拼接与时间轴。 |
| `scripts/resume_vimax_track_ab_failed_videos.py` | 已有节点下的失败视频任务恢复。 |
| `scripts/run_joint_cameramaster_vimax_case.sh` | 同一 case 串联 CameraMaster 与 ViMax，按阶段标记恢复。 |
| `ViMax/` | ViMax 基线的独立源码目录，包含原仓库许可证、benchmark、改造后的 pipeline 和评估脚本。 |
| `evaluation/run_cameramaster_vimax_comparison.py` | 规范化两个方法输出、生成定性对照、执行定量评估、汇总表格。 |
| `evaluation/run_vimax_bench_eval.py`, `evaluation/run_vimax_planning_judge.py`, `evaluation/sss_smoke_test.py` | 视觉一致性、规划评判、SSS 指标子评估。 |
| `evaluation/generate_experiment_html.py`, `scripts/generate_experiment_results_overview_html.py` | 实验 HTML 展示入口。 |
| `tests/`, `evaluation/test_*.py` | 离线回归测试。 |

`agents/`, `narrative/`, `scripts/`, `evaluation/` 顶层脚本、`configs/`, `prompt_library/`, `Static/`, `Dynamic/`, `rag/components/`, `tool_hub/code/` 构成当前可提交源码。`evaluation/ViCLIP/` 等子目录为外部或衍生材料，不在本次源码版本内。

## 同一 case 的 Track 定义

CameraMaster Track A 读取 benchmark 的每个 shot 的 `first_frame` 和 `video_prompt`；每个源 shot 对应一个生成节点和关键帧。Track B 将 `story_overview` 作为叙事内容，shot 文本只用于抽取角色名单，不作为推演事件锚点。Track B 的世界模型分发当前可观察状态，角色基于状态选择动作；StoryControl v4 用叙事证据检查必需事件。`--max-steps` 是最多推演轮数，`--max-nodes` 是最多选取节点数，两者均不是必须达到的数量。若必需事件缺失，覆盖门禁阻止视频生成。

ViMax 的对应定义见 `ViMax/docs/vimax_pipeline_evaluation.md`：Track A 将源 shot 文本送入 Script2Video，Track B 只将 `story_overview` 送入 Idea2Video。四组输出应使用同一 benchmark case，评估时按 Track 分组。

## 运行与恢复

在 CameraMaster 目录、`mifage` 环境下运行：

```bash
bash scripts/run_joint_cameramaster_vimax_case.sh \
  --case artist_extreme_weather_typeA.json \
  --camera-track B --vimax-track both \
  --camera-output-root /data1/AgentG/hhb/CameraMaster/outputs/vimax_track_ab_4030_storycontrol_v4 \
  --max-steps 40 --max-nodes 30
```

脚本固定 CameraMaster 使用 `mifage` Python、ViMax 使用 `Vimax` Python；同一命令重跑时利用 `.joint_run_state` 成功标记跳过已完成阶段。CameraMaster 叙事 checkpoint 与 ViMax `--resume` 在各自目录内恢复。StoryControl v3 checkpoint 与 v4 不兼容，需要新的 CameraMaster 输出根目录。标记代表阶段进程成功退出；评估仍需检查视频、manifest 和结果文件。

联合脚本默认读取 `CameraMaster/ViMax/vimax_benchmark`，把 ViMax 新输出写入 `CameraMaster/ViMax/outputs/vimax_benchmark/all_cases`。旧运行仍在 `/data1/AgentG/hhb/ViMax/outputs` 时，可同时设置 `VIMAX_ROOT=/data1/AgentG/hhb/ViMax` 与 `VIMAX_OUTPUT_DIR=/data1/AgentG/hhb/ViMax/outputs/vimax_benchmark/all_cases` 来恢复；不要把新旧输出路径混用。

已有四组输出后运行比较：

```bash
python -u evaluation/run_cameramaster_vimax_comparison.py \
  --benchmark-case /data1/AgentG/hhb/CameraMaster/ViMax/vimax_benchmark/artist_extreme_weather_typeA.json \
  --cameramaster-output-root /data1/AgentG/hhb/CameraMaster/outputs/vimax_track_ab_4030_storycontrol_v4/artist_extreme_weather_typeA \
  --vimax-output-root /data1/AgentG/hhb/CameraMaster/ViMax/outputs/vimax_benchmark/all_cases/artist_extreme_weather \
  --output-dir /data1/AgentG/hhb/CameraMaster/outputs/comparisons/artist_extreme_weather_typeA_v4_vs_vimax \
  --run-all --resume
```

`--resume` 跳过已经存在且能解析为非空 JSON 对象的子评估结果；`--no-resume` 重算。脚本输出 `comparison_manifest.json`、`quantitative/` 下各子评估 JSON、CSV/Markdown 指标表，以及 `qualitative/` 对照材料。`--run-all` 会调用外部 LLM、embedding、ViCLIP `/embed` 和视频处理，属于在线实验；本次整理提交不会重跑该实验。

## 指标与解释边界

比较表包含 IC、CC、GC（ViCLIP 视觉一致性），CBC、NC、PPR、VSU、STQ、NF（规划/叙事评判字段）及 `sss_emb`（SSS embedding）。指标由各子评估汇总，不可把不同 Track 的分数直接当作同一输入条件下的因果差异。`sss_emb` 的层级明细保存在 SSS 原始 JSON；表格只保留主分数。Track B 不使用源 scene 标签，不能把源 shot 对齐分数当作其主指标。对比指标为本项目实验实现，ViMax 视觉评估并非官方 ViMax-Bench 结果。

## 交付边界

Git 保存源代码、配置模板、脚本、测试与文档；不保存 `.env`、API Key、输出视频、缓存、第三方模型权重和 MovieBench 大型数据。远端服务地址、模型和密钥由各运行环境配置。此源码提交不代表现有案例已全部成功生成或所有在线评估已通过；具体进度以各次输出目录的 manifest、coverage report 与评估 JSON 为准。
