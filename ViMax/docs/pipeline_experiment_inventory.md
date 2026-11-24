# ViMax 基线 pipeline 与实验交付清单

本分支在 HKUDS/ViMax 基础上加入本地 benchmark 适配、CameraMaster 可比的图像/视频后端、恢复脚本与离线评估。`origin` 指向公开上游；本次只在服务器本地建立提交，不向上游推送。

| 位置 | 用途 |
| --- | --- |
| `vimax_benchmark/benchmark_index.json` 与 case JSON | 两个方法共用的测试用例。 |
| `scripts/run_vimax_benchmark_pipeline.py` | 按 case 运行 Track A/B，可使用 `--run --resume`。 |
| `pipelines/script2video_pipeline.py` | Track A：使用 `story_overview` 与源 shot 的 `first_frame`、`video_prompt`。 |
| `pipelines/idea2video_pipeline.py` | Track B：仅使用 `story_overview`，不传源 shot 描述。 |
| `agent_runtime/`, `tools/`, `utils/`, `configs/` | OpenAI 兼容聊天、Nano Banana 图像、MiniMax H3 视频提供方适配。 |
| `scripts/run_vimax_pipeline_eval.py`, `scripts/vimax_pipeline_eval.py` | ViMax 输出的自定义 ViCLIP IC/CC/GC 评估。 |
| `scripts/run_vimax_moviebench_pipeline.py`, `scripts/run_vimax_moviebench_arc01_baseline.py` | MovieBench 扩展实验。 |
| `docs/vimax_pipeline_evaluation.md`, `docs/moviebench_vimax_native_pipeline.md` | 评估协议与扩展实验说明。 |

单 case 运行：

```bash
cd /data1/AgentG/hhb/ViMax
conda run -n Vimax python -u scripts/run_vimax_benchmark_pipeline.py \
  --benchmark-root /data1/AgentG/hhb/ViMax/vimax_benchmark \
  --case artist_extreme_weather_typeA.json \
  --track both \
  --output-dir /data1/AgentG/hhb/ViMax/outputs/vimax_benchmark/all_cases \
  --run --resume --log-level INFO
```

`--resume` 允许使用已存在的 case 输出目录。跨方法的阶段跳过由 CameraMaster `scripts/run_joint_cameramaster_vimax_case.sh` 管理；它分别调用 ViMax Track A、B 并写成功标记。`outputs/`、临时计划、缓存和本地密钥不进入本提交。视觉评估是项目自定义实验，不是官方 ViMax-Bench 评分。最终四组横向比较由 CameraMaster `evaluation/run_cameramaster_vimax_comparison.py` 产生。
