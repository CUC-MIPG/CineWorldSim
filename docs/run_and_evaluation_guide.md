# CineWorldSim / ViMax 运行与评估指南

本指南对应仓库根目录的 CameraMaster 和 `ViMax/` 基线快照，面向服务器 `172.17.34.44`。命令均在服务器上执行；先以 `ssh AgentG@172.17.34.44` 登录，再进入 `/data1/AgentG/hhb/CameraMaster`。`outputs/`、`ViMax/outputs/`、`.env`、模型和缓存均不进入 Git。

## 1. 实验流程与输入边界

```text
ViMax/vimax_benchmark/<case>.json
    ├─ CameraMaster Track A：源 shot 的 first_frame + video_prompt → 关键帧/视频
    ├─ CameraMaster Track B：story_overview → StoryControl v4 世界推演 → 节点/视频
    ├─ ViMax Track A：story_overview + 源 shot → Script2VideoPipeline
    └─ ViMax Track B：仅 story_overview → Idea2VideoPipeline
四组现有视频 + 同一 case → comparison → 视觉/规划/SSS → CSV、Markdown、定性网页
```

CameraMaster Track B 的 shot 文本只用于抽取角色名称，不作为叙事事件输入。世界模型逐步提供角色可观察的当前状态；`--max-steps` 是推演轮数上限，`--max-nodes` 是选取节点上限，都不承诺恰好生成指定数量。StoryControl v4 的必需事件缺少证据时，`story_coverage_failure.json` 会记录缺项并阻止视频阶段。Track A 与 B 的输入条件不同，应在同一 Track 内比较 CameraMaster 与 ViMax。

`consistency_type`（Type A/B/C）是 benchmark 用例类型，和实验的 Track A/B 不是同一个概念。ViMax 输出目录用 `benchmark_index.json` 里的 `theme` 命名；本文示例 `artist_extreme_weather_typeA.json` 的 theme 为 `artist_extreme_weather`。

## 2. 环境与服务准备

```bash
cd /data1/AgentG/hhb/CameraMaster
git rev-parse --short HEAD
test -x /data1/AgentG/anaconda3/envs/mifage/bin/python
test -x /data1/AgentG/anaconda3/envs/Vimax/bin/python
test -f ViMax/vimax_benchmark/artist_extreme_weather_typeA.json
command -v ffmpeg
bash scripts/run_joint_cameramaster_vimax_case.sh --help
```

联合脚本默认使用 `mifage` 运行 CameraMaster、`Vimax` 运行 ViMax。环境路径不同的机器可分别设置 `MIFAGE_PY` 和 `VIMAX_PY`。`CameraMaster/.env` 是服务器私有配置；嵌套 ViMax 保留进程环境变量优先级，随后读取 `ViMax/.env`，最后从父目录 `.env` 补足缺少的变量。请从 `.env.example` 确认变量名，不把真实密钥写进 Git。

| 用途 | 主要配置 |
| --- | --- |
| LLM / VLM | `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`；ViMax benchmark 的聊天配置也可从这些变量补全。 |
| Banana 图像 | `IMAGE_GEN_PROVIDER=nano_banana`、`NANO_BANANA_API_KEY`、`NANO_BANANA_BASE_URL`、`NANO_BANANA_MODEL_NAME`。 |
| MiniMax H3 视频 | `VIDEO_GEN_PROVIDER=minimax_h3`、`MINIMAX_H3_VLLM_ENABLED=true`、`MINIMAX_H3_VLLM_API_URL`；当前模板指向 `http://127.0.0.1:9001/v1/videos/sync`。 |
| SSS embedding | 建议单独配置 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL_NAME`；未配置时会回退到 OpenAI 兼容变量及默认模型。 |
| ViCLIP 视觉评估 | 比较脚本 `--embed-url`，当前默认 `http://172.17.34.43:8890/embed`。 |

开始在线运行前，确认上述 LLM、图像、视频和 ViCLIP 服务可用。`--strict-online` 遇到服务失败会报错；模型不存在的 404 需检查模型名及对应接口，网络超时或 5xx 可在服务恢复后按第 5 节恢复。在线调用与视频生成有时间、算力和 API 成本。

## 3. 定义一次可复现的运行

以下变量供本指南后续命令共用。为不同模型、参数或代码版本选择新的输出根目录；不要把不同实验写进同一个比较目录。

```bash
cd /data1/AgentG/hhb/CameraMaster
export CM_ROOT="$PWD"
export CASE="artist_extreme_weather_typeA.json"
export CASE_STEM="artist_extreme_weather_typeA"
export VIMAX_THEME="artist_extreme_weather"
export CM_OUT="$CM_ROOT/outputs/vimax_track_ab_4030_storycontrol_v4_artist_run1"
export VIMAX_OUT="$CM_ROOT/ViMax/outputs/vimax_benchmark/artist_run1"
export CMP_OUT="$CM_ROOT/outputs/comparisons/artist_extreme_weather_typeA_artist_run1"
```

### 3.1 一条命令运行完整四组输出

```bash
bash scripts/run_joint_cameramaster_vimax_case.sh \
  --case "$CASE" \
  --camera-track both \
  --vimax-track both \
  --camera-output-root "$CM_OUT" \
  --vimax-output-dir "$VIMAX_OUT" \
  --backend all_banana \
  --max-steps 40 \
  --max-nodes 30 \
  --max-reference-images 4 \
  --retries 3 \
  --log-level INFO
```

联合脚本依次运行 CameraMaster、ViMax A、ViMax B。其默认 `--camera-track` 是 `B`，因此四组实验必须显式传 `--camera-track both`。只做 Track B 时，把两个 Track 参数均改为 `B`，评估时也要传 `--tracks B`。`--camera-track` 目前只接受 `B` 或 `both`；只做 CameraMaster Track A 时使用下一节的独立入口。

### 3.2 独立运行两个方法

CameraMaster（`mifage`）：

```bash
/data1/AgentG/anaconda3/envs/mifage/bin/python -u scripts/run_vimax_track_ab_pipeline.py \
  --benchmark-root "$CM_ROOT/ViMax/vimax_benchmark" \
  --case "$CASE" --track both --output-root "$CM_OUT" \
  --backend all_banana --max-steps 40 --max-nodes 30 \
  --max-reference-images 4 --strict-online --retries 3 --log-level INFO
```

ViMax（`Vimax`）：

```bash
/data1/AgentG/anaconda3/envs/Vimax/bin/python -u ViMax/scripts/run_vimax_benchmark_pipeline.py \
  --benchmark-root "$CM_ROOT/ViMax/vimax_benchmark" \
  --case "$CASE" --track both --output-dir "$VIMAX_OUT" \
  --run --resume --log-level INFO
```

ViMax 不带 `--run` 时只准备输入和 manifest，不调用在线生成。CameraMaster 可用 `--dry-run` 检查部分流程；Track B 未显式传 `--track-b-characters` 时仍可能调用 LLM 抽取角色，不能将其视作完全离线预检。

## 4. 确认生成结果

运行结束后检查以下文件及状态，而不只看终端最后一行：

| 方法 | 关键产物 |
| --- | --- |
| CameraMaster | `$CM_OUT/$CASE_STEM/vimax_track_ab_manifest.json`；`track_A/`、`track_B/` 各自的 `narrative_nodes.json`、`path_plan.json`、`video/event_video_summary.json`、`video/final_orchestrated_events.mp4`。 |
| CameraMaster B | `track_B/story_coverage_report.json`；推演 checkpoint 位于 `track_B/narrative_simulation/all_banana/narrative_checkpoint.json`。 |
| ViMax | `$VIMAX_OUT/$VIMAX_THEME/case_manifest.json`；Track A 镜头在 `track_A/vimax_pipeline/shots/<idx>/video.mp4`，Track B 在 `track_B/vimax_pipeline/scene_<n>/shots/<idx>/video.mp4`。 |
| 联合脚本 | `$CM_OUT/.joint_run_state/v4/.../*.done` 只在相应阶段成功退出后写入，日志位于同目录的 `logs/`。 |

CameraMaster 视频完成条件是 `event_video_summary.json` 中 `success == total`、`failed == 0`、`concat_ok == true`，最终 MP4 非空；ViMax 要检查 `case_manifest.json` 的每个 Track 状态与 `final_video` 路径。Track A 的 clip 数还应与 benchmark 源 shot 数一致，否则比较脚本会拒绝对齐。

## 5. 中断与恢复

1. **联合脚本中断：**在同一代码版本、case、参数和输出路径下原样重跑第 3.1 节命令。`.done` 阶段会跳过；未完成阶段会再执行。ViMax 的 `--resume` 允许使用已有目录，但不代表所有内部任务都会被自动跳过。
2. **CameraMaster 仅视频失败：**已有 `track_X/video/event_video_tasks.json` 和 `event_video_summary.json` 时，先预览缺失任务，再恢复。把 `X` 改为 `A` 或 `B`。

   ```bash
   export VIDEO_DIR="$CM_OUT/$CASE_STEM/track_B/video"
   /data1/AgentG/anaconda3/envs/mifage/bin/python -u scripts/resume_vimax_track_ab_failed_videos.py \
     --video-dir "$VIDEO_DIR" --dry-run
   /data1/AgentG/anaconda3/envs/mifage/bin/python -u scripts/resume_vimax_track_ab_failed_videos.py \
     --video-dir "$VIDEO_DIR" --retries 3 --log-level INFO
   ```

   恢复脚本只重试缺失 MP4，全部完成后才重新拼接。随后重新检查 summary；联合阶段尚未写 `.done` 时，再原样重跑联合命令以完成 manifest。
3. **Track B 剧情覆盖失败：**读取 `track_B/story_coverage_failure.json` 的 `missing_beat_ids` 及 `story_coverage_report.json`。区分推演未发生事件与节点选择遗漏；`--max-steps` 和 `--max-nodes` 都是上限，单纯加大数值不保证修复。不要为正式对比实验关闭覆盖门禁来掩盖缺失事件。StoryControl v3 checkpoint 不能用于 v4；改版本时使用新输出根目录。
4. **从头重算：**另取新的 `CM_OUT`、`VIMAX_OUT` 和 `CMP_OUT`。联合脚本的 `--force` 会重新调用阶段，但不会清空已有生成文件，因此不等同于全新实验。

## 6. 统一定量与定性比较

四组完整时，在 `mifage` 环境执行：

```bash
/data1/AgentG/anaconda3/envs/mifage/bin/python -u evaluation/run_cameramaster_vimax_comparison.py \
  --benchmark-case "$CM_ROOT/ViMax/vimax_benchmark/$CASE" \
  --cameramaster-output-root "$CM_OUT/$CASE_STEM" \
  --vimax-output-root "$VIMAX_OUT/$VIMAX_THEME" \
  --output-dir "$CMP_OUT" \
  --run-all --resume --timeout 900
```

只生成了 Track B 的两组视频时，在上述命令中加入 `--tracks B`。如 ViCLIP 地址不同，加 `--embed-url http://HOST:PORT/embed`；`--num-frames` 默认 8、`--resolution` 默认 224。`--timeout 900` 是子评估单次请求的 15 分钟超时。`--run-all` 等价于启用 `--run-visual`、`--run-planning`、`--run-sss`；不启用这些开关时，只生成规范化输入与定性对照。评估默认读取 `$CM_ROOT/.env`，也可用 `--env-file` 指定。

`--resume` 默认开启：已有非空、可解析为 JSON 对象的子评估结果会跳过，SSS 的 `sss_result.json` 可被复制回汇总位置。它不会检查旧结果是否对应本次模型、视频或参数；只有确认输出目录对应同一次实验时才能复用。需要全部重算时用 `--no-resume`，并优先使用新的比较目录。

主要产物：

| 路径（相对 `$CMP_OUT`） | 内容 |
| --- | --- |
| `comparison_manifest.json`、`reference_story.txt` | 四组样本元数据、作为 proxy 的 benchmark `story_overview`。 |
| `normalized/` | 每个方法 × Track 的统一评估输入 JSON。 |
| `quantitative/*_visual.json` | ViCLIP 视觉一致性结果。 |
| `quantitative/*_planning.json` | LLM 规划评分、理由和 batch 记录。 |
| `quantitative/*_sss.json`、`quantitative/sss/` | SSS 主结果和原始分层重建、诊断。 |
| `quantitative/comparison.csv`、`comparison.md` | 统一表格；空值表示没有可报告结果。 |
| `qualitative/index.html`、`qualitative/media/` | 按 Track 和序号并排展示视频与分镜文本。`media/` 在 Linux 上通常是指向原视频的符号链接，网页离开原机器后不能单独播放。 |
| `derived_media/` | SSS 使用的顺序拼接视频，不改动两种方法的原始输出。 |

定性页面的 Track B 行按**生成序号**并排展示；两方法镜头数可不同，这不是剧情事件的一对一语义对齐。若只需无视频的结果文档，可在四组评分齐备后先运行：

```bash
/data1/AgentG/anaconda3/envs/mifage/bin/python -u evaluation/export_comparison_error_analysis.py \
  --comparison-dir "$CMP_OUT"
/data1/AgentG/anaconda3/envs/mifage/bin/python -u evaluation/generate_experiment_html.py \
  --comparison-dir "$CMP_OUT"
```

该旧版 `experiment_report.html` 模板要求四组结果及 `analysis/` 文件，含针对既有 Artist 案例的静态描述（例如镜头数、SSS 模式、embedding 模型）。迁移到新 case 或新配置时，需逐项核对模板文字与原始 JSON；它不是通用的自动事实校验器。网页本身不嵌入视频，但附带的 `analysis/` 下载链接应与 HTML 一同交付。

## 7. 指标解释与报告规则

| 指标 | 含义与范围 |
| --- | --- |
| IC / CC / GC | ViCLIP 视觉 embedding 的同 scene 镜头对、跨 scene 镜头对、全故事镜头对余弦相似度均值。值越高表示视觉越相似，不直接证明剧情正确；Track B 缺少可靠的源 scene 标签时，主报告只解释 GC。 |
| CBC | Character Behavior Consistency：角色身份、目标、行为连贯性，LLM 1–5 分。 |
| NC | Narrative Coherence：镜头的因果与时间连贯性，LLM 1–5 分。 |
| PPR | Plot Pacing and Rhythm：节奏、升级、停顿和镜头粒度，LLM 1–5 分。 |
| VSU | Visual Specificity and Shot Usability：分镜描述是否具体、可生成，LLM 1–5 分。 |
| STQ | Scene Transition and Continuity Quality：转场中的空间、时间和状态连续性，LLM 1–5 分。 |
| NF | Narrative Faithfulness：需另给完整原始故事文本；当前比较脚本只调用 `planning` 模式，表中 NF 为 N/A，不应写成已评估。 |
| SSS_emb | benchmark `story_overview` 代理参考文本与从视频逐层重建得到的 L4 故事文本的 embedding 余弦相似度；不是 L1/L2/L3 三层分数的平均。 |

比较脚本对 SSS 调用 Track A `v3`、Track B `v2`，并显式标记 `summary_provenance=proxy`。SSS 的 L1/L2/L3 用于逐层重建，最终主分数对比 reference 与 L4；结构化对齐及事件覆盖保存在原始结果中作诊断。SSS 模型默认 `OPENAI_MODEL`，embedding 模型默认 `EMBEDDING_MODEL_NAME`，缺失时回退到 `text-embedding-3-small`。规划评判同样默认 `OPENAI_MODEL`。记录模型名、服务版本、case、Track、提交哈希和输出路径后，再做跨方法比较。

IC/CC/GC 在单 case 表里是该故事内部镜头对的平均；多 case 聚合应区分 story macro mean 与 pair weighted mean。规划分数可能按 batch 汇总，不能未经核对就称为所有镜头的逐镜头平均。Track A/B 输入约束、生成镜头数和时长不同，不把 A 与 B 的差值解释为单变量因果效应。以上视觉评估与 LLM 提示词是本项目的 ViMax-Bench 风格适配，并非官方 ViMax-Bench 榜单结果。

## 8. 最终检查与常见报错

- 比较脚本在输入发现阶段失败：核对 `CASE_STEM` 与 `VIMAX_THEME`，并检查四组 MP4 是否齐全且非空。Track A 源 shot 数必须与生成 clip 数相同。
- ViCLIP `/embed` 连接失败：检查 `--embed-url` 服务及网络，再以同一 `CMP_OUT` 执行 `--resume`；未完成的 visual 子评估会继续。
- SSS chat 超时、SSL EOF 或 embedding 503：确认 `.env` 中 chat 与 embedding 指向各自可用接口，服务恢复后 `--resume`。SSS 可复用已写入的中间结果；`--timeout` 可增加单次等待时间。
- LLM 返回非法 JSON：查看 SSS 对应输出目录中的原始响应与失败细节；代码会对部分结构化响应做重试，持续失败需检查模型兼容性。
- 比较表出现 N/A：先检查该子评估 JSON 的原始字段和 `comparison_manifest.json`，再判断是协议不适用、尚未运行，还是结果映射/生成失败；不要把 N/A 当作 0 分。
- 交付前检查：四组路径、case 版本与输出目录一致；`comparison.csv` 与四组 JSON 齐备；Track B coverage 通过；保存 `git rev-parse HEAD`、实验命令、`.env` 中的模型名（不记录密钥）与服务版本。

源码入口和研究背景见 `docs/pipeline_experiment_inventory.md`、`ViMax/INTEGRATION.md`、`ViMax/docs/vimax_pipeline_evaluation.md`。本指南描述当前脚本行为，不代表既有 `outputs/` 中的所有历史运行均已通过上述检查。
