# ConStory-Bench、DirectorBench、LoCoT2V-Bench 评测体系与使用方法

本文基于三个本地仓库的 README、评测脚本和指标实现整理。三者面向的对象不同：ConStory-Bench 评测长文本故事，DirectorBench 评测约一分钟的带音视频生成结果，LoCoT2V-Bench 评测长时、多场景 text-to-video 视频。

## 1. 总览

| Benchmark | 被测对象 | 主要问题 | 核心输出 |
|---|---|---|---|
| ConStory-Bench | 长篇文本故事 | 故事在长文本中是否自洽 | CED、GRR、5 类/19 子类错误统计 |
| DirectorBench | 约 1 分钟视频，可带 prompt/script/storyboard/audio | 叙事、镜头、视觉连续性、音视频同步和生成稳定性 | 0-1 子指标、维度分数、总体分数、A-F 等级、诊断建议 |
| LoCoT2V-Bench | 长时、多场景 text-to-video 视频 | 文本对齐、角色/背景长期一致性、动态质量和感知质量 | 8 类评测结果及归一化综合分数 |

三者的分数不可直接横向比较：ConStory 的 CED/GRR 主要衡量文本错误，DirectorBench 是可配置的多智能体诊断框架，LoCoT2V 是面向长视频的多模型自动评测套件。

---

## 2. ConStory-Bench

### 2.1 评测内容

ConStory-Bench 的 ConStory-Checker 使用 LLM 检查长故事中的跨段落一致性。评测 taxonomy 有 5 类、19 个子类：

| 类别 | 子类 |
|---|---|
| Characterization | memory contradictions、knowledge contradictions、skill/power fluctuations、forgotten abilities |
| Factual Detail | appearance mismatches、nomenclature confusions、quantitative mismatches |
| Narrative Style | perspective confusions、tone inconsistencies、style shifts |
| Timeline & Plot | absolute-time contradictions、duration/timeline contradictions、simultaneity contradictions、causeless effects、causal-logic violations、abandoned plot elements |
| World-building & Setting | core-rule violations、social-norm conflicts、geographical contradictions |

每个类别由一个独立 prompt 评测。LLM 输出结构化错误实例；错误实例需要包含 `exact_quote`，因此可以统计错误数量、类别和位置。评测结果保存为 CSV。

### 2.2 指标

#### CED：Consistency Error Density

对每个故事，统计 19 个子类中是否存在错误，并按文本长度归一化：

`CED_story = error_count / (word_count / 10000)`

代码中的 overall CED 是各故事 CED 的平均值，同时还报告 median、标准差、平均词数、平均错误数和五个类别的 category CED。数值越低越好。注意：实现按“某子类是否有错误”计数，而不是把一个 cell 中 JSON 数组的所有错误实例都累加到 CED；`count_errors_in_cell` 虽能统计数组长度，但 `compute_ced_single` 实际调用的是 `check_error_exists`。

#### GRR：Group Relative Rank

在同一 story id 上比较多个模型：

`efficiency = word_count / (1 + error_count)`

按 efficiency 降序排名，GRR 是模型在所有可比较故事上的平均 rank。数值越低越好。它会奖励生成更长文本，同时惩罚错误，因此不能单独解释为纯一致性指标。

### 2.3 使用方法

安装并进入仓库：

```bash
cd ConStory-Bench-main
pip install -r requirements.txt
```

生成故事（输入通常为包含 prompt 的 parquet）：

```bash
export OPENAI_API_KEY="..."
python -m constory.generate \
  --input data/prompts.parquet \
  --output data/stories/my_model.parquet \
  --model gpt-4o --concurrent 5
```

评测生成故事。`--story-column` 指定故事文本列，judge 可使用任意 OpenAI-compatible API：

```bash
python -m constory.judge \
  --input data/stories/my_model.parquet \
  --story-column generated_story \
  --model-name my_model \
  --judge-model o4-mini \
  --api-base https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --concurrent 3
```

计算 CED/GRR：

```bash
python -m constory.metrics \
  --eval-dir output/ \
  --config configs/models.yaml \
  --mode both
```

也可运行 `constory.correlation` 计算类别条件概率 `P(B|A)`，运行 `constory.positional` 分析错误在故事中的位置、原事实与矛盾出现点之间的间隔。

### 2.4 输入输出与复现注意事项

- 官方数据包括 `prompts.parquet`、`stories.parquet` 和 `evaluations/*.csv`；README 标称 prompts 约 2000 条。
- judge 默认 temperature=0.5、每个故事并行调用 5 个类别 prompt，并支持失败重试和增量保存。
- 应保存 judge 模型、prompt 版本、temperature、最大 token、API endpoint 和解析失败记录。
- CED 适合报告总体及五类分项；GRR 必须在多个模型共享同一组 story id 时使用。

---

## 3. DirectorBench

### 3.1 评测内容与执行流程

DirectorBench 是 LangGraph 多智能体评测框架，目标是约一分钟的多镜头视频。输入可以包括：

- 生成视频 `video_path`；
- 原始用户 prompt；
- script/storyboard（可选）；
- 独立音频（可选）；
- 用户 profile（影响总体权重）。

流程为：

1. 预处理：ffprobe、PySceneDetect 镜头检测、代表帧抽取、边界帧指标、ffmpeg 音频抽取、可选 AudioShake 音轨分离、Whisper ASR。
2. 并行 specialist agents：Script、Video、Audio、Stability。
3. Cross-Modal agent：文本-视频、视频-音频、文本-音频及整体多模态和谐。
4. Diagnosis Synthesizer：按置信度和 profile 权重聚合，输出瓶颈、证据和建议。

没有提供 script 时，Script Agent 会用 VLM+ASR 从视频中抽取叙事，因此属于 reference-free 模式；提供 script/storyboard 时属于 reference 模式。

### 3.2 子指标

代码实际包含以下指标（README 中“14 sub-metrics”没有把 cross-modal 的整体项完整展开）：

| Agent | 指标 | 评测内容与实现 |
|---|---|---|
| Script | `script_reasonableness` | 事件逻辑、因果和情节连贯性；LLM/VLM checkpoint 评分 |
| Script | `script_novelty` | 原创性、避免陈词滥调 |
| Script | `user_requirement_consistency` | 叙事是否满足用户 prompt |
| Script | `script_video_fidelity` | reference 模式下 script/storyboard 与视频的忠实度 |
| Video | `user_demand_fulfillment` | 画面是否实现用户要求，支持逐 shot 分数 |
| Video | `temporal_coherence` | 跨镜头角色脸、服装、物体、背景、比例、运动方向和时间逻辑；算法证据+VLM checkpoints |
| Video | `lighting_consistency` | 光源方向、阴影、色温、曝光连续性 |
| Video | `transition_quality` | cut 边界的 SSIM、颜色直方图距离、Farneback 光流；可疑边界再交给 VLM 判断是否为有意场景切换或 splice artifact |
| Audio | `narration_reasonableness` | 对白/旁白时序、自然度和情绪匹配；无语音时跳过 |
| Audio | `bgm_consistency` | BGM 与场景情绪、节奏、转场和对白混音；无 BGM 时跳过 |
| Stability | `generation_stability` | 视频前后段画质是否下降；LLM checkpoint + BRISQUE/质量退化率 |
| Cross-modal | `text_video_consistency` | script/prompt 与视频；VLM checkpoint 与 ViCLIP 相似度组合，代码中通常 `0.6*checkpoint + 0.4*ViCLIP` |
| Cross-modal | `video_audio_consistency` | lip-sync、动作与声音事件匹配；checkpoint 与轻量音视频同步分数组合 |
| Cross-modal | `text_audio_consistency` | script 与 ASR/语音内容；LLM 分数与 Sentence-BERT 语义相似度组合 |
| Cross-modal | `overall_multimodal_harmony` | 对跨模态整体协调性的 LLM 总结评分 |

每个指标最终输出 `EvalResult`：`score` 为 0-1，另有 confidence、evidence chain、suggestions 和可选 checkpoint 结果。checkpoint 通常是 1-5 Likert 或 binary，再归一化到 0-1；不适用 checkpoint 会被动态跳过。

### 3.3 总分、维度和等级

Diagnosis 阶段先对同一 agent 的指标做 confidence-weighted average，得到 `script`、`video`、`audio`、`stability`、`crossmodal` 五个维度。没有数据的维度（例如静音视频的 audio）不参与分母。

总体分数默认使用 `UserProfile.priority_weights`，而不是简单平均：

- script -> `text_story_arc`
- video 和 stability -> 共同使用 `visual_camera`
- audio -> `audio_emotion`
- crossmodal -> `cross_modal_sync`

默认 `config/default_config.yaml` 还提供六个展示权重（shot_transition、camera_control、consistency、physical_realism、narrative、storyboard、audio），但最终 Diagnosis 代码的实际聚合以 profile priority weights 为准，使用时应明确配置来源。默认等级边界为：A >= 0.85，B >= 0.70，C >= 0.55，D >= 0.40，F < 0.40；低于 0.5 的指标会被列为 bottleneck。

### 3.4 使用方法

```bash
cd DirectorBench-master
pip install -r requirements.txt
```

带 script/profile：

```bash
python -m directorbench.main \
  --video path/to/video.mp4 \
  --prompt "A dramatic chase through a neon-lit city" \
  --script "Scene 1: ..." \
  --profile-id 2 \
  --output-dir ./eval_outputs
```

reference-free：

```bash
python -m directorbench.main \
  --video path/to/video.mp4 \
  --prompt "A dramatic chase through a neon-lit city" \
  --output-dir ./eval_outputs
```

Python API：

```python
from directorbench.main import evaluate_video
report = evaluate_video(
    video_path="output.mp4",
    user_prompt="A dramatic chase scene",
    script_text="Scene 1: ...",       # 可选
    storyboard=[...],                  # 可选
    output_dir="./eval_outputs",
)
print(report.overall_score, report.overall_grade)
```

结果追加到 `results.jsonl`，工具调用追踪保存到 `tool_traces.jsonl`。可用 `render_report.py --latest --out reports/latest.md` 渲染报告。外部 API 主要包括 OpenAI/VLM、Whisper 和可选 AudioShake；缺少可选 API 时框架使用 fallback 或跳过对应指标，报告中的 confidence 会下降。

---

## 4. LoCoT2V-Bench

### 4.1 数据和输入格式

`data/prompt.json` 为每个 theme/sample 提供 prompt metadata：

- `source_description`、`prompt`、`purified_prompt`；
- `character_infos`、`background_infos`、`scene_infos`；
- `fga_questions`（细粒度 VQA）；
- `herd_expectations`、`herd_questions`（叙事期望）。

评测视频列表是 JSON 映射，例如 `eval_videos/example.json`：sample id -> video path。视频应放在 `videos/` 或使用绝对路径。

### 4.2 指标和实际计算方式

| 指标 | 内容 | 实际计算 |
|---|---|---|
| Overall Alignment | 视频整体是否实现文本描述 | Qwen3-VL 直接看视频和 `purified_prompt`，输出 0-100 整数，除以 100 归一化；样本均值为 overall |
| Fine-Grained Alignment | 场景存在、角色属性/动作、背景类型/约束、镜头运动 | 预生成树状 VQA；Qwen3-VL 对问题只回答 Yes/No。按 scene、角色和问题计数，输出 scene existence、character attr/action、background type/conformity、camera movement 及 sample score |
| Character Consistency | 跨场景角色身份和外观稳定性 | SAM3 根据 `sam_prompt` 找角色区域；FG-CLIP2 比较同一角色各图像与 anchor 图像的相似度；每角色平均，再对角色平均 |
| Background Consistency | 相邻采样帧背景稳定 | FG-CLIP2 对连续代表帧的图像相似度求平均 |
| Warping Error | 运动/时序形变错误 | RAFT 光流+视频帧 warping error；代码将原始误差转换为 `exp(-alpha*error)`，因此结果越高越好 |
| Perceptual Quality | 帧级感知画质 | DeQA-Score-Mix3 对采样帧打分；按 scale/window 平均后再跨 scale 平均，并除以上限归一化 |
| Dynamic Degree | 动态程度 | RAFT 光流半径的高分位统计，按阈值截断到 1 后对帧对平均；高分表示更明显的运动 |
| Motion Smoothness | 运动平滑度 | AMT 视频插帧模型计算相邻帧差异，`(255 - mean_difference)/255`，高分表示更平滑 |
| DEVIL score | 长视频动态/多样性和稳定性 | segment-level：DINOv2 与 ViCLIP 分数分别 power-normalize 后取均值；video-level：feature information variance 与 temporal entropy 分数取均值；同时输出 segment/video 分项 |
| HERD | Human Expectation Realization Degree | Qwen3-VL 两阶段：Auditor 先生成客观视频报告；Evaluator 按 1-5 评六维，再除以 5：Emotional Response、Narrative Flow、Character Development、Visual Style、Themes Expression、Overall Impression。六维均值为 sample score |
| Dynamic Quality (DQ) | 综合动态质量 | `compute_dq_scores.py` 使用 Dynamic Degree、Motion Smoothness、DEVIL scores 和已提供的线性回归 scaler/model 合并，生成 `eval_dq_scores.json` |

所有脚本通常输出 `overall_score` 和 `sample_scores`；FGA、HERD、DEVIL 还输出分维度 overall 字段。

### 4.3 环境和 checkpoint

仓库将指标拆到四个环境：

| 环境 | 指标 |
|---|---|
| `pq_eval` | Perceptual Quality |
| `locot2v_bench` | Overall/Fine-Grained Alignment、Background、HERD、DEVIL |
| `sam3_env` | Character Consistency |
| `vbench_env` | Warping Error、Dynamic Degree、Motion Smoothness |

先按 README 创建环境并下载 checkpoint：

```bash
cd LoCoT2V-Bench-main
conda env create -f environments/environment_locot2v_bench.yml
conda env create -f environments/environment_vbench.yml
conda env create -f environments/environment_sam3.yml
conda env create -f environments/environment_pq.yml
bash ckpts/download.sh
```

DEVIL 所需的 DINOv2、ViCLIP、OpenCLIP 权重需按 README 手动放到 `ckpts/devil_weights/`；RAFT、AMT、SAM3、FG-CLIP2、Qwen3-VL、DeQA 权重也必须存在于 README 指定路径。

### 4.4 使用方法

生成视频列表：

```bash
python scripts/make_eval_list.py \
  --video_dir ./videos \
  --output_json ./eval_videos/my_videos.json
```

运行单项：

```bash
bash scripts/eval_overall_alignment.sh \
  ./eval_videos/my_videos.json ./results/my_videos/
bash scripts/eval_fine_grained_alignment.sh \
  ./eval_videos/my_videos.json ./results/my_videos/
bash scripts/eval_character_consistency.sh \
  ./eval_videos/my_videos.json ./results/my_videos/
```

运行全部指标：修改 `scripts/eval_all_metrics.sh` 中的 `VIDEO_DIR`、`EVAL_PATH`、`RESULT_DIR` 后执行：

```bash
bash scripts/eval_all_metrics.sh
```

也可分别运行 perceptual quality、background consistency、warping error、HERD 和 dynamic quality 脚本。Dynamic Quality 脚本会先计算 dynamic degree、motion smoothness、DEVIL，再调用 `compute_dq_scores.py` 合并。

### 4.5 复现注意事项

- Overall Alignment 使用 `purified_prompt`，不是带专有名词的原 prompt；否则通用 MLLM 可能无法正确理解角色名。
- FGA 和 HERD 依赖 `prompt.json` 中与 sample id 对齐的元数据，不能只提供视频路径。
- Character Consistency 是“分割后区域的 embedding 相似度”，不是 LLM 的角色判定；Background Consistency 只比较相邻代表帧。
- Dynamic Degree 高不一定代表质量高，它衡量运动程度；应和 Motion Smoothness、DEVIL 一起解读。
- Warping Error 在代码中已指数变换为“高分更好”的 score，报告时不要再把它当原始误差。
- 三个 benchmark 都建议保留 per-sample JSON，而不仅报告 overall mean；同时记录模型 checkpoint、环境、采样策略和失败样本。

---

## 5. 面向后续 CameraMaster 实验的建议

1. 如果输出是长篇文本/剧本，使用 ConStory-Bench，主报 CED 及五类分项；只有在比较多个模型且 story id 对齐时再报 GRR。
2. 如果需要一分钟视频的综合诊断（包含音频、script/storyboard 和用户偏好），使用 DirectorBench；固定 profile 和 config，否则总体分数不具备可比性。
3. 如果重点是长视频 text-to-video 的文本对齐、角色/背景 consistency 和动态质量，使用 LoCoT2V-Bench；至少运行 Overall Alignment、Fine-Grained Alignment、Character Consistency、Background Consistency、Perceptual Quality、Warping Error、HERD 和 Dynamic Quality。
4. 统一实验记录：视频文件、prompt、storyboard/script、音轨、sample id、指标脚本版本、模型 checkpoint、环境名称、结果 JSON 和运行日志。
5. 不要把三个 benchmark 的“consistency”分数合成一个未经校准的总分；它们的测量单位、样本粒度和方向不同。若必须综合，应先在人工标注集上做尺度校准并报告原始分数。

## 6. 关键入口文件

- ConStory-Bench：`constory/judge.py`、`constory/metrics.py`、`prompts/*.md`、`scripts/run_judge.sh`
- DirectorBench：`directorbench/main.py`、`directorbench/preprocessing.py`、`directorbench/agents/*.py`、`directorbench/checkpoints.py`、`config/default_config.yaml`
- LoCoT2V-Bench：`data/prompt.json`、`scripts/eval_*.sh`、`locot2v_eval/eval_suite/*.py`

