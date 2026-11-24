# ViMax-Bench 评测脚本

本目录实现了 ViMax Appendix A Figure 12/13/14 的复刻评测协议，以及 Figure 15/ViMax-Bench 自动视觉一致性指标的 ViCLIP 计算。

## 指标边界

### LLM 规划评测

`CBC`、`NC`、`PPR`、`VSU`、`STQ` 和 `NF` 均为 1--5 分整数，由 LLM judge 输出 JSON 后校验。

- CBC：Character Behavior Consistency
- NC：Narrative Coherence
- PPR：Plot Pacing and Rhythm
- VSU：Visual Specificity and Shot Usability
- STQ：Scene Transition and Continuity Quality
- NF：Narrative Faithfulness

规划维度使用 Figure 12/13 的 storyboard batch；NF 使用 Figure 14 的完整 source text 加 storyboard batch。每个 batch 默认 20 条 storyboard。论文协议中 baseline 取前 20 条，ViMax 取前 100 条并分成 5 个 batch，最后对 batch 分数取算术平均。

### ViCLIP 自动评测

对一个 story 内的每个 shot 单独提取一条 video-level ViCLIP feature，再对所有 shot pair 计算 cosine similarity：

- `IC`：同一 scene 内 pair 的平均 cosine；
- `CC`：不同 scene 间 pair 的平均 cosine；
- `GC`：所有 pair 的平均 cosine。

没有完整 `scene_id` 时，`IC` 和 `CC` 返回 `null`，不会错误地返回 0；`GC` 仍可计算。三项指标只表示视觉 embedding 稳定性，不等同于叙事一致性或事件召回。

## 1. 生成 ViMax-style shot/scene 输入

现有 CameraMaster/DirectorBench 输入：

```bash
python vimax_bench_adapter.py \
  --inputs-json adapted_cases/0001_American_Beauty_arc_001_real/directorbench/inputs.json \
  --track A \
  --shots-per-sample 8 \
  --output /tmp/vimax_samples.json
```

`--shots-per-sample` 只负责按源 clip 顺序切 batch，不会推断 scene。要计算 IC/CC，提供 shot/node 到 scene 的显式映射：

```json
{
  "node_id_or_shot_id_1": "scene_001",
  "node_id_or_shot_id_2": "scene_001",
  "node_id_or_shot_id_3": "scene_002"
}
```

然后增加 `--scene-map scene_map.json`。如果一个样本有 8--16 shots、2--4 scenes 且 scene 标签完整，adapter 会标记 `official_vimax_result_eligible=true`；这仍然是 CameraMaster custom adaptation，不是官方 35-story ViMax-Bench 结果。

adapter 也接受已经规范化的 JSON：

```json
{
  "samples": [{
    "sample_id": "story_001",
    "shots": [{"shot_id": "s1", "scene_id": "scene_1", "video_path": "/path/a.mp4", "storyboard_description": "..."}]
  }]
}
```

## 2. ViCLIP CC/IC/GC

先用 fake encoder 检查流程：

```bash
python run_vimax_bench_eval.py \
  --inputs-json adapted_cases/0001_American_Beauty_arc_001_real/directorbench/inputs.json \
  --track A --shots-per-sample 8 \
  --fake-encoder --cache-dir /tmp/vimax_viclip_cache \
  --output /tmp/vimax_visual_smoke.json
```

正式运行默认通过已经验证的 HTTP `/embed` 服务提取 embedding；该接口返回单位归一化的 768 维向量，以及 `sampled_frames`、`resolution` 和 `model` 元数据。示例：

```bash
python run_vimax_bench_eval.py \
  --inputs-json /path/to/vimax_samples_or_directorbench_inputs.json \
  --scene-map /path/to/scene_map.json \
  --backend http \
  --embed-url http://172.17.34.43:8890/embed \
  --num-frames 8 --resolution 224 --timeout 600 \
  --cache-dir /path/to/vimax_feature_cache \
  --output /path/to/vimax_visual_metrics.json
```

每个 shot 独立上传到 `/embed`。IC/CC/GC 使用返回的 video embedding 两两计算 cosine；数值因浮点舍入略超出范围时裁剪到 `[-1, 1]`。结果包括每个 story 的 pair 详情、pair 数、均值/标准差，以及跨 story 的 `story_macro_mean` 和 `pair_weighted_mean`。缓存 key 包含视频路径/大小/修改时间、backend、endpoint、抽帧数和分辨率；更换服务或参数会自动产生新缓存项。

如需在本机 ViCLIP 环境运行，可显式使用 `--backend local --checkpoint /path/to/ViClip-InternVid-10M-FLT.pth --device cuda`。`--fake-encoder` 只用于离线流程 smoke test，禁止用于实验结果。

`ViCLIP/viclip_client.py` 的 `/predict` 是另一种任务：根据一段视频在候选文本中排序，适用于 video-text retrieval/top-k 命中；它返回的排序分数不能替代 `/embed` 的 video embedding cosine。LoCoT2V 的 DEVIL `segment_viclip_score` 会先按 temporal block 切分帧，再对 block ViCLIP features 求内部两两相似度，和 ViMax 的 shot-pair IC/CC/GC 不是同一指标；当前 `/embed` whole-video API 不足以声称复现完整 DEVIL 管线。

## 3. CBC/NC/PPR/VSU/STQ/NF LLM judge

先生成 prompt 而不调用 API：

```bash
python run_vimax_planning_judge.py \
  --samples-json /tmp/vimax_samples.json \
  --metric planning --max-items 20 --dry-run \
  --output /tmp/vimax_planning_prompt.json
```

使用 `.env` 中的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`/`OPENAI_API_BASE` 和 `OPENAI_MODEL`：

```bash
set -a; source /data1/AgentG/hhb/CameraMaster/.env; set +a
python run_vimax_planning_judge.py \
  --samples-json /tmp/vimax_samples.json \
  --metric both \
  --source-text /path/to/source_novel.txt \
  --batch-size 20 --max-items 100 \
  --output /tmp/vimax_llm_judge.json
```

CameraMaster 当前 L4 `movie_bible.json` 不是 ViMax Figure 14 要求的完整原始 novel text。若暂时使用 L4 synopsis，只能标记为 `proxy NF`，不能声称复现论文 NF。

## 4. 结果解释与记录

不要把 `CBC/NC/PPR/VSU/STQ/NF` 和 `CC/IC/GC` 合成一个未经论文定义的总分。每次正式实验至少保存：输入 manifest、shot 顺序、scene map、ViCLIP checkpoint、帧采样和 transform、LLM 模型/endpoint、原始 judge response、环境信息。

`Track A` 当前是 13 个 L3 arc clips，`Track B` 当前是 45 个 global-story clips；它们不能直接冒充官方 8--16 shot ViMax-Bench story。Track B 超出官方 shot 范围时，应作为长序列诊断，或先用显式规则切成 custom samples。
