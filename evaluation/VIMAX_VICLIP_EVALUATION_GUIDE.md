# ViMax-Bench 风格 ViCLIP 评测说明

本文记录如何将 CameraMaster 输出适配为 ViMax-Bench 风格的 shot 序列，并使用 ViCLIP 计算视觉一致性指标。当前示例使用 `outputs/vimax_track_ab_3020/artist_extreme_weather_typeA`。这是 CameraMaster 自定义样本上的复刻/适配评估，不是官方 ViMax-Bench 测试集结果。

## 1. 评测指标

每个生成镜头（shot）对应一个独立视频片段。ViCLIP `/embed` 为每个片段提取一个 768 维、L2 归一化的视觉向量。对任意两个镜头 (i,j)，计算余弦相似度：

```text
sim(i, j) = clip(dot(z_i, z_j), -1, 1)
```

输入向量已归一化，因此点积等价于余弦相似度；代码仍会做归一化和 `[-1, 1]` 裁剪，以消除浮点误差。

| 指标 | 计算范围 | 含义 |
| --- | --- | --- |
| IC (Intra-Consistency) | 同一 scene 内所有不同 shot pair 的相似度均值 | 同场景镜头的视觉一致性 |
| CC (Cross-Consistency) | 不同 scene 间所有 shot pair 的相似度均值 | 跨场景镜头的视觉一致性 |
| GC (Global-Consistency) | 故事内所有不同 shot pair 的相似度均值 | 整体镜头序列的视觉一致性 |

IC 和 CC 需要每个 shot 都有可靠的 `scene_id`。任何场景标签缺失时，程序将两者记为 `null`，不会用 0 替代；GC 不依赖 scene 标签，仍可计算。分数是视觉 embedding 的相似程度，不等同于文本-视频语义匹配，也不直接衡量剧情正确性。

多样本汇总同时报告：

- `story_macro_mean`：先按故事计算指标，再对故事分数等权平均。
- `pair_weighted_mean`：按每个故事实际参与的 pair 数加权平均。

单样本时，两种汇总值相同。正式比较应优先按预先确定的 benchmark 聚合协议报告，不能在看到结果后选择更有利的汇总口径。

## 2. 脚本职责

- `build_vimax_output_inputs.py`：读取 CameraMaster Track A/B 的 `narrative_nodes.json`、`path_plan.json` 和镜头视频，生成统一输入 JSON；Track A 从视频文件名读取 scene，Track B 可选择使用 location 作为明确标注的代理 scene，或完全不提供 scene。
- `vimax_bench_adapter.py`：将统一输入转成 evaluator 使用的 shot 列表，检查镜头数量与 scene 标签完整性，并标记协议资格。
- `run_vimax_bench_eval.py`：调用 ViCLIP 提取特征，计算 IC/CC/GC 和数据集汇总，写出结果 JSON。
- `vimax_metrics.py`：封装 ViCLIP 本地/HTTP encoder、特征校验、余弦计算和指标聚合。

评测以镜头级视频片段为单位，不应先将整条 Track 拼成单个视频再提特征。HTTP 服务已验证 `/embed` 合约：输入视频，参数为抽帧数和分辨率，返回 768 维归一化 embedding；示例采用 8 帧、224 分辨率。

## 3. 运行流程

在 `evaluation` 目录执行。先从 CameraMaster 输出构建输入文件：

```bash
python build_vimax_output_inputs.py \
  --output-root /data1/AgentG/hhb/CameraMaster/outputs/vimax_track_ab_3020/artist_extreme_weather_typeA \
  --output vimax_results/artist_extreme_weather_typeA_inputs.json
```

上面默认包含 Track A 和 Track B。若只评 Track A，可加 `--tracks A`。若需要生成不使用 Track B 场景代理的严格输入：

```bash
python build_vimax_output_inputs.py \
  --output-root /data1/AgentG/hhb/CameraMaster/outputs/vimax_track_ab_3020/artist_extreme_weather_typeA \
  --tracks B \
  --track-b-scene-source none \
  --output vimax_results/artist_extreme_weather_typeA_trackB_noscene_inputs.json
```

确认 ViCLIP 服务可用后运行正式特征评测。服务地址可按部署环境调整：

```bash
curl -s http://172.17.34.43:8890/health

python run_vimax_bench_eval.py \
  --inputs-json vimax_results/artist_extreme_weather_typeA_inputs.json \
  --backend http \
  --embed-url http://172.17.34.43:8890/embed \
  --num-frames 8 \
  --resolution 224 \
  --timeout 600 \
  --cache-dir vimax_feature_cache/artist_extreme_weather_typeA \
  --output vimax_results/artist_extreme_weather_typeA_vimax_visual_metrics.json
```

只评某个 Track 时，可在命令中增加 `--track A` 或 `--track B`。特征缓存以视频路径、文件大小/修改时间和 encoder 参数生成 key；复跑时可复用缓存。若视频内容被覆盖但文件大小和时间戳被人为保留，应清理对应缓存或换新缓存目录。

离线 smoke test 只验证输入适配、流程和 JSON 写出，不产生有效 ViCLIP 分数：

```bash
python run_vimax_bench_eval.py \
  --inputs-json vimax_results/artist_extreme_weather_typeA_inputs.json \
  --fake-encoder \
  --output /tmp/vimax_smoke_test.json
```

`--fake-encoder` 输出不可用于论文、模型比较或 benchmark 报告。若不使用远程服务，也可指定 `--backend local --checkpoint <ViCLIP checkpoint>`，但当前建议复用已验证的 HTTP `/embed` 服务。

## 4. 结果文件

输出 JSON 顶层包含 `schema_version`、`encoder`、`backend`、`dataset_aggregate` 和 `samples`。每个 sample 记录：

- `protocol`：shot 数是否在 8–16 范围、scene 标签是否完整、scene 数量及协议资格标记。
- `metrics.IC/CC/GC.score`：指标均值；`pair_count`：参与均值的镜头对数；`std`：pair 分数标准差；`pairs`：逐 pair 分数。
- `feature_extraction`：encoder、接口、抽帧数、分辨率、缓存命中数及镜头数。
- `shots`：镜头 ID、场景 ID 和源视频路径，便于追溯。

`protocol.official_vimax_result_eligible=true` 只代表该样本满足脚本检查的 shot/scene 结构条件，不会将自定义样本变成官方 ViMax-Bench 样本。报告时应保留 `results_are_official_vimax_bench=false` 和来源说明。

## 5. 本次样例结果

本次结果来自 `vimax_results/artist_extreme_weather_typeA_vimax_visual_metrics.json`，ViCLIP HTTP 服务返回的特征均为 768 维，8 帧、224 分辨率。

| Track / 场景处理 | 镜头数 | IC | CC | GC | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| A，文件名中的真实 scene 分组 | 12 | 0.559361 (18 pairs) | 0.492492 (48) | 0.510729 (66) | 数量/标注结构满足脚本协议条件；仍是自定义样本 |
| B，location 作为 scene 代理 | 20 | 0.754508 (126) | 0.612060 (64) | 0.706525 (190) | IC/CC 仅作代理标签诊断；20 shots 超出 8–16 范围 |
| B，不提供 scene 标签 | 20 | N/A | N/A | 0.706525 (190) | 推荐作为 B 的严格报告口径；IC/CC 无法定义 |

Track A 的 pair 数为 `12 * 11 / 2 = 66`，其中同场景 18 对、跨场景 48 对。Track B 的 GC 为 `20 * 19 / 2 = 190` 对。location 代理把相同地点的不同剧情段落视作同 scene，可能改变 IC/CC 的解释，因此不能与有正式场景标注的结果等同。

## 6. 报告边界与注意事项

1. Track A 满足脚本设定的 8–16 shots、2–4 scenes、scene 完整等结构检查，但它不是 ViMax-Bench 官方样本集；只能写作 CameraMaster 自定义样例上的 ViMax-Bench 风格评估。
2. Track B 当前为 20 shots，违反每个 benchmark sample 的 shot 数限制。若要进行严格的官方协议复现，应按 benchmark 规定重新构造符合范围的样本，而不是将长 Track 直接切块并默认切块等价于官方故事样本。
3. Track B 的 location scene 是人工设定的代理，不是官方 scene annotation。主结果使用无场景标签版本报告 GC；代理 IC/CC 可放在补充诊断中，需显式命名为 location-proxy。
4. ViCLIP embedding cosine 是视觉表征一致性；不能替代 ViMax-Bench 其他可能的评价项，也不是视频-文本检索分数。ViCLIP `/predict` 文本排序、LoCoT2V 的 DEVIL segment score 均为不同评测任务/协议，不能混用。
5. 固定并记录 ViCLIP checkpoint/服务版本、抽帧数、分辨率、镜头切分和 scene 标注来源。不同配置产生的分数不应直接横向比较。
