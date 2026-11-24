# CineWorldSim 层级定义与数据重构说明

## 1. 目的

CineWorldSim 的叙事层级定义为：

```text
L4：整部电影的全局故事线与全片设定
└── L3：全局故事线中的局部故事弧
    └── L2：事件级叙事单元
        └── L1：镜头级视觉证据
```

L4 是单部电影唯一的全局层，描述全片总体剧情、人物关系演化、反复出现的地点/物体以及有证据支持的全局约束。L3 是局部 arc，不是全片总故事。L2 是供事件级分析及视频生成使用的事件单位；5 秒视频是实验时长设置，不代表每个原始 L2 事件都天然满足 5 秒的叙事完整性。L1 保留镜头级观察和证据。

## 2. 当前源数据与定义偏差

以 `0001_American_Beauty` 为例，现有数据已经有一份 `l4_story/movie_bible.json`，其中包含全片 `movie_synopsis`、全局人物档案、重复地点/物体及证据约束；另外 `l4_story/story_arcs.json` 有 16 条局部故事弧，但记录字段将其命名为 `l4_story_arc_id`。现有 `l3_scenes/final.json` 则含 143 条场景 occurrence，不是所需定义中的局部故事弧层。

因此本次重构不是重新调用 LLM 生成叙事，而是纠正分层命名和组织方式：将 Movie Bible 提升/固定为唯一 L4，将原 16 条 story arcs 转成 L3 arcs，并把旧 L3 scene occurrences 保留为来源证据上下文。L2/L1 及 Track A/B 样本正文不做改写。

## 3. 重构映射

| 重构层/产物 | 输入来源 | 处理规则 |
|---|---|---|
| L4 全局故事 | `l4_story/movie_bible.json` | 保留现有 `result` 内容，标记 `hierarchy_level=L4_global_movie_story`；不合成或补写 synopsis |
| L3 局部故事弧 | `l4_story/story_arcs.json:result.story_arcs` | 逐条保留弧内容，生成稳定 ID `{movie_id}_L3ARC_NNN`；旧 ID 保存到 `source_l4_story_arc_id` |
| L3 -> 来源场景证据 | arc 的 `l3_scene_ids`、事件的 `source_l3_ids` | 改名为 `source_scene_occurrence_ids`，明确它们是旧场景 occurrence 来源，不是新 L3 arc ID |
| L2 事件 | `l2_events/` | 原样复制，保留原 ID；每条 L3 arc 的 `l2_event_ids` 仍指向这些事件 |
| L1 镜头证据 | `l1_shots/` | 原样复制，保留原 ID；通过 arc 的 `l1_shot_ids` 以及 L2 的 `source_shot_ids` 追溯 |
| Track A/B | `track_a.json`、`track_b.json` | 若存在则原样复制；报告提供旧 arc ID 到新 L3 arc ID 的映射，不更改 Track 样本字段 |

脚本会验证 arc 声明的 L2/L1 引用，以及 essential event 中的 L2/场景来源引用。默认任一引用无法解析时立即失败，不生成一个表面成功但链接断裂的数据集。若要保留有引用缺口的源材料供审计，可显式传入 `--allow-unresolved-references`；此时不会改写或删除原 ID，报告会列出未解析 ID，并将状态标记为 `incomplete`。输入与输出目录必须不同，且输出目录必须尚不存在。

## 4. 输出结构

```text
<output-root>/
  l4_global/
    movie_bible.json
  l3_arcs/
    story_arcs.json
  l2_events/                  # 从输入原样复制
  l1_shots/                   # 从输入原样复制
  source_context/
    l3_scene_occurrences.json # 旧场景 occurrence，作为来源证据保留
  track_a.json                # 存在时原样复制
  track_b.json                # 存在时原样复制
  hierarchy_restructure_report.json
```

此旁路目录不覆盖原始 `l4_story/`、`l3_scenes/` 或任何既有实验输出。`hierarchy_restructure_report.json` 记录来源/输出路径、层级说明、数量、Track arc 映射和验证结果。

## 5. 执行

从项目根目录运行：

```bash
python -u scripts/restructure_cineworldsim_hierarchy.py \
  --input-root outputs/moviebench_cineworldsim/0001_American_Beauty \
  --output-root outputs/moviebench_cineworldsim_restructured/0001_American_Beauty
```

默认运行是离线文件转换，不访问 `.env`、不调用 LLM/图像/视频 API。输出目录已存在时会拒绝覆盖；如需重新生成，应选择一个新的输出目录名。

### 可选：用 LLM 重新生成 L4 全局故事

显式添加 `--generate-l4` 才会调用 `.env` 中配置的 OpenAI-compatible API。脚本使用现有 Movie Bible 作为全局先验，并将 L3 场景与 L2 事件转成保留来源 ID 的压缩摘要；不直接发送全部 L1 原始标注。调用范围默认使用 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`，也支持 `--llm-scope` 指向 scope-specific 配置。

为避免 143 个 L3 场景和 598 个 L2 事件造成上下文膨胀，发送给 LLM 的 L3/L2 证据使用紧凑数组 schema，而不是逐条重复 JSON 字段名。L3 保留场景 ID、部分 L2 ID 及其数量、时间/地点/人物、场景叙事和状态转移；L2 保留事件 ID、全部来源 L1 shot ID、短标题/描述，以及首条动作和状态变化摘要。真实样本的当前 prompt 规模约为 230,676 字符，实际 API 调用前仍会把完整请求写入审计文件。

```bash
python -u scripts/restructure_cineworldsim_hierarchy.py \
  --input-root outputs/moviebench_cineworldsim/0001_American_Beauty \
  --output-root outputs/moviebench_cineworldsim_restructured/0001_American_Beauty_llm_l4 \
  --allow-unresolved-references \
  --generate-l4 \
  --llm-scope OPENAI \
  --llm-retries 3 \
  --llm-timeout 180
```

生成的 L4 写入 `l4_global/movie_bible.json`，请求和响应审计分别写入 `l4_global/l4_generation_request.json` 与 `l4_global/l4_generation_response.json`；审计文件不会包含 API key。失败时默认终止，不写入重构输出；只有显式添加 `--fallback-existing-bible` 才会使用源 Movie Bible 回退，并在结果及报告中标记 `source_fallback`。`--allow-unresolved-references` 只控制源数据引用校验，不改变 LLM 失败策略。

当前源数据存在少量旧 arc 引用不一致：L2 events 的 749 个唯一 `source_shot_ids` 均能在 L1 单元中解析；但旧 story arcs 自身有 31 个 `l1_shot_ids` 无法解析，另有 25 个 `l3_scene_ids`/essential-event 场景引用无法在 `l3_scenes/final.json` 解析。因此，对该样本生成旁路副本时需要明确选择：严格模式会失败并报告缺口；传入 `--allow-unresolved-references` 可以生成带 `incomplete` 标记的副本，但该副本不能被描述为引用完整的数据集。脚本不进行时间戳近似匹配或猜测映射。

## 6. 评测和视频生成的后续约束

修正数据组织不等于现有评测脚本已经改为按 L2 生成视频。当前 `scripts/run_moviebench_arc_video_eval.py` 仍按旧的 L4 arc 输入，并从 arc 关联的 L3 scene records 构造视频节点。后续若要按目标层级评测，应另行更新/新增 evaluator：以每条 L2 event 为候选视频节点，将其所属 L3 arc 和 L4 movie bible 用作上下文，将 L1 shots 用作证据/视觉 grounding，并处理 L2 事件过短、过长或缺少可视动作的筛选策略。该脚本不会在本次数据重构中自动变更。

Track A/B 当前是按旧 arc 构建的样本。重构报告只保留它们与 L3 arc 的 ID 映射，不代表其输入语义已经自动改为新层级；采用新层级训练或评测前仍需单独检查 `full_story_S`、`short_storyline_L`、anchor 和约束文本是否泄漏目标事件。

## 7. 验收标准

- 重构输出只含一个 L4 global movie story。
- 每个源 story arc 恰好映射到一个 L3 arc，且新 ID 唯一。
- L2 event、L1 shot、来源 scene occurrence 引用均可解析。
- L1/L2/Track A/B 文件内容与源文件一致。
- 输入树内容不变；目标目录存在时命令安全失败。
- 重构不宣称改变了下游 evaluator 的视频节点粒度。
