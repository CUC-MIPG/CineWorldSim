# MovieBench 数据集特征与已完成处理

本文根据 CameraMaster 工作区中 MovieBench 数据目录、索引文件、脚本和现存输出整理。统计只描述当前磁盘文件，不代表 MovieBench 官方数据集完整性认证，也不把小样本实验外推为全量处理。

## 1. 数据集内容与组织

工作区数据根目录：

```text
/data1/AgentG/hhb/CameraMaster/MovieBench
```

当前主要包含：

| 路径 | 内容 | 当前用途 |
|---|---|---|
| `Annotation_Shot_Desc_11.15_V2_160movies/` | 按电影分目录的镜头级 JSON 标注 | 原始镜头描述输入 |
| `Annotation_Shot_Des  c_11.15_V2_160movies.zip` | 上述标注目录的压缩包 | 原始归档 |
| `Continuous_Clip_Groups_11.15_V2_160movies/` | 按时间相邻关系整理的连续片段序列 JSON | 事件描述生成输入 |
| `Continuous_Event_Descriptions_11.15_V2_160movies_dryrun_sample/` | 5 个候选序列的规则式 dry-run 描述 | 检查输出结构和流程，不是实际 LLM 推理结果 |
| `Continuous_Event_Descriptions_11.15_V2_160movies_gpt52_test5/` | American Beauty 的 5 个 GPT 事件描述 | 小样本模型测试 |
| `Continuous_Event_Descriptions_11.15_V2_160movies_gpt52_test5_v2/` | 同一组 5 个候选序列的第二版 GPT 描述 | 提示/结果迭代测试 |

### 1.1 镜头级标注字段

抽查的单镜头 JSON 包含：

- `Characters`：角色名称及当前镜头中的状态/描述，可能为空对象。
- `Style Elements`：视觉风格或氛围标签列表。
- `Plot`：镜头中事件或动作的自然语言描述。
- `Background Description`：背景和空间描述。
- `Camera Motion`：镜头运动/静止方式。
- 部分已处理样本还包含 `clip_meta`，例如电影 ID、起止时间码、时长、片内顺序、前一镜头关系等。

文件名也编码镜头时间范围，形式为：

```text
<movie-prefix>_<HH>.<MM>.<SS>.<mmm>-<HH>.<MM>.<SS>.<mmm>.json
```

例如 `0001_American_Beauty_00.01.57.944-00.02.01.546.json` 表示该镜头的开始和结束时间码。

## 2. 当前可确认的数据统计

统计来源是当前目录和 `Continuous_Clip_Groups_11.15_V2_160movies/continuous_dataset_index.json`：

- 原始镜头标注目录当前有 **166 个电影子目录**。
- 原始目录当前有 **107,726 个 JSON 文件**；该计数包含辅助文件，不等同于有效镜头数。
- 连续片段目录当前有 **166 个电影子目录**。
- 连续片段目录索引记载 **56,929 个序列、107,667 个镜头**。
- 构建索引时记载 **3 个无效 JSON 被跳过**。
- 当前连续片段分组索引使用相邻时间间隔小于 **1000 ms** 的规则。
- 原始目录名带有 `160movies`，但当前本地实际目录数为 166；此处保留观察到的目录名和计数，不推断二者差异的原因。

注意：原始文件数量是通过递归计数获得，镜头总数则使用连续片段索引记录值；因统计对象不同，不应直接用原始 JSON 总数减索引镜头数解释差额。

## 3. 已完成的处理

### 3.1 镜头元数据处理脚本

脚本：

```text
scripts/process_moviebench_clip_metadata.py
```

实现能力：

1. 从镜头 JSON 文件名解析起止时间码。
2. 换算起止毫秒、镜头时长。
3. 按电影内时间顺序生成镜头序号。
4. 计算与前一镜头的时间差，并以 1000 ms 阈值判断连续关系。
5. 使用 `--write` 时将结果写回镜头 JSON 的 `clip_meta` 字段；不带该参数为 dry-run。

当前样例原始 JSON 中可看到 `clip_meta`，且连续片段构建产物记录了该元数据。该脚本本身有写回能力；仅凭当前文件状态不能独立还原每次写回执行的命令或时间。

### 3.2 连续镜头分组

脚本：

```text
scripts/build_moviebench_continuous_dataset.py
```

输入：

```text
MovieBench/Annotation_Shot_Desc_11.15_V2_160movies/
```

输出：

```text
MovieBench/Continuous_Clip_Groups_11.15_V2_160movies/
```

处理逻辑：

- 收集符合文件名时间码格式的镜头 JSON。
- 按电影分组，并依镜头起始时间排序。
- 比较相邻镜头的 `current.start_ms - previous.end_ms`。
- 当两者时间差绝对值小于 1000 ms 时，将镜头归入同一连续序列。
- 每个序列写成一个 JSON，保留序列时间范围、镜头数、来源文件名及镜头字段。
- 写出 `continuous_dataset_index.json` 汇总索引。

索引的当前结果为 166 部电影、56,929 个序列、107,667 个镜头，跳过 3 个无效 JSON。

### 3.3 连续序列到事件描述

脚本：

```text
scripts/build_moviebench_event_descriptions_from_continuous_shots.py
```

输入为连续片段分组 JSON；默认只选择至少包含 4 个镜头的序列。脚本将每个镜头的角色、风格、剧情、背景、镜头运动和时间信息按顺序交给 LLM，要求生成一个完整的连续事件描述，并输出：

- `event_title`
- `event_description`
- `main_characters`
- `setting`
- `temporal_progression`
- `shot_to_event_mapping`
- 原始镜头标注及时间信息
- LLM 状态和模型记录

目前磁盘上有三组各 **5 个候选序列**的输出，均来自 `0001_American_Beauty`：

| 输出目录 | 模式 | 候选/写出/失败 | 覆盖范围 |
|---|---|---:|---|
| `Continuous_Event_Descriptions_11.15_V2_160movies_dryrun_sample/` | dry-run 规则式合并 | 5 / 5 / 0 | American Beauty，5 个序列 |
| `Continuous_Event_Descriptions_11.15_V2_160movies_gpt52_test5/` | 实际 LLM 输出 | 5 / 5 / 0 | American Beauty，5 个序列 |
| `Continuous_Event_Descriptions_11.15_V2_160movies_gpt52_test5_v2/` | 第二版实际 LLM 输出 | 5 / 5 / 0 | American Beauty，5 个序列 |

两组 GPT 目录的索引均记录 `dry_run=false`、`written=5`、`failed=0`，单个事件文件的模型字段记录为 `gpt-5.2`。这些证据只支持“5 个候选样例已生成”，不表示 56,929 个序列都已运行。

### 3.4 片长分析与场景分段工具

仓库中还存在以下脚本：

| 脚本 | 能力概述 |
|---|---|
| `scripts/analyze_moviebench_duration_plot.py` | 按片段时长分析并进行一维聚类，输出类别/统计报告；包含 LLM 相关逻辑 |
| `scripts/llm_scene_segment_moviebench.py` | 基于相邻镜头的 Plot 和 Background Description 请求 LLM 判断场景/故事线是否延续 |

当前 `outputs/moviebench_analysis/` 可见 `3075_sitter_smoke` 和 `3075_sitter_llm_strict` 的报告及 CSV 文件。它们是针对特定电影 `3075_SITTER` 的分析结果，不是全数据集统计结果。当前检查未在上述结果目录中确认场景分段脚本的全量输出，因此本文不将该部分记为全量完成。

## 4. 典型处理链路

```text
镜头级 JSON 标注
  -> 解析文件名中的时间码并补充 clip_meta（脚本支持写回）
  -> 按电影排序镜头
  -> 以相邻间隔 < 1000 ms 构建连续镜头序列
  -> 筛选至少 4 镜头的序列
  -> LLM 将序列中的逐镜头标注整合为连续事件文本
  -> 保存事件 JSON、shot-to-event 对应关系及索引
```

## 5. 复现命令模板

以下是脚本参数构成的命令模板，不代表本次检查期间执行过这些命令。

### 5.1 构建连续片段分组

```bash
python -u scripts/build_moviebench_continuous_dataset.py \
  --root MovieBench/Annotation_Shot_Desc_11.15_V2_160movies \
  --output-root MovieBench/Continuous_Clip_Groups_11.15_V2_160movies \
  --threshold-ms 1000 \
  --min-clips 1
```

### 5.2 小样本 LLM 事件描述测试

```bash
python -u scripts/build_moviebench_event_descriptions_from_continuous_shots.py \
  --input-root MovieBench/Continuous_Clip_Groups_11.15_V2_160movies \
  --output-root MovieBench/Continuous_Event_Descriptions_11.15_V2_160movies_gpt52_test5_v2 \
  --min-shots 4 \
  --limit 5 \
  --model gpt-5.2 \
  --retries 3 \
  --no-dry-run \
  --strict-online
```

执行真实 LLM 调用前，应确认项目 `.env` 中对应 API 配置可用；不要将密钥写入命令或文档。

## 6. 当前未完成或不能据现有证据确认的事项

- 没有证据显示所有符合条件的连续序列都完成了 LLM 事件描述；现存两组实际模型输出各只有 5 个序列。
- 没有证据显示 MovieBench 上已完成图像/视频生成基准实验或官方指标评测。
- `3075_SITTER` 的分析报告是单片分析，不代表整个 MovieBench 的统计结论。
- 原始目录名标称 160 部电影，而本地目录和连续分组索引当前显示 166 部；原因尚未确认。
- 统计索引将 3 个无效输入记为跳过；如需发布或正式评测，应先核验其文件清单和影响。

## 7. 核心文件索引

- 原始镜头标注：`MovieBench/Annotation_Shot_Desc_11.15_V2_160movies/`
- 连续片段分组：`MovieBench/Continuous_Clip_Groups_11.15_V2_160movies/`
- 连续分组索引：`MovieBench/Continuous_Clip_Groups_11.15_V2_160movies/continuous_dataset_index.json`
- LLM 事件描述第二版样例索引：`MovieBench/Continuous_Event_Descriptions_11.15_V2_160movies_gpt52_test5_v2/event_description_index.json`
- 分析结果：`outputs/moviebench_analysis/`
- 元数据处理：`scripts/process_moviebench_clip_metadata.py`
- 连续分组：`scripts/build_moviebench_continuous_dataset.py`
- 事件描述：`scripts/build_moviebench_event_descriptions_from_continuous_shots.py`
- 时长分析：`scripts/analyze_moviebench_duration_plot.py`
- 场景分段：`scripts/llm_scene_segment_moviebench.py`
