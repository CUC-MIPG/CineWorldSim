# CineWorldSim Full Context Handoff

更新时间：2026-09-11（Asia/Shanghai）

本文档用于在新对话中恢复 CineWorldSim 的完整项目上下文，包括研究目标、系统架构、当前实现、实验结果、失败恢复机制和下一步计划。本文档中的内容区分“已实现/已验证”和“目标/计划”，不能把设计目标误认为运行事实。除本文件外，本次交接不修改代码、配置、已有输出或运行中的任务。

## 1. 当前任务

当前主任务是为 CineWorldSim 建立一套面向所有后续任务的视频生成前恢复机制：

```text
world initialization
  -> narrative simulation
  -> world state / world delta
  -> event keyframe
  -> narrative node selection
  -> LLM path orchestration / MCTS path orchestration
  -> VideoAgent
```

恢复机制需要能够：

- 识别结构化的 LLM 调用失败和 fallback；
- 判断失败发生在哪个 step、角色或阶段；
- 不把 fallback 产物误认为成功产物；
- 对已知失败节点进行重跑；
- 在失败后从新目录恢复后续 pre-video 阶段；
- 在进入视频生成前完成健康检查。

当前最具体的故障背景是 2026-09-10 的 demo1 运行中，step 13 及后续相关 LLM 调用出现 503 或连接/DNS 错误，并使用了 fallback。

## 2. 已完成的代码工作

### 2.1 LLM fallback 记录

已修改 `narrative/simulation_scaffold.py`：

- `ModelConfig` 增加 `call_history`；
- 每次模型调用记录 `call_id`、`call_label`、`status`、`fallback_used`、`error`；
- 角色级记录 `model_calls`、`fallback_used`、`fallback_reasons`；
- step 级记录 fallback 状态；
- 为 `character_action`、`action_evaluation`、`story_anchor`、`storyline_completion` 等调用增加标签。

### 2.2 全流程 pre-video 检查和恢复

已新增或完成 `scripts/recover_pre_video_pipeline.py`，提供：

```text
check -> plan -> repair
```

检查阶段覆盖：

- world model 的 locations、items、portraits 及图像文件；
- narrative simulation 的 step 连续性、checkpoint、role logs；
- `world_state_stepN.json`；
- `world_deltas/world_delta_stepN.json`；
- event keyframes；
- narrative node selection；
- LLM path 和 MCTS path；
- 日志、JSON 状态字段及调用记录中的 fallback/error 标记。

恢复阶段的约束：

- 必须输出到新的目录；
- 不覆盖源 pipeline 目录；
- 初始化、推演、world delta 或 event keyframe 失败时，保守地重建后续 pre-video 阶段；
- node selection 失败时重建两个 path 阶段；
- 仅 LLM path 或仅 MCTS path 失败时，只重建对应路径；
- 不负责生成视频，也不负责最终视频拼接。

对应说明文档为 [`docs/pre_video_recovery.md`](./pre_video_recovery.md)。

### 2.3 已知 fallback 节点重跑工具

已完成 `scripts/rerun_fallback_narrative_nodes.py`：

- 使用 `STEP:ACTOR:SOURCE` 指定节点，例如 `13:Mara_Venn:evaluation`；
- 支持 `--rerun-downstream`，从目标 step 重跑到原始最大 step；
- 不覆盖源目录；
- 失败时写入 `fallback_rerun_manifest.json`；
- 记录 `status`、`error_type`、`error`、请求节点和已产生的 partial artifacts。

### 2.4 测试和语法验证

前序会话已执行并记录以下验证：

```bash
pytest -q tests/test_pre_video_recovery.py tests/test_rerun_fallback_narrative_nodes.py
```

结果：`14 passed`

```bash
pytest -q tests
```

结果：`135 passed`

```bash
python -m py_compile \
  narrative/simulation_scaffold.py \
  scripts/recover_pre_video_pipeline.py \
  scripts/rerun_fallback_narrative_nodes.py
```

结果：通过。

## 3. 当前产物状态

### 3.1 原始 demo1 pre-video 目录

目录：

```text
outputs/gpt55_pre_video_demo1/all_banana
```

只读检查命令：

```bash
python scripts/recover_pre_video_pipeline.py \
  --pipeline-dir outputs/gpt55_pre_video_demo1/all_banana \
  --mode check \
  --report-dir /tmp/cameramaster_final_gpt55_check \
  --log-level INFO
```

检查报告：

```text
/tmp/cameramaster_final_gpt55_check/pre_video_health_report.json
/tmp/cameramaster_final_gpt55_check/pre_video_recovery_plan.json
```

报告结果为：

```text
status=healthy
```

持久化产物结构包括：

- 15 个 simulation steps；
- 16 个 world states，即 step 0 到 step 15；
- 15 个 world deltas；
- 45 个 event keyframes；
- 3 个角色肖像；
- narrative nodes 和 path 产物；
- LLM path 和 MCTS path 产物；
- 检查报告中没有结构化 fallback 标记。

重要解释：这个 `healthy` 只证明当前旧产物的结构完整，不能证明用户提供的 2026-09-10 原始运行日志没有发生 fallback。该目录是在结构化 fallback 记录接入前生成的，因此旧 `step_logs.json` 没有保留那次日志中的失败状态。

### 3.2 demo1 step 13 恢复目录

目录：

```text
outputs/gpt55_pre_video_demo1_repaired_step13/all_banana
```

恢复尝试使用的命令：

```bash
PYTHONUNBUFFERED=1 \
conda run -n mifage python -u scripts/rerun_fallback_narrative_nodes.py \
  --pipeline-dir outputs/gpt55_pre_video_demo1/all_banana \
  --node 13:Mara_Venn:evaluation \
  --node 13:Ilya_Trent:action \
  --rerun-downstream \
  --output-dir outputs/gpt55_pre_video_demo1_repaired_step13/all_banana \
  --strict-online \
  --retries 5 \
  --log-level INFO \
  2>&1 | tee outputs/gpt55_pre_video_demo1_repaired_step13.log
```

实际结果：

- step 13、14、15 被重新执行；
- 相关 LLM 调用均出现 fallback；
- 重建 world delta 时失败；
- 核心错误为 `httpx.ConnectError: [Errno -3] Temporary failure in name resolution`；
- 失败位置涉及 `narrative/world_delta_judge.py` 和 `agents/llm_tool.py`；
- 该目录只有部分 checkpoint、partial step logs、部分 world states/deltas 和角色肖像；
- 没有完整 `pipeline_result.json`；
- 该目录不能用于 node selection、path orchestration 或视频生成。

日志：

```text
outputs/gpt55_pre_video_demo1_repaired_step13.log
```

### 3.3 step 13 重试目录

目录：

```text
outputs/gpt55_pre_video_demo1_repaired_step13_retry/all_banana
```

该目录包含：

- `narrative_checkpoint.json`，记录 completed step 15；
- `step_logs.partial.json`；
- `world_state_step0.json` 到 `world_state_step15.json`；
- `world_deltas/world_delta_step1.json` 到 `world_delta_step15.json`；
- 角色肖像。

但没有完整 pipeline manifest，不能仅凭 `completed_steps=15` 判定恢复成功。对应日志文件为空：

```text
outputs/gpt55_pre_video_demo1_repaired_step13_retry.log
```

该目录仍不能进入视频生成阶段，除非重新执行 pre-video health check 并得到健康报告。

### 3.4 视频产物

对以下三个目录进行过只读视频文件检查：

```text
outputs/gpt55_pre_video_demo1
outputs/gpt55_pre_video_demo1_repaired_step13
outputs/gpt55_pre_video_demo1_repaired_step13_retry
```

检查目标为 `.mp4`、`.webm`、`.mov`，没有输出文件。因此这三个目录当前没有被该检查发现的视频产物或最终拼接视频。

## 4. 当前卡点

当前主要卡点不是恢复脚本的语法问题，而是 LLM/API 网络不可用：

```text
httpx.ConnectError: [Errno -3] Temporary failure in name resolution
```

此前原始日志还明确出现：

```text
Error code: 503 - Service temporarily unavailable
```

所以目前不能把 fallback 生成的动作、评估、story anchor 或 world delta 当作可靠结果。即使 checkpoint 的 step 数已经达到 15，也不能跳过 fallback 检查。

另外，当前工作区不是一个有效的 Git 仓库：

```text
.git: directory
```

但 `.git` 目录为空，执行 `git rev-parse --is-inside-work-tree`、`git status --short` 和 `git log -1 --oneline` 均返回：

```text
fatal: not a git repository (or any parent up to mount point /)
```

因此无法提供可靠的 commit、分支、modified/untracked 文件状态。本结论不是“工作区干净”，而是“当前目录缺少可用 Git 元数据”。

## 5. 下一步计划

### 第一步：恢复 LLM/API 网络

先确认 `api.xi-ai.cn` 或当前 `.env` 配置的 LLM endpoint 可以解析并返回正常响应。不要在网络仍然失败时重复长时间运行完整 pipeline。

### 第二步：在新目录重跑 step 13 及下游

不要复用两个已经失败的恢复目录。建议使用新的目录：

```bash
cd /data1/AgentG/hhb/CameraMaster

PYTHONUNBUFFERED=1 \
conda run -n mifage python -u scripts/rerun_fallback_narrative_nodes.py \
  --pipeline-dir outputs/gpt55_pre_video_demo1/all_banana \
  --node 13:Mara_Venn:evaluation \
  --node 13:Ilya_Trent:action \
  --rerun-downstream \
  --output-dir outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana \
  --strict-online \
  --retries 5 \
  --log-level INFO \
  2>&1 | tee outputs/gpt55_pre_video_demo1_repaired_step13_retry2.log
```

### 第三步：检查恢复结果

先确认 manifest：

```bash
test -f \
  outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/fallback_rerun_manifest.json \
  && cat outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana/fallback_rerun_manifest.json \
  || echo rerun_failed_or_incomplete
```

再执行统一 pre-video 检查：

```bash
python scripts/recover_pre_video_pipeline.py \
  --pipeline-dir outputs/gpt55_pre_video_demo1_repaired_step13_retry2/all_banana \
  --mode check \
  --report-dir /tmp/gpt55_step13_retry2_check \
  --log-level INFO
```

只有报告为 `healthy`，并且确认没有 fallback，才继续：

1. 重建 narrative node selection；
2. 重建 LLM path 和 MCTS path；
3. 检查所有关键帧、world states、world deltas；
4. 单独启动 VideoAgent；
5. 检查每个 event video、辅助镜头和最终 concat。

## 6. 高成本重复问题和注意事项

- 不要把 `completed_steps=15` 当作 LLM 调用成功的证明；必须检查 `fallback_used`、`model_calls`、`fallback_reasons` 和错误日志。
- 旧版本产物可能没有结构化 fallback 字段；对这类产物只能说明“当前结构检查通过”，不能追溯证明历史请求成功。
- `rerun_fallback_narrative_nodes.py` 是窄范围节点恢复工具；`recover_pre_video_pipeline.py` 是通用的 stage 级检查、计划和恢复工具。未知失败阶段优先使用后者。
- 所有恢复都写新目录，严禁覆盖 `outputs/gpt55_pre_video_demo1/all_banana`。
- fallback 恢复结果不能直接用于视频生成；必须重新跑 pre-video health check。
- `rerun_fallback_narrative_nodes.py` 不负责生成视频和最终拼接；恢复成功后仍需单独运行 VideoAgent。
- 网络/DNS/503 问题不能误判成 world delta 或 pipeline 逻辑本身的问题。
- 长时间运行前先做单次 LLM endpoint 健康检查，避免再次消耗完整 step 13-15 的运行时间。
- 当前 `.git` 不可用，后续若需要变更审计，应先恢复或确认正确的 Git 仓库元数据；不要据此声称代码没有修改。
- 不要输出 `.env`、API key 或 `test.txt` 中的密钥。
- 当前已检查的 gpt55 demo1 目录没有发现视频文件，不要将 pre-video 健康报告误解为视频生成完成。

## 7. 关键文件和路径索引

代码：

```text
narrative/simulation_scaffold.py
narrative/world_delta_judge.py
narrative/event_keyframe_renderer.py
agents/llm_tool.py
scripts/recover_pre_video_pipeline.py
scripts/rerun_fallback_narrative_nodes.py
scripts/run_full_narrative_world_video_pipeline.py
scripts/run_narrative_world_path_pipeline.py
scripts/run_dual_backend_narrative_world_pipeline.py
```

文档：

```text
docs/pre_video_recovery.md
iclr/experiment_implementation_protocol.md
docs/handoff_pre_video_recovery_20260911.md
```

主要输出：

```text
outputs/gpt55_pre_video_demo1/all_banana
outputs/gpt55_pre_video_demo1_repaired_step13/all_banana
outputs/gpt55_pre_video_demo1_repaired_step13_retry/all_banana
```

检查报告：

```text
/tmp/cameramaster_final_gpt55_check/pre_video_health_report.json
/tmp/cameramaster_final_gpt55_check/pre_video_recovery_plan.json
```

## 8. 本次交接实际操作边界

本次交接期间执行的操作为只读检查和新增本文档：

- 读取技能文件路径时发现会话列出的 `using-superpowers` 本地路径不存在，未使用该技能；
- 读取关键代码、日志、manifest、输出目录和现有文档；
- 检查进程，没有发现 CineWorldSim pipeline、恢复脚本或视频生成进程；
- 检查 Git，确认 `.git` 为空目录且不是有效仓库；
- 检查视频扩展名文件，没有发现目标三个 gpt55 目录中的视频文件；
- 新增本 handoff 文档；
- 没有修改现有代码、配置、旧文档、输出目录或运行产物。

新会话恢复时，优先阅读本文件的第 3、4、5 节，然后执行新的 endpoint 健康检查和新的恢复目录检查。
