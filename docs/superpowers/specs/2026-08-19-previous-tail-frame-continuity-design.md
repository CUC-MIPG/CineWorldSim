# Previous Tail-Frame Continuity Design

## Goal

让 `VideoAgent` 根据前后事件的剧情、人物、场景和物品关系，决定是否使用上一个事件视频的尾帧作为当前事件的参考图；默认不使用，判断失败时安全回退为不使用。

## Behavior

- `use_previous_tail_frame=False` 时，完全不使用上一事件尾帧。
- `use_previous_tail_frame=True` 时：
  - 第一个事件不使用尾帧。
  - 后续事件调用 VideoAgent 的现有 LLM client 做连续性判断。
  - 只有结构化结果明确为 `use_previous_tail_frame=true` 且上一尾帧文件存在时，才把尾帧加入当前参考图。
  - LLM 调用失败、超时、返回非法 JSON 或结果不明确时，不使用尾帧。
- 当前事件自身的角色关键帧、世界空间图像和物品图像收集逻辑不变。
- `event_video_tasks.json` 的 `continuity_reference` 记录判断结果、分数、理由和最终是否使用。

## LLM Decision Contract

System prompt must instruct the model to compare:

- previous event description and current event description;
- character overlap and relationship continuity;
- location/environment overlap and spatial continuity;
- important object continuity;
- whether the previous tail frame would help preserve visual continuity rather than introduce a conflicting state.

The response must be a JSON object:

```json
{
  "use_previous_tail_frame": false,
  "continuity_score": 0.0,
  "reason": "short explanation"
}
```

`use_previous_tail_frame` is treated as true only when the parsed value is exactly boolean `true`.

## Data Flow

1. Generate or load the previous event's tail frame.
2. Build a compact comparison context from the previous and current narrative nodes.
3. Ask the LLM for a structured continuity decision.
4. If approved, prepend the tail frame to the current reference image list.
5. Persist the decision in `continuity_reference`.

## Failure Handling

- No previous node, no previous tail frame, or missing tail frame: no decision and no tail-frame reference.
- LLM exception: decision `use_previous_tail_frame=false`, reason records the failure class.
- Invalid JSON or invalid boolean: decision `false`.
- Existing output resume path still extracts continuity assets and applies the same decision logic to the next event.

## Testing

- Default option does not call the continuity judge and does not add a tail frame.
- Explicit option with an affirmative LLM response prepends the tail frame.
- Explicit option with a negative response does not add the tail frame.
- LLM failure and malformed response do not add the tail frame.
- Task metadata contains the decision and reason.
