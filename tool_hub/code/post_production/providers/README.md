# Provider Runtime Slots

- `ace_step/`: local runtime code for BGM generation
- `omnivoice/`: local runtime code for dubbing/narration generation

Model root mapping is defined in:
`tool_hub/configs/post_production_model_registry.json`

## ACE-Step Scripts

- Local infer script:
  - `tool_hub/code/post_production/providers/acestep_local_infer.py`
- Minimal API service:
  - `tool_hub/code/post_production/providers/acestep_api_service.py`

Run examples:

```bash
cd /data1/AgentG/hhb/ACE-Step-1.5
uv run python /data1/AgentG/hhb/CameraMaster/tool_hub/code/post_production/providers/acestep_local_infer.py \
  --prompt "cinematic ambient underscore, gentle strings and piano" \
  --duration 20 \
  --audio-format wav
```

```bash
cd /data1/AgentG/hhb/ACE-Step-1.5
uv run uvicorn tool_hub.code.post_production.providers.acestep_api_service:app \
  --app-dir /data1/AgentG/hhb/CameraMaster \
  --host 0.0.0.0 \
  --port 8015
```
