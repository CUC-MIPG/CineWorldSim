# DigitalHuman ScriptAgent Bridge

This bridge now includes a built-in compatibility shim for `XD-MU-ScriptAgent` on `swift + qwen_omni_utils`.
It auto-injects `sitecustomize` via `PYTHONPATH` and sets writable cache paths (`/tmp/...`) to avoid read-only lock errors.
You can disable the shim with `SCRIPTAGENT_ENABLE_OMNI_COMPAT=0`.

## Smoke test
```bash
bash /data1/AgentG/hhb/CameraMaster/scripts/digitalhuman_scriptagent/smoke_test.sh
```

## Verify Omni compatibility
```bash
bash /data1/AgentG/hhb/CameraMaster/scripts/digitalhuman_scriptagent/verify_omni_compat.sh
```

## Director
```bash
bash /data1/AgentG/hhb/CameraMaster/scripts/digitalhuman_scriptagent/run_director.sh \
  --script_path /data1/AgentG/hhb/digitalhuman/ScriptAgent/code/storyscript.txt \
  --model sora2-pro \
  --style anime \
  --output_dir /data1/AgentG/hhb/CameraMaster/outputs/scriptagent_director_test
```

## Critic Script
```bash
bash /data1/AgentG/hhb/CameraMaster/scripts/digitalhuman_scriptagent/run_critic_script.sh \
  --scripts_jsonl /abs/path/infer_script_result.jsonl \
  --dialogues_json /abs/path/test.json \
  --output_json /data1/AgentG/hhb/CameraMaster/outputs/scriptagent_critic/script_eval.json
```

## Critic Video
```bash
bash /data1/AgentG/hhb/CameraMaster/scripts/digitalhuman_scriptagent/run_critic_video.sh \
  --video_folder /abs/path/final_video \
  --mapping_jsonl /abs/path/video_dialogues.jsonl \
  --output_json /data1/AgentG/hhb/CameraMaster/outputs/scriptagent_critic/video_eval.json
```
