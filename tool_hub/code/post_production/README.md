# Post-Production Tools

This folder hosts standalone CLIs for post-production tool testing while reusing
the production implementations under `agents/post_production_tools.py`.

## Run

```bash
conda run -n mifage python tool_hub/code/post_production/run_post_tools.py -h
```

### Audio Service Test

```bash
conda run -n mifage python tool_hub/code/post_production/run_post_tools.py audio \
  --mode both \
  --text "你好，这是工具中心音频联调测试。" \
  --output-dir outputs/tool_hub/post_audio_test \
  --strict-service
```

### Dubbing Local OmniVoice Test

```bash
conda run -n mifage python tool_hub/code/post_production/run_post_tools.py audio \
  --mode dubbing \
  --dubbing-provider omnivoice \
  --voice-profile "calm male narrator" \
  --text "你好，这是OmniVoice本地推理测试。" \
  --output-dir outputs/tool_hub/post_audio_test
```

### Video Edit Test

```bash
conda run -n mifage python tool_hub/code/post_production/run_post_tools.py edit \
  --input-videos outputs/e2e_text_only_full_run/videos/video_SC001_SH001.mp4,outputs/e2e_text_only_full_run/videos/video_SC001_SH002.mp4 \
  --output-video outputs/tool_hub/post_edit_test/final_master_post.mp4
```
