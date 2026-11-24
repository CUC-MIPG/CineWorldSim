# MiniMax H3 Video Backend Implementation Plan

> **For agentic workers:** This plan is being executed inline in the current session. Steps use checkbox syntax for tracking.

**Goal:** Replace ViMax's selected video generation API with CameraMaster's MiniMax H3 synchronous vLLM endpoint without changing the pipeline generator contract or image generation.

**Architecture:** Add a dedicated async video generator that posts multipart form data to the configured vLLM sync URL and returns MP4 bytes as `VideoOutput`. Its request configuration comes from `MINIMAX_H3_*` process environment variables, while `VIDEO_GEN_PROVIDER=minimax_h3` selects it in Agent runtime. All direct pipeline config variants select the new class; old generator classes remain available.

**Tech Stack:** Python 3.12, aiohttp, Pillow-independent file uploads, existing `VideoOutput`, unittest.

---

### Task 1: Define MiniMax H3 vLLM behavior with tests

**Files:**
- Create: `tests/test_minimax_h3_vllm_video_generator.py`
- Modify: `tests/test_generator_protocol.py`
- Modify: `tests/test_vimax_adapters.py`

- [x] Test auto task mapping: zero references to `t2va`, one to `fl2va`, multiple to `ref2va`.
- [x] Test multipart fields and image field names for text, first-frame, and multi-reference requests.
- [x] Test response bytes become `VideoOutput` and non-2xx/empty responses fail clearly.
- [x] Test the new class satisfies the existing generator protocol and provider configuration returns `minimax_h3`.
- [x] Run each focused test before implementation and confirm expected failures.

### Task 2: Implement the async vLLM adapter

**Files:**
- Create: `tools/video_generator_minimax_h3_vllm.py`
- Modify: `tools/__init__.py`

- [x] Read URL and generation options from the documented `MINIMAX_H3_*` environment variables, using CameraMaster defaults.
- [x] Build multipart form fields and upload the correct image fields; pass `extra_params` as JSON with task and audio flow settings.
- [x] Use an aiohttp total timeout, report non-success response text without credentials, reject empty output, and return MP4 bytes.
- [x] Write a small JSON sidecar beside the generated video with URL, status, content type, field names, reference count, byte count, and elapsed time; do not include prompt or file content.
- [x] Run focused adapter tests and protocol tests.

### Task 3: Wire Agent runtime and pipeline configs

**Files:**
- Modify: `agent_runtime/vimax_adapters.py`
- Modify: `configs/idea2video.yaml`
- Modify: `configs/idea2video_minimax.yaml`
- Modify: `configs/script2video.yaml`
- Modify: `configs/script2video_minimax.yaml`
- Modify: `tests/test_vimax_adapters.py`
- Modify: `tests/test_provider_presets.py`

- [x] Select the MiniMax H3 adapter when `VIDEO_GEN_PROVIDER=minimax_h3`; keep current URL-based selection unchanged otherwise.
- [x] Set the video generator `class_path` to `tools.VideoGeneratorMinimaxH3VLLM` in the four existing direct workflow configs.
- [x] Keep chat and image generator settings unchanged.
- [x] Test YAML class paths and provider selection.

### Task 4: Document configuration and verify regressions

**Files:**
- Modify: `README_ZH.md`
- Modify: `readme.md`

- [x] Document endpoint startup prerequisite and sourcing the CameraMaster `.env` before starting ViMax, without copying or printing secret values.
- [x] Document task auto mapping and configurable H3 dimensions, frame count, sampling, and timeout.
- [x] Run focused tests, relevant adapter/config tests, and the repository unittest suite available in `Vimax`; five modules remain blocked by the environment's LangChain import mismatch.
- [x] Review `git diff` and preserve unrelated worktree changes.
