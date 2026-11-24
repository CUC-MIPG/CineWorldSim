# CineWorldSim LLM L4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in LLM path that synthesizes a new movie-level L4 global story from the existing Movie Bible plus compact L3/L2 evidence.

**Architecture:** Keep `restructure_dataset()` as the orchestration boundary. Add pure prompt/projection/validation helpers to the same script, use the existing `LLMJsonTool` for OpenAI-compatible calls, and write generation audit artifacts under `l4_global/`. The default path continues to copy the source Movie Bible without an API call.

**Tech Stack:** Python 3.12, `argparse`, JSON files, `agents.llm_tool.LLMJsonTool`, pytest, existing CineWorldSim hierarchy script.

---

### Task 1: Add Failing Tests for L4 Generation Helpers

**Files:**
- Modify: `tests/test_restructure_cineworldsim_hierarchy.py`
- Test: `tests/test_restructure_cineworldsim_hierarchy.py`

- [ ] **Step 1: Add a test for compact L4 input projections**

```python
from scripts.restructure_cineworldsim_hierarchy import build_l4_generation_prompt


def test_l4_prompt_contains_prior_and_compact_scene_event_ids() -> None:
    system, user = build_l4_generation_prompt(
        "film_01",
        {"movie_id": "film_01", "movie_synopsis": {"short_synopsis": "prior"}},
        [{"l3_scene_id": "scene_01", "l2_event_ids": ["event_01"], "scene_narrative": "scene"}],
        [{"l2_event_id": "event_01", "source_shot_ids": ["shot_01"], "event_description": "event"}],
    )
    assert "film_01" in user
    assert "scene_01" in user
    assert "event_01" in user
    assert "shot_01" in user
    assert "movie_synopsis" in user
    assert "Return valid JSON only" in system
```

- [ ] **Step 2: Add a test for generated L4 validation**

```python
from scripts.restructure_cineworldsim_hierarchy import validate_generated_l4


def test_validate_generated_l4_requires_global_storyline_and_synopsis() -> None:
    valid = validate_generated_l4(
        {
            "movie_id": "film_01",
            "movie_synopsis": {"short_synopsis": "whole film"},
            "global_storyline": "A complete global storyline.",
            "evidence_trace": {"source_l3_ids": [], "source_l2_ids": [], "source_l1_ids": []},
        },
        "film_01",
    )
    assert valid["movie_id"] == "film_01"
    with pytest.raises(ValueError, match="global_storyline"):
        validate_generated_l4({"movie_id": "film_01", "movie_synopsis": "x"}, "film_01")
```

- [ ] **Step 3: Add an integration test proving `--generate-l4` uses an injected LLM**

```python
def test_restructure_generate_l4_writes_llm_bible_and_audits(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    target = tmp_path / "restructured"
    _source_tree(source)
    response = {
        "movie_id": "film_01",
        "movie_synopsis": {"short_synopsis": "generated whole film"},
        "global_storyline": "Generated global storyline.",
        "evidence_trace": {"source_l3_ids": ["scene_01"], "source_l2_ids": ["event_01"], "source_l1_ids": ["shot_01"]},
    }

    class FakeTool:
        def __init__(self, **kwargs):
            self.model = "fake-model"

        def run_json(self, **kwargs):
            return response

    monkeypatch.setattr("scripts.restructure_cineworldsim_hierarchy.LLMJsonTool", FakeTool)
    report = restructure_dataset(source, target, generate_l4=True)
    generated = json.loads((target / "l4_global/movie_bible.json").read_text())
    assert generated["result"]["global_storyline"] == "Generated global storyline."
    assert generated["result"]["generation_method"] == "llm_compaction"
    assert (target / "l4_global/l4_generation_request.json").exists()
    assert (target / "l4_global/l4_generation_response.json").exists()
    assert report["l4_generation"]["method"] == "llm_compaction"
```

- [ ] **Step 4: Run the new tests and verify they fail because the feature is absent**

Run:

```bash
pytest -q tests/test_restructure_cineworldsim_hierarchy.py
```

Expected: collection or test failure because `build_l4_generation_prompt`,
`validate_generated_l4`, and the `generate_l4` argument do not yet exist.

### Task 2: Implement Prompt Projection and Validation

**Files:**
- Modify: `scripts/restructure_cineworldsim_hierarchy.py`
- Test: `tests/test_restructure_cineworldsim_hierarchy.py`

- [ ] **Step 1: Add bounded JSON and compact projection helpers**

Import `hashlib`, `datetime`, `LLMJsonTool`, `compact_l2_event_for_l3`, and
`compact_l3_scene_for_l4`. Implement helpers with these signatures:

Implement these helpers with the following exact interfaces:

```python
def _compact_value(value: Any, limit: int) -> Any
def _compact_movie_bible(bible: dict[str, Any]) -> dict[str, Any]
def build_l4_generation_prompt(
    movie_id: str,
    movie_bible: dict[str, Any],
    l3_scenes: list[dict[str, Any]],
    l2_events: list[dict[str, Any]],
) -> tuple[str, str]
def validate_generated_l4(payload: Any, movie_id: str) -> dict[str, Any]
```

`_compact_value` must bound strings and list/object payloads while retaining
keys ending in `_id` or `_ids`. `_compact_movie_bible` must retain the movie
ID, synopsis, character profiles, recurring locations/objects, constraints,
unresolved questions, and evidence trace. `build_l4_generation_prompt` must
serialize the compact prior, `compact_l3_scene_for_l4` results, and
`compact_l2_event_for_l3` results. `validate_generated_l4` must raise
`ValueError` for a non-object response, a mismatched movie ID, an empty
synopsis, or an empty global storyline, and must return a normalized object.

The prompt must request existing Movie Bible-compatible fields plus a non-empty
`global_storyline`, and it must require source L3/L2/L1 IDs for claims.

- [ ] **Step 2: Run the prompt and validation tests**

Run:

```bash
pytest -q tests/test_restructure_cineworldsim_hierarchy.py::test_l4_prompt_contains_prior_and_compact_scene_event_ids tests/test_restructure_cineworldsim_hierarchy.py::test_validate_generated_l4_requires_global_storyline_and_synopsis
```

Expected: PASS.

### Task 3: Integrate Optional LLM Generation and Audit Artifacts

**Files:**
- Modify: `scripts/restructure_cineworldsim_hierarchy.py`
- Test: `tests/test_restructure_cineworldsim_hierarchy.py`

- [ ] **Step 1: Load L2 and L3 rows without changing source files**

Add a helper that reads `l3_scenes/final.json` and all JSON files below
`l2_events/units`, returning the scene rows and event rows used only for prompt
construction. Keep `_collect_l2_ids()` as the source-reference validator.

- [ ] **Step 2: Add L4 generation parameters to `restructure_dataset()`**

Use this interface:

Use this exact `restructure_dataset()` interface:

```python
def restructure_dataset(
    input_root: Path | str,
    output_root: Path | str,
    allow_unresolved_references: bool = False,
    *,
    generate_l4: bool = False,
    llm_scope: str = "OPENAI",
    llm_model: str | None = None,
    llm_retries: int = 3,
    llm_timeout: float | None = None,
    fallback_existing_bible: bool = False,
) -> dict[str, Any]:
```

The function body must return the existing report structure after writing the
target tree; its only new behavior is selecting the generated or source L4
payload and recording `l4_generation` metadata.

When `generate_l4=False`, preserve the current source-bible path. When true,
build the prompt and call `LLMJsonTool(scope=llm_scope, model=llm_model,
retries=llm_retries)`. Validate the returned JSON before writing it.

- [ ] **Step 3: Write request, response, and report metadata**

Write the following files only in the new output directory:

```text
l4_global/l4_generation_request.json
l4_global/l4_generation_response.json
```

Record model/scope, prompt SHA-256, source paths, counts, status, and parsed
response. Never serialize API keys. Add `l4_generation` to the report with
`method`, `status`, and `fallback_used`.

- [ ] **Step 4: Implement explicit fallback behavior**

If generation raises an exception or validation fails, re-raise by default.
If `fallback_existing_bible=True`, use the source Movie Bible, set
`method="source_fallback"`, set `fallback_used=true`, and write the failure
details to the response audit file.

- [ ] **Step 5: Add CLI flags and wire them to `main()`**

Add:

```python
parser.add_argument("--generate-l4", action="store_true")
parser.add_argument("--llm-scope", default="OPENAI")
parser.add_argument("--llm-model", default=None)
parser.add_argument("--llm-retries", type=int, default=3)
parser.add_argument("--llm-timeout", type=float, default=None)
parser.add_argument("--fallback-existing-bible", action="store_true")
```

Pass all values into `restructure_dataset()`.

- [ ] **Step 6: Run the integration test**

Run:

```bash
pytest -q tests/test_restructure_cineworldsim_hierarchy.py::test_restructure_generate_l4_writes_llm_bible_and_audits
```

Expected: PASS.

### Task 4: Regression Verification and Documentation

**Files:**
- Modify: `docs/CineWorldSim_Hierarchy_Redefinition.md`
- Modify: `tests/test_restructure_cineworldsim_hierarchy.py`

- [ ] **Step 1: Verify default mode does not construct an LLM client**

Run the existing default-mode test suite with a monkeypatch that raises if
`LLMJsonTool` is constructed. The default restructuring test must still pass.

- [ ] **Step 2: Run all hierarchy tests**

```bash
pytest -q tests/test_restructure_cineworldsim_hierarchy.py
```

Expected: all hierarchy tests pass.

- [ ] **Step 3: Run related LLM builder tests**

```bash
pytest -q tests/test_moviebench_llm_builder.py tests/test_llm_tool.py
```

Expected: all related tests pass.

- [ ] **Step 4: Update the hierarchy documentation**

Document `--generate-l4`, the generated audit files, the source-fallback flag,
and the fact that normal invocation remains offline.

- [ ] **Step 5: Run a syntax check and inspect the final diff**

```bash
python -m py_compile scripts/restructure_cineworldsim_hierarchy.py
git diff --check -- scripts/restructure_cineworldsim_hierarchy.py tests/test_restructure_cineworldsim_hierarchy.py docs/CineWorldSim_Hierarchy_Redefinition.md
```

The current workspace has an empty `.git` directory, so Git commands may be
unavailable; if so, report that fact rather than treating it as a code failure.
