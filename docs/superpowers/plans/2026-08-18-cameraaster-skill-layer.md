# CineWorldSim Skill Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared skill registry, agent allowlists, and skill-aware prompt adapters so all CineWorldSim agents use one skill system with explicit permission boundaries.

**Architecture:** Introduce a small `agents.skills` package that loads skill manifests from local docs and the MiniMax-H3 repo, validates per-agent allowlists, and resolves a skill into either a prompt string or a structured payload. Keep each agent's external JSON contract unchanged and preserve the current prompt-library and H3 fallback paths as compatibility layers.

**Tech Stack:** Python 3.12, `pathlib`, `yaml`/front-matter parsing, `pytest`, existing CineWorldSim agent classes, existing MiniMax-H3 prompt adapter.

---

### Task 1: Add shared skill manifest loading and registry

**Files:**
- Create: `agents/skills/__init__.py`
- Create: `agents/skills/types.py`
- Create: `agents/skills/loader.py`
- Create: `agents/skills/registry.py`
- Create: `tests/test_skill_registry.py`

- [ ] **Step 1: Write the failing test**

```python
def test_registry_loads_local_and_minimax_skills(tmp_path, monkeypatch):
    # Arrange a local docs/skills tree and a MiniMax-H3 skills tree with SKILL.md front matter.
    # Assert that registry.list() returns both skill names and preserves source_root/skill_path.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_registry.py -v`

Expected: fail because `agents.skills.registry` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add:

```python
@dataclass(frozen=True)
class SkillManifest:
    name: str
    description: str
    kind: str
    source_root: str
    skill_path: str
    compatible_agents: tuple[str, ...] = ()
    prompt_template_paths: tuple[str, ...] = ()
    fallback_policy: str = "fallback"

class SkillRegistry:
    def __init__(self, roots: list[Path]): ...
    def load(self) -> None: ...
    def get(self, name: str) -> SkillManifest | None: ...
    def list(self) -> list[SkillManifest]: ...
```

Implement a loader that can read:

- YAML front matter in `SKILL.md`
- `meta.yaml` when present

Normalize skill names from the `name:` field and store the originating file path.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill_registry.py -v`

Expected: pass.

- [ ] **Step 5: Add a smoke check for the existing local H3 skill**

```python
def test_local_h3_prompt_writing_is_discoverable():
    reg = SkillRegistry([Path("docs/skills"), Path("/data1/AgentG/hhb/minimax_h3/minimaxh3/MiniMax-H3/skills")])
    reg.load()
    assert reg.get("h3-prompt-writing") is not None
```

---

### Task 2: Add per-agent allowlists and skill resolution

**Files:**
- Create: `agents/skills/policy.py`
- Create: `agents/skills/resolver.py`
- Create: `configs/agent_skill_policies.yaml`
- Create: `tests/test_agent_skill_policy.py`

- [ ] **Step 1: Write the failing test**

```python
def test_policy_blocks_disallowed_skill():
    policy = AgentSkillPolicy.from_yaml(Path("configs/agent_skill_policies.yaml"))
    assert policy.allows("VideoAgent", "h3-prompt-writing")
    assert not policy.allows("ShotAgent", "h3-prompt-writing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_skill_policy.py -v`

Expected: fail because `agents.skills.policy` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add:

```python
class AgentSkillPolicy:
    def __init__(self, allowlists: dict[str, set[str]]): ...
    @classmethod
    def from_yaml(cls, path: Path) -> "AgentSkillPolicy": ...
    def allows(self, agent_name: str, skill_name: str) -> bool: ...
    def default_skill(self, agent_name: str) -> str | None: ...

class SkillResolver:
    def __init__(self, registry: SkillRegistry, policy: AgentSkillPolicy): ...
    def resolve(self, agent_name: str, requested_skill: str | None, context: dict[str, Any]) -> SkillManifest: ...
```

Use explicit allowlists and fail closed. If a requested skill is not allowed, fall back to the agent default only when the default is allowed.

Proposed initial allowlist shape:

```yaml
DirectorAgent:
  default: cameraatist-story-design
  allowed:
    - cameraatist-story-design
    - cameraatist-scene-creation
    - cameraatist-world-setup
ShotAgent:
  default: cameraatist-shot-creation
  allowed:
    - cameraatist-shot-creation
StorySupervisorAgent:
  default: cameraatist-shot-planning
  allowed:
    - cameraatist-shot-planning
    - cameraatist-post-production-planning
VideoAgent:
  default: h3-prompt-writing
  allowed:
    - h3-prompt-writing
    - cameraatist-event-video-prompt
    - cameraatist-event-multishot-prompt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_skill_policy.py -v`

Expected: pass.

- [ ] **Step 5: Add a resolver test for fallback behavior**

```python
def test_resolver_falls_back_to_agent_default_when_requested_skill_missing():
    # Requested skill is absent from registry.
    # Resolver should return the allowed default skill for that agent.
```

---

### Task 3: Move VideoAgent prompt creation onto the skill layer first

**Files:**
- Modify: `agents/video_agent.py`
- Modify: `agents/h3_prompt_adapter.py`
- Create: `agents/skills/adapter.py`
- Create: `tests/test_skill_adapter.py`
- Update: `tests/test_orchestrated_event_video_generation.py`

- [ ] **Step 1: Write the failing test**

```python
def test_video_agent_records_skill_metadata_for_official_ref(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_H3_PROMPT_STYLE", "official_ref")
    # Run a dry-run event build and read event_video_tasks.json.
    # Assert the generated task row records skill_name == "h3-prompt-writing".
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrated_event_video_generation.py -v`

Expected: existing prompt path has no skill metadata yet.

- [ ] **Step 3: Write minimal implementation**

Add a `SkillAdapter` that wraps the current H3 prompt builder:

```python
class SkillAdapter:
    def build(self, agent_name: str, skill: SkillManifest, context: dict[str, Any]) -> dict[str, Any]: ...
```

For `VideoAgent`:

- `MINIMAX_H3_PROMPT_STYLE=official_ref` maps to `h3-prompt-writing`
- `compact` maps to a built-in event prompt skill
- multi-shot prompt assembly stays in `VideoAgent`, but each shot prompt is emitted through the same adapter

Return both:

- the prompt string
- skill metadata attached to the task JSON

Keep the current `generate_video_minimax_h3` and `generate_video_minimax_h3_vllm` calls unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrated_event_video_generation.py -v`

Expected: pass, with `task["skill_name"] == "h3-prompt-writing"` for official ref mode.

- [ ] **Step 5: Preserve existing fallback behavior**

If the registry cannot load a skill or the selected skill is disallowed, `VideoAgent` must still:

- fall back to the current compact prompt builder
- or fall back to `build_h3_ref_prompt`
- without changing the video backend call signature

---

### Task 4: Move DirectorAgent, ShotAgent, and StorySupervisorAgent onto the same skill path

**Files:**
- Modify: `agents/director_agent.py`
- Modify: `agents/shot_agent.py`
- Modify: `agents/story_supervisor_agent.py`
- Modify: `agents/prompts.py`
- Create: `tests/test_agent_skill_integration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_director_shot_supervisor_can_resolve_allowed_defaults(monkeypatch):
    # Build each agent with a shared registry and policy.
    # Assert each agent resolves its own default skill and emits the same payload shape as before.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_skill_integration.py -v`

Expected: fail until each agent accepts a resolver or skill context.

- [ ] **Step 3: Write minimal implementation**

Inject the resolver into each agent with a default constructed from env/config.

For each agent:

- keep the current public methods and return values
- move prompt string generation into a skill-aware helper
- keep the current prompt-library and JSON repair logic as fallback

Suggested mappings:

- `DirectorAgent.fast_story_cot()` -> `cameraatist-scene-creation`
- `DirectorAgent.story_design_structured()` -> `cameraatist-story-design`
- `ShotAgent.slow_shot_cot()` -> `cameraatist-shot-creation`
- `StorySupervisorAgent._generate_scene_plan_online()` -> `cameraatist-shot-planning`

If a requested skill is unavailable, the current hardcoded prompt builders remain the fallback path.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent_skill_integration.py -v`

Expected: pass.

- [ ] **Step 5: Keep prompt-library compatibility**

`agents/prompts.py` should still load `prompt_query_template.py`, but registry-backed skills may optionally point to prompt templates in a manifest. Existing prompt libraries must remain usable during migration.

---

### Task 5: Add trace metadata, CLI wiring, and regression coverage

**Files:**
- Modify: `scripts/run_orchestrated_event_video_generation.py`
- Modify: `scripts/test_minimax_h3_video_tool.py`
- Modify: `agents/video_agent.py`
- Modify: `tests/test_four_genre_demo_pipeline.py`
- Create: `tests/test_skill_trace_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
def test_task_json_records_skill_metadata(tmp_path):
    # Run the VideoAgent in dry-run mode.
    # Assert event_video_tasks.json records skill_name, skill_source, and skill_kind.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill_trace_metadata.py -v`

Expected: fail because task JSON has no skill metadata yet.

- [ ] **Step 3: Write minimal implementation**

Add skill trace fields to generated task rows:

```python
task["skill_name"] = skill.name
task["skill_source"] = skill.source_root
task["skill_kind"] = skill.kind
```

Add optional env wiring for skill overrides:

- `CAMERAMASTER_SKILL_ROOTS`
- `CAMERAMASTER_SKILL_POLICY_PATH`
- `DIRECTOR_SKILL_NAME`
- `SHOT_SKILL_NAME`
- `SUPERVISOR_SKILL_NAME`
- `VIDEO_SKILL_NAME`

Do not change the current backend CLI unless the skill override cannot be expressed through env vars.

- [ ] **Step 4: Run regression tests**

Run:

```bash
pytest tests/test_skill_trace_metadata.py -v
pytest tests/test_minimax_h3_video_provider.py -v
pytest tests/test_orchestrated_event_video_generation.py -v
pytest tests/test_four_genre_demo_pipeline.py -v
```

Expected: pass, with no change to existing video backend behavior.

- [ ] **Step 5: Validate full pipeline compatibility**

Run the existing event-video pipeline in dry-run or a small subset mode and confirm:

- task JSON still writes successfully
- prompt text still reaches MiniMax-H3
- disallowed skills never bypass the allowlist
- all existing demo paths still work

---

## Self-Review Checklist

- The registry task covers local CineWorldSim skills and MiniMax-H3 skills.
- The policy task covers agent-level permission boundaries.
- The adapter task keeps `VideoAgent` backward compatible.
- The agent integration task covers `DirectorAgent`, `ShotAgent`, and `StorySupervisorAgent`.
- The trace-metadata task covers observability and CLI/env wiring.
- No task depends on a function or file that is not introduced earlier in the plan.
