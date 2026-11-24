# CineWorldSim Skill Layer Design

## Goal

Build a shared skill layer for CineWorldSim so that `DirectorAgent`, `ShotAgent`, `StorySupervisorAgent`, and `VideoAgent` all use the same skill discovery and prompt-construction system, while each agent is restricted to an explicit allowlist of permitted skills.

The new layer must preserve current pipeline behavior by default, add a clean place to register MiniMax-H3 and CineWorldSim-local skills, and keep workflow-style skills as reusable planning/specification assets rather than trying to execute MiniMax Hub canvas logic inside CineWorldSim.

## Scope

### In Scope

- One shared skill registry for local and imported skill manifests
- Per-agent allowlists that block disallowed skill usage
- A lightweight skill adapter interface for prompt building and validation
- Backward-compatible fallbacks to current prompt builders and existing agent logic
- Skill metadata surfaced in generated tasks and logs for traceability
- Test coverage for registry loading, allowlist enforcement, adapter selection, and fallback behavior

### Out of Scope

- Reproducing MiniMax Hub canvas runtime inside CineWorldSim
- Interactive choice-card workflow execution
- Remote skill marketplace syncing
- Dynamic Python code execution from skill files
- Rewriting the existing narrative pipeline into a new agent framework

## Current State

CineWorldSim already has partial prompt specialization:

- `agents/h3_prompt_adapter.py` builds a MiniMax-H3 full-reference prompt for event video generation
- `agents/director_agent.py` uses a prompt library for scene creation and story design
- `agents/shot_agent.py` uses a prompt library for shot creation
- `agents/story_supervisor_agent.py` builds duration and post-production plans from shot JSON
- `agents/video_agent.py` chooses between compact event prompts and the H3 full-reference prompt when `MINIMAX_H3_PROMPT_STYLE=official_ref`

The missing piece is a shared skill abstraction. Today, each agent knows only its own prompt shape. There is no common registry, no formal per-agent allowlist, and no place to record which skill produced a given prompt.

## Design Principles

1. Skill selection must be explicit, not implicit.
2. Agent permissions must be enforced centrally.
3. Skill outputs must not change the existing JSON contracts unless the agent already owns that contract.
4. Portable skills should work in CineWorldSim even when they were authored for a different runtime, as long as their logic can be mapped to local prompt generation.
5. Workflow-style skills should be treated as planning blueprints and prompt policies, not as direct canvas runtimes.

## Architecture

### 1. Skill Registry

Create one registry that discovers skills from configured roots and normalizes them into a common metadata object.

The registry should support at least:

- local CineWorldSim skill docs under `docs/skills`
- bundled MiniMax-H3 skills under `/data1/AgentG/hhb/minimax_h3/minimaxh3/MiniMax-H3/skills`

The registry should normalize these fields:

- `name`
- `description`
- `source_root`
- `skill_path`
- `kind` (`prompt`, `workflow`, `hybrid`)
- `compatible_agents`
- `input_contract`
- `output_contract`
- `prompt_template_paths`
- `fallback_policy`

### 2. Agent Skill Policy

Create one policy file that defines which agents may use which skills.

The policy must be enforced before any skill is executed. If a skill is not allowed for the current agent, the agent must fail closed and fall back to its existing prompt builder or raise a clear error if no fallback exists.

Default policy intent:

- `DirectorAgent`: story setup, scene creation, story design, world-model oriented skills
- `ShotAgent`: shot creation and shot normalization skills
- `StorySupervisorAgent`: duration planning, narration planning, post-production planning
- `VideoAgent`: H3 prompt-writing, event-video prompt, render-oriented skills

### 3. Skill Adapter

Create a small adapter interface so each agent can ask for a skill-generated payload without knowing the internal format of that skill.

Required operations:

- `build_prompt(context)`
- `validate(context)`
- `fallback(context)`

The adapter should be able to return either:

- a raw prompt string
- a structured prompt payload
- a prompt plus metadata bundle

### 4. Agent Runtime Integration

Each agent keeps its existing output contract, but the prompt-building step becomes skill-aware:

- `DirectorAgent` selects a skill for scene creation or story design, then emits the same storyboard / scene JSON as today
- `ShotAgent` selects a shot-creation skill, then emits the same shot JSON as today
- `StorySupervisorAgent` selects a planning skill, then emits the same scene plan JSON as today
- `VideoAgent` selects a video-prompt skill, then emits the same task JSON and video backend calls as today

Every generated task should record:

- `skill_name`
- `skill_source`
- `skill_kind`
- `skill_version` or manifest revision when available

## File Structure

### New Files

- `agents/skills/__init__.py`
- `agents/skills/registry.py`
- `agents/skills/policy.py`
- `agents/skills/adapter.py`
- `agents/skills/context.py`
- `configs/agent_skill_policies.yaml`
- `tests/test_skill_registry.py`
- `tests/test_agent_skill_policy.py`
- `tests/test_skill_adapter.py`

### Existing Files to Modify

- `agents/director_agent.py`
- `agents/shot_agent.py`
- `agents/story_supervisor_agent.py`
- `agents/video_agent.py`
- `agents/h3_prompt_adapter.py`
- `agents/prompts.py` if registry-backed prompt templates need to coexist with the current prompt library
- `scripts/run_orchestrated_event_video_generation.py` only if CLI flags or env wiring are added for skill selection
- `scripts/test_minimax_h3_video_tool.py` if a skill-aware prompt selection flag is exposed there

## Runtime Flow

1. Load skill manifests from configured roots.
2. Load `agent_skill_policies.yaml`.
3. Resolve the requested skill for the current agent.
4. Check the allowlist.
5. Build the prompt or structured payload through the skill adapter.
6. Fall back to the current hardcoded builder if the skill is unavailable or disabled.
7. Attach skill metadata to logs and generated task JSON.

## Compatibility Strategy

The first release should be additive.

- Existing env vars like `MINIMAX_H3_PROMPT_STYLE` remain valid.
- Existing prompt-library logic remains the fallback path.
- Existing video backend selection remains unchanged.
- The registry can initially wrap existing prompt builders as built-in skills before more skill files are added.

## Initial Skill Coverage

The first implementation should cover these categories:

- `DirectorAgent`
  - story setup
  - scene creation
  - story design
- `ShotAgent`
  - shot creation
- `StorySupervisorAgent`
  - shot duration planning
  - scene plan generation
- `VideoAgent`
  - compact event prompt
  - MiniMax-H3 ref2va prompt writing
  - multi-shot event prompt planning

MiniMax-H3 workflow-style skills should be imported as prompt/plan blueprints only. CineWorldSim does not need to execute Hub canvas steps.

## Testing Plan

Tests must verify:

- registry loads skills from both roots
- skill names are normalized consistently
- allowlist blocks disallowed skills
- each agent resolves its permitted default skill
- fallback is used when a skill is missing or disabled
- `VideoAgent` still produces the same task shape when the skill layer is active
- `MINIMAX_H3_PROMPT_STYLE=official_ref` still yields the full-reference H3 prompt

## Migration Plan

1. Add registry and policy types without changing agent behavior.
2. Wrap current prompt builders as built-in skills.
3. Move `VideoAgent` prompt selection onto the skill adapter.
4. Move `DirectorAgent`, `ShotAgent`, and `StorySupervisorAgent` onto the same path.
5. Add trace metadata to generated prompts and task JSON.
6. Expand supported skills gradually, starting with the ones already reflected in current behavior.

## Risks

- Overfitting the skill layer to one backend would make it brittle.
- Loading workflow-style MiniMax-H3 skills too literally could create a fake Hub runtime.
- A large allowlist surface without tests would make accidental skill drift likely.
- Prompt changes could alter video behavior, so the fallback path must remain stable.
