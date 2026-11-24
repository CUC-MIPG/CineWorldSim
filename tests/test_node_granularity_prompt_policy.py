from __future__ import annotations


def test_node_selection_policy_explicitly_blocks_same_step_role_merging() -> None:
    from narrative.visual_prompt_constraints import build_node_selection_prompt_policy

    policy = build_node_selection_prompt_policy().lower()
    assert "one role action per step" in policy
    assert "one video" in policy
    assert "same step" in policy
    assert "same location" in policy
    assert "direct interaction" in policy
    assert "step::1::charactera" in policy


def test_path_policy_preserves_single_role_nodes_as_atomic_videos() -> None:
    from narrative.visual_prompt_constraints import build_path_orchestration_prompt_policy

    policy = build_path_orchestration_prompt_policy().lower()
    assert "atomic narrative node" in policy
    assert "do not split or merge" in policy
    assert "single_role_step" in policy
    assert "one video task" in policy
