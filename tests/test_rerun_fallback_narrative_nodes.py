from __future__ import annotations

from pathlib import Path

import pytest

from scripts.rerun_fallback_narrative_nodes import (
    build_default_output_dir,
    parse_node_spec,
    select_rerun_range,
)


def test_parse_node_spec_normalizes_actor_and_source() -> None:
    spec = parse_node_spec("13:Ilya_Trent:action")

    assert spec.step == 13
    assert spec.actor == "Ilya Trent"
    assert spec.source == "action"


def test_parse_node_spec_rejects_invalid_source() -> None:
    with pytest.raises(ValueError, match="source"):
        parse_node_spec("13:Ilya_Trent:unknown")


def test_select_rerun_range_only_replays_target_step_by_default() -> None:
    assert select_rerun_range(target_step=13, max_step=15, rerun_downstream=False) == (13, 13)


def test_select_rerun_range_replays_to_end_when_requested() -> None:
    assert select_rerun_range(target_step=13, max_step=15, rerun_downstream=True) == (13, 15)


def test_default_output_dir_does_not_overwrite_source() -> None:
    source = Path("/tmp/example/all_banana")

    output = build_default_output_dir(source)

    assert output != source
    assert output == Path("/tmp/example_fallback_rerun/all_banana")
