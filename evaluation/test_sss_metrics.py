import numpy as np
import pytest

from sss_metrics import (
    build_event_coverage_prompt,
    build_grounding_prompt,
    build_reference_events_prompt,
    build_structured_alignment_prompt,
    calculate_caption_grounding,
    calculate_event_coverage,
    calculate_structured_alignment,
    cosine_similarity,
    get_embedding,
    validate_embedding_payload,
    validate_reference_events,
    validate_event_matches,
    validate_grounding_payload,
)


def test_embedding_cosine_is_raw_cosine_and_rejects_bad_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1, 0], [1])
    with pytest.raises(ValueError, match="zero norm"):
        cosine_similarity([0, 0], [1, 0])


def test_embedding_response_validation_and_openai_compatible_transport(monkeypatch):
    assert validate_embedding_payload({"data": [{"embedding": [0.1, 0.2]}]}) == [0.1, 0.2]
    with pytest.raises(ValueError, match="embedding"):
        validate_embedding_payload({"data": []})

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [1.0, 0.0]}]}

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return Response()

    monkeypatch.setattr("sss_metrics.requests.post", fake_post)
    assert get_embedding("story", "http://localhost/v1", "secret", "text-embedding-3-small") == [1.0, 0.0]
    assert captured["url"] == "http://localhost/v1/embeddings"
    assert captured["json"]["model"] == "text-embedding-3-small"


def test_structured_alignment_scores_separate_five_dimensions():
    result = calculate_structured_alignment({
        "scores": {"characters": 5, "conflict": 4, "key_events": 3, "relationships_goals": 4, "ending": 2},
        "rationale": {"ending": "The ending is not visible."},
    })
    assert result["score"] == pytest.approx(0.72)
    assert result["scores"]["ending"] == 2


def test_event_coverage_uses_assessable_reference_events_only():
    result = calculate_event_coverage(
        ["e1", "e2", "e3"],
        [{"event_id": "e1", "status": "recovered"}, {"event_id": "e2", "status": "not_assessable"}, {"event_id": "e3", "status": "missed"}],
    )
    assert result["coverage"] == pytest.approx(0.5)
    assert result["assessable_event_count"] == 2
    assert result["recovered_count"] == 1
    with pytest.raises(ValueError, match="exactly one"):
        calculate_event_coverage(["e1"], [])


def test_caption_grounding_reports_unsupported_claim_rate():
    result = calculate_caption_grounding([
        {"claim_id": "c1", "grounded": True},
        {"claim_id": "c2", "grounded": False},
        {"claim_id": "c3", "grounded": None},
    ])
    assert result["grounded_count"] == 1
    assert result["unsupported_count"] == 1
    assert result["hallucination_rate"] == pytest.approx(0.5)
    assert result["evaluable_claim_count"] == 2


def test_diagnostic_prompts_keep_grounding_blind_to_reference_story():
    ref_prompt = build_structured_alignment_prompt("REFERENCE SECRET", "RECON STORY")
    coverage_prompt = build_event_coverage_prompt([{"event_id": "e1", "text": "REFERENCE EVENT"}], [])
    grounding_prompt = build_grounding_prompt([{"shot_id": "s1", "caption": "VISIBLE CAPTION"}])
    assert "REFERENCE SECRET" in ref_prompt
    assert "REFERENCE EVENT" in coverage_prompt
    assert "VISIBLE CAPTION" in grounding_prompt
    assert "REFERENCE" not in grounding_prompt


def test_reference_event_extraction_and_match_schemas_are_validated():
    prompt = build_reference_events_prompt("A woman enters and leaves.")
    assert "A woman enters and leaves." in prompt
    events = validate_reference_events({"events": [
        {"event_id": "e1", "text": "A woman enters."},
        {"event_id": "e2", "text": "A woman leaves."},
    ]})
    assert len(events) == 2
    assert validate_event_matches({"matches": [
        {"event_id": "e1", "status": "recovered"},
        {"event_id": "e2", "status": "missed"},
    ]}, ["e1", "e2"])[1]["status"] == "missed"


def test_grounding_payload_has_one_boolean_or_unknown_judgment_per_claim():
    payload = validate_grounding_payload({"claims": [
        {"claim_id": "c1", "grounded": True, "text": "The person opens a door."},
        {"claim_id": "c2", "grounded": None, "text": "The person feels worried."},
    ]})
    assert payload[1]["grounded"] is None
    with pytest.raises(ValueError, match="grounded"):
        validate_grounding_payload({"claims": [{"claim_id": "c1", "grounded": "yes", "text": "claim"}]})
