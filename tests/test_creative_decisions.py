import json

import pytest

from studio_agent.creative_decisions import normalize_decision, resolve_decision
from studio_agent.orchestrator.project import Project


def test_normalize_decision_keeps_two_choices_and_matching_default():
    decision = normalize_decision(
        {
            "question": "Should the scene feel exposed or intimate?",
            "why_it_matters": "This changes camera distance and blocking.",
            "choices": [
                {"value": "exposed", "label": "Exposed", "description": "More spatial distance"},
                {"value": "intimate", "label": "Intimate", "description": "Closer observation"},
            ],
            "default": "intimate",
            "evidence": ["The idea emphasizes a private confession."],
        },
        stage="storyboard",
        gap={"dimension": "camera_language", "default": "intimate"},
    )

    assert decision["stage"] == "storyboard"
    assert decision["dimension"] == "camera_language"
    assert [choice["value"] for choice in decision["choices"]] == ["exposed", "intimate"]
    assert decision["default"] == "intimate"
    assert decision["resolution"] is None


def test_malformed_decision_uses_deterministic_fallback():
    decision = normalize_decision(
        {"choices": [], "default": "unknown"},
        stage="bible",
        gap={
            "dimension": "character_treatment",
            "default": "restrained and lived-in",
            "why": "Identity direction affects every later image.",
        },
    )

    assert len(decision["choices"]) == 2
    assert decision["default"] == "restrained and lived-in"
    assert decision["question"]


def test_resolve_decision_rejects_unknown_choice(tmp_path):
    project = Project.create("idea", root=tmp_path, stages=["bible"])
    path = project.path("story", "decisions", "bible.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "choices": [{"value": "restrained"}, {"value": "expressive"}],
        "default": "restrained",
        "resolution": None,
    }))

    with pytest.raises(ValueError, match="unknown decision choice"):
        resolve_decision(project, stage="bible", choice="random")

    resolved = resolve_decision(project, stage="bible", choice="expressive")
    assert resolved["resolution"] == {
        "value": "expressive",
        "source": "user",
        "locked": True,
    }
