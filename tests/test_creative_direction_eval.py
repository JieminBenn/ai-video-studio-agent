import json
from pathlib import Path

import yaml

from studio_agent.creative_direction_eval import run_eval
from studio_agent.providers.fake import FakeLLM
from studio_agent.stages.base import Providers


SCENARIOS = Path(__file__).parents[1] / "evals" / "creative_direction" / "scenarios.yaml"


def test_eval_manifest_has_eight_bilingual_scenarios():
    scenarios = yaml.safe_load(SCENARIOS.read_text())["scenarios"]

    assert len(scenarios) == 8
    assert {item["language"] for item in scenarios} == {"en", "zh"}
    assert {item["focus"] for item in scenarios} >= {
        "intimate_dialogue",
        "power_reversal",
        "suspense",
        "kinetic_action",
        "anime",
        "stylized_3d",
        "two_character_identity",
        "location_lighting_continuity",
    }


def test_eval_writes_deterministic_report_and_unscored_ab_sheet(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = run_eval(
        scenarios_path=SCENARIOS,
        out_dir=first_dir,
        providers=Providers(llm=FakeLLM()),
    )
    second = run_eval(
        scenarios_path=SCENARIOS,
        out_dir=second_dir,
        providers=Providers(llm=FakeLLM()),
    )

    assert first == second
    assert first["summary"] == {"scenarios": 8, "deterministic_passes": 8, "failures": 0}
    assert all(item["passed"] for item in first["results"])
    assert all(item["retrieval_ids"] for item in first["results"])
    assert "human_preference_wins" not in first["summary"]
    assert json.loads(first_dir.joinpath("eval-report.json").read_text()) == first

    worksheet = first_dir.joinpath("human-ab-review.md").read_text()
    assert "camera motivation (1–5)" in worksheet
    assert "identity specificity (1–5)" in worksheet
    assert "prompt usefulness (1–5)" in worksheet
    assert "Human result: _unscored_" in worksheet
