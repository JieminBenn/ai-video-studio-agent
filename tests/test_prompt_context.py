"""Tests for compact story context used by prompt compilers."""

import json

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_context import project_prompt_context


def test_project_prompt_context_extracts_story_scene_and_neighbors(tmp_path):
    p = Project.create(
        "一个灯塔守望者遇见会说话的海鸥",
        root=tmp_path,
        stages=["storyboard", "video"],
        model_config={
            "language": "zh",
            "product_format": {"name": "short_film", "label": "Short film"},
        },
    )
    p.path("story", "idea.md").write_text("一个灯塔守望者遇见会说话的海鸥")
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "一部关于孤独与勇气的短片。",
        "synopsis": "守望者必须决定是否相信海鸥的警告。",
        "themes": ["孤独", "勇气"],
    }))
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{
            "episode": 1,
            "scenes": [{
                "scene": 1,
                "heading": "EXT. LIGHTHOUSE - DAWN",
                "beats": ["海风变强", "海鸥发出警告"],
                "dialogue": [{"character": "海鸥", "line": "暴风雨来了。"}],
            }],
        }]
    }))
    shots = [
        {"id": "sh-001", "scene": 1, "camera": "wide", "action": "守望者推开门"},
        {"id": "sh-002", "scene": 1, "camera": "close-up", "action": "海鸥发出警告"},
        {"id": "sh-003", "scene": 1, "camera": "wide", "action": "灯塔灯光旋转"},
    ]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))

    ctx = project_prompt_context(p, shots[1], shots=shots)

    assert ctx["language"] == "zh"
    assert "会说话的海鸥" in ctx["idea"]
    assert "孤独与勇气" in ctx["story"]["logline"]
    assert ctx["scene"]["heading"] == "EXT. LIGHTHOUSE - DAWN"
    assert "暴风雨来了" in ctx["scene"]["dialogue"][0]["line"]
    assert ctx["previous_shot"]["id"] == "sh-001"
    assert ctx["next_shot"]["id"] == "sh-003"


def test_project_prompt_context_contains_compact_creative_brief(tmp_path):
    p = Project.create("idea", root=tmp_path, stages=["storyboard"])
    p.path("story", "creative_brief.json").write_text(json.dumps({
        "version": 2,
        "visual_direction": {
            "tone": {"value": "tender dread"},
            "camera_language": {"value": "patient and intimate"},
        },
        "hard_avoidances": ["flat lighting"],
    }))

    ctx = project_prompt_context(p, {"id": "sh-001", "scene": 1}, shots=[])

    assert ctx["creative_brief"] == {
        "tone": "tender dread",
        "camera_language": "patient and intimate",
        "hard_avoidances": ["flat lighting"],
    }
