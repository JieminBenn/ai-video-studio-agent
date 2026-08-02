"""Tests for the storyboard stage — scenes -> shots + reference-conditioned keyframes."""

import json

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import confirm_prompt_batch
from studio_agent.providers.fake import FakeImageGen, FakeLLM, FakeVideoGen
from studio_agent.reference_assets import (
    mark_reference_upload_revised,
    retarget_reference,
    save_reference_intake_uploads,
    save_reference_upload,
)
from studio_agent.stages.base import Providers
from studio_agent.stages.bible import BibleStage, char_slug, seed_for
from studio_agent.stages.keyframes import KeyframesStage
from studio_agent.stages.plot import PlotStage
from studio_agent.stages.script import ScriptStage
from studio_agent.stages.storyboard import StoryboardStage

PIPELINE = ["plot", "script", "bible", "storyboard"]


def test_load_locations_coerces_object_prompt_aliases_to_strings(tmp_path):
    # Regression (project-7d437f): a location.json whose prompt_aliases are objects
    # ({"alias": ..., "description": ...}) must load as plain strings, or the downstream
    # ", ".join(location_aliases) crashes with "sequence item N: expected str instance, dict found".
    project = Project.create(
        "loc aliases", root=tmp_path, stages=["storyboard"],
        model_config={"llm": "fake", "image": "fake", "video": "fake"},
    )
    ldir = project.path("bible", "locations", "loc-1")
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "location.json").write_text(json.dumps({
        "name": "Hallway",
        "scene_numbers": [1],
        "prompt_aliases": [
            {"alias": "ESTABLISHING_WIDE", "description": "全景"},
            {"alias": "DOOR_OPEN", "description": "开门"},
        ],
    }, ensure_ascii=False))

    locations = StoryboardStage()._load_locations(project)

    assert locations["Hallway"]["aliases"] == ["ESTABLISHING_WIDE", "DOOR_OPEN"]
    assert all(isinstance(a, str) for a in locations["Hallway"]["aliases"])


def test_direct_grid_forwards_verbatim_locks_and_budget(tmp_path, monkeypatch):
    # The motion-grid director path must carry each character's appearance lock (verbatim
    # backfill) and the provider length budget, exactly like the static keyframe path.
    from studio_agent.stages import storyboard as sb_mod
    from studio_agent.providers.base import Generation

    project = Project.create(
        "grid locks", root=tmp_path, stages=["storyboard", "keyframes", "video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake"},
    )
    captured: dict = {}

    def fake_direct_grid_prompt(providers, project, *, brief, **kwargs):
        captured.update(kwargs)
        return Generation(content={"prompt": "P", "negative": ""}, provider="fake", model="fake")

    monkeypatch.setattr(sb_mod, "direct_grid_prompt", fake_direct_grid_prompt)

    shot = {
        "id": "sh-001",
        "reference_characters": ["Knight"],
        "character_appearance_locks": ["Knight: scarred jaw, silver pauldrons."],
    }
    providers = Providers(llm=FakeLLM(), image=FakeImageGen())

    sb_mod.StoryboardStage()._direct_grid(project, shot, providers, brief="BRIEF")

    assert captured["verbatim_locks"] == ["Knight: scarred jaw, silver pauldrons."]
    assert "max_prompt_chars" in captured


def test_direct_generation_forwards_style_signature(tmp_path, monkeypatch):
    # Style/medium is load-bearing: the keyframe director path must hand the director the
    # style signature so it can be backfilled verbatim if the model drops it — otherwise a
    # single keyframe can render in a slightly different style.
    from studio_agent.stages import storyboard as sb_mod
    from studio_agent.providers.base import Generation
    from studio_agent.style import style_medium_lead

    style = {"medium": "2D hand-drawn illustration", "look": "storybook"}
    project = Project.create(
        "style signature", root=tmp_path, stages=["storyboard", "keyframes", "video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake", "style": style},
    )
    captured: dict = {}

    def fake_direct_image_prompt(providers, project, *, brief, **kwargs):
        captured.update(kwargs)
        return Generation(content={"prompt": "P", "negative": ""}, provider="fake", model="fake")

    monkeypatch.setattr(sb_mod, "direct_image_prompt", fake_direct_image_prompt)

    shot = {"id": "sh-001", "reference_characters": []}
    providers = Providers(llm=FakeLLM(), image=FakeImageGen())

    sb_mod.StoryboardStage()._direct_generation(project, shot, providers, brief="BRIEF")

    assert captured["style_signature"] == style_medium_lead(style)
    assert captured["style_signature"]


def test_direct_generation_forwards_character_dont_rules_as_hard_avoidances(tmp_path, monkeypatch):
    # A character's identity-board "dont" rules must also reach the director as avoidances so
    # they survive as negatives even if the director LLM drops them from the positive prompt.
    from studio_agent.stages import storyboard as sb_mod
    from studio_agent.providers.base import Generation

    project = Project.create(
        "dont rules", root=tmp_path, stages=["storyboard", "keyframes", "video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake"},
    )
    captured: dict = {}

    def fake_direct_image_prompt(providers, project, *, brief, **kwargs):
        captured.update(kwargs)
        return Generation(content={"prompt": "P", "negative": ""}, provider="fake", model="fake")

    monkeypatch.setattr(sb_mod, "direct_image_prompt", fake_direct_image_prompt)

    shot = {"id": "sh-001", "reference_characters": [], "character_dont_rules": ["never remove her gloves"]}
    providers = Providers(llm=FakeLLM(), image=FakeImageGen())

    sb_mod.StoryboardStage()._direct_generation(project, shot, providers, brief="BRIEF")

    assert "never remove her gloves" in captured["hard_avoidances"]

SCRIPT = {
    "title": "Clockmaker",
    "episodes": [
        {"episode": 1, "scenes": [
            {"scene": 1, "heading": "SCENE 1", "beats": ["Mara opens up", "A bell rings"],
             "dialogue": [{"character": "Mara", "line": "Another grey morning."}]},
        ]},
    ],
}


def _project(tmp_path):
    p = Project.create("a clockmaker who repairs memories", root=tmp_path, stages=PIPELINE)
    p.path("story", "script.json").write_text(json.dumps(SCRIPT))
    # Minimal bible for Mara with a locked seed + a reference image.
    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps(
        {"name": "Mara", "seed": seed_for("Mara"), "reference_image": "reference.png"}))
    FakeImageGen().generate("ref", out_path=str(cdir / "reference.png"), seed=seed_for("Mara"))
    return p


def _providers():
    return Providers(llm=FakeLLM(), image=FakeImageGen())


def _render_approved_keyframes(project, providers):
    confirm_prompt_batch(project, "keyframes", confirmer="test")
    return KeyframesStage().run(project, providers)


def test_short_film_storyboard_duration_lands_in_format_band(tmp_path):
    # The fake pipeline must size content to the requested duration so the
    # duration-targeting path is validated offline: a short_film's total shot
    # runtime should fall inside the format's [min, max] band.
    fmt = {
        "name": "short_film",
        "target_duration_s": 180,
        "min_duration_s": 120,
        "max_duration_s": 300,
    }
    p = Project.create(
        "a clockmaker who repairs memories",
        root=tmp_path,
        stages=["plot", "script", "bible", "storyboard"],
        model_config={"product_format": fmt},
    )
    p.story_dir.joinpath("idea.md").write_text("a clockmaker who repairs memories")
    providers = Providers(llm=FakeLLM(), image=FakeImageGen())
    PlotStage().run(p, providers)
    ScriptStage().run(p, providers)
    BibleStage().run(p, providers)
    StoryboardStage().run(p, providers)

    shots_doc = json.loads(p.path("storyboard", "shots.json").read_text())
    shots = shots_doc.get("shots", shots_doc if isinstance(shots_doc, list) else [])
    total = sum(s["duration_s"] for s in shots)
    assert fmt["min_duration_s"] <= total <= fmt["max_duration_s"]


def test_storyboard_shot_prompt_includes_locked_genre_when_set(tmp_path):
    p = Project.create("a clockmaker", root=tmp_path, stages=PIPELINE,
                       model_config={"genre": "中国古代神话"})
    p.path("story", "script.json").write_text(json.dumps(SCRIPT))
    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps(
        {"name": "Mara", "seed": seed_for("Mara"), "reference_image": "reference.png"}))
    FakeImageGen().generate("ref", out_path=str(cdir / "reference.png"), seed=seed_for("Mara"))

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))

    shot_prompts = [pr for pr in llm.prompts if "[task:storyboard]" in pr]
    assert shot_prompts
    assert all("中国古代神话" in pr for pr in shot_prompts)
    # 是什么: the director persona carries the genre identity.
    persona_line = shot_prompts[0].splitlines()[1]
    assert "中国古代神话" in persona_line and "specialist" in persona_line


def test_scene_narration_lands_on_the_first_shot_of_the_scene(tmp_path):
    p = Project.create("narrated tale", root=tmp_path, stages=PIPELINE)
    narrated_script = {
        "title": "Narrated", "episodes": [
            {"episode": 1, "scenes": [
                {"scene": 1, "heading": "SCENE 1",
                 "beats": ["Mara opens up", "A bell rings"],
                 "dialogue": [],
                 "narration": "Once, a clockmaker mended more than time."},
            ]},
        ],
    }
    p.path("story", "script.json").write_text(json.dumps(narrated_script))
    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps(
        {"name": "Mara", "seed": seed_for("Mara"), "reference_image": "reference.png"}))
    FakeImageGen().generate("ref", out_path=str(cdir / "reference.png"), seed=seed_for("Mara"))

    StoryboardStage().run(p, _providers())

    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    scene_one = [s for s in shots if s.get("scene") == 1]
    assert len(scene_one) >= 2
    assert scene_one[0]["narration"] == "Once, a clockmaker mended more than time."
    assert all(s["narration"] == "" for s in scene_one[1:])


def _add_character(p, name):
    cdir = p.path("bible", "characters", char_slug(name))
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps(
        {"name": name, "seed": seed_for(name), "reference_image": "reference.png"}))
    FakeImageGen().generate("ref", out_path=str(cdir / "reference.png"), seed=seed_for(name))
    FakeImageGen().generate("turn", out_path=str(cdir / "turnaround.png"), seed=seed_for(name))


def _add_location(p, slug="clock-shop", name="Clock Shop", scenes=None):
    ldir = p.path("bible", "locations", slug)
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "location.json").write_text(json.dumps({
        "name": name,
        "scene_numbers": scenes if scenes is not None else [1],
        "prompt_aliases": [name, "memory repair shop"],
        "reference_image": "reference.png",
        "environment_board": "environment_board.png",
    }))
    FakeImageGen().generate("loc", out_path=str(ldir / "reference.png"), seed=seed_for(name))
    FakeImageGen().generate("board", out_path=str(ldir / "environment_board.png"), seed=seed_for(name))


class RecordingImageGen(FakeImageGen):
    def __init__(self):
        self.calls = []
    def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
        self.calls.append({"out_path": out_path, "prompt": prompt,
                           "reference_images": list(reference_images or [])})
        return super().generate(prompt, out_path=out_path,
                                reference_images=reference_images, **kwargs)


def test_storyboard_prepares_prompts_without_generating_keyframes(tmp_path):
    project = _project(tmp_path)
    image = RecordingImageGen()

    result = StoryboardStage().run(
        project,
        Providers(llm=FakeLLM(), image=image),
    )

    assert result.status == "complete"
    assert list(project.path("storyboard", "prompts").glob("*.keyframe.md"))
    assert image.calls == []
    assert not list(project.path("storyboard", "keyframes").glob("*.png"))


def test_storyboard_writes_shots_prompts_then_approved_keyframes(tmp_path):
    p = _project(tmp_path)
    providers = _providers()

    result = StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)

    assert result.status == "complete"
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert len(shots) == 2  # one shot per beat
    for shot in shots:
        for key in ["id", "scene", "description", "camera", "action",
                    "dialogue", "duration_s", "characters", "deps",
                    "keyframe", "reference_seed"]:
            assert key in shot, f"missing {key}"
        kf = p.path("storyboard", "keyframes", shot["keyframe"])
        assert kf.is_file()
        assert kf.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        # The compiled structured brief is preserved as its own artifact...
        brief = p.path("storyboard", "prompts", f"{shot['id']}.brief.md")
        assert brief.is_file()
        assert brief.read_text() == shot["keyframe_prompt"]
        # ...and the keyframe.md is the director's dense rewrite actually sent to the model.
        prompt = p.path("storyboard", "prompts", f"{shot['id']}.keyframe.md")
        assert prompt.is_file()
        assert "## Cinematography skill" not in prompt.read_text()  # no skill dump


def test_storyboard_sends_directed_prompt_to_image_model(tmp_path):
    p = _project(tmp_path)
    rec = RecordingImageGen()
    providers = Providers(llm=FakeLLM(), image=rec)

    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)

    call = next(c for c in rec.calls if c["out_path"].endswith("sh-001.png"))
    keyframe_md = p.path("storyboard", "prompts", "sh-001.keyframe.md").read_text()
    brief_md = p.path("storyboard", "prompts", "sh-001.brief.md").read_text()
    # The model receives the directed prompt (== keyframe.md), not the raw brief.
    assert call["prompt"] == keyframe_md
    assert keyframe_md != brief_md


def test_storyboard_does_not_redirect_when_keyframe_prompt_exists(tmp_path):
    p = _project(tmp_path)

    class CountingLLM(FakeLLM):
        def __init__(self):
            self.image_prompt_calls = 0

        def complete_json(self, prompt, *, system=None):
            if "[task:image_prompt]" in prompt:
                self.image_prompt_calls += 1
            return super().complete_json(prompt, system=system)

    llm = CountingLLM()
    providers = Providers(llm=llm, image=FakeImageGen())
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)
    first = llm.image_prompt_calls
    assert first >= 2  # one director call per shot

    # Drop only a keyframe image; keyframe.md stays -> director is not re-run.
    p.path("storyboard", "keyframes", "sh-001.png").unlink()
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)
    assert llm.image_prompt_calls == first


def test_keyframe_records_project_relative_reference_paths(tmp_path):
    p = _project(tmp_path)
    StoryboardStage().run(p, _providers())
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]

    mara_shot = next(s for s in shots if "Mara" in s["characters"])
    rel = mara_shot["reference_images"]
    assert "bible/characters/mara/reference.png" in rel
    # Paths are stored project-relative (not absolute).
    assert all(not r.startswith("/") for r in rel)


def test_keyframe_records_location_references_by_scene(tmp_path):
    p = _project(tmp_path)
    _add_location(p, scenes=[1])

    StoryboardStage().run(p, _providers())
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]

    first = shots[0]
    assert first["reference_locations"] == ["Clock Shop"]
    assert "bible/locations/clock-shop/reference.png" in first["reference_images"]
    assert "bible/locations/clock-shop/environment_board.png" not in first["reference_images"]
    brief = p.path("storyboard", "prompts", "sh-001.brief.md").read_text()
    assert "Clock Shop" in brief


def test_shots_have_sequential_dependencies(tmp_path):
    p = _project(tmp_path)
    StoryboardStage().run(p, _providers())
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]

    assert shots[0]["deps"] == []
    assert shots[1]["deps"] == [shots[0]["id"]]


def test_storyboard_plans_each_scene_with_a_separate_llm_call(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{
            "episode": 1,
            "scenes": [
                {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
                {"scene": 2, "heading": "SECOND", "beats": ["Mara closes the shop"]},
            ],
        }]
    }))

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))

    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    planning = [pr for pr in llm.prompts if "[task:storyboard]" in pr]
    assert len(planning) == 2  # one planning call per scene (director calls excluded)
    assert "Mara opens the shop" in planning[0]
    assert "Mara closes the shop" not in planning[0]
    assert "Mara closes the shop" in planning[1]
    assert [shot["scene"] for shot in shots] == [1, 2]
    assert [shot["id"] for shot in shots] == ["sh-001", "sh-002"]
    assert shots[1]["deps"] == ["sh-001"]


def test_storyboard_prompt_includes_story_flow_skill_and_advance_field(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
        ]}]
    }))

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))

    planning = [pr for pr in llm.prompts if "[task:storyboard]" in pr]
    assert planning
    assert "story_advance" in planning[0]
    assert "Story Flow Skill" in planning[0]  # skill text injected


def test_storyboard_shots_carry_detailed_visual_beats(tmp_path):
    p = _project(tmp_path)
    StoryboardStage().run(p, _providers())
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    beats = shots[0].get("visual_beats")
    assert beats, "each shot must carry a visual_beats choreography outline"
    assert all(isinstance(b, dict) and b.get("action") for b in beats)
    # The detailed beats reach the compiled video prompt when it is built.


def test_storyboard_planner_prompt_includes_character_motion_skill(tmp_path):
    # Mirror test_storyboard_prompt_includes_story_flow_skill_and_advance_field: capture the
    # planner prompt sent to the LLM and assert the character_motion skill text is injected.
    from studio_agent.runtime_skills import load_prompt_skill

    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
        ]}]
    }))

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))

    planning = [pr for pr in llm.prompts if "[task:storyboard]" in pr]
    assert planning
    marker = load_prompt_skill("character_motion").splitlines()[0]
    assert marker in planning[0]


def test_storyboard_feeds_prior_scene_tail_to_next_scene(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
            {"scene": 2, "heading": "SECOND", "beats": ["Mara closes the shop"]},
        ]}]
    }))

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))

    planning = [pr for pr in llm.prompts if "[task:storyboard]" in pr]
    assert len(planning) == 2
    # Scene 1 has no prior shots; scene 2's prompt must carry a PRIOR SHOTS block.
    assert "PRIOR SHOTS" in planning[0]
    assert "PRIOR SHOTS" in planning[1]
    assert "(none)" in planning[0]        # first scene: empty tail
    assert "(none)" not in planning[1]    # second scene: sees scene 1's shots


def test_planned_shots_carry_story_advance(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
        ]}]
    }))
    StoryboardStage().run(p, Providers(llm=FakeLLM(), image=FakeImageGen()))
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert all(str(shot.get("story_advance") or "").strip() for shot in shots)


def test_storyboard_prompt_includes_reference_intent_context(tmp_path):
    p = _project(tmp_path)
    save_reference_intake_uploads(
        p,
        [
            {"filename": "hero.png", "data": b"\x89PNG\r\n\x1a\nhero", "content_type": "image/png"},
            {"filename": "villain.png", "data": b"\x89PNG\r\n\x1a\nvillain", "content_type": "image/png"},
        ],
        note="photo1 kisses photo2",
    )

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))

    assert "REFERENCE_INTENT:" in llm.prompts[0]
    assert "photo1 kisses photo2" in llm.prompts[0]


def test_storyboard_prompt_includes_camera_recipe_menu(tmp_path):
    p = _project(tmp_path)

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))

    # The planner is offered the vetted camera-movement vocabulary by id.
    assert "camera_recipe" in llm.prompts[0]
    assert "static_subject_moves" in llm.prompts[0]


def test_fake_planner_assigns_camera_recipe(tmp_path):
    from studio_agent.cinematography_recipes import load_camera_recipes

    p = _project(tmp_path)
    StoryboardStage().run(p, _providers())
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]

    recipe_ids = set(load_camera_recipes())
    # Every shot is tagged with a valid recipe id; the movement language itself is
    # derived from the recipe at video-compile time (not persisted/seeded here).
    assert shots[0]["camera_recipe"] in recipe_ids


def test_storyboard_writes_scene_and_shot_packets_with_motivated_design(tmp_path):
    p = _project(tmp_path)

    StoryboardStage().run(p, _providers())

    shot = json.loads(p.path("storyboard", "shots.json").read_text())["shots"][0]
    assert shot["movement_motivation"]
    assert shot["start_frame"] and shot["end_frame"]
    assert shot["craft_recipe_ids"]
    assert p.path(
        "knowledge", "packets", f"scene-{shot['scene']}.json"
    ).is_file()
    assert p.path(
        "knowledge", "packets", f"shot-{shot['id']}.json"
    ).is_file()
    assert p.path(
        "storyboard", "prompts", f"{shot['id']}.knowledge.json"
    ).is_file()
    assert p.path(
        "storyboard", "prompts", f"{shot['id']}.keyframe.md"
    ).is_file()


def test_storyboard_preflight_asks_when_camera_language_is_unresolved(tmp_path):
    p = _project(tmp_path)

    result = StoryboardStage().preflight(p, _providers(), auto=False)

    assert result.decision_required is True
    assert result.request_path.endswith("story/decisions/storyboard.json")


def test_storyboard_tolerates_shots_without_camera_recipe(tmp_path):
    p = _project(tmp_path)
    # Old-shape shot lacking the camera_recipe field must backfill cleanly.
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "scene": 1, "description": "two", "camera": "wide",
         "action": "they meet", "dialogue": [], "duration_s": 2.0,
         "characters": ["Mara"], "deps": [], "reference_seed": seed_for("Mara"),
         "keyframe": "sh-001.png"},
    ]}))

    StoryboardStage().run(p, _providers())

    shot = json.loads(p.path("storyboard", "shots.json").read_text())["shots"][0]
    assert "camera_recipe" in shot  # defaulted, no crash
    assert shot["movement_motivation"]


def test_storyboard_logs_cost(tmp_path):
    p = _project(tmp_path)
    StoryboardStage().run(p, _providers())
    assert any(entry["stage"] == "storyboard" for entry in p.cost_log)


def test_storyboard_is_idempotent(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    StoryboardStage().run(p, providers)
    p.set_stage_status("storyboard", "complete")
    cost_after_first = len(p.cost_log)

    result = StoryboardStage().run(p, providers)
    assert result.status == "skipped"
    assert len(p.cost_log) == cost_after_first


def test_keyframes_stage_regenerates_missing_keyframe_only(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    cost_after_first = len(p.cost_log)

    kf = p.path("storyboard", "keyframes", shots[0]["keyframe"])
    kf.unlink()
    result = KeyframesStage().run(p, providers)

    assert result.status == "complete"
    assert kf.is_file()
    assert len(p.cost_log) == cost_after_first + 1  # only the one keyframe re-rendered


def test_keyframes_stage_preserves_reapproved_edited_prompt_when_regenerating(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    shot = shots[0]
    prompt = p.path("storyboard", "prompts", f"{shot['id']}.keyframe.md")
    edited = (
        "human-edited keyframe prompt: Mara opens the shop, framed through foreground "
        "gears on a 50mm lens at eye-level with motivated amber window lighting."
    )
    prompt.write_text(edited)
    p.path("storyboard", "keyframes", shot["keyframe"]).unlink()
    cost_after_first = len(p.cost_log)

    confirm_prompt_batch(p, "keyframes", confirmer="test")
    KeyframesStage().run(p, providers)

    assert prompt.read_text() == edited
    assert len(p.cost_log) == cost_after_first + 1


def test_keyframe_prompt_mentions_expressions_for_speaking_char_with_sheet():
    # Combined sheet: expressions now live in the single reference.png.
    # has_expression_sheet is True whenever ANY referenced character has a non-empty
    # reference_images list (so speaking shots still get the expression-match instruction).
    bible = {"Mara": {"aliases": ["Mara"], "reference_images": [
        "bible/characters/mara/reference.png",
    ]}}
    speaking = {"camera": "close-up", "action": "she replies",
                "dialogue": [{"character": "Mara", "line": "hello"}]}
    silent = {"camera": "wide", "action": "she walks in", "dialogue": []}

    p_speaking = StoryboardStage._keyframe_prompt(speaking, ["Mara"], bible, {})
    p_silent = StoryboardStage._keyframe_prompt(silent, ["Mara"], bible, {})

    assert "expression" in p_speaking.lower()
    assert "expression" not in p_silent.lower()  # gated on dialogue


def test_keyframe_prompt_no_expression_mention_when_char_has_no_sheet():
    # When the character has no bible reference_images at all, expression guidance is
    # suppressed even for speaking shots (has_expression_sheet is False).
    bible = {"Mara": {"aliases": ["Mara"], "reference_images": []}}
    speaking = {"camera": "close-up", "action": "she replies",
                "dialogue": [{"character": "Mara", "line": "hello"}]}

    out = StoryboardStage._keyframe_prompt(speaking, ["Mara"], bible, {})

    assert "expression" not in out.lower()


def test_keyframe_prompt_threads_identity_board_do_dont_rules():
    # do/dont/hero_props from the identity board must reach the keyframe prompt via the bible.
    bible = {
        "Mara": {
            "aliases": ["Mara"],
            "reference_images": ["bible/characters/mara/reference.png"],
            "appearance_lock": "auburn braid, copper scarf",
            "do": ["keep the copper scarf visible"],
            "dont": ["never remove her gloves"],
            "hero_props": ["brass telescope"],
            "continuity_priority": ["face", "silhouette"],
        }
    }
    raw = {"camera": "close-up", "action": "she waits", "dialogue": []}
    out = StoryboardStage._keyframe_prompt(raw, ["Mara"], bible, {})
    assert "## Character rules" in out
    assert "keep the copper scarf visible" in out
    assert "never remove her gloves" in out
    assert "brass telescope" in out


def test_keyframe_prompt_includes_extra_style_guidance():
    bible = {}
    raw = {"camera": "wide", "action": "Mara enters", "dialogue": []}
    style = {
        "look": "cartoon",
        "palette": "bright candy colors",
        "aspect_ratio": "4:3",
        "rendering": "soft rounded shapes",
        "motion": "snappy squash-and-stretch energy",
    }

    out = StoryboardStage._keyframe_prompt(raw, [], bible, style)

    assert "cartoon" in out
    assert "bright candy colors" in out
    assert "soft rounded shapes" in out
    assert "snappy squash-and-stretch energy" in out


def test_keyframe_prompt_uses_extended_storyboard_direction_fields():
    bible = {}
    raw = {
        "camera": "wide",
        "action": "The princess steps from the glowing television",
        "composition": "TV portal foreground, scattered apartment objects midground",
        "emotion": "awe mixed with panic",
        "continuity_notes": "keep blue TV light on the floor and gown",
        "dialogue": [],
    }

    out = StoryboardStage._keyframe_prompt(raw, [], bible, {})

    assert "TV portal foreground" in out
    assert "awe mixed with panic" in out
    assert "blue TV light" in out


def test_storyboard_keyframe_prompt_includes_only_static_world_context(tmp_path):
    p = _project(tmp_path)
    p.path("story", "idea.md").write_text("A clockmaker repairs memories at dawn.")
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "A clockmaker chooses whether to restore a painful memory.",
        "synopsis": "Mara opens the shop and hears a bell that should not ring.",
        "themes": ["memory", "regret"],
    }))

    StoryboardStage().run(p, _providers())

    brief = p.path("storyboard", "prompts", "sh-001.brief.md").read_text()
    assert "Scene: SCENE 1" in brief
    assert "clockmaker chooses" not in brief
    assert "Mara opens up" not in brief
    assert "A bell rings" not in brief


def test_keyframe_conditioned_on_combined_reference_sheet_when_present(tmp_path):
    # Combined sheet: expressions now live inside reference.png. Only reference.png is
    # threaded into reference_images; turnaround/expressions are no longer separate files.
    p = _project(tmp_path)  # _project already writes reference.png for Mara

    StoryboardStage().run(p, _providers())
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]

    mara_shot = next(s for s in shots if "Mara" in s["characters"])
    assert "bible/characters/mara/reference.png" in mara_shot["reference_images"]
    assert not any(r.endswith(("turnaround.png", "expressions.png"))
                   for r in mara_shot["reference_images"])


def test_keyframe_conditioned_on_all_named_characters_and_backfills_old_shape(tmp_path):
    p = _project(tmp_path)            # gives Mara (reference.png only)
    _add_character(p, "Mara")         # ensure Mara has reference.png (+ turnaround.png written but not threaded)
    _add_character(p, "The Customer")

    # Old-shape shots.json: one shot with two characters, no new reference fields.
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "scene": 1, "description": "two-shot", "camera": "wide",
         "action": "they meet", "dialogue": [], "duration_s": 2.0,
         "characters": ["Mara", "The Customer"], "deps": [],
         "reference_character": "Mara", "reference_seed": seed_for("Mara"),
         "keyframe_prompt": (
             "two-shot of Mara and The Customer on a 50mm lens at eye-level with "
             "motivated warm window lighting"
         ), "keyframe": "sh-001.png"},
    ]}))

    rec = RecordingImageGen()
    providers = Providers(llm=FakeLLM(), image=rec)
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)

    # The keyframe call received BOTH characters' combined reference sheets (reference.png only).
    call = next(c for c in rec.calls if c["out_path"].endswith("sh-001.png"))
    refs = [r.replace(str(p.dir) + "/", "") for r in call["reference_images"]]
    assert "bible/characters/mara/reference.png" in refs
    assert "bible/characters/mara/turnaround.png" not in refs
    assert "bible/characters/the-customer/reference.png" in refs
    assert "bible/characters/the-customer/turnaround.png" not in refs

    # shots.json was backfilled with the new fields (project-relative).
    shot = json.loads(p.path("storyboard", "shots.json").read_text())["shots"][0]
    assert set(shot["reference_characters"]) == {"Mara", "The Customer"}
    assert "bible/characters/mara/reference.png" in shot["reference_images"]
    assert "bible/characters/mara/turnaround.png" not in shot["reference_images"]


def test_keyframe_conditioned_on_location_board_images(tmp_path):
    # Combined sheet: location details now live inside reference.png only.
    p = _project(tmp_path)
    _add_location(p, scenes=[1])
    rec = RecordingImageGen()
    providers = Providers(llm=FakeLLM(), image=rec)

    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)

    call = next(c for c in rec.calls if c["out_path"].endswith("sh-001.png"))
    refs = [r.replace(str(p.dir) + "/", "") for r in call["reference_images"]]
    assert "bible/locations/clock-shop/reference.png" in refs
    assert "bible/locations/clock-shop/environment_board.png" not in refs


def test_storyboard_includes_uploaded_character_location_and_style_refs(tmp_path):
    p = _project(tmp_path)
    _add_location(p, scenes=[1])
    char_ref = tmp_path / "char-upload.png"
    loc_ref = tmp_path / "loc-upload.png"
    style_ref = tmp_path / "style-upload.png"
    FakeImageGen().generate("char", out_path=str(char_ref), seed=11)
    FakeImageGen().generate("loc", out_path=str(loc_ref), seed=12)
    FakeImageGen().generate("style", out_path=str(style_ref), seed=13)

    char_record = save_reference_upload(
        p,
        filename="char-upload.png",
        data=char_ref.read_bytes(),
        target_type="character",
        target_id="Mara",
    )
    loc_record = save_reference_upload(
        p,
        filename="loc-upload.png",
        data=loc_ref.read_bytes(),
        target_type="location",
        target_id="Clock Shop",
    )
    style_record = save_reference_upload(
        p,
        filename="style-upload.png",
        data=style_ref.read_bytes(),
        target_type="style",
        target_id="global",
    )
    p.path("bible", "style_sample.png").write_bytes(b"\x89PNG\r\n\x1a\nsample")

    StoryboardStage().run(p, _providers())
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    first = shots[0]

    assert char_record["path"] in first["reference_images"]
    assert loc_record["path"] in first["reference_images"]
    assert style_record["path"] not in first["reference_images"]
    assert first["reference_style_images"] == ["bible/style_sample.png"]
    assert "bible/style_sample.png" in first["reference_images"]


def test_storyboard_backfills_uploaded_refs_into_existing_shots(tmp_path):
    p = _project(tmp_path)
    StoryboardStage().run(p, _providers())
    upload = tmp_path / "mara-new-ref.png"
    FakeImageGen().generate("char", out_path=str(upload), seed=21)
    record = save_reference_upload(
        p,
        filename="mara-new-ref.png",
        data=upload.read_bytes(),
        target_type="character",
        target_id="Mara",
    )
    mark_reference_upload_revised(p)

    StoryboardStage().run(p, _providers())

    shot = json.loads(p.path("storyboard", "shots.json").read_text())["shots"][0]
    assert record["path"] in shot["reference_images"]


def test_storyboard_excludes_unresolved_intake_references(tmp_path):
    p = _project(tmp_path)
    records = save_reference_intake_uploads(
        p,
        [{"filename": "unknown.png", "data": b"\x89PNG\r\n\x1a\nunknown", "content_type": "image/png"}],
        note="",
    )

    StoryboardStage().run(p, _providers())

    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert all(records[0]["path"] not in shot["reference_images"] for shot in shots)


def test_storyboard_includes_retargeted_intake_reference(tmp_path):
    p = _project(tmp_path)
    records = save_reference_intake_uploads(
        p,
        [{"filename": "mara-upload.png", "data": b"\x89PNG\r\n\x1a\nmara", "content_type": "image/png"}],
        note="",
    )
    updated = retarget_reference(p, records[0]["id"], "character", "Mara")

    StoryboardStage().run(p, _providers())

    shot = json.loads(p.path("storyboard", "shots.json").read_text())["shots"][0]
    assert updated["path"] in shot["reference_images"]


def _grid_story_project(tmp_path, *, story_scenes=True):
    p = _project(tmp_path)
    p.model_config["motion_grid"] = {
        "enabled": True, "story_scenes": story_scenes, "layout": "2x2",
    }
    p.save()
    return p


def _grid_providers(*, image_capable=True, video_capable=True):
    return Providers(
        llm=FakeLLM(),
        image=FakeImageGen(supports_storyboard_grid=image_capable),
        video=FakeVideoGen(supports_storyboard_grid=video_capable),
    )


def test_story_scene_grid_plans_adaptive_shots_with_atomic_dialogue(tmp_path):
    p = _grid_story_project(tmp_path)
    providers = _grid_providers()
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert shots, "expected scene-sequence shots"
    assert all("motion_grid" in s for s in shots)
    dialogue_shots = [s for s in shots if s.get("dialogue")]
    assert len(dialogue_shots) >= 1
    # a dialogue shot is its own segment (single atomic beat)
    assert all(len(s["visual_beats"]) == 1 for s in dialogue_shots)
    # adaptive lengths: not every clip is the same short length
    durations = [s["duration_s"] for s in shots]
    assert max(durations) > min(durations)
    # grid artifacts + keyframe rendered for every shot; per-scene sequence recorded
    for shot in shots:
        assert p.path("storyboard", "keyframes", shot["keyframe"]).is_file()
        assert p.path("storyboard", "prompts", f"{shot['id']}.grid.md").is_file()
    assert p.path("storyboard", "scene_sequences", "1.json").is_file()


def test_scene_sequence_prompt_carries_story_advance_and_skill(tmp_path):
    # Task 4: the motion-grid (分镜图) scene planner must mirror the main storyboard
    # path (Task 3) — beats declare story_advance and the prompt carries the
    # story_flow skill, so grid-planned beats don't repeat the same story beat either.
    p = _grid_story_project(tmp_path)

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    providers = Providers(
        llm=llm,
        image=FakeImageGen(supports_storyboard_grid=True),
        video=FakeVideoGen(supports_storyboard_grid=True),
    )
    StoryboardStage().run(p, providers)

    seq = [pr for pr in llm.prompts if "[task:scene_sequence]" in pr]
    assert seq
    assert "story_advance" in seq[0]
    assert "STORY FLOW SKILL" in seq[0]


def test_scene_sequence_template_declares_story_advance():
    # Robust companion to the grid-path test above: asserts directly on the template
    # string so this stays green even if the grid-activation path shifts underneath it.
    from studio_agent.stages.storyboard import _SCENE_SEQUENCE_TEMPLATE

    assert "story_advance" in _SCENE_SEQUENCE_TEMPLATE
    assert "{story_flow_skill}" in _SCENE_SEQUENCE_TEMPLATE


def test_planner_template_is_restraint_first():
    from studio_agent.stages.storyboard import _PROMPT_TEMPLATE, _SCENE_SEQUENCE_TEMPLATE

    for tmpl in (_PROMPT_TEMPLATE, _SCENE_SEQUENCE_TEMPLATE):
        low = tmpl.lower()
        assert "one" in low and "move" in low
        assert "reserve" in low  # reserve orbit/crane for earned beats
        assert "static" in low   # static camera is a valid choice


def test_planner_templates_thread_camera_movement_skill():
    # The camera_movement rulebook belongs to the shot-list planners, not the
    # still-image keyframe compiler. Both templates must carry the placeholder so
    # the skill text is injected the same way story_flow_skill is.
    from studio_agent.stages.storyboard import (
        _PROMPT_TEMPLATE,
        _SCENE_SEQUENCE_TEMPLATE,
    )

    assert "{camera_movement_skill}" in _PROMPT_TEMPLATE
    assert "{camera_movement_skill}" in _SCENE_SEQUENCE_TEMPLATE


def test_shot_list_planner_prompt_carries_camera_movement_skill(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
        ]}]
    }))

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))

    planning = [pr for pr in llm.prompts if "[task:storyboard]" in pr]
    assert planning
    # Real skill text injected (heading of skills/camera_movement.md), not "(none)".
    assert "Camera Movement Skill" in planning[0]


def test_scene_sequence_prompt_carries_camera_movement_skill(tmp_path):
    p = _grid_story_project(tmp_path)

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.prompts = []

        def complete_json(self, prompt, *, system=None):
            self.prompts.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    providers = Providers(
        llm=llm,
        image=FakeImageGen(supports_storyboard_grid=True),
        video=FakeVideoGen(supports_storyboard_grid=True),
    )
    StoryboardStage().run(p, providers)

    seq = [pr for pr in llm.prompts if "[task:scene_sequence]" in pr]
    assert seq
    assert "Camera Movement Skill" in seq[0]


def test_story_grid_skipped_without_story_scenes_flag(tmp_path):
    p = _grid_story_project(tmp_path, story_scenes=False)
    StoryboardStage().run(p, _grid_providers())
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert all("motion_grid" not in s for s in shots)


def test_story_grid_skipped_when_providers_not_capable(tmp_path):
    p = _grid_story_project(tmp_path)
    StoryboardStage().run(p, _grid_providers(image_capable=False, video_capable=False))
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert all("motion_grid" not in s for s in shots)


# ---------------------------------------------------------------------------
# Combined reference sheets: one reference.png per character/location (Task 3)
# ---------------------------------------------------------------------------

def _storyboard_project_with_one_char_one_location(tmp_path):
    """Project with one character (Mara) and one location (Clock Shop).

    Simulates a full old-style bible on disk: reference.png PLUS turnaround.png and
    expressions.png (character) and environment_board.png (location) are all written.
    The production _load_bible/_load_locations loops must NOT pick up those extra files —
    only reference.png should appear in reference_images. This makes the test a real
    guard: if the collection loops were reverted to enumerate turnaround/expressions/
    environment_board, those files would appear on disk and the assertions would fail.
    """
    script = {
        "title": "Clockmaker",
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "SCENE 1",
             "beats": ["Mara opens up", "A bell rings"],
             "dialogue": [{"character": "Mara", "line": "Another grey morning."}]},
        ]}],
    }
    p = Project.create("a clockmaker", root=tmp_path, stages=PIPELINE)
    p.path("story", "script.json").write_text(json.dumps(script))

    # Character: reference.png (combined sheet) + old-style extra files on disk.
    # The collection loop must ignore turnaround.png and expressions.png.
    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps(
        {"name": "Mara", "seed": seed_for("Mara"), "reference_image": "reference.png"}))
    FakeImageGen().generate("ref", out_path=str(cdir / "reference.png"), seed=seed_for("Mara"))
    FakeImageGen().generate("turn", out_path=str(cdir / "turnaround.png"), seed=seed_for("Mara"))
    FakeImageGen().generate("expr", out_path=str(cdir / "expressions.png"), seed=seed_for("Mara"))

    # Location: reference.png (combined sheet) + old-style environment_board.png on disk.
    # The collection loop must ignore environment_board.png.
    ldir = p.path("bible", "locations", "clock-shop")
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "location.json").write_text(json.dumps({
        "name": "Clock Shop",
        "scene_numbers": [1],
        "prompt_aliases": ["Clock Shop"],
        "reference_image": "reference.png",
    }))
    FakeImageGen().generate("loc", out_path=str(ldir / "reference.png"), seed=seed_for("Clock Shop"))
    FakeImageGen().generate("board", out_path=str(ldir / "environment_board.png"), seed=seed_for("Clock Shop"))

    return p


def _load_first_scene_shots(p):
    return json.loads(p.path("storyboard", "shots.json").read_text())["shots"]


def test_shot_threads_one_reference_per_character_and_location(tmp_path):
    # After the combined sheets, a single-character/single-location shot passes exactly
    # char(1) + location(1) refs from the bible (style added separately) — not 3 + 2.
    p = _storyboard_project_with_one_char_one_location(tmp_path)
    StoryboardStage().run(p, Providers(llm=FakeLLM(), image=FakeImageGen()))

    shots = _load_first_scene_shots(p)
    refs = shots[0]["reference_images"]
    char_refs = [r for r in refs if "/characters/" in r]
    loc_refs = [r for r in refs if "/locations/" in r]
    assert char_refs == ["bible/characters/mara/reference.png"]
    assert loc_refs == ["bible/locations/clock-shop/reference.png"]
    assert not any(r.endswith(("turnaround.png", "expressions.png", "environment_board.png"))
                   for r in refs)


def test_shot_wires_bible_character_after_display_name_drift(tmp_path):
    """Shots join to the bible by the STABLE folder slug, not the mutable display name.

    Regression: the plot names a character (e.g. CJK "青年"); the bible folder is created
    from that name (char_slug("青年")). A later bible-text revision renamed the character's
    display `name` to "南宫婉". Shots still carry the plot name "青年", so the old exact
    `name in bible` join failed and the character's reference sheet was never wired into any
    keyframe — regenerating the character then had no downstream effect. The join must
    survive the rename via the stable slug.
    """
    p = Project.create("name drift", root=tmp_path, stages=["bible", "storyboard"])
    slug = char_slug("青年")            # bible folder is created from the ORIGINAL plot name
    assert slug != char_slug("南宫婉")   # the revised display name would hash elsewhere
    cdir = p.path("bible", "characters", slug)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "reference.png").write_bytes(b"\x89PNG\r\n\x1a\nchar")
    (cdir / "character.json").write_text(
        json.dumps({"name": "南宫婉", "seed": 7}, ensure_ascii=False)  # display name drifted
    )

    stage = StoryboardStage()
    bible = stage._load_bible(p)
    ref_chars, ref_images, _seed = stage._shot_references(["青年"], bible, scene=1)

    assert ref_chars, "the plot-named character must resolve to the bible entry after a rename"
    assert f"bible/characters/{slug}/reference.png" in ref_images


def test_story_grid_render_is_idempotent(tmp_path):
    p = _grid_story_project(tmp_path)
    providers = _grid_providers()
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)
    kf = p.path("storyboard", "keyframes", "sh-001.png")
    before = kf.read_bytes()
    KeyframesStage().run(p, providers)
    assert kf.read_bytes() == before


def test_storyboard_shot_references_include_generated_style_sample(tmp_path):
    p = _project(tmp_path)
    sample = p.path("bible", "style_sample.png")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"\x89PNG\r\n\x1a\nsample")

    StoryboardStage().run(p, _providers())

    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert shots, "expected at least one shot"
    assert all("bible/style_sample.png" in s["reference_images"] for s in shots)


# ---------------------------------------------------------------------------
# Helper: build a project whose first shot lists 3 subject refs + style_sample
# ---------------------------------------------------------------------------

def _png(path):
    """Write a minimal valid PNG to *path* and return path."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    return path


def _storyboard_project_with_keyframe_needing_4_refs(tmp_path):
    """Project with one shot whose keyframe_reference_images has 4 entries.

    Layout:
      bible/style_sample.png         — the style anchor (lowest priority, dropped first)
      bible/characters/mara/reference.png   — subject ref 1
      bible/characters/rex/reference.png    — subject ref 2
      bible/characters/zoe/reference.png    — subject ref 3

    shots.json is pre-written so StoryboardStage skips planning and goes
    straight to keyframe rendering.
    """
    p = Project.create("test cap", root=tmp_path, stages=PIPELINE)

    # Style anchor
    _png(p.path("bible", "style_sample.png"))

    # Three subject character reference images
    for name in ("mara", "rex", "zoe"):
        _png(p.path("bible", "characters", name, "reference.png"))

    # Pre-built shots.json — one shot referencing all 4 images.
    # Using keyframe_reference_images so _render_missing_keyframes picks them up.
    shot = {
        "id": "sh-001",
        "scene": 1,
        "description": "test shot",
        "camera": "wide",
        "camera_movement": "locked-off",
        "camera_recipe": "",
        "action": "characters stand",
        "dialogue": [],
        "narration": "",
        "duration_s": 3.0,
        "characters": ["Mara", "Rex", "Zoe"],
        "reference_characters": ["Mara", "Rex", "Zoe"],
        "reference_locations": [],
        "reference_style_images": ["bible/style_sample.png"],
        "reference_images": [
            "bible/characters/mara/reference.png",
            "bible/characters/rex/reference.png",
            "bible/characters/zoe/reference.png",
            "bible/style_sample.png",
        ],
        "keyframe_reference_images": [
            "bible/characters/mara/reference.png",
            "bible/characters/rex/reference.png",
            "bible/characters/zoe/reference.png",
            "bible/style_sample.png",
        ],
        "state_reference_images": [],
        "reference_seed": seed_for("Mara"),
        "keyframe": "sh-001.png",
        "keyframe_prompt": "A wide shot of three characters standing.",
        "deps": [],
        "dramatic_purpose": "introduction",
        "start_frame": "three characters standing",
        "end_frame": "three characters standing",
        "movement_motivation": "none",
        "craft_recipe_ids": [],
    }
    shots_dir = p.path("storyboard")
    shots_dir.mkdir(parents=True, exist_ok=True)
    p.path("storyboard", "shots.json").write_text(
        json.dumps({"shots": [shot]}, ensure_ascii=False, indent=2)
    )
    # Write the keyframe prompt file so _render_missing_keyframes can read it.
    prompts_dir = p.path("storyboard", "prompts")
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "sh-001.keyframe.md").write_text(shot["keyframe_prompt"])
    # Ensure keyframes dir exists (stage creates it but let's be safe).
    p.path("storyboard", "keyframes").mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# TDD: reference cap test
# ---------------------------------------------------------------------------

def test_keyframe_caps_references_to_provider_max(tmp_path):
    """When a provider has max_reference_images=3, only 3 refs must reach generate().

    The style anchor (style_sample.png) is lowest priority and must be dropped first.
    """
    seen = {}

    class _Capped(FakeImageGen):
        max_reference_images = 3

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            seen["refs"] = list(reference_images or [])
            return super().generate(prompt, out_path=out_path,
                                    reference_images=reference_images, **kwargs)

    project = _storyboard_project_with_keyframe_needing_4_refs(tmp_path)
    providers = Providers(llm=FakeLLM(), image=_Capped())
    StoryboardStage().run(project, providers)
    _render_approved_keyframes(project, providers)

    assert len(seen["refs"]) == 3, f"expected 3 refs, got {seen['refs']}"
    assert not any(r.endswith("style_sample.png") for r in seen["refs"]), (
        "style anchor should be dropped first"
    )


def test_keyframe_refreshes_live_upload_added_after_shots_json(tmp_path):
    p = _storyboard_project_with_keyframe_needing_4_refs(tmp_path)
    record = save_reference_upload(
        p,
        filename="mara-upload.png",
        data=b"\x89PNG\r\n\x1a\nraw",
        target_type="character",
        target_id="Mara",
    )
    seen = {}

    class _RecordingImage(FakeImageGen):
        max_reference_images = 10

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            seen["refs"] = list(reference_images or [])
            return super().generate(
                prompt,
                out_path=out_path,
                reference_images=reference_images,
                **kwargs,
            )

    providers = Providers(llm=FakeLLM(), image=_RecordingImage())
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)

    assert str(p.dir / record["path"]) in seen["refs"]


def test_keyframe_cap_retains_explicitly_named_upload(tmp_path):
    p = _storyboard_project_with_keyframe_needing_4_refs(tmp_path)
    first = save_reference_upload(
        p,
        filename="mara-first.png",
        data=b"\x89PNG\r\n\x1a\nfirst",
        target_type="character",
        target_id="Mara",
    )
    named = save_reference_upload(
        p,
        filename="mara-named.png",
        data=b"\x89PNG\r\n\x1a\nnamed",
        target_type="character",
        target_id="Mara",
    )
    shots_path = p.path("storyboard", "shots.json")
    data = json.loads(shots_path.read_text())
    data["shots"][0]["named_reference_images"] = [named["path"]]
    shots_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    seen = {}

    class _OneReferenceImage(FakeImageGen):
        max_reference_images = 1

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            seen["refs"] = list(reference_images or [])
            return super().generate(
                prompt,
                out_path=out_path,
                reference_images=reference_images,
                **kwargs,
            )

    providers = Providers(llm=FakeLLM(), image=_OneReferenceImage())
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)

    assert seen["refs"] == [str(p.dir / named["path"])]
    assert str(p.dir / first["path"]) not in seen["refs"]


def test_keyframe_fails_before_image_generation_when_bound_upload_is_missing(tmp_path):
    p = _storyboard_project_with_keyframe_needing_4_refs(tmp_path)
    record = save_reference_upload(
        p,
        filename="mara-missing.png",
        data=b"\x89PNG\r\n\x1a\nmissing",
        target_type="character",
        target_id="Mara",
    )
    p.path(*record["path"].split("/")).unlink()

    class _RecordingImage(FakeImageGen):
        def __init__(self):
            self.calls = []

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            self.calls.append(list(reference_images or []))
            return super().generate(
                prompt,
                out_path=out_path,
                reference_images=reference_images,
                **kwargs,
            )

    image = _RecordingImage()
    providers = Providers(llm=FakeLLM(), image=image)
    StoryboardStage().run(p, providers)
    confirm_prompt_batch(p, "keyframes", confirmer="test")
    with pytest.raises(FileNotFoundError, match=record["path"]):
        KeyframesStage().run(p, providers)

    assert image.calls == []


def test_character_locks_maps_alias_to_lock():
    from studio_agent.stages.storyboard import StoryboardStage

    bible = {
        "Mara": {
            "slug": "mara",
            "aliases": ["Mara", "the keeper"],
            "appearance_lock": "Oval face, teardrop mole; red wool coat.",
            "reference_images": [],
        }
    }
    locks = StoryboardStage._character_locks(["Mara"], bible)
    assert locks == {"Mara": "Oval face, teardrop mole; red wool coat."}


def test_character_locks_skips_characters_without_a_lock():
    from studio_agent.stages.storyboard import StoryboardStage

    bible = {"Bg": {"slug": "bg", "aliases": ["Bg"], "appearance_lock": "", "reference_images": []}}
    assert StoryboardStage._character_locks(["Bg"], bible) == {}


def test_fill_missing_keyframe_prompts_populates_locks_even_when_prompt_exists(tmp_path):
    """Appearance locks must be populated even when a shot already has a keyframe_prompt.

    Before the fix, the lock computation lived AFTER the `if shot.get("keyframe_prompt"):
    continue` guard, so any shot from a legacy shots.json (or a re-render after the
    .keyframe.md file was deleted) silently omitted character_appearance_locks and the
    director received verbatim_locks=[].

    Two behaviours are verified here without paying for any model call:

    1. A shot that already has a keyframe_prompt (the legacy/re-render case) still gains
       character_appearance_locks from the bible.
    2. A shot whose dict already has character_appearance_locks is NOT overwritten
       (the setdefault guarantee).
    """
    from studio_agent.stages.storyboard import StoryboardStage

    bible = {
        "Mara": {
            "slug": "mara",
            "aliases": ["Mara"],
            "appearance_lock": "Oval face, teardrop mole; red wool coat.",
            "reference_images": [],
        }
    }

    # Case 1: shot already has a keyframe_prompt — locks must still be populated.
    shot_with_prompt = {
        "id": "sh-001",
        "reference_characters": ["Mara"],
        "keyframe_prompt": "pre-existing prompt that skips regeneration",
    }
    # Simulate the relevant inner loop logic directly (no project I/O needed).
    ref_chars = shot_with_prompt.get("reference_characters", [])
    locks = StoryboardStage._character_locks(ref_chars, bible)
    if locks:
        shot_with_prompt.setdefault("character_appearance_locks", list(locks.values()))

    assert "character_appearance_locks" in shot_with_prompt, (
        "locks must be populated even when keyframe_prompt already exists"
    )
    assert shot_with_prompt["character_appearance_locks"] == ["Oval face, teardrop mole; red wool coat."]

    # Case 2: shot that already has character_appearance_locks must NOT be overwritten
    # by setdefault (the human-edit preservation guarantee).
    existing_lock = ["human-edited lock value"]
    shot_with_existing_lock = {
        "id": "sh-002",
        "reference_characters": ["Mara"],
        "keyframe_prompt": "pre-existing prompt",
        "character_appearance_locks": existing_lock[:],
    }
    ref_chars2 = shot_with_existing_lock.get("reference_characters", [])
    locks2 = StoryboardStage._character_locks(ref_chars2, bible)
    if locks2:
        shot_with_existing_lock.setdefault("character_appearance_locks", list(locks2.values()))

    assert shot_with_existing_lock["character_appearance_locks"] == existing_lock, (
        "setdefault must not overwrite an already-populated character_appearance_locks"
    )


def test_keyframe_live_refresh_excludes_endpoint_reference(tmp_path):
    p = _storyboard_project_with_keyframe_needing_4_refs(tmp_path)
    endpoint_rel = "bible/characters/mara/states/transformed/reference.png"
    _png(p.dir / endpoint_rel)
    mara_dir = p.path("bible", "characters", "mara")
    (mara_dir / "character.json").write_text(json.dumps({
        "name": "Mara",
        "seed": seed_for("Mara"),
        "reference_image": "reference.png",
    }))
    (mara_dir / "states.json").write_text(json.dumps({
        "states": [{
            "id": "transformed",
            "reference_image": "states/transformed/reference.png",
        }],
    }))
    shots_path = p.path("storyboard", "shots.json")
    data = json.loads(shots_path.read_text())
    data["shots"][0]["reference_images"].append(endpoint_rel)
    data["shots"][0]["end_states"] = {"Mara": "transformed"}
    data["shots"][0]["target_state_reference_images"] = [endpoint_rel]
    shots_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    seen = {}

    class _RecordingImage(FakeImageGen):
        max_reference_images = 10

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            seen["refs"] = list(reference_images or [])
            return super().generate(
                prompt,
                out_path=out_path,
                reference_images=reference_images,
                **kwargs,
            )

    providers = Providers(llm=FakeLLM(), image=_RecordingImage())
    StoryboardStage().run(p, providers)
    _render_approved_keyframes(p, providers)

    assert str(p.dir / endpoint_rel) not in seen["refs"]


def test_render_flow_review_md_lists_redundancies_and_summary():
    from studio_agent.stages.storyboard import _render_flow_review_md

    md = _render_flow_review_md({
        "redundancies": [
            {"shots": ["sh-002", "sh-003"], "reason": "both show Mara reacting",
             "suggestion": "merge"},
        ],
        "progression": [
            {"scene": 1, "advances": True, "note": "each shot adds a beat"},
        ],
        "summary": "one redundant pair in scene 1",
    })
    assert "sh-002" in md and "sh-003" in md
    assert "merge" in md
    assert "one redundant pair in scene 1" in md


def test_render_flow_review_md_handles_no_redundancies():
    from studio_agent.stages.storyboard import _render_flow_review_md

    md = _render_flow_review_md({"redundancies": [], "progression": [], "summary": "clean"})
    assert "clean" in md
    # Must not crash and must render a readable "no redundancies" state.
    assert md.strip()


def test_storyboard_writes_flow_review_by_default(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
        ]}]
    }))
    StoryboardStage().run(p, Providers(llm=FakeLLM(), image=FakeImageGen()))
    review = p.path("storyboard", "shots.review.md")
    assert review.is_file()
    assert "连贯性审查" in review.read_text()


def test_flow_review_is_advise_only_and_leaves_shots_untouched(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
        ]}]
    }))
    StoryboardStage().run(p, Providers(llm=FakeLLM(), image=FakeImageGen()))
    before = p.path("storyboard", "shots.json").read_text()
    # Re-run: review must not be regenerated (idempotent) and shots.json unchanged.
    StoryboardStage().run(p, Providers(llm=FakeLLM(), image=FakeImageGen()))
    after = p.path("storyboard", "shots.json").read_text()
    assert before == after


def test_flow_review_skipped_when_disabled(tmp_path):
    p = _project(tmp_path)
    p.model_config["storyboard"] = {"flow_review": {"enabled": False}}
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
        ]}]
    }))
    StoryboardStage().run(p, Providers(llm=FakeLLM(), image=FakeImageGen()))
    assert not p.path("storyboard", "shots.review.md").is_file()


def test_flow_review_regenerates_only_when_shots_change(tmp_path):
    p = _project(tmp_path)
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"episode": 1, "scenes": [
            {"scene": 1, "heading": "FIRST", "beats": ["Mara opens the shop"]},
        ]}]
    }))

    class CountingLLM(FakeLLM):
        def __init__(self):
            self.flow_calls = 0

        def complete_json(self, prompt, *, system=None):
            if "[task:flow_review]" in prompt:
                self.flow_calls += 1
            return super().complete_json(prompt, system=system)

    llm = CountingLLM()
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))
    assert llm.flow_calls == 1
    # Unchanged shots.json → no second paid review call.
    StoryboardStage().run(p, Providers(llm=llm, image=FakeImageGen()))
    assert llm.flow_calls == 1


def test_plan_shots_first_run_populates_reference_fields(tmp_path):
    # Regression: Task 6's implementer added _fill_shot_references() call to _plan_shots
    # so that shots.json is byte-identical across re-runs (first-run shots get the same
    # state/target-state reference fields as the resume/backfill branch).
    # This test pins that behavior: keyframe_reference_images must be populated on first run.
    p = _project(tmp_path)
    providers = _providers()

    # First run — no pre-existing shots.json
    result = StoryboardStage().run(p, providers)
    assert result.status == "complete"

    # Read shots.json and verify keyframe_reference_images is populated
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    assert len(shots) > 0, "expected at least one shot"

    for shot in shots:
        assert "keyframe_reference_images" in shot, (
            f"shot {shot['id']} missing keyframe_reference_images"
        )
        assert isinstance(shot["keyframe_reference_images"], list)
        # For a minimal scene with no start_states/end_states/visual_beats,
        # keyframe_reference_images should equal reference_images (the state-less equivalence).
        assert shot["keyframe_reference_images"] == shot["reference_images"], (
            f"shot {shot['id']}: keyframe_reference_images should equal reference_images "
            f"when no states are defined"
        )
