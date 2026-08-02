"""Tests for the video stage — image-to-video clip per shot with last-frame carry."""

import json
from pathlib import Path

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_approvals import PromptApprovalError, confirm_prompt_batch
from studio_agent.providers.base import Generation, VideoCapabilities
from studio_agent.providers.fake import FakeImageGen, FakeLLM, FakeVideoGen
from studio_agent.reference_assets import save_reference_upload
from studio_agent.stages.base import Providers
from studio_agent.stages.video import VideoStage
from studio_agent.stages.video_prompts import VideoPromptsStage

PIPELINE = ["storyboard", "video"]

SHOTS = {
    "shots": [
        {"id": "sh-001", "scene": 1, "description": "wide", "camera": "wide", "action": "open",
         "dialogue": [], "duration_s": 3.0, "characters": ["Mara"], "deps": [],
         "reference_seed": 11, "keyframe": "sh-001.png"},
        {"id": "sh-002", "scene": 1, "description": "cu", "camera": "close-up", "action": "react",
         "dialogue": [], "duration_s": 2.0, "characters": ["Mara"], "deps": ["sh-001"],
         "reference_seed": 11, "keyframe": "sh-002.png"},
    ]
}


def _project(tmp_path):
    p = Project.create("clip test", root=tmp_path, stages=PIPELINE)
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    # Keyframes the video stage animates from.
    for shot in SHOTS["shots"]:
        kf = p.path("storyboard", "keyframes", shot["keyframe"])
        FakeImageGen().generate("kf", out_path=str(kf), seed=shot["reference_seed"])
    return p


def _styled_project(tmp_path):
    p = Project.create(
        "styled clip test",
        root=tmp_path,
        stages=PIPELINE,
        model_config={"style": {
            "look": "anime",
            "palette": "moonlit teal and warm brass",
            "rendering": "clean cel shading",
            "motion": "controlled limited-animation timing",
        }},
    )
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    for shot in SHOTS["shots"]:
        kf = p.path("storyboard", "keyframes", shot["keyframe"])
        FakeImageGen().generate("kf", out_path=str(kf), seed=shot["reference_seed"])
    return p


def _single_shot_project(tmp_path):
    project = _project(tmp_path)
    project.path("storyboard", "shots.json").write_text(
        json.dumps({"shots": [SHOTS["shots"][0]]})
    )
    return project


def _providers():
    return Providers(video=FakeVideoGen())


def _run_approved_video(project, providers, *, stage=None):
    VideoPromptsStage().run(project, providers)
    confirm_prompt_batch(project, "videos", confirmer="test")
    return (stage or VideoStage()).run(project, providers)


class RecordingVideoGen(FakeVideoGen):
    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, *, out_path: str, **kwargs):
        self.calls.append({"out_path": out_path, "prompt": prompt, **kwargs})
        return super().generate(prompt, out_path=out_path, **kwargs)


class CappedRecordingVideoGen(RecordingVideoGen):
    def __init__(
        self,
        max_image_inputs: int,
        *,
        supports_reference_images: bool = True,
        supports_last_frame: bool = True,
    ):
        super().__init__()
        self._capabilities = VideoCapabilities(
            supports_reference_images=supports_reference_images,
            supports_last_frame=supports_last_frame,
            max_image_inputs=max_image_inputs,
        )

    @property
    def capabilities(self):
        return self._capabilities

    def generate(self, prompt: str, *, out_path: str, **kwargs):
        refs = list(kwargs.get("reference_images") or [])
        target_refs = list(kwargs.get("target_state_reference_images") or [])
        assert set(target_refs).issubset(refs)
        image_inputs = [kwargs.get("keyframe_path"), *refs]
        if kwargs.get("last_frame_ref"):
            image_inputs.append(kwargs["last_frame_ref"])
        assert len(image_inputs) <= self.capabilities.max_image_inputs
        return super().generate(prompt, out_path=out_path, **kwargs)


class ConflictingPromptLLM(FakeLLM):
    def complete_json(self, prompt, *, system=None):
        if "[task:video_prompt]" in prompt:
            return Generation(
                content={
                    "prompt": "locked-off camera, then orbit around Mara",
                    "negative": "",
                },
                provider="fake-llm",
                model="fake-conflict",
            )
        return super().complete_json(prompt, system=system)


def test_video_requires_approved_existing_prompts(tmp_path):
    project = _single_shot_project(tmp_path)
    video = RecordingVideoGen()

    with pytest.raises(PromptApprovalError):
        VideoStage().run(project, Providers(video=video))

    assert video.calls == []


def test_video_uses_approved_prompt_verbatim(tmp_path):
    project = _single_shot_project(tmp_path)
    video = RecordingVideoGen()
    prompt = "EXACT APPROVED MOTION PROMPT"
    project.path("storyboard", "prompts", "sh-001.video.md").write_text(prompt)
    confirm_prompt_batch(project, "videos", confirmer="human")

    VideoStage().run(project, Providers(video=video))

    assert video.calls[0]["prompt"] == prompt


def test_video_does_not_invoke_llm_when_prompt_is_missing(tmp_path):
    project = _single_shot_project(tmp_path)
    video = RecordingVideoGen()

    class RecordingLLM(FakeLLM):
        def __init__(self):
            self.calls = []

        def complete_json(self, prompt, *, system=None):
            self.calls.append(prompt)
            return super().complete_json(prompt, system=system)

    llm = RecordingLLM()
    with pytest.raises(PromptApprovalError):
        VideoStage().run(project, Providers(video=video, llm=llm))

    assert llm.calls == []
    assert video.calls == []


def test_video_prompt_file_includes_camera_recipe_text(tmp_path):
    from studio_agent.cinematography_recipes import fill_recipe_slots, load_camera_recipes

    recipe_id = next(iter(load_camera_recipes()))
    recipe_prompt = load_camera_recipes()[recipe_id]["prompt_en"]

    p = Project.create("recipe clip", root=tmp_path, stages=PIPELINE)
    shots = {"shots": [
        {"id": "sh-001", "scene": 1, "description": "wide", "camera": "wide",
         "action": "open", "dialogue": [], "duration_s": 3.0, "characters": ["Mara"],
         "deps": [], "reference_seed": 11, "keyframe": "sh-001.png",
         "camera_recipe": recipe_id},
    ]}
    p.path("storyboard", "shots.json").write_text(json.dumps(shots))
    kf = p.path("storyboard", "keyframes", "sh-001.png")
    FakeImageGen().generate("kf", out_path=str(kf), seed=11)

    _run_approved_video(p, _providers())

    brief = p.path("storyboard", "prompts", "sh-001.video.brief.md").read_text()
    # The recipe text is woven in with its [slots] filled from the shot (subject "Mara"),
    # so the filled form appears and no raw placeholder token leaks through.
    assert fill_recipe_slots(recipe_prompt, subject="Mara") in brief
    assert "[the subject]" not in brief


def test_video_sends_directed_prompt_to_video_model(tmp_path):
    p = _project(tmp_path)
    rec = RecordingVideoGen()

    _run_approved_video(p, Providers(video=rec, llm=FakeLLM()))

    final = p.path("storyboard", "prompts", "sh-001.video.md").read_text()
    brief = p.path("storyboard", "prompts", "sh-001.video.brief.md").read_text()
    sent = next(c for c in rec.calls if c["out_path"].endswith("sh-001.mp4"))["prompt"]
    # The model receives the directed prompt (== video.md), not the raw brief.
    assert sent == final
    assert final != brief


def test_directed_video_prompt_is_trusted_and_sent_without_a_qc_gate(tmp_path):
    # The deterministic QC gate is gone: the director's output is trusted and sent
    # straight to the provider — no prompt-qc artifact, no pre-paid rejection.
    p = _project(tmp_path)
    rec = RecordingVideoGen()

    _run_approved_video(p, Providers(video=rec, llm=ConflictingPromptLLM()))

    sent = next(c for c in rec.calls if c["out_path"].endswith("sh-001.mp4"))["prompt"]
    final = p.path("storyboard", "prompts", "sh-001.video.md").read_text()
    assert sent == final
    assert not p.path("storyboard", "prompts", "sh-001.prompt-qc.json").exists()


def test_video_writes_a_clip_per_shot_and_a_manifest(tmp_path):
    p = _project(tmp_path)

    result = _run_approved_video(p, _providers())

    assert result.status == "complete"
    for shot in SHOTS["shots"]:
        assert p.path("assets", "clips", f"{shot['id']}.mp4").is_file()

    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())["clips"]
    assert [c["id"] for c in manifest] == ["sh-001", "sh-002"]
    assert manifest[0]["duration_s"] == 3.0


def test_video_writes_editable_video_prompt(tmp_path):
    p = _project(tmp_path)
    _run_approved_video(p, _providers())
    prompt = p.path("storyboard", "prompts", "sh-001.video.md")
    assert prompt.is_file()
    assert len(prompt.read_text()) > 0


class RestrictedRefVideoGen(FakeVideoGen):
    """A fake whose capabilities mirror the BytePlus international profile: it cannot mix
    a first-frame keyframe with extra reference media or a carried last frame."""

    @property
    def capabilities(self):
        return VideoCapabilities(
            supports_reference_images=False, supports_last_frame=False
        )


def test_video_prompt_has_motion_beats_and_avoids_static(tmp_path):
    p = _project(tmp_path)

    _run_approved_video(p, _providers())

    prompt = p.path("storyboard", "prompts", "sh-001.video.brief.md").read_text()
    assert "Beat 1" in prompt
    assert "Camera" in prompt
    assert "avoid" in prompt.lower()
    assert "still image" in prompt.lower()


def test_video_prompt_respects_restricted_provider_capabilities(tmp_path):
    p = _project(tmp_path)

    # sh-002 has a previous shot, so an unrestricted provider would add continuity +
    # reference-mixing language. The restricted provider must not.
    _run_approved_video(p, Providers(video=RestrictedRefVideoGen()))

    prompt = p.path("storyboard", "prompts", "sh-002.video.brief.md").read_text().lower()
    assert "previous shot" not in prompt
    assert "reference image" not in prompt
    assert "keyframe" in prompt  # identity still anchored on the starting frame
    assert "beat 1" in prompt


def test_video_preserves_hand_edited_prompt(tmp_path):
    p = _project(tmp_path)
    prompt_path = p.path("storyboard", "prompts", "sh-001.video.md")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    edited = (
        "HAND EDITED PROMPT — keep me. Mara opens the door. Beat 1 establishes her, "
        "then Beat 2 resolves the action. Gentle push-in motivated by the opening door. "
        "50mm lens at eye-level with motivated window light. No voices or dialogue."
    )
    prompt_path.write_text(edited)

    _run_approved_video(p, _providers())

    assert prompt_path.read_text() == edited


def test_video_prompt_includes_style_guidance(tmp_path):
    p = _styled_project(tmp_path)

    _run_approved_video(p, _providers())

    prompt = p.path("storyboard", "prompts", "sh-001.video.brief.md").read_text()
    assert "anime" in prompt
    assert "clean cel shading" in prompt
    assert "controlled limited-animation timing" in prompt


def test_video_prompt_includes_story_and_neighbor_context(tmp_path):
    p = _project(tmp_path)
    p.path("story", "idea.md").write_text("A lighthouse keeper trusts a warning gull.")
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "A keeper must believe a gull before the storm arrives.",
        "synopsis": "The warning interrupts an ordinary morning.",
        "themes": ["trust", "survival"],
    }))
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{
            "episode": 1,
            "scenes": [{
                "scene": 1,
                "heading": "EXT. LIGHTHOUSE - DAWN",
                "beats": ["The keeper opens the door", "The gull reacts"],
            }],
        }]
    }))

    _run_approved_video(p, _providers())

    prompt = p.path("storyboard", "prompts", "sh-001.video.brief.md").read_text()
    assert "Story beat" in prompt
    assert "keeper must believe" in prompt
    assert "EXT. LIGHTHOUSE - DAWN" in prompt
    assert "react" in prompt


def test_video_passes_existing_reference_images_to_provider(tmp_path):
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    ref = p.path("bible", "characters", "mara", "reference.png")
    ref.parent.mkdir(parents=True, exist_ok=True)
    FakeImageGen().generate("mara ref", out_path=str(ref), seed=11)
    shots[0]["reference_images"] = [
        "bible/characters/mara/reference.png",
        "bible/characters/mara/missing.png",
    ]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    rec = RecordingVideoGen()

    _run_approved_video(p, Providers(video=rec))

    assert rec.calls[0]["reference_images"] == [str(ref)]


def test_video_refreshes_live_upload_added_after_shots_json(tmp_path):
    p = _project(tmp_path)
    record = save_reference_upload(
        p,
        filename="mara-upload.png",
        data=b"\x89PNG\r\n\x1a\nraw",
        target_type="character",
        target_id="Mara",
    )
    rec = RecordingVideoGen()

    _run_approved_video(p, Providers(video=rec))

    assert str(p.dir / record["path"]) in rec.calls[0]["reference_images"]


def test_video_passes_target_state_reference_images_to_provider(tmp_path):
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    target = p.path("bible", "characters", "mara", "states", "dragon", "reference.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    FakeImageGen().generate("dragon ref", out_path=str(target), seed=99)
    shots[0]["target_state_references"] = [
        {
            "character": "Mara",
            "state": "dragon",
            "image": "bible/characters/mara/states/dragon/reference.png",
        }
    ]
    shots[0]["target_state_reference_images"] = [
        "bible/characters/mara/states/dragon/reference.png"
    ]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    rec = RecordingVideoGen()

    _run_approved_video(p, Providers(video=rec))

    assert str(target) in rec.calls[0]["reference_images"]
    assert rec.calls[0]["target_state_reference_images"] == [str(target)]


def test_video_cap_prioritizes_named_then_bible_sheet(tmp_path):
    # Priority under a tight cap: explicitly named refs first, then the derived bible sheet
    # (the downstream identity anchor, invariant #4), then the raw upload that fed the bible.
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    raw = save_reference_upload(
        p,
        filename="mara-upload.png",
        data=b"\x89PNG\r\n\x1a\nraw",
        target_type="character",
        target_id="Mara",
    )
    named = p.path("bible", "characters", "mara", "named.png")
    derived = p.path("bible", "characters", "mara", "reference.png")
    style = p.path("bible", "style_sample.png")
    for path, seed in ((named, 21), (derived, 22), (style, 23)):
        FakeImageGen().generate("ref", out_path=str(path), seed=seed)
    shots[0]["reference_images"] = [
        "bible/characters/mara/reference.png",
        "bible/characters/mara/named.png",
        "bible/style_sample.png",
    ]
    shots[0]["named_reference_images"] = ["bible/characters/mara/named.png"]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    rec = CappedRecordingVideoGen(max_image_inputs=3)

    _run_approved_video(p, Providers(video=rec))

    # capacity 2 -> named sheet + derived bible sheet win; the raw upload is shed last.
    assert rec.calls[0]["reference_images"] == [
        str(named),
        str(derived),
    ]
    assert str(p.dir / raw["path"]) not in rec.calls[0]["reference_images"]


def test_video_cap_reserves_keyframe_and_last_frame_before_extra_references(tmp_path):
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    ref = p.path("bible", "characters", "mara", "reference.png")
    FakeImageGen().generate("ref", out_path=str(ref), seed=31)
    for shot in shots:
        shot["reference_images"] = ["bible/characters/mara/reference.png"]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    rec = CappedRecordingVideoGen(max_image_inputs=2)

    _run_approved_video(p, Providers(video=rec))

    assert rec.calls[0]["reference_images"] == [str(ref)]
    assert rec.calls[0]["last_frame_ref"] is None
    assert rec.calls[1]["reference_images"] == []
    assert rec.calls[1]["last_frame_ref"].endswith("sh-001.last_frame.png")


def test_video_prompt_does_not_claim_carry_when_cap_has_no_slot(tmp_path):
    p = _project(tmp_path)
    rec = CappedRecordingVideoGen(max_image_inputs=1)

    _run_approved_video(p, Providers(video=rec))

    prompt = p.path("storyboard", "prompts", "sh-002.video.brief.md").read_text().lower()
    assert "begin on the previous final frame" not in prompt
    assert rec.calls[1]["last_frame_ref"] is None


def test_video_cap_prioritizes_target_state_before_other_derived_refs(tmp_path):
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    base = p.path("bible", "characters", "mara", "reference.png")
    location = p.path("bible", "locations", "clock-shop", "reference.png")
    target = p.path("bible", "characters", "mara", "states", "dragon", "reference.png")
    for path, seed in ((base, 31), (location, 32), (target, 33)):
        FakeImageGen().generate("ref", out_path=str(path), seed=seed)
    shots[0]["reference_images"] = [
        "bible/characters/mara/reference.png",
        "bible/locations/clock-shop/reference.png",
    ]
    shots[0]["target_state_reference_images"] = [
        "bible/characters/mara/states/dragon/reference.png"
    ]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    rec = CappedRecordingVideoGen(max_image_inputs=2)

    _run_approved_video(p, Providers(video=rec))

    assert rec.calls[0]["reference_images"] == [str(target)]
    assert rec.calls[0]["target_state_reference_images"] == [str(target)]


def test_video_prompt_distinguishes_selected_and_dropped_target_states(tmp_path):
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    dragon = p.path("bible", "characters", "mara", "states", "dragon", "reference.png")
    wolf = p.path("bible", "characters", "theo", "states", "wolf", "reference.png")
    FakeImageGen().generate("dragon", out_path=str(dragon), seed=34)
    FakeImageGen().generate("wolf", out_path=str(wolf), seed=35)
    shots[0]["target_state_references"] = [
        {
            "character": "Mara",
            "state": "dragon",
            "image": "bible/characters/mara/states/dragon/reference.png",
        },
        {
            "character": "Theo",
            "state": "wolf",
            "image": "bible/characters/theo/states/wolf/reference.png",
        },
    ]
    shots[0]["target_state_reference_images"] = [
        "bible/characters/mara/states/dragon/reference.png",
        "bible/characters/theo/states/wolf/reference.png",
    ]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    rec = CappedRecordingVideoGen(max_image_inputs=2)

    _run_approved_video(p, Providers(video=rec))

    assert rec.calls[0]["target_state_reference_images"] == [str(dragon)]
    prompt = p.path("storyboard", "prompts", "sh-001.video.brief.md").read_text()
    media_section = prompt.split("Condition on the supplied target-state media", 1)[0]
    assert "Mara: dragon" in media_section
    assert "Theo: wolf" not in media_section
    assert "Theo: wolf" in prompt
    assert "without supplied target-state media" in prompt.lower()


def test_video_cap_filters_dropped_target_state_references(tmp_path):
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    raw = save_reference_upload(
        p,
        filename="mara-upload.png",
        data=b"\x89PNG\r\n\x1a\nraw",
        target_type="character",
        target_id="Mara",
    )
    named = p.path("bible", "characters", "mara", "named.png")
    target = p.path("bible", "characters", "mara", "states", "dragon", "reference.png")
    FakeImageGen().generate("named", out_path=str(named), seed=41)
    FakeImageGen().generate("target", out_path=str(target), seed=42)
    shots[0]["reference_images"] = ["bible/characters/mara/named.png"]
    shots[0]["named_reference_images"] = ["bible/characters/mara/named.png"]
    shots[0]["target_state_reference_images"] = [
        "bible/characters/mara/states/dragon/reference.png"
    ]
    shots[0]["target_state_references"] = [
        {
            "character": "Mara",
            "state": "dragon",
            "image": "bible/characters/mara/states/dragon/reference.png",
        }
    ]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    # capacity 1 (only the named ref fits): the target-state anchor and raw upload are both
    # shed, and a dropped target-state must be pruned from the media list + prompt.
    rec = CappedRecordingVideoGen(max_image_inputs=2)

    _run_approved_video(p, Providers(video=rec))

    assert rec.calls[0]["reference_images"] == [str(named)]
    assert rec.calls[0]["target_state_reference_images"] == []
    prompt = p.path("storyboard", "prompts", "sh-001.video.brief.md").read_text().lower()
    assert "condition on the supplied target-state media" not in prompt
    assert "provider cannot receive extra target-state media" in prompt


def test_video_omits_extra_references_when_provider_does_not_support_them(tmp_path):
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())["shots"]
    target = p.path("bible", "characters", "mara", "states", "dragon", "reference.png")
    FakeImageGen().generate("target", out_path=str(target), seed=51)
    shots[0]["reference_images"] = [
        "bible/characters/mara/states/dragon/reference.png"
    ]
    shots[0]["target_state_reference_images"] = [
        "bible/characters/mara/states/dragon/reference.png"
    ]
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    rec = CappedRecordingVideoGen(
        max_image_inputs=1,
        supports_reference_images=False,
        supports_last_frame=False,
    )

    _run_approved_video(p, Providers(video=rec))

    assert all(call["reference_images"] == [] for call in rec.calls)
    assert all(call["target_state_reference_images"] == [] for call in rec.calls)


def test_video_carries_last_frame_forward(tmp_path):
    p = _project(tmp_path)
    _run_approved_video(p, _providers())
    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())["clips"]

    # First clip has no carried frame; the second carries the first shot's *real* last
    # frame artifact (not its keyframe / first frame).
    assert manifest[0]["last_frame_ref"] in (None, "")
    assert manifest[1]["last_frame_ref"].endswith("sh-001.last_frame.png")


def test_video_saves_a_last_frame_artifact_per_shot(tmp_path):
    p = _project(tmp_path)
    _run_approved_video(p, _providers())
    for shot in SHOTS["shots"]:
        artifact = p.path("assets", "clips", f"{shot['id']}.last_frame.png")
        assert artifact.is_file()
        assert artifact.stat().st_size > 0


def test_video_feeds_previous_last_frame_into_next_generation(tmp_path):
    p = _project(tmp_path)
    rec = RecordingVideoGen()

    _run_approved_video(p, Providers(video=rec))

    # First shot has nothing to carry; the second is conditioned on shot one's last frame.
    assert rec.calls[0]["last_frame_ref"] is None
    assert rec.calls[1]["last_frame_ref"].endswith("sh-001.last_frame.png")


def test_video_prefers_provider_returned_last_frame(tmp_path):
    p = _project(tmp_path)

    class UrlVideoGen(FakeVideoGen):
        def generate(self, prompt, *, out_path, **kwargs):
            gen = super().generate(prompt, out_path=out_path, **kwargs)
            gen.meta["last_frame_url"] = "https://example.test/last.png"
            return gen

    def fake_downloader(url, out_path):
        Path(out_path).write_bytes(b"DOWNLOADED:" + url.encode())
        return True

    _run_approved_video(
        p,
        Providers(video=UrlVideoGen()),
        stage=VideoStage(downloader=fake_downloader),
    )

    artifact = p.path("assets", "clips", "sh-001.last_frame.png")
    assert artifact.read_bytes().startswith(b"DOWNLOADED:")


def test_video_extracts_last_frame_from_real_mp4_when_no_url(tmp_path):
    p = _project(tmp_path)

    class Mp4VideoGen(FakeVideoGen):
        def generate(self, prompt, *, out_path, **kwargs):
            gen = super().generate(prompt, out_path=out_path, **kwargs)
            Path(out_path).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
            return gen

    def fake_extractor(clip_path, out_path):
        Path(out_path).write_bytes(b"EXTRACTED:" + Path(clip_path).name.encode())
        return True

    _run_approved_video(
        p,
        Providers(video=Mp4VideoGen()),
        stage=VideoStage(frame_extractor=fake_extractor),
    )

    artifact = p.path("assets", "clips", "sh-001.last_frame.png")
    assert artifact.read_bytes().startswith(b"EXTRACTED:")


def test_video_omits_last_frame_ref_when_provider_cannot_carry(tmp_path):
    p = _project(tmp_path)

    class RestrictedRecordingVideoGen(RecordingVideoGen):
        @property
        def capabilities(self):
            return VideoCapabilities(supports_last_frame=False)

    rec = RestrictedRecordingVideoGen()
    _run_approved_video(p, Providers(video=rec))

    # The provider cannot accept a carried last frame, so we never feed one in...
    assert all(call["last_frame_ref"] is None for call in rec.calls)
    # ...but the honest last-frame artifact is still stored for inspection/QC.
    assert p.path("assets", "clips", "sh-001.last_frame.png").is_file()


def test_video_logs_cost(tmp_path):
    p = _project(tmp_path)
    _run_approved_video(p, _providers())
    assert sum(1 for e in p.cost_log if e["stage"] == "video") == len(SHOTS["shots"])


def test_video_is_idempotent(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    _run_approved_video(p, providers)
    p.set_stage_status("video", "complete")
    cost_after_first = len(p.cost_log)

    result = VideoStage().run(p, providers)
    assert result.status == "skipped"
    assert len(p.cost_log) == cost_after_first


def test_video_regenerates_missing_clip_only(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    _run_approved_video(p, providers)
    cost_after_first = len(p.cost_log)

    clip = p.path("assets", "clips", "sh-001.mp4")
    clip.unlink()
    result = VideoStage().run(p, providers)

    assert result.status == "complete"
    assert clip.is_file()
    assert len(p.cost_log) == cost_after_first + 1  # only the one clip re-rendered


def test_native_mode_passes_generate_audio_and_records_fake_sidecar(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    rec = RecordingVideoGen()

    _run_approved_video(p, Providers(video=rec))

    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())["clips"]
    assert rec.calls[0]["generate_audio"] is True
    assert manifest[0]["native_audio"] == "sh-001.native.source.wav"
    assert p.path("assets", "clips", manifest[0]["native_audio"]).is_file()


def test_native_mode_rejects_unsupported_provider_before_generation(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"

    class Unsupported(RecordingVideoGen):
        @property
        def capabilities(self):
            return VideoCapabilities(supports_native_audio=False)

    provider = Unsupported()

    with pytest.raises(ValueError, match="native audio"):
        _run_approved_video(p, Providers(video=provider))

    assert provider.calls == []


def test_native_mode_discovers_existing_sidecar_on_resume(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    rec = RecordingVideoGen()
    clips_dir = p.path("assets", "clips")
    clips_dir.joinpath("sh-001.mp4").write_bytes(b"existing clip")
    clips_dir.joinpath("sh-001.native.source.wav").write_bytes(b"existing audio")

    _run_approved_video(p, Providers(video=rec))

    manifest = json.loads(clips_dir.joinpath("clips.json").read_text())["clips"]
    assert rec.calls[0]["out_path"].endswith("sh-002.mp4")
    assert manifest[0]["native_audio"] == "sh-001.native.source.wav"


def test_native_mode_records_sidecars_with_relative_project_root_on_fresh_run_and_resume(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    p = Project.create(
        "relative root clip test",
        root=Path("projects"),
        stages=PIPELINE,
        model_config={"audio_mode": "native_video"},
    )
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    for shot in SHOTS["shots"]:
        FakeImageGen().generate(
            "kf",
            out_path=str(p.path("storyboard", "keyframes", shot["keyframe"])),
            seed=shot["reference_seed"],
        )

    providers = Providers(video=FakeVideoGen())
    _run_approved_video(p, providers)
    manifest_path = p.path("assets", "clips", "clips.json")
    fresh = json.loads(manifest_path.read_text())["clips"]
    assert fresh[0]["native_audio"] == "sh-001.native.source.wav"

    rec = RecordingVideoGen()
    VideoStage().run(p, Providers(video=rec))
    resumed = json.loads(manifest_path.read_text())["clips"]
    assert rec.calls == []
    assert resumed[0]["native_audio"] == "sh-001.native.source.wav"


def test_native_mode_rejects_provider_sidecar_outside_project(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"unsafe audio")

    class UnsafeSidecar(FakeVideoGen):
        def generate(self, prompt, *, out_path, **kwargs):
            gen = super().generate(prompt, out_path=out_path, **kwargs)
            gen.meta["native_audio_path"] = str(outside)
            return gen

    with pytest.raises(ValueError, match="outside project"):
        _run_approved_video(p, Providers(video=UnsafeSidecar()))


def test_native_mode_rejects_provider_sidecar_outside_clips_directory(tmp_path):
    p = _project(tmp_path)
    p.model_config["audio_mode"] = "native_video"
    misplaced = p.path("assets", "audio", "sh-001.native.source.wav")
    misplaced.write_bytes(b"misplaced audio")

    class MisplacedSidecar(FakeVideoGen):
        def generate(self, prompt, *, out_path, **kwargs):
            gen = super().generate(prompt, out_path=out_path, **kwargs)
            gen.meta["native_audio_path"] = str(misplaced)
            return gen

    with pytest.raises(ValueError, match="assets/clips"):
        _run_approved_video(p, Providers(video=MisplacedSidecar()))


def test_legacy_mode_does_not_request_or_record_native_audio(tmp_path):
    p = _project(tmp_path)
    rec = RecordingVideoGen()

    _run_approved_video(p, Providers(video=rec))

    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())["clips"]
    assert rec.calls[0]["generate_audio"] is False
    assert "native_audio" not in manifest[0]


def test_legacy_mode_ignores_provider_native_audio_metadata(tmp_path):
    p = _project(tmp_path)
    outside = tmp_path / "legacy-provider.wav"
    outside.write_bytes(b"legacy provider audio")

    class LegacySidecar(FakeVideoGen):
        def generate(self, prompt, *, out_path, **kwargs):
            gen = super().generate(prompt, out_path=out_path, **kwargs)
            gen.meta["native_audio_path"] = str(outside)
            return gen

    _run_approved_video(p, Providers(video=LegacySidecar()))

    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())["clips"]
    assert "native_audio" not in manifest[0]
