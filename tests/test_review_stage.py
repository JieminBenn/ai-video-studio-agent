"""Tests for the review (QC) stage — a report per clip, gate blocks on severe failure."""

import json
from pathlib import Path

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.providers.base import (
    Generation,
    QC_DIMENSIONS,
    VLMCheck,
    qc_failure_blocks,
)
from studio_agent.providers.fake import FakeImageGen, FakeVideoGen, FakeVLMCheck
from studio_agent.stages.base import Providers
from studio_agent.stages.review import ReviewStage

PIPELINE = ["video", "review"]

SHOTS = {"shots": [
    {"id": "sh-001", "duration_s": 2.0, "reference_seed": 1, "keyframe": "sh-001.png",
     "keyframe_prompt": "wide shot", "description": "wide"},
    {"id": "sh-002", "duration_s": 2.0, "reference_seed": 1, "keyframe": "sh-002.png",
     "keyframe_prompt": "close up", "description": "cu"},
]}

EXPECTED_AUDIO_QC_DIMENSIONS = (
    "dialogue_accuracy", "music_absence", "sound_design", "audio_sync"
)


def _project(tmp_path, *, audio_mode="", with_native_audio=False):
    model_config = {"audio_mode": audio_mode} if audio_mode else None
    p = Project.create(
        "qc test", root=tmp_path, stages=PIPELINE, model_config=model_config
    )
    p.path("storyboard", "shots.json").write_text(json.dumps(SHOTS))
    clips = []
    for shot in SHOTS["shots"]:
        kf = p.path("storyboard", "keyframes", shot["keyframe"])
        FakeImageGen().generate("kf", out_path=str(kf), seed=shot["reference_seed"])
        clip = p.path("assets", "clips", f"{shot['id']}.mp4")
        generation = FakeVideoGen().generate(
            "v", out_path=str(clip), keyframe_path=str(kf),
            seed=shot["reference_seed"], duration_s=shot["duration_s"],
            generate_audio=with_native_audio,
        )
        entry = {"id": shot["id"], "clip": clip.name, "duration_s": shot["duration_s"]}
        if generation.meta.get("native_audio_path"):
            entry["native_audio"] = Path(generation.meta["native_audio_path"]).name
        clips.append(entry)
    p.path("assets", "clips", "clips.json").write_text(json.dumps({"clips": clips}))
    return p


def _providers():
    return Providers(vlm=FakeVLMCheck())


def test_review_writes_a_qc_report_per_clip(tmp_path):
    p = _project(tmp_path)

    result = ReviewStage().run(p, _providers())

    assert result.status == "complete"
    for shot in SHOTS["shots"]:
        report = p.path("assets", "qc", f"{shot['id']}.json")
        assert report.is_file()
        data = json.loads(report.read_text())
        assert data["overall_pass"] is True
        assert len(data["checks"]) == len(QC_DIMENSIONS)


def test_review_blocks_when_a_clip_fails_qc(tmp_path):
    p = _project(tmp_path)
    # Corrupt one clip so QC flags it.
    p.path("assets", "clips", "sh-002.mp4").write_bytes(b"FAKECLIP\nQC_FAIL\n")

    result = ReviewStage().run(p, _providers())

    assert result.status == "failed"
    report = json.loads(p.path("assets", "qc", "sh-002.json").read_text())
    assert report["overall_pass"] is False
    assert report["max_severity"] == "high"


def test_review_logs_cost_per_clip(tmp_path):
    p = _project(tmp_path)
    ReviewStage().run(p, _providers())
    assert sum(1 for e in p.cost_log if e["stage"] == "review") == len(SHOTS["shots"])


def test_review_is_idempotent(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    ReviewStage().run(p, providers)
    p.set_stage_status("review", "complete")
    cost_after_first = len(p.cost_log)

    result = ReviewStage().run(p, providers)
    assert result.status == "skipped"
    assert len(p.cost_log) == cost_after_first


def test_review_rereviews_when_report_missing(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    ReviewStage().run(p, providers)
    cost_after_first = len(p.cost_log)

    p.path("assets", "qc", "sh-001.json").unlink()
    result = ReviewStage().run(p, providers)

    assert result.status == "complete"
    assert p.path("assets", "qc", "sh-001.json").is_file()
    assert len(p.cost_log) == cost_after_first + 1  # only the missing report re-reviewed


def test_review_refreshes_stale_report_when_clip_changes(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    ReviewStage().run(p, providers)
    cost_after_first = len(p.cost_log)

    # Regenerate a clip with different content; its old report is now stale.
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"FAKECLIP\nQC_FAIL\n")
    ReviewStage().run(p, providers)

    assert len(p.cost_log) == cost_after_first + 1
    assert json.loads(p.path("assets", "qc", "sh-001.json").read_text())["overall_pass"] is False


def test_qc_dimensions_prioritize_story_and_instructions():
    assert list(QC_DIMENSIONS) == [
        "story_alignment",
        "shot_instruction_adherence",
        "continuity",
        "identity_drift",
        "motion_anatomy",
        "artifacts",
        "dialogue_accuracy",
        "music_absence",
        "sound_design",
        "audio_sync",
        "safety",
    ]


@pytest.mark.parametrize(
    "dimension", ["dialogue_accuracy", "music_absence", "audio_sync"]
)
def test_native_audio_policy_failures_block(dimension):
    assert qc_failure_blocks({
        "dimension": dimension, "passed": False, "severity": "high"
    }) is True


def test_low_sound_design_issue_is_warning():
    assert qc_failure_blocks({
        "dimension": "sound_design", "passed": False, "severity": "low"
    }) is False


def test_medium_sound_design_issue_blocks():
    assert qc_failure_blocks({
        "dimension": "sound_design", "passed": False, "severity": "medium"
    }) is True


class CapturingVLM(VLMCheck):
    name = "capture"
    model = "capture-vlm"

    def __init__(self):
        self.prompts = []

    @property
    def supports_audio_review(self):
        return True

    def review(self, clip_path: str, *, prompt: str, **kwargs):
        self.prompts.append(prompt)
        checks = [
            {
                "dimension": dim,
                "passed": True,
                "severity": "none",
                "detail": "ok",
                "timestamp": None,
            }
            for dim in QC_DIMENSIONS
        ]
        return Generation(
            content={
                "clip": "clip.mp4",
                "prompt": prompt,
                "checks": checks,
                "overall_pass": True,
                "max_severity": "none",
                "recommendation": "approve",
            },
            provider=self.name,
            model=self.model,
        )


class CapabilityVLM(CapturingVLM):
    name = "switchable"
    model = "same-model"

    def __init__(self, supports_audio_review):
        super().__init__()
        self._supports_audio_review = supports_audio_review

    @property
    def supports_audio_review(self):
        return self._supports_audio_review


def test_native_review_cache_invalidates_when_reviewer_loses_audio_capability(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    capable = CapabilityVLM(True)
    assert ReviewStage().run(p, Providers(vlm=capable)).status == "complete"

    visual_only = CapabilityVLM(False)
    result = ReviewStage().run(p, Providers(vlm=visual_only))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())

    assert result.status == "failed"
    assert len(visual_only.prompts) == len(SHOTS["shots"])
    assert report["vlm_supports_audio_review"] is False


def test_native_review_cache_invalidates_when_reviewer_gains_audio_capability(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    visual_only = CapabilityVLM(False)
    assert ReviewStage().run(p, Providers(vlm=visual_only)).status == "failed"

    capable = CapabilityVLM(True)
    result = ReviewStage().run(p, Providers(vlm=capable))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())

    assert result.status == "complete"
    assert len(capable.prompts) == len(SHOTS["shots"])
    assert report["vlm_supports_audio_review"] is True


def test_review_cache_invalidates_when_audio_mode_changes(tmp_path):
    p = _project(tmp_path)
    vlm = CapabilityVLM(True)
    assert ReviewStage().run(p, Providers(vlm=vlm)).status == "complete"
    assert len(vlm.prompts) == len(SHOTS["shots"])

    p.model_config["audio_mode"] = "native_video"
    result = ReviewStage().run(p, Providers(vlm=vlm))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())

    assert result.status == "failed"
    assert report["audio_mode"] == "native_video"
    assert report["summary"] == "Native audio is missing or undecodable."


def test_review_cache_invalidates_when_reviewer_model_changes(tmp_path):
    p = _project(tmp_path)
    first = CapabilityVLM(True)
    first.model = "model-a"
    ReviewStage().run(p, Providers(vlm=first))

    second = CapabilityVLM(True)
    second.model = "model-b"
    result = ReviewStage().run(p, Providers(vlm=second))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())

    assert result.status == "complete"
    assert len(second.prompts) == len(SHOTS["shots"])
    assert report["vlm_provider"] == "switchable"
    assert report["vlm_model"] == "model-b"


def test_audio_ineligible_reviewer_cannot_auto_pass(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)

    class VisualOnly(FakeVLMCheck):
        @property
        def supports_audio_review(self):
            return False

    result = ReviewStage().run(p, Providers(vlm=VisualOnly()))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())
    music = next(c for c in report["checks"] if c["dimension"] == "music_absence")

    assert result.status == "failed"
    assert music["status"] == "not_checked"
    assert music["passed"] is False
    assert music["severity"] == "high"


def test_missing_native_stream_blocks_without_provider_call(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=False)
    vlm = CapturingVLM()

    result = ReviewStage().run(p, Providers(vlm=vlm))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())

    assert result.status == "failed"
    assert vlm.prompts == []
    assert report["overall_pass"] is False
    assert report["blocking_failures"] == list(EXPECTED_AUDIO_QC_DIMENSIONS)
    assert report["clip_hash"]
    assert report["qc_context_hash"]
    assert report["qc_context_version"] == 5


def test_undecodable_native_sidecar_blocks_without_provider_call(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())
    for entry in manifest["clips"]:
        p.path("assets", "clips", entry["native_audio"]).write_bytes(b"not audio")
    vlm = CapturingVLM()

    result = ReviewStage().run(p, Providers(vlm=vlm))

    assert result.status == "failed"
    assert vlm.prompts == []


def test_native_review_uses_embedded_clip_audio_when_sidecar_is_absent(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    manifest_path = p.path("assets", "clips", "clips.json")
    manifest = json.loads(manifest_path.read_text())
    first = manifest["clips"][0]
    sidecar = p.path("assets", "clips", first.pop("native_audio"))
    p.path("assets", "clips", first["clip"]).write_bytes(sidecar.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    vlm = CapturingVLM()

    result = ReviewStage().run(p, Providers(vlm=vlm))

    assert result.status == "complete"
    assert len(vlm.prompts) == len(SHOTS["shots"])
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())
    assert report["native_audio_source"] == "sh-001.mp4"


@pytest.mark.parametrize("invalid_sidecar", ["missing", "corrupt"])
def test_native_review_falls_back_from_invalid_sidecar_to_embedded_audio(
    tmp_path, invalid_sidecar
):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())
    for entry in manifest["clips"]:
        sidecar = p.path("assets", "clips", entry["native_audio"])
        audio_bytes = sidecar.read_bytes()
        p.path("assets", "clips", entry["clip"]).write_bytes(audio_bytes)
        if invalid_sidecar == "missing":
            sidecar.unlink()
        else:
            sidecar.write_bytes(b"not audio")
    vlm = CapturingVLM()

    result = ReviewStage().run(p, Providers(vlm=vlm))

    assert result.status == "complete"
    assert len(vlm.prompts) == len(SHOTS["shots"])
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())
    assert report["native_audio_source"] == "sh-001.mp4"


def test_native_review_prefers_valid_sidecar_over_embedded_audio(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())
    for entry in manifest["clips"]:
        sidecar = p.path("assets", "clips", entry["native_audio"])
        p.path("assets", "clips", entry["clip"]).write_bytes(sidecar.read_bytes())

    result = ReviewStage().run(p, Providers(vlm=CapturingVLM()))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())

    assert result.status == "complete"
    assert report["native_audio_source"] == "sh-001.native.source.wav"


def test_native_review_rereviews_when_selected_audio_source_changes(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    manifest = json.loads(p.path("assets", "clips", "clips.json").read_text())
    saved_audio = {}
    for entry in manifest["clips"]:
        sidecar = p.path("assets", "clips", entry["native_audio"])
        saved_audio[entry["id"]] = sidecar.read_bytes()
        p.path("assets", "clips", entry["clip"]).write_bytes(saved_audio[entry["id"]])
        sidecar.write_bytes(b"not audio")
    vlm = CapturingVLM()
    ReviewStage().run(p, Providers(vlm=vlm))
    assert len(vlm.prompts) == len(SHOTS["shots"])

    for entry in manifest["clips"]:
        p.path("assets", "clips", entry["native_audio"]).write_bytes(
            saved_audio[entry["id"]]
        )
    result = ReviewStage().run(p, Providers(vlm=vlm))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())

    assert result.status == "complete"
    assert len(vlm.prompts) == len(SHOTS["shots"]) * 2
    assert report["native_audio_source"] == "sh-001.native.source.wav"


@pytest.mark.parametrize("unsafe_name", ["absolute", "traversal", "nested"])
def test_native_review_never_hashes_or_probes_unsafe_sidecar_paths(
    tmp_path, monkeypatch, unsafe_name
):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    manifest_path = p.path("assets", "clips", "clips.json")
    manifest = json.loads(manifest_path.read_text())
    original_sidecar = p.path(
        "assets", "clips", manifest["clips"][0]["native_audio"]
    )
    external = tmp_path / "external.wav"
    external.write_bytes(original_sidecar.read_bytes())
    nested = p.path("assets", "clips", "nested", "audio.wav")
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(original_sidecar.read_bytes())
    names = {
        "absolute": str(external.resolve()),
        "traversal": "../../../external.wav",
        "nested": "nested/audio.wav",
    }
    for entry in manifest["clips"]:
        entry["native_audio"] = names[unsafe_name]
    manifest_path.write_text(json.dumps(manifest))

    hashed = []
    probed = []
    original_hash = ReviewStage._hash

    def recording_hash(path):
        hashed.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(ReviewStage, "_hash", staticmethod(recording_hash))
    monkeypatch.setattr(
        "studio_agent.stages.review.has_audio_stream",
        lambda path: probed.append(Path(path)) or False,
    )
    result = ReviewStage().run(p, Providers(vlm=CapturingVLM()))

    assert result.status == "failed"
    assert external not in hashed and external not in probed
    assert nested not in hashed and nested not in probed
    assert all(path.parent == p.path("assets", "clips") for path in probed)


def test_native_review_rejects_looped_sidecar_symlink_without_provider_call(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    manifest_path = p.path("assets", "clips", "clips.json")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["clips"]:
        loop_name = f"{entry['id']}.loop.wav"
        loop = p.path("assets", "clips", loop_name)
        loop.symlink_to(loop.name)
        entry["native_audio"] = loop_name
    manifest_path.write_text(json.dumps(manifest))
    vlm = CapturingVLM()

    result = ReviewStage().run(p, Providers(vlm=vlm))

    assert result.status == "failed"
    assert vlm.prompts == []
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())
    assert report["summary"] == "Native audio is missing or undecodable."


def test_native_review_marks_omitted_audio_checks_not_checked(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)

    class OmitsAudio(CapturingVLM):
        def review(self, clip_path: str, *, prompt: str, **kwargs):
            generation = super().review(clip_path, prompt=prompt, **kwargs)
            generation.content["checks"] = [
                check for check in generation.content["checks"]
                if check["dimension"] not in EXPECTED_AUDIO_QC_DIMENSIONS
            ]
            return generation

    result = ReviewStage().run(p, Providers(vlm=OmitsAudio()))
    report = json.loads(p.path("assets", "qc", "sh-001.json").read_text())

    assert result.status == "failed"
    for dimension in EXPECTED_AUDIO_QC_DIMENSIONS:
        check = next(c for c in report["checks"] if c["dimension"] == dimension)
        assert check == {
            "dimension": dimension,
            "passed": False,
            "severity": "high",
            "detail": "audio-capable QC response omitted this required dimension",
            "status": "not_checked",
            "timestamp": None,
        }


def test_native_review_prompt_includes_exact_dialogue_and_sound_policy(tmp_path):
    p = _project(tmp_path, audio_mode="native_video", with_native_audio=True)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())
    shots["shots"][0]["dialogue"] = [
        {"character": "Mara", "line": "别回头。"}
    ]
    p.path("storyboard", "shots.json").write_text(
        json.dumps(shots, ensure_ascii=False)
    )
    vlm = CapturingVLM()

    ReviewStage().run(p, Providers(vlm=vlm))

    prompt = vlm.prompts[0]
    assert '"character": "Mara"' in prompt
    assert '"line": "别回头。"' in prompt
    assert "reject invented speech" in prompt
    assert "synchronized diegetic SFX and natural ambience" in prompt
    assert "score, song, singing, beat, melody, or underscore" in prompt


def test_review_sends_story_aware_context_to_vlm(tmp_path):
    p = _project(tmp_path)
    shots = json.loads(p.path("storyboard", "shots.json").read_text())
    shots["shots"][0].update({
        "camera_movement": "slow dolly from the glowing TV to her bare feet",
        "composition": "TV portal foreground, lonely sofa midground, city window behind",
        "emotion": "wonder turning into fear",
        "continuity_notes": "blue TV light spills across the carpet and gown",
    })
    p.path("storyboard", "shots.json").write_text(json.dumps(shots))
    p.path("story", "idea.md").write_text("A princess leaves a TV and enters a lonely home.\n")
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "A princess crosses from television fantasy into a real apartment.",
        "synopsis": "The lonely homeowner must decide whether to help her return.",
        "themes": ["loneliness", "wonder"],
    }))
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{
            "episode": 1,
            "scenes": [{
                "scene": None,
                "heading": "INT. HOME - NIGHT",
                "beats": ["The television glows", "The princess steps onto the carpet"],
                "dialogue": [{"character": "Princess", "line": "Where am I?"}],
            }],
        }]
    }))
    p.path("storyboard", "prompts", "sh-001.video.md").write_text(
        "VIDEO PROMPT: the princess exits the glowing television."
    )
    p.path("storyboard", "prompts", "sh-001.keyframe.md").write_text(
        "KEYFRAME PROMPT: glowing TV portal in a man's living room."
    )
    vlm = CapturingVLM()

    ReviewStage().run(p, Providers(vlm=vlm))

    prompt = vlm.prompts[0]
    assert "# QC context" in prompt
    assert "A princess leaves a TV" in prompt
    assert "A princess crosses from television fantasy" in prompt
    assert "The princess steps onto the carpet" in prompt
    assert "slow dolly from the glowing TV" in prompt
    assert "TV portal foreground" in prompt
    assert "wonder turning into fear" in prompt
    assert "blue TV light spills" in prompt
    assert "VIDEO PROMPT: the princess exits" in prompt
    assert "KEYFRAME PROMPT: glowing TV portal" in prompt
    assert "Previous shot" in prompt and "Next shot" in prompt


def test_review_threads_bible_identity_and_plausibility_checks(tmp_path):
    p = _project(tmp_path)
    cdir = p.path("bible", "characters", "mara")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({
        "name": "Mara",
        "visual_description": "tall woman with a copper braid",
        "wardrobe": "green field jacket",
        "palette": "olive and rust",
    }))
    (cdir / "identity_board.json").write_text(json.dumps({
        "canonical_face": "freckled, sharp jaw, green eyes",
        "canonical_body": "lean, five foot nine",
        "hair": "copper braid",
        "wardrobe": "green field jacket",
        "palette": "olive and rust",
        "do": ["always copper braid"],
        "dont": ["never blonde"],
        "prompt_aliases": ["Mara"],
    }))
    shots = json.loads(p.path("storyboard", "shots.json").read_text())
    shots["shots"][0]["characters"] = ["Mara"]
    shots["shots"][0]["reference_characters"] = ["Mara"]
    p.path("storyboard", "shots.json").write_text(json.dumps(shots))

    vlm = CapturingVLM()
    ReviewStage().run(p, Providers(vlm=vlm))

    prompt = vlm.prompts[0]
    # The locked Bible identity is threaded in so identity_drift checks against ground truth.
    assert "## Bible character identity" in prompt
    assert "freckled, sharp jaw, green eyes" in prompt
    assert "copper braid" in prompt
    assert "never blonde" in prompt
    # Explicit anatomy + scene-logic guidance for the motion_anatomy dimension.
    lowered = prompt.lower()
    assert "anatom" in lowered
    assert "twisted" in lowered
    assert "physically" in lowered or "logical" in lowered

    # Shots with no Bible character do not get an empty identity section.
    assert "## Bible character identity" not in vlm.prompts[1]
