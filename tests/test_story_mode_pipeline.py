import json
import shutil
import subprocess
import wave

from studio_agent import cli
from studio_agent.formats import resolve_format
from studio_agent.orchestrator.project import Project
from studio_agent.prompt_conversation import revise_prompt_turn


def test_full_story_mode_run_on_fakes_has_native_audio(tmp_path):
    config = cli.load_config()
    resolved_format = resolve_format(config, "short_film")
    profile = dict(config["profiles"]["fake"])
    project = Project.create(
        "a lighthouse keeper hears a talking gull warn of a storm",
        root=tmp_path,
        stages=cli.pipeline_for(resolved_format),
        cost_cap=config.get("cost_cap"),
        model_config={
            "profile": "fake",
            "audio_mode": "native_video",
            "style": {},
            "format_name": resolved_format.name,
            "product_format": resolved_format.to_project_config(),
            "language": "en",
            **profile,
        },
    )
    project.story_dir.joinpath("idea.md").write_text(project.idea + "\n")
    project.save()

    result = cli._machine().run(project, cli.build_providers(profile), auto=True)
    assert result.done is True

    assert project.path("story", "creative_brief.json").is_file()
    assert list(project.path("knowledge", "packets").glob("*.json"))
    assert list(project.path("storyboard", "prompts").glob("*.keyframe.md"))

    reloaded = Project.load(project.dir)
    for stage in cli.pipeline_for(resolved_format):
        assert reloaded.stage_status(stage) in ("complete", "approved")

    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert all(shot.get("movement_motivation") and shot.get("lens_intent") for shot in shots)
    manifest = json.loads(project.path("assets", "audio", "audio.json").read_text())
    assert manifest["mode"] == "native_video"
    assert [track["id"] for track in manifest["tracks"]] == [shot["id"] for shot in shots]
    assert "music" not in manifest
    assert len(manifest["tracks"]) == len(shots)
    for track in manifest["tracks"]:
        wav_path = project.path("assets", "audio", track["file"])
        assert wav_path.is_file()
        with wave.open(str(wav_path), "rb") as wav:
            assert wav.getnchannels() == 2
            assert wav.getframerate() == 48_000
            assert wav.getnframes() > 0

    assert not project.path("assets", "audio", "music.wav").exists()
    assert not project.path("storyboard", "prompts", "music.audio.md").exists()
    # Native track extraction is free; the only audio generations are the off-screen
    # narration (旁白) VO stems, one per narrated shot, synthesized via TTS (zero fake cost).
    narrated = [shot for shot in shots if str(shot.get("narration") or "").strip()]
    narration_tracks = manifest.get("narration", [])
    assert [n["id"] for n in narration_tracks] == [shot["id"] for shot in narrated]
    audio_cost_entries = [e for e in reloaded.cost_log if e["stage"] == "audio"]
    assert len(audio_cost_entries) == len(narrated)
    assert all(entry["cost_usd"] == 0 for entry in audio_cost_entries)
    for track in narration_tracks:
        assert project.path("assets", "audio", track["file"]).is_file()

    output = project.path("output", f"{project.project_id}.mp4")
    assert output.is_file()
    if shutil.which("ffprobe"):
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert len([line for line in probe.stdout.splitlines() if line.strip()]) == 1


def _run_story_mode(tmp_path, *, motion_grid=None):
    config = cli.load_config()
    resolved_format = resolve_format(config, "short_film")
    profile = dict(config["profiles"]["fake"])
    model_config = {
        "profile": "fake",
        "style": {},
        "format_name": resolved_format.name,
        "product_format": resolved_format.to_project_config(),
        "language": "en",
        **profile,
    }
    if motion_grid is not None:
        model_config["motion_grid"] = motion_grid
    project = Project.create(
        "a lighthouse keeper hears a talking gull warn of a storm",
        root=tmp_path,
        stages=cli.pipeline_for(resolved_format),
        cost_cap=config.get("cost_cap"),
        model_config=model_config,
    )
    project.story_dir.joinpath("idea.md").write_text(project.idea + "\n")
    project.save()
    result = cli._machine().run(project, cli.build_providers(profile), auto=True)
    assert result.done is True
    return project


def test_full_story_mode_grid_run_on_fakes(tmp_path):
    project = _run_story_mode(
        tmp_path,
        motion_grid={"enabled": True, "story_scenes": True, "layout": "2x2"},
    )
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert shots and all("motion_grid" in s for s in shots)
    # adaptive lengths: clips are not all the same short duration
    durations = [s["duration_s"] for s in shots]
    assert max(durations) > min(durations)
    # the grid is the keyframe + a playable final cut exists
    assert project.path("storyboard", "keyframes", shots[0]["keyframe"]).is_file()
    assert project.path("storyboard", "scene_sequences", "1.json").is_file()
    assert project.path("output", f"{project.project_id}.mp4").is_file()


def test_story_mode_without_grid_has_no_motion_grid(tmp_path):
    project = _run_story_mode(tmp_path)  # motion_grid absent -> today's behavior
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert all("motion_grid" not in s for s in shots)
    assert project.path("output", f"{project.project_id}.mp4").is_file()


def test_reviewed_story_mode_pauses_at_both_prompt_and_media_gates(tmp_path):
    config = cli.load_config()
    resolved = resolve_format(config, "short_film")
    profile = dict(config["profiles"]["fake"])
    project = Project.create(
        "reviewed story prompt gates",
        root=tmp_path,
        stages=cli.pipeline_for(resolved),
        model_config={
            "profile": "fake",
            "style": {},
            "format_name": resolved.name,
            "product_format": resolved.to_project_config(),
            "language": "en",
            **profile,
        },
    )
    project.story_dir.joinpath("idea.md").write_text(project.idea)
    providers = cli.build_providers(profile)
    machine = cli._machine()
    for name in project.stages[:project.stages.index("storyboard")]:
        machine._by_name[name].run(project, providers)
        project.set_stage_status(name, "approved")
    project.current_stage = "storyboard"
    project.save()

    assert machine.run(project, providers, auto=False).paused_at == "storyboard"
    assert not list(project.path("storyboard", "keyframes").glob("*.png"))
    first_shot = json.loads(project.path("storyboard", "shots.json").read_text())["shots"][0]
    revise_prompt_turn(
        project,
        gate="keyframes",
        shot_id=first_shot["id"],
        message="Clarify the static opening composition while preserving the project look.",
        apply_to_all=False,
        providers=providers,
    )
    assert machine.approve(project) == "keyframes"
    assert machine.run(project, providers, auto=False).paused_at == "keyframes"
    assert list(project.path("storyboard", "keyframes").glob("*.png"))
    assert machine.approve(project) == "video_prompts"
    assert machine.run(project, providers, auto=False).paused_at == "video_prompts"
    assert list(project.path("storyboard", "prompts").glob("*.video.md"))
    assert not list(project.path("assets", "clips").glob("*.mp4"))
    revise_prompt_turn(
        project,
        gate="videos",
        shot_id=first_shot["id"],
        message="Use restrained motion and preserve the approved frame identity.",
        apply_to_all=False,
        providers=providers,
    )
    assert machine.approve(project) == "video"
    assert machine.run(project, providers, auto=False).paused_at == "video"
    assert list(project.path("assets", "clips").glob("*.mp4"))

    reloaded = Project.load(project.dir)
    assert reloaded.path("storyboard", "conversations", "prompt-review.jsonl").is_file()
    assert reloaded.path("storyboard", "prompt_approvals", "keyframes.json").is_file()
    assert reloaded.path("storyboard", "prompt_approvals", "videos.json").is_file()
