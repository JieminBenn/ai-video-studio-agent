import json
import shutil
import subprocess
import wave

from studio_agent import cli
from studio_agent.formats import resolve_format
from studio_agent.orchestrator.project import Project
from studio_agent.prompt_conversation import revise_prompt_turn
from studio_agent.prompt_validation import validate_grid_prompt


def _assert_valid_native_wav(path):
    assert path.is_file()
    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getframerate() == 48_000
        assert wav.getnframes() > 0


def _assert_one_audio_stream(path):
    if not shutil.which("ffprobe"):
        return
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert len([line for line in probe.stdout.splitlines() if line.strip()]) == 1


def _run_clip_mode(tmp_path, extra_config=None):
    """Run the full clip-mode pipeline on fakes and return (project, providers).

    ``extra_config`` is merged into model_config so tests can inject motion_grid
    or any other per-run config without repeating the full setup boilerplate.
    """
    config = cli.load_config()
    resolved_format = resolve_format(config, "short_video")
    profile = config["profiles"]["fake"]

    model_config = {
        "profile": "fake",
        "style": {},
        "format_name": resolved_format.name,
        "product_format": resolved_format.to_project_config(),
        "language": "en",
        "clip_count": 2,
        "clip_seconds": 15,
        **profile,
    }
    if extra_config:
        model_config.update(extra_config)

    project = Project.create(
        "a dancer spinning in slow motion under falling cherry blossoms",
        root=tmp_path,
        stages=cli.pipeline_for(resolved_format),
        cost_cap=config.get("cost_cap"),
        model_config=model_config,
    )
    project.story_dir.joinpath("idea.md").write_text(project.idea + "\n")
    project.save()

    providers = cli.build_providers(profile)
    cli._machine().run(project, providers, auto=True)
    return project, providers


def test_full_clip_mode_run_on_fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)
    project, providers = _run_clip_mode(tmp_path)

    reloaded = Project.load(project.dir)
    assert project.path("story", "creative_brief.json").is_file()
    assert list(project.path("knowledge", "packets").glob("*.json"))
    assert list(project.path("storyboard", "prompts").glob("*.video.md"))
    assert reloaded.model_config["audio_mode"] == "native_video"
    assert "plot" not in reloaded.stages
    assert "script" not in reloaded.stages
    assert "audio" in reloaded.stages
    # concept + bible + clip + video + audio + assemble all ran to approved
    for stage in (
        "concept", "bible", "clip", "keyframes", "video_prompts", "video",
        "audio", "assemble",
    ):
        assert reloaded.stage_status(stage) in ("complete", "approved")

    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert len(shots) == 2
    assert sum(s["duration_s"] for s in shots) == 30
    audio = json.loads(project.path("assets", "audio", "audio.json").read_text())
    assert audio["mode"] == "native_video"
    assert [track["id"] for track in audio["tracks"]] == ["sh-001", "sh-002"]
    assert "music" not in audio
    for track in audio["tracks"]:
        _assert_valid_native_wav(project.path("assets", "audio", track["file"]))
    assert not project.path("assets", "audio", "music.wav").exists()
    assert not project.path("storyboard", "prompts", "music.audio.md").exists()
    assert not any(entry["stage"] == "audio" for entry in reloaded.cost_log)
    outputs = list(project.path("output").glob("*.mp4"))
    assert outputs, "clip mode must still assemble a playable mp4"
    _assert_one_audio_stream(outputs[0])


def test_regenerated_character_reaches_clips_even_after_display_name_drift(tmp_path):
    """End-to-end guard: a regenerated bible character always reaches the clips.

    Locks the whole staleness class the earlier fixes closed. A bible-text revision renames the
    character's display `name` (青年 -> 南宫婉) while shots keep the plot name; the character is
    then regenerated. The resume must (a) re-wire the character's reference sheet into the shot
    by stable slug and (b) re-render the keyframe against the FRESH sheet — not leave the old one.
    """
    from studio_agent.asset_regeneration import regenerate_bible_asset

    project, providers = _run_clip_mode(tmp_path)
    slug = sorted((project.dir / "bible" / "characters").iterdir())[0].name
    cjson = project.path("bible", "characters", slug, "character.json")
    data = json.loads(cjson.read_text())
    data["name"] = "南宫婉"                      # display name drifts away from the plot name
    cjson.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    kf_dir = project.path("storyboard", "keyframes")
    before = {p.name: p.read_bytes() for p in sorted(kf_dir.glob("*.png"))}
    assert before, "expected rendered keyframes from the initial run"

    regenerate_bible_asset(project, f"bible/characters/{slug}/reference.png")
    project = Project.load(project.dir)
    machine = cli._machine()
    for _ in range(20):
        result = machine.run(project, providers, auto=False)
        if getattr(result, "paused_at", None) is None:
            break
        machine.approve(project)
        project = Project.load(project.dir)
        if project.status == "done":
            break

    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    # (a) the character's bible sheet is wired into every shot that features it
    for shot in shots:
        char_refs = [r for r in shot.get("reference_images", []) if "/characters/" in r]
        assert any(slug in r for r in char_refs), (
            f"regenerated character sheet must be wired into {shot['id']} after the rename"
        )
    # (b) each keyframe re-rendered against the fresh sheet (fake output folds ref content)
    after = {p.name: p.read_bytes() for p in sorted(kf_dir.glob("*.png"))}
    for name, old in before.items():
        assert after.get(name) != old, f"keyframe {name} must re-render against the regenerated sheet"


def test_clip_mode_grid_end_to_end_on_fakes(tmp_path):
    # Build a clip-mode project with motion_grid enabled and fake (grid-capable) providers,
    # using the same helper the other end-to-end test in this file uses, then run all stages.
    project, providers = _run_clip_mode(
        tmp_path, extra_config={"motion_grid": {"enabled": True, "layout": "2x2"}}
    )
    reloaded = Project.load(project.dir)
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert shots[0]["motion_grid"]["panel_count"] == 4
    # the grid is the keyframe, and a clip + output still get produced
    assert project.path("storyboard", "keyframes", shots[0]["keyframe"]).is_file()
    grid_prompt = project.path(
        "storyboard", "prompts", f"{shots[0]['id']}.grid.md"
    ).read_text()
    assert validate_grid_prompt(
        grid_prompt,
        shots[0],
        panel_count=shots[0]["motion_grid"]["panel_count"],
    ).valid
    assert list(project.path("assets", "clips").glob("*.mp4"))
    assert list(project.dir.glob("output/*.mp4"))
    # full pipeline completion assertion: all stages reached complete or approved
    for stage in (
        "concept", "bible", "clip", "keyframes", "video_prompts", "video",
        "audio", "assemble",
    ):
        assert reloaded.stage_status(stage) in ("complete", "approved")


def test_clip_mode_no_grid_when_disabled(tmp_path):
    project, providers = _run_clip_mode(tmp_path)  # motion_grid off by default
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert "motion_grid" not in shots[0]


def test_reviewed_clip_mode_pauses_at_both_prompt_and_media_gates(tmp_path):
    config = cli.load_config()
    resolved = resolve_format(config, "short_video")
    profile = config["profiles"]["fake"]
    project = Project.create(
        "reviewed clip prompt gates",
        root=tmp_path,
        stages=cli.pipeline_for(resolved),
        model_config={
            "profile": "fake",
            "style": {},
            "format_name": resolved.name,
            "product_format": resolved.to_project_config(),
            "language": "en",
            "clip_count": 1,
            "clip_seconds": 8,
            **profile,
        },
    )
    project.story_dir.joinpath("idea.md").write_text(project.idea)
    providers = cli.build_providers(profile)
    machine = cli._machine()
    for name in project.stages[:project.stages.index("clip")]:
        machine._by_name[name].run(project, providers)
        project.set_stage_status(name, "approved")
    project.current_stage = "clip"
    project.save()

    assert machine.run(project, providers, auto=False).paused_at == "clip"
    assert not list(project.path("storyboard", "keyframes").glob("*.png"))
    first_shot = json.loads(project.path("storyboard", "shots.json").read_text())["shots"][0]
    revise_prompt_turn(
        project,
        gate="keyframes",
        shot_id=first_shot["id"],
        message="Keep the opening frame locked and make the silhouette clearer.",
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
        message="Make the turn more deliberate without changing identity.",
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


def test_clip_default_plan_yields_one_model_length_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)
    project, providers = _run_clip_mode(
        tmp_path, extra_config={"clip_plan_mode": "default"}
    )
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert len(shots) == 1
    assert shots[0]["duration_s"] is None  # the video model's default length governs
    clips = json.loads(project.path("assets", "clips", "clips.json").read_text())["clips"]
    # The manifest records the real rendered seconds (FakeVideo resolves None to 4.0).
    assert clips[0]["duration_s"] == 4.0
    # The compiled video prompt defers pacing to the model instead of claiming ~2s.
    prompt = project.path("storyboard", "prompts", "sh-001.video.md").read_text()
    assert "Duration: use your default clip length." in prompt


def test_add_shot_mid_pipeline_renders_only_the_new_shot(tmp_path, monkeypatch):
    from studio_agent.shot_insertion import add_shot

    monkeypatch.setattr(cli, "PROJECTS_ROOT", tmp_path)
    project, providers = _run_clip_mode(tmp_path)  # completes with sh-001 + sh-002

    keyframes = project.path("storyboard", "keyframes")
    clips_dir = project.path("assets", "clips")
    kf_before = {p.name: p.stat().st_mtime_ns for p in keyframes.glob("*.png")}
    clips_before = {p.name: p.stat().st_mtime_ns for p in clips_dir.glob("*.mp4")}

    result = add_shot(
        project, description="插入一个特写镜头", after_shot_id="sh-001",
        providers=providers,
    )
    cli._machine().run(project, providers, auto=True)

    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    assert [s["id"] for s in shots] == ["sh-001", result.shot_id, "sh-002"]
    assert keyframes.joinpath(f"{result.shot_id}.png").is_file()
    assert clips_dir.joinpath(f"{result.shot_id}.mp4").is_file()
    # Localized regeneration: no existing paid artifact was re-rendered.
    for name, mtime in kf_before.items():
        assert keyframes.joinpath(name).stat().st_mtime_ns == mtime
    for name, mtime in clips_before.items():
        assert clips_dir.joinpath(name).stat().st_mtime_ns == mtime
    clips = json.loads(clips_dir.joinpath("clips.json").read_text())["clips"]
    assert [c["id"] for c in clips] == ["sh-001", result.shot_id, "sh-002"]
