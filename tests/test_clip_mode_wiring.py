from studio_agent import cli
from studio_agent.formats import resolve_format


def test_pipeline_for_clip_mode_includes_native_audio_gate():
    config = cli.load_config()
    pipeline = cli.pipeline_for(resolve_format(config, "short_video"))
    assert pipeline == [
        "style", "concept", "bible", "clip", "keyframes", "video_prompts",
        "video", "audio", "assemble",
    ]
    assert "plot" not in pipeline and "script" not in pipeline


def test_pipeline_for_story_mode_is_full_pipeline():
    config = cli.load_config()
    pipeline = cli.pipeline_for(resolve_format(config, "short_film"))
    assert pipeline == cli.PIPELINE
    assert pipeline[0] == "style"
    assert pipeline[1] == "plot"


def test_machine_can_resolve_clip_stages():
    machine = cli._machine()
    names = {s.name for s in machine.stages}
    assert {"concept", "clip", "keyframes", "video_prompts"} <= names
