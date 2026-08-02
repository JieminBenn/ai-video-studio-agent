"""The style stage: profile the user's input, render a sample, gate the look.

Runs first so a human approves the project-wide style (especially one extracted from
an uploaded image) before any character/keyframe is generated in it.
"""

import json
from pathlib import Path

from studio_agent import cli
from studio_agent.orchestrator.project import Project
from studio_agent.stages.style import StyleStage


def _project(tmp_path, model_config):
    base = {"profile": "fake", **cli.load_config()["profiles"]["fake"]}
    base.update(model_config)
    project = Project.create(
        "a lighthouse keeper meets a talking gull",
        root=tmp_path,
        stages=["style"],
        model_config=base,
    )
    return project


def _providers():
    return cli.build_providers(cli.load_config()["profiles"]["fake"])


def test_style_stage_profiles_text_into_the_project_style(tmp_path):
    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "1970s grainy analog sci-fi"},
    })

    result = StyleStage().run(project, _providers())

    assert result.status == "complete"
    style = project.model_config["style"]
    assert "sci-fi" in json.dumps(style).lower()
    # The human-facing label is kept off the style dict (so it never leaks into prompts).
    assert "label" not in style
    assert project.model_config["style_label"].strip()


def test_style_feedback_refines_even_without_an_original_description(tmp_path):
    # A custom-but-blank style (no description, no uploaded image) must still refine when
    # the user gives style feedback — otherwise "Refine style" is a silent no-op. This is
    # the common clip-mode path: the dashboard sets style_input={"description": ""} when the
    # user doesn't type a custom style, and the refine gate only ever supplies feedback.
    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "", "feedback": "make it brighter and warmer"},
    })

    result = StyleStage().run(project, _providers())

    assert result.status == "complete"
    style = project.model_config["style"]
    assert style, "feedback-only refine must produce a resolved style, not leave it blank"
    assert "brighter" in json.dumps(style).lower()  # the feedback reached the profiler


def test_blank_style_refine_routes_feedback_as_description(tmp_path):
    # Real VLM profilers reject an empty description with no image
    # ("style profiling needs a description or an image"). When the user refines a blank
    # style (feedback only, no description/image), the stage must route the feedback AS the
    # description so the profiler gets a valid base instead of crashing.
    from studio_agent.providers.base import Generation

    seen = {}

    class GuardedProfiler:
        name = "guarded"
        model = "m"

        def profile(self, *, description="", image_path=None, language="en", feedback=""):
            if not (description or "").strip() and not image_path:
                raise ValueError("style profiling needs a description or an image")
            seen["description"] = description
            seen["feedback"] = feedback
            return Generation(
                content={"look": description, "label": "refined"},
                provider="guarded", model="m", cost_usd=0.0, seconds=0.0,
                meta={"saw_image": False},
            )

    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "", "feedback": "make it brighter and warmer"},
    })
    providers = _providers()
    providers.style_profiler = GuardedProfiler()

    result = StyleStage().run(project, providers)  # must NOT raise the guard error

    assert result.status == "complete"
    assert "brighter" in seen["description"].lower()  # feedback routed as the description
    assert "brighter" in json.dumps(project.model_config["style"]).lower()


def test_style_stage_writes_style_md_and_a_sample_frame(tmp_path):
    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "dreamy pastel watercolor"},
    })

    StyleStage().run(project, _providers())

    style_md = project.path("bible", "style.md").read_text().lower()
    assert "watercolor" in style_md or "pastel" in style_md
    assert project.path("bible", "style_sample.png").is_file()


def test_style_stage_is_idempotent_and_skips_when_complete(tmp_path):
    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "dreamy pastel watercolor"},
    })
    providers = _providers()

    StyleStage().run(project, providers)
    project.set_stage_status("style", "complete")
    cost_before = len(project.cost_log)

    second = StyleStage().run(project, providers)

    assert second.status == "skipped"
    assert len(project.cost_log) == cost_before


def test_style_stage_profiles_a_custom_style_via_the_llm_profiler(tmp_path):
    # Reproduces the dashboard model-mix case: a real LLM, no vision profiler. The style
    # stage must profile the description through the LLM-backed default, not hard-fail.
    from studio_agent.providers.base import Generation
    from studio_agent.providers.llm_style import LLMStyleProfiler

    class _StubLLM:
        def complete(self, prompt, *, system=None):  # pragma: no cover - unused
            raise NotImplementedError

        def complete_json(self, prompt, *, system=None):
            return Generation(
                content={"look": "neon noir", "label": "neon noir",
                         "prompt_playbook": "Rain-slicked neon streets..."},
                provider="stub-llm", model="stub-1", cost_usd=0.001, seconds=0.02,
            )

    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "一个霓虹黑色电影风格"},
    })
    providers = _providers()
    providers.style_profiler = LLMStyleProfiler(_StubLLM())

    result = StyleStage().run(project, providers)

    assert result.status == "complete"
    assert project.model_config["style"]["look"] == "neon noir"
    assert "neon noir" in project.path("bible", "style.md").read_text().lower()


def test_style_stage_does_not_pass_raw_style_photo_to_image_provider(tmp_path):
    # The uploaded style photo may contain people/objects. It should be read by the style
    # profiler, then the sample should be generated from the extracted style text only.
    from studio_agent.providers.base import Generation, ImageGen
    from studio_agent.providers.fake import FakeImageGen
    from studio_agent.reference_assets import save_reference_upload

    class _RecordingImageGen(ImageGen):
        def __init__(self):
            self.seen_refs: list[str] = []

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            self.seen_refs = list(reference_images or [])
            for ref in self.seen_refs:
                if not Path(ref).is_file():
                    raise FileNotFoundError(f"image reference file not found: {ref}")
            Path(out_path).write_bytes(b"\x89PNG\r\n\x1a\n")
            return Generation(content=out_path, provider="rec", model="rec-1")

    project = _project(tmp_path, {"style_name": "custom", "style": {}, "style_input": {}})
    style_png = tmp_path / "src.png"
    FakeImageGen().generate("style", out_path=str(style_png), seed=1)
    save_reference_upload(
        project, target_type="style", target_id="global",
        data=style_png.read_bytes(), filename="moody.png", label="style reference",
    )

    providers = _providers()
    recorder = _RecordingImageGen()
    providers.image = recorder

    result = StyleStage().run(project, providers)

    assert result.status == "complete"
    assert recorder.seen_refs == []


def test_style_stage_passes_absolute_existing_reference_path_to_the_profiler(tmp_path):
    # Vision style profilers (e.g. doubao-seed VLM) resolve the reference path against the
    # cwd and require the file to exist, so the profiler must get an absolute, on-disk path —
    # not the project-relative path stored in the manifest (which raised
    # "style reference image not found: references/uploads/...").
    from studio_agent.providers.base import Generation
    from studio_agent.providers.fake import FakeImageGen
    from studio_agent.reference_assets import save_reference_upload

    class _RecordingProfiler:
        name = "rec"

        def __init__(self):
            self.seen_path: str | None = None

        def profile(self, *, description="", image_path=None, language="en", feedback=""):
            self.seen_path = image_path
            if image_path and not Path(image_path).is_file():
                raise FileNotFoundError(f"style reference image not found: {image_path}")
            return Generation(
                content={"look": "extracted", "label": "Extracted"},
                provider="rec", model="m", cost_usd=0.0, seconds=0.0,
                meta={"saw_image": True},
            )

    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "moody analog sci-fi"},
    })
    style_png = tmp_path / "style_src.png"
    FakeImageGen().generate("style", out_path=str(style_png), seed=42)
    save_reference_upload(
        project, target_type="style", target_id="global",
        data=style_png.read_bytes(), filename="moody.png", label="style reference",
    )

    providers = _providers()
    recorder = _RecordingProfiler()
    providers.style_profiler = recorder

    result = StyleStage().run(project, providers)

    assert result.status == "complete"
    assert recorder.seen_path, "profiler should be conditioned on the uploaded reference"
    assert Path(recorder.seen_path).is_absolute() and Path(recorder.seen_path).is_file()


def _make_style_project(tmp_path, *, with_style_image: bool = False):
    """Build a project configured for custom-style profiling.

    When *with_style_image* is True, also save a dummy style reference so the
    stage resolves a non-empty ``image_path`` when calling the profiler.
    """
    from studio_agent.providers.fake import FakeImageGen
    from studio_agent.reference_assets import save_reference_upload

    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "moody analog sci-fi"},
    })
    if with_style_image:
        style_png = tmp_path / "style_src.png"
        FakeImageGen().generate("style", out_path=str(style_png), seed=42)
        save_reference_upload(
            project, target_type="style", target_id="global",
            data=style_png.read_bytes(), filename="moody.png", label="style reference",
        )
    return project


def test_style_md_discloses_when_image_was_not_seen(tmp_path):
    from studio_agent.providers.base import Generation
    from studio_agent.stages.style import StyleStage

    class _BlindProfiler:
        name = "blind"

        def profile(self, *, description="", image_path=None, language="en", feedback=""):
            return Generation(
                content={"look": "inferred", "label": "Inferred"},
                provider="blind", model="m", cost_usd=0.0, seconds=0.0,
                meta={"saw_image": False},
            )

    project = _make_style_project(tmp_path, with_style_image=True)
    providers = _providers()
    providers.style_profiler = _BlindProfiler()
    StyleStage().run(project, providers)

    text = project.path("bible", "style.md").read_text()
    assert "inferred from your text" in text.lower()
    assert "no vision model" in text.lower()


def test_style_md_has_no_disclosure_when_image_was_seen(tmp_path):
    from studio_agent.providers.base import Generation
    from studio_agent.stages.style import StyleStage

    class _SeeingProfiler:
        name = "seeing"

        def profile(self, *, description="", image_path=None, language="en", feedback=""):
            return Generation(
                content={"look": "extracted", "label": "Extracted"},
                provider="seeing", model="m", cost_usd=0.0, seconds=0.0,
                meta={"saw_image": True},
            )

    project = _make_style_project(tmp_path, with_style_image=True)
    providers = _providers()
    providers.style_profiler = _SeeingProfiler()
    StyleStage().run(project, providers)

    text = project.path("bible", "style.md").read_text()
    assert "no vision model" not in text.lower()


def test_style_stage_passes_through_a_preset_without_a_profiler_call(tmp_path):
    # A named preset (CLI back-compat) carries a resolved dict already; the stage just
    # records it and renders a sample — it must not require profiling.
    project = _project(tmp_path, {
        "style_name": "anime",
        "style": {"look": "anime", "palette": "moonlit teal", "aspect_ratio": "16:9"},
    })
    providers = _providers()
    providers.style_profiler = None  # prove no profiling is attempted

    result = StyleStage().run(project, providers)

    assert result.status == "complete"
    assert "anime" in project.path("bible", "style.md").read_text().lower()
    assert project.path("bible", "style_sample.png").is_file()


def test_style_stage_threads_feedback_to_profiler(tmp_path):
    project = _project(tmp_path, {
        "style_name": "custom",
        "style": {},
        "style_input": {"description": "moody analog sci-fi", "feedback": "make it warmer"},
    })
    StyleStage().run(project, _providers())
    style_md = project.path("bible", "style.md").read_text().lower()
    assert "warmer" in style_md
