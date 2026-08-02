import json
from pathlib import Path

import pytest

from studio_agent.image_generation import (
    ImagePromptLimitError,
    generate_project_image,
    prepare_image_prompt,
)
from studio_agent.keyframe_prompt import STYLE_REFERENCE_GUARD
from studio_agent.orchestrator.project import Project
from studio_agent.providers.base import Generation, ImageCapabilities, ImageGen


def test_under_limit_prompt_is_unchanged_after_existing_strip_behavior():
    prompt = "  保持原样 cinematic lighting  "

    result = prepare_image_prompt(
        prompt,
        ImageCapabilities(
            max_prompt_length=100,
            prompt_length_unit="utf8_bytes",
        ),
    )

    assert result.submitted == prompt.strip()
    assert result.was_fitted is False
    assert result.measured_canonical == len(prompt.strip().encode("utf-8"))


def test_chinese_is_measured_as_utf8_bytes_not_characters():
    prompt = "中" * 10

    character_result = prepare_image_prompt(
        prompt,
        ImageCapabilities(
            max_prompt_length=10,
            prompt_length_unit="characters",
        ),
    )

    assert character_result.was_fitted is False
    with pytest.raises(ImagePromptLimitError, match=r"30 utf8_bytes.*20"):
        prepare_image_prompt(
            prompt,
            ImageCapabilities(
                max_prompt_length=20,
                prompt_length_unit="utf8_bytes",
            ),
        )


def test_over_limit_prompt_drops_only_packaged_skill_section_and_keeps_guard():
    prompt = (
        "Location sheet for a modern home gaming room. Preserve the laptop, desk, "
        "RGB lighting, black-blue palette, and fixed camera.\n\n"
        "# Location Identity Skill\n"
        + ("generic production guidance. " * 30)
        + "\n\n"
        + STYLE_REFERENCE_GUARD
    )

    result = prepare_image_prompt(
        prompt,
        ImageCapabilities(
            max_prompt_length=500,
            prompt_length_unit="utf8_bytes",
        ),
    )

    assert result.was_fitted is True
    assert result.measured_submitted <= 500
    assert "modern home gaming room" in result.submitted
    assert "RGB lighting" in result.submitted
    assert STYLE_REFERENCE_GUARD in result.submitted
    assert "Location Identity Skill" not in result.submitted
    assert result.removed_skill_sections == ("Location Identity Skill",)


def test_inline_packaged_skill_heading_can_be_removed_without_losing_prior_text():
    prompt = (
        "Designed location with fixed practical lighting. # Location Identity Skill\n"
        + ("generic guidance " * 40)
        + "\n\n"
        + STYLE_REFERENCE_GUARD
    )

    result = prepare_image_prompt(
        prompt,
        ImageCapabilities(
            max_prompt_length=400,
            prompt_length_unit="utf8_bytes",
        ),
    )

    assert "Designed location with fixed practical lighting." in result.submitted
    assert STYLE_REFERENCE_GUARD in result.submitted
    assert result.measured_submitted <= 400


def test_unique_over_limit_content_fails_instead_of_being_truncated():
    prompt = "unique identity setting effect " * 100

    with pytest.raises(ImagePromptLimitError, match="cannot be fitted safely"):
        prepare_image_prompt(
            prompt,
            ImageCapabilities(
                max_prompt_length=100,
                prompt_length_unit="characters",
            ),
        )


class RecordingLimitedImage(ImageGen):
    name = "limited"
    model = "limited-1"
    max_prompt_length = 500
    prompt_length_unit = "utf8_bytes"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "out_path": out_path,
            "reference_images": list(reference_images or []),
            "kwargs": dict(kwargs),
        })
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return Generation(
            content=out_path,
            provider=self.name,
            model=self.model,
            cost_usd=0.25,
            seconds=1.0,
        )


def test_shared_boundary_fits_before_call_and_saves_exact_projection(tmp_path):
    project = Project.create("idea", root=tmp_path, stages=["bible"])
    provider = RecordingLimitedImage()
    output = project.path("bible", "locations", "room", "reference.png")
    prompt = (
        "Modern gaming room, preserve RGB lighting and black-blue palette.\n\n"
        "# Location Identity Skill\n"
        + ("generic guidance " * 100)
    )

    result = generate_project_image(
        project,
        provider,
        prompt,
        out_path=str(output),
        reference_images=[],
        seed=9,
    )

    submitted = provider.calls[0]["prompt"]
    assert len(submitted.encode("utf-8")) <= 500
    assert "Modern gaming room" in submitted
    assert result.meta["prompt_projection"]["was_fitted"] is True
    assert result.meta["prompt_projection"]["submitted_prompt"] == submitted
    assert output.with_suffix(".provider-prompt.md").read_text() == submitted
    metadata = json.loads(output.with_suffix(".provider-prompt.json").read_text())
    assert metadata["provider"] == "limited"
    assert metadata["model"] == "limited-1"
    assert metadata["original_length"] > metadata["submitted_length"]
    assert provider.calls[0]["reference_images"] == []
    assert provider.calls[0]["kwargs"] == {"seed": 9}


def test_shared_boundary_leaves_under_limit_call_and_files_unchanged(tmp_path):
    project = Project.create("idea", root=tmp_path, stages=["style"])
    provider = RecordingLimitedImage()
    output = project.path("bible", "style_sample.png")
    prompt_sidecar = output.with_suffix(".provider-prompt.md")
    metadata_sidecar = output.with_suffix(".provider-prompt.json")
    prompt_sidecar.parent.mkdir(parents=True, exist_ok=True)
    prompt_sidecar.write_text("stale fitted prompt")
    metadata_sidecar.write_text("{}")

    result = generate_project_image(
        project,
        provider,
        "short exact prompt",
        out_path=str(output),
        seed=7,
    )

    assert provider.calls[0]["prompt"] == "short exact prompt"
    assert result.meta["prompt_projection"]["was_fitted"] is False
    assert not prompt_sidecar.exists()
    assert not metadata_sidecar.exists()


@pytest.mark.parametrize(
    ("relative_path", "expected_boundary_calls"),
    [
        ("studio_agent/stages/style.py", 1),
        ("studio_agent/stages/bible.py", 2),
        ("studio_agent/stages/storyboard.py", 2),
    ],
)
def test_all_stage_image_calls_use_shared_boundary(
    relative_path,
    expected_boundary_calls,
):
    source = (Path(__file__).parents[1] / relative_path).read_text()

    assert "providers.image.generate(" not in source
    assert source.count("generate_project_image(") == expected_boundary_calls


def test_fitter_drops_skill_section_but_keeps_appearance_lock():
    lock_section = (
        "## Character appearance lock (verbatim — do not alter)\n"
        "Mara: Oval face, teardrop mole under left eye; red wool coat."
    )
    skill_section = "## Character identity skill\n" + ("blah " * 400)
    prompt = f"# Keyframe prompt\n\n{lock_section}\n\n{skill_section}"

    caps = ImageCapabilities(
        max_prompt_length=400,
        prompt_length_unit="characters",
    )
    prepared = prepare_image_prompt(prompt, caps)

    assert "Mara: Oval face, teardrop mole under left eye; red wool coat." in prepared.submitted
    assert "Character identity skill" not in prepared.submitted
    assert prepared.measured_submitted <= 400
