"""Artifact Studio helpers: readable review + scoped LLM revisions."""

import json

import pytest

from studio_agent.artifact_studio import (
    apply_artifact_revision,
    downstream_stages_for_artifact,
)
from studio_agent.orchestrator.project import Project
from studio_agent.providers.base import Generation, LLM
from studio_agent.stages.base import Providers


PIPELINE = [
    "plot", "script", "bible", "storyboard", "keyframes", "video_prompts",
    "video", "review", "audio", "assemble",
]


class JsonRevisionLLM(LLM):
    name = "test-llm"
    model = "test-reviser"

    def __init__(self, content):
        self.content = content
        self.calls = []

    def complete(self, prompt: str, *, system: str | None = None) -> Generation:
        self.calls.append(prompt)
        return Generation(content=str(self.content), provider=self.name, model=self.model)

    def complete_json(self, prompt: str, *, system: str | None = None) -> Generation:
        self.calls.append(prompt)
        return Generation(content=self.content, provider=self.name, model=self.model)


def _project(tmp_path):
    p = Project.create("artifact studio", root=tmp_path, stages=PIPELINE)
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "old",
        "synopsis": "old synopsis",
        "themes": [],
    }))
    for stage in PIPELINE:
        p.set_stage_status(stage, "approved")
    p.current_stage = None
    p.status = "done"
    p.save()
    return p


def test_downstream_stages_for_artifact():
    assert downstream_stages_for_artifact("story/plot.json") == [
        "script", "bible", "storyboard", "keyframes", "video_prompts", "video",
        "review", "audio", "assemble"
    ]
    assert downstream_stages_for_artifact("story/script.json") == [
        "storyboard", "keyframes", "video_prompts", "video", "review", "audio",
        "assemble"
    ]
    assert downstream_stages_for_artifact("bible/characters/mara/identity_board.json") == [
        "storyboard", "keyframes", "video_prompts", "video", "review", "audio",
        "assemble"
    ]
    assert downstream_stages_for_artifact("storyboard/prompts/sh-001.video.md") == [
        "video_prompts", "video", "review", "audio", "assemble"
    ]


def test_apply_artifact_revision_updates_json_marks_stages_and_writes_history(tmp_path):
    p = _project(tmp_path)
    llm = JsonRevisionLLM({
        "logline": "new",
        "synopsis": "new synopsis",
        "themes": ["power"],
    })

    result = apply_artifact_revision(
        p,
        "story/plot.json",
        "make it more dramatic",
        Providers(llm=llm),
    )

    updated = json.loads(p.path("story", "plot.json").read_text())
    assert result.applied is True
    assert updated["logline"] == "new"
    assert p.stage_status("plot") == "complete"
    assert p.stage_status("script") == "pending"
    assert p.stage_status("assemble") == "pending"
    assert p.current_stage == "plot"
    history = list(p.path("adjustments").glob("*.json"))
    assert len(history) == 1
    record = json.loads(history[0].read_text())
    assert record["artifact_path"] == "story/plot.json"
    assert record["instruction"] == "make it more dramatic"
    assert record["applied"] is True
    assert record["provider"] == "test-llm"


def test_apply_artifact_revision_keeps_old_file_on_invalid_json(tmp_path):
    p = _project(tmp_path)
    before = p.path("story", "plot.json").read_text()
    llm = JsonRevisionLLM(["wrong shape"])

    with pytest.raises(ValueError, match="same JSON shape"):
        apply_artifact_revision(
            p,
            "story/plot.json",
            "break it",
            Providers(llm=llm),
        )

    assert p.path("story", "plot.json").read_text() == before
    history = list(p.path("adjustments").glob("*.json"))
    assert len(history) == 1
    record = json.loads(history[0].read_text())
    assert record["applied"] is False
    assert "same JSON shape" in record["error"]


def _bible_project_with_fake_llm(tmp_path):
    p = Project.create("bible test", root=tmp_path, stages=["bible", "storyboard"])
    p.path("story").mkdir(parents=True, exist_ok=True)
    p.path("story", "plot.json").write_text(json.dumps({"logline": "a gull", "synopsis": "brief"}))
    char_dir = p.path("bible", "characters", "gull")
    char_dir.mkdir(parents=True, exist_ok=True)
    char_dir.joinpath("character.json").write_text(
        json.dumps({"name": "Gull", "description": "young woman"})
    )
    char_dir.joinpath("reference.png").write_bytes(b"img")
    p.set_stage_status("bible", "complete")
    p.save()
    providers = Providers(llm=JsonRevisionLLM({"name": "Gull", "description": "older woman"}))
    return p, providers


def test_apply_artifact_revision_can_skip_invalidation(tmp_path):
    project, providers = _bible_project_with_fake_llm(tmp_path)
    rel = "bible/characters/gull/character.json"
    ref = project.path("bible", "characters", "gull", "reference.png")
    assert ref.is_file()

    result = apply_artifact_revision(project, rel, "make her older", providers, invalidate=False)

    assert result.applied is True
    assert result.invalidated == []
    assert result.archived == 0
    assert ref.is_file()  # image untouched when invalidation is skipped


def test_apply_artifact_revision_uses_vision_when_images_supplied(tmp_path):
    import json
    from studio_agent.artifact_studio import apply_artifact_revision
    from studio_agent.orchestrator.project import Project
    from studio_agent.providers.fake import FakeLLM, FakeReferenceAnalyzer
    from studio_agent.stages.base import Providers

    p = Project.create("vision revise", root=tmp_path, stages=["bible"])
    rel = "bible/characters/mara/character.json"
    cpath = p.path("bible", "characters", "mara", "character.json")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({"name": "Mara", "wardrobe": ""}))
    img = p.path("references", "uploads", "hero.png")
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\nhero")

    result = apply_artifact_revision(
        p, rel, "keep the same outfit",
        Providers(llm=FakeLLM(), reference_analyzer=FakeReferenceAnalyzer()),
        invalidate=False, image_paths=[str(img)],
    )

    assert result.applied
    revised = json.loads(cpath.read_text())
    assert revised["grounded_in"] == ["hero"]  # the fake revise saw the image


def test_apply_artifact_revision_stays_text_only_without_images(tmp_path):
    import json
    from studio_agent.artifact_studio import apply_artifact_revision
    from studio_agent.orchestrator.project import Project
    from studio_agent.providers.fake import FakeLLM
    from studio_agent.stages.base import Providers

    p = Project.create("text only", root=tmp_path, stages=["bible"])
    rel = "bible/characters/mara/character.json"
    cpath = p.path("bible", "characters", "mara", "character.json")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({"name": "Mara", "wardrobe": ""}))

    apply_artifact_revision(
        p, rel, "make it warmer",
        Providers(llm=FakeLLM()),
        invalidate=False, image_paths=[],
    )

    revised = json.loads(cpath.read_text())
    assert "grounded_in" not in revised  # text path, no vision
    assert revised["revision_note"] == "make it warmer"


def test_apply_artifact_revision_blocks_when_requested_images_have_no_analyzer(tmp_path):
    import json
    from studio_agent.orchestrator.project import Project

    p = Project.create("vision unavailable", root=tmp_path, stages=["bible"])
    rel = "bible/characters/mara/character.json"
    cpath = p.path("bible", "characters", "mara", "character.json")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"name": "Mara", "wardrobe": "red coat"})
    cpath.write_text(original)
    img = p.path("references", "uploads", "hero.png")
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\nhero")
    llm = JsonRevisionLLM({"name": "Mara", "wardrobe": "wrong fallback"})

    with pytest.raises(ValueError, match="reference analyzer"):
        apply_artifact_revision(
            p,
            rel,
            "keep the same outfit",
            Providers(llm=llm),
            invalidate=False,
            image_paths=[str(img)],
        )

    assert llm.calls == []
    assert cpath.read_text() == original
    history = list(p.path("adjustments").glob("*.json"))
    assert len(history) == 1
    record = json.loads(history[0].read_text())
    assert record["applied"] is False
    assert "reference analyzer" in record["error"]


def test_apply_artifact_revision_blocks_when_analyzer_cannot_revise(tmp_path):
    """A requested vision revision must never degrade into a text-only rewrite."""
    import json
    from studio_agent.artifact_studio import apply_artifact_revision
    from studio_agent.orchestrator.project import Project
    from studio_agent.providers.fake import FakeReferenceAnalyzer
    from studio_agent.stages.base import Providers

    # --- project + artifact setup (mirrors the vision tests above) ---
    p = Project.create("vision fallback", root=tmp_path, stages=["bible"])
    rel = "bible/characters/mara/character.json"
    cpath = p.path("bible", "characters", "mara", "character.json")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({"name": "Mara", "wardrobe": ""}))
    img = p.path("references", "uploads", "hero.png")
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\nhero")

    llm = JsonRevisionLLM({"name": "Mara", "wardrobe": "wrong fallback"})

    # --- analyzer whose revise() always raises NotImplementedError ---
    class FailingAnalyzer(FakeReferenceAnalyzer):
        def revise(self, image_paths, *, prompt, language="en"):
            raise NotImplementedError("vision not supported")

    with pytest.raises(ValueError, match="vision-grounded artifact revision"):
        apply_artifact_revision(
            p, rel, "keep the same outfit",
            Providers(llm=llm, reference_analyzer=FailingAnalyzer()),
            invalidate=False, image_paths=[str(img)],
        )

    assert llm.calls == []


def test_apply_artifact_revision_rejects_each_missing_requested_image(tmp_path):
    import json
    import re

    from studio_agent.orchestrator.project import Project
    from studio_agent.providers.fake import FakeReferenceAnalyzer

    p = Project.create("missing vision input", root=tmp_path, stages=["bible"])
    rel = "bible/characters/mara/character.json"
    cpath = p.path("bible", "characters", "mara", "character.json")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({"name": "Mara", "wardrobe": ""}))
    existing = p.path("references", "uploads", "hero.png")
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"\x89PNG\r\n\x1a\nhero")
    missing = p.path("references", "uploads", "missing.png")
    llm = JsonRevisionLLM({"name": "Mara", "wardrobe": "wrong fallback"})

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        apply_artifact_revision(
            p,
            rel,
            "keep the same outfit",
            Providers(llm=llm, reference_analyzer=FakeReferenceAnalyzer()),
            invalidate=False,
            image_paths=[str(existing), str(missing)],
        )

    assert llm.calls == []


def test_apply_text_prompt_revision_returns_to_prompt_gate_without_deleting_clip(tmp_path):
    p = _project(tmp_path)
    p.path("storyboard", "shots.json").write_text(json.dumps({
        "shots": [{"id": "sh-001", "keyframe": "sh-001.png"}],
    }))
    prompt = p.path("storyboard", "prompts", "sh-001.video.md")
    prompt.write_text("old prompt")
    clip = p.path("assets", "clips", "sh-001.mp4")
    clip.write_bytes(b"old paid clip")

    result = apply_artifact_revision(
        p,
        "storyboard/prompts/sh-001.video.md",
        "make motion stronger",
        Providers(llm=JsonRevisionLLM("new prompt")),
    )

    assert result.applied is True
    assert prompt.read_text().strip() == "new prompt"
    assert clip.read_bytes() == b"old paid clip"
    assert list(p.path("history").rglob("sh-001.mp4")) == []
    assert p.stage_status("video") == "pending"
    assert p.current_stage == "video_prompts"


def test_apply_artifact_revision_caps_images_to_provider_max_reference_images(tmp_path):
    """Vision images must be capped to provider.image.capabilities.max_reference_images."""
    import json
    from types import SimpleNamespace

    from studio_agent.artifact_studio import apply_artifact_revision
    from studio_agent.orchestrator.project import Project
    from studio_agent.providers.fake import FakeLLM, FakeReferenceAnalyzer
    from studio_agent.stages.base import Providers

    # --- project + artifact setup ---
    p = Project.create("cap test", root=tmp_path, stages=["bible"])
    rel = "bible/characters/mara/character.json"
    cpath = p.path("bible", "characters", "mara", "character.json")
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps({"name": "Mara", "wardrobe": ""}))

    # Create THREE real existing image files
    uploads = p.path("references", "uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    img_paths = []
    for i in range(1, 4):
        img = uploads / f"ref{i}.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + f"image{i}".encode())
        img_paths.append(str(img))

    # Recording analyzer: tracks exactly which image_paths it received
    received_image_lists: list[list[str]] = []

    class RecordingAnalyzer(FakeReferenceAnalyzer):
        def revise(self, image_paths, *, prompt, language="en"):
            received_image_lists.append(list(image_paths))
            return super().revise(image_paths, prompt=prompt, language=language)

    # Image provider stub with max_reference_images = 2
    image_stub = SimpleNamespace(
        capabilities=SimpleNamespace(max_reference_images=2)
    )

    apply_artifact_revision(
        p, rel, "update wardrobe details",
        Providers(
            llm=FakeLLM(),
            reference_analyzer=RecordingAnalyzer(),
            image=image_stub,
        ),
        invalidate=False,
        image_paths=img_paths,  # 3 images supplied
    )

    assert len(received_image_lists) == 1, "revise() should have been called exactly once"
    assert len(received_image_lists[0]) == 2, (
        f"Expected 2 images (capped), but analyzer received {len(received_image_lists[0])}"
    )
    # Subjects-first ordering preserved — first two paths survive
    assert received_image_lists[0] == img_paths[:2]
