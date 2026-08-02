import json

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.prompt_conversation import (
    build_prompt_context,
    conversation_records,
    revise_prompt_turn,
)
from studio_agent.prompt_approvals import PromptApprovalError
from studio_agent.providers.base import Generation
from studio_agent.providers.fake import FakeLLM, FakeReferenceAnalyzer
from studio_agent.stages.base import Providers


class RecordingLLM(FakeLLM):
    def __init__(self):
        self.prompts = []

    def complete(self, prompt, *, system=None):
        self.prompts.append(prompt)
        current = prompt.split("CURRENT PROMPT:\n", 1)[1].split(
            "\n\nRETURN ONLY", 1
        )[0]
        return Generation(
            content=current + "\nStatic revision: held pose, warmer light.",
            provider="recording-llm",
            model="recording-1",
        )


class RecordingAnalyzer:
    def __init__(self):
        self.received_paths = []

    def revise(self, image_paths, *, prompt, language="en"):
        self.received_paths = list(image_paths)
        current = prompt.split("CURRENT PROMPT:\n", 1)[1].split(
            "\n\nRETURN ONLY", 1
        )[0]
        return Generation(
            content=current + "\nStatic revision: identity matched in a held pose.",
            provider="recording-vision",
            model="vision-1",
        )


def _project(tmp_path):
    project = Project.create(
        "PROJECT IDEA: Mara rebuilds a clockwork bird",
        root=tmp_path,
        stages=["clip", "keyframes", "video_prompts", "video"],
    )
    project.path("story", "idea.md").write_text(project.idea)
    project.path("story", "creative_brief.json").write_text(json.dumps({
        "intent_summary": "intimate mechanical wonder",
        "visual_direction": {"tone": "warm and precise"},
    }))
    project.path("storyboard", "visual_sequence.json").write_text(json.dumps({
        "intent_class": "micro-arc",
        "beats": ["repair", "awakening"],
    }))
    identity = project.path("bible", "characters", "mara", "identity_board.json")
    identity.parent.mkdir(parents=True, exist_ok=True)
    identity.write_text(json.dumps({
        "name": "Mara",
        "canonical_appearance": "short black hair, red work coat",
    }))
    reference = identity.parent / "reference.png"
    reference.write_bytes(b"reference")
    upload = project.path("references", "uploads", "mara-face.png")
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(b"upload")
    project.path("references", "references.json").write_text(json.dumps({
        "references": [{
            "alias": "@image1",
            "status": "resolved",
            "target_type": "character",
            "target_id": "Mara",
            "path": "references/uploads/mara-face.png",
            "note": "match her face",
        }]
    }))
    shots = [
        {
            "id": "sh-001",
            "scene": 1,
            "keyframe": "sh-001.png",
            "characters": ["Mara"],
            "reference_images": [
                "bible/characters/mara/reference.png",
                "references/uploads/mara-face.png",
            ],
            "start_frame": "Mara sits with her back facing us beside the mouse",
            "camera": "medium rear view",
            "camera_movement": "slow push-in",
            "action": "Mara repairs the clockwork bird",
        },
        {
            "id": "sh-002",
            "scene": 1,
            "keyframe": "sh-002.png",
            "characters": ["Mara"],
            "reference_images": ["bible/characters/mara/reference.png"],
            "start_frame": "The clockwork bird rests in Mara's hands",
            "camera": "close-up",
            "camera_movement": "locked-off",
            "action": "The clockwork bird wakes",
        },
    ]
    project.path("storyboard", "shots.json").write_text(json.dumps({"shots": shots}))
    prompts = project.path("storyboard", "prompts")
    prompts.joinpath("sh-001.keyframe.md").write_text(
        "Mara sits in a static rear-view pose beside the workbench mouse."
    )
    prompts.joinpath("sh-002.keyframe.md").write_text("unchanged")
    prompts.joinpath("sh-001.video.md").write_text(
        "Mara repairs the bird as the camera slowly pushes in."
    )
    prompts.joinpath("sh-002.video.md").write_text(
        "The bird wakes while the camera remains locked-off."
    )
    return project


def test_selected_turn_receives_project_memory_and_changes_only_selected_prompt(tmp_path):
    project = _project(tmp_path)
    llm = RecordingLLM()

    result = revise_prompt_turn(
        project,
        gate="keyframes",
        shot_id="sh-001",
        message="Keep her back facing us and put the hand beside the mouse.",
        apply_to_all=False,
        providers=Providers(llm=llm),
    )

    assert result.changed_paths == ["storyboard/prompts/sh-001.keyframe.md"]
    sent = llm.prompts[-1]
    for token in ("PROJECT IDEA", "IDENTITY BOARD", "sh-001", "sh-002", "CURRENT PROMPT"):
        assert token in sent
    assert project.path("storyboard", "prompts", "sh-002.keyframe.md").read_text() == "unchanged"


def test_conversation_persists_across_reload_and_video_gate(tmp_path):
    project = _project(tmp_path)
    providers = Providers(llm=RecordingLLM())
    revise_prompt_turn(
        project, gate="keyframes", shot_id="sh-001", message="warmer light",
        apply_to_all=False, providers=providers,
    )

    reopened = Project.load(project.dir)
    revise_prompt_turn(
        reopened, gate="videos", shot_id="sh-001", message="start slower",
        apply_to_all=False, providers=providers,
    )

    records = conversation_records(reopened)
    assert [record["gate"] for record in records if record["role"] == "user"] == [
        "keyframes", "videos"
    ]


def test_vision_capable_turn_receives_actual_reference_paths(tmp_path):
    project = _project(tmp_path)
    analyzer = RecordingAnalyzer()

    result = revise_prompt_turn(
        project,
        gate="keyframes",
        shot_id="sh-001",
        message="match the uploaded face",
        apply_to_all=False,
        providers=Providers(llm=RecordingLLM(), reference_analyzer=analyzer),
    )

    assert analyzer.received_paths == [
        str(project.path("bible", "characters", "mara", "reference.png")),
        str(project.path("references", "uploads", "mara-face.png")),
    ]
    assert result.vision_mode == "vision"


def test_text_only_turn_discloses_images_were_not_seen(tmp_path):
    project = _project(tmp_path)

    result = revise_prompt_turn(
        project,
        gate="keyframes",
        shot_id="sh-001",
        message="match the face",
        apply_to_all=False,
        providers=Providers(llm=RecordingLLM()),
    )

    assert result.vision_mode == "text_only"
    assert "did not directly inspect images" in result.disclosure


def test_non_vision_analyzer_falls_back_without_claiming_it_saw_images(tmp_path):
    project = _project(tmp_path)
    llm = RecordingLLM()

    class NonVisionAnalyzer:
        def revise(self, image_paths, *, prompt, language="en"):
            raise NotImplementedError

    result = revise_prompt_turn(
        project,
        gate="keyframes",
        shot_id="sh-001",
        message="match the face",
        apply_to_all=False,
        providers=Providers(llm=llm, reference_analyzer=NonVisionAnalyzer()),
    )

    assert result.vision_mode == "text_only"
    assert "VISION AVAILABLE FOR THIS TURN: False" in llm.prompts[-1]


def test_build_context_reads_current_prompt_fresh(tmp_path):
    project = _project(tmp_path)
    prompt_path = project.path("storyboard", "prompts", "sh-001.keyframe.md")
    prompt_path.write_text("fresh hand edit")

    context, _images = build_prompt_context(
        project,
        gate="keyframes",
        shot_id="sh-001",
        message="keep this",
        providers=Providers(llm=RecordingLLM()),
    )

    assert "fresh hand edit" in context


def test_apply_all_writes_nothing_when_one_static_candidate_is_invalid(tmp_path):
    project = _project(tmp_path)
    before = {
        shot_id: project.path(
            "storyboard", "prompts", f"{shot_id}.keyframe.md"
        ).read_text()
        for shot_id in ("sh-001", "sh-002")
    }

    class InvalidSecondLLM(RecordingLLM):
        def complete(self, prompt, *, system=None):
            if "SELECTED SHOT: sh-002" in prompt:
                # An empty revision is the remaining invalid case (keyframe prompts are no longer
                # prose-validated); apply-to-all must still roll back all writes atomically.
                return Generation(
                    content="   ",
                    provider="bad",
                    model="bad-1",
                )
            return super().complete(prompt, system=system)

    with pytest.raises(PromptApprovalError):
        revise_prompt_turn(
            project,
            gate="keyframes",
            shot_id="sh-001",
            message="apply warm light everywhere",
            apply_to_all=True,
            providers=Providers(llm=InvalidSecondLLM()),
        )

    assert {
        shot_id: project.path(
            "storyboard", "prompts", f"{shot_id}.keyframe.md"
        ).read_text()
        for shot_id in ("sh-001", "sh-002")
    } == before


def test_fake_vision_revision_preserves_static_prompt_contract(tmp_path):
    project = _project(tmp_path)

    result = revise_prompt_turn(
        project,
        gate="keyframes",
        shot_id="sh-001",
        message="make the light warmer",
        apply_to_all=False,
        providers=Providers(
            llm=FakeLLM(),
            reference_analyzer=FakeReferenceAnalyzer(),
        ),
    )

    assert result.vision_mode == "vision"
    revised = project.path("storyboard", "prompts", "sh-001.keyframe.md").read_text()
    assert "frozen held pose" in revised
