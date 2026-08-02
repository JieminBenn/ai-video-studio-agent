"""Tests for the LLM image-prompt director (turns a structured brief into a dense,
model-optimized image prompt). Deterministic on fakes."""

from studio_agent.image_prompt_director import (
    build_director_prompt,
    direct_image_prompt,
    direct_video_prompt,
    directed_prompt_text,
    director_enabled,
)
from studio_agent.orchestrator.project import Project
from studio_agent.providers.base import Generation
from studio_agent.providers.fake import FakeLLM
from studio_agent.stages.base import Providers


class _DroppingLLM:
    """A model that ignores identity anchors and hard avoidances — and, like a cheap
    cross-language model, rewrites a Chinese brief into English. Reproduces the case
    where 神秘女子 / the verbatim avoidances vanish from the directed prompt, so the
    director's backfill must restore them."""

    name = "drop"
    model = "drop-1"

    def complete_json(self, prompt, *, system=None):
        # A realistic, detailed prompt that nonetheless translated the name and dropped
        # the avoidances — exactly the real DeepSeek output the backfill must repair.
        return Generation(
            content={
                "prompt": (
                    "a mysterious woman in a dim industrial loft, soft motivated key light "
                    "from a bare bulb, 85mm portrait lens at eye-level, shallow depth of field"
                ),
                "negative": "blurry, deformed hands",
            },
            provider="drop",
            model="drop-1",
        )

BRIEF = (
    "# Keyframe prompt\n\n"
    "## Subject & moment\n"
    "A clockmaker repairs a glowing pocket watch. Visible emotion: quiet awe.\n\n"
    "## Style\n"
    "Style: cartoon, bright candy colors, aspect 16:9.\n\n"
    "## Avoid\n"
    "text, watermark, blurry."
)


def _project(tmp_path):
    return Project.create("a sexy noir detective in neon rain", root=tmp_path, stages=["plot"])


def test_build_prompt_carries_task_skill_playbook_intent_and_aliases():
    prompt = build_director_prompt(
        brief=BRIEF,
        style={"look": "3d cg"},
        style_name="3d",
        intent="tasteful cinematic allure, realistic face",
        purpose="keyframe",
        reference_aliases=["Mara", "The Customer"],
        language="en",
        playbook="CG PLAYBOOK: subsurface skin, cloth simulation.",
        quality_baseline="Highest fidelity, correct hands.",
    )
    assert "[task:image_prompt]" in prompt
    assert "CG PLAYBOOK" in prompt
    assert "tasteful cinematic allure, realistic face" in prompt
    assert "Mara" in prompt and "The Customer" in prompt
    assert "Highest fidelity" in prompt
    assert "keyframe" in prompt
    # The structured brief is handed to the model under a findable BRIEF: marker.
    assert "BRIEF:" in prompt
    assert "A clockmaker repairs a glowing pocket watch" in prompt


def test_direct_image_prompt_returns_prompt_and_negative_on_fake(tmp_path):
    p = _project(tmp_path)
    gen = direct_image_prompt(
        Providers(llm=FakeLLM()),
        p,
        stage="storyboard",
        brief=BRIEF,
        style={"look": "cartoon"},
        style_name="cartoon",
        intent="",
        purpose="keyframe",
        reference_aliases=[],
        language="en",
    )
    assert isinstance(gen.content, dict)
    assert gen.content["prompt"].strip()
    assert "prompt" in gen.content and "negative" in gen.content
    # The fake distills the brief's subject + style (no markdown headers / skill dumps).
    assert "clockmaker repairs a glowing pocket watch" in gen.content["prompt"]
    assert "cartoon" in gen.content["prompt"]
    assert "##" not in gen.content["prompt"]


def test_direct_image_prompt_backfills_missing_identity_and_avoidances(tmp_path):
    """The director's own safety net: every required identity anchor and every hard
    avoidance is present verbatim — even when the model drops or translates them."""
    p = _project(tmp_path)
    gen = direct_image_prompt(
        Providers(llm=_DroppingLLM()),
        p,
        stage="bible",
        brief=BRIEF,
        purpose="character_reference",
        reference_aliases=["神秘女子"],
        language="zh",
        hard_avoidances=["避免血腥、暴力或过度恐怖元素", "避免低俗色情或露骨暗示"],
    )
    text = directed_prompt_text(gen.content)
    assert "神秘女子" in text
    assert "避免血腥、暴力或过度恐怖元素" in text
    assert "避免低俗色情或露骨暗示" in text


def test_direct_image_prompt_records_cost_and_guards_budget(tmp_path):
    p = _project(tmp_path)
    direct_image_prompt(
        Providers(llm=FakeLLM()),
        p,
        stage="bible",
        brief=BRIEF,
        purpose="character_reference",
    )
    assert any(entry["stage"] == "bible" for entry in p.cost_log)


VIDEO_BRIEF = (
    "# Video prompt — sh-001\n\n"
    "## Subject & action\n"
    "A keeper opens the lighthouse door. open.\n\n"
    "## Motion beats (~3s)\n"
    "- Beat 1 (0-1.5s): establish the frame.\n"
    "- Beat 2 (1.5-3s): the motion resolves.\n\n"
    "## Camera\n"
    "wide, with a gentle drifting push-in.\n\n"
    "## Style\n"
    "Style: cinematic, natural.\n\n"
    "## Avoid\n"
    "static pose, still image."
)


def test_video_mode_uses_video_task_and_skill():
    prompt = build_director_prompt(brief=VIDEO_BRIEF, media="video")
    assert "[task:video_prompt]" in prompt
    # The video discipline must require preserving motion + dialogue constraints.
    assert "motion beats" in prompt.lower()
    assert "dialogue" in prompt.lower()
    assert "target-state" in prompt.lower()


def test_direct_video_prompt_keeps_motion_on_fake(tmp_path):
    p = _project(tmp_path)
    gen = direct_video_prompt(
        Providers(llm=FakeLLM()),
        p,
        brief=VIDEO_BRIEF,
        style={"look": "cinematic"},
    )
    assert "Beat 1" in gen.content["prompt"]  # motion beats preserved
    assert "drifting push-in" in gen.content["prompt"]  # camera move preserved
    assert "##" not in gen.content["prompt"]
    assert any(e["stage"] == "video" for e in p.cost_log)


def test_directed_prompt_text_folds_negative_into_prompt():
    gen_content = {"prompt": "a vivid scene", "negative": "blurry, extra fingers"}
    text = directed_prompt_text(gen_content)
    assert "a vivid scene" in text
    assert "blurry, extra fingers" in text


def test_director_enabled_defaults_true_and_respects_toggle():
    assert director_enabled({}) is True
    assert director_enabled({"image_prompt_director": {"enabled": False}}) is False
    assert director_enabled({"image_prompt_director": {"enabled": True}}) is True


def test_build_director_prompt_grid_media():
    prompt = build_director_prompt(brief="BRIEF BODY", media="grid")
    assert "[task:grid_prompt]" in prompt
    assert "storyboard grid" in prompt.lower()
    # grid discipline: preserve layout + panels + consistency
    assert "layout" in prompt.lower()
    assert "panel" in prompt.lower()


VIDEO_BRIEF_WITH_GRID = (
    "# Video prompt — sh-grid\n\n"
    "## Subject & action\n"
    "A keeper opens the lighthouse door.\n\n"
    "## Storyboard grid\n"
    "The supplied image is a single 4-panel storyboard grid (2x2). Generate "
    "ONE continuous shot that follows the panels in order, left-to-right then "
    "top-to-bottom, interpolating smooth motion between them. Do not show the grid, "
    "the gutters, or multiple frames in the output — render only the live scene.\n\n"
    "## Motion beats (~3s)\n"
    "- Beat 1 (0-1.5s): establish the frame.\n"
    "- Beat 2 (1.5-3s): the motion resolves.\n\n"
    "## Camera\n"
    "wide, with a gentle drifting push-in.\n\n"
    "## Style\n"
    "Style: cinematic, natural.\n\n"
    "## Avoid\n"
    "static pose, still image."
)


def test_direct_video_prompt_preserves_storyboard_grid_on_fake(tmp_path):
    """The storyboard-grid directive must survive the video prompt director on the fake path.

    Root cause: ``_directed_prompt`` (FakeLLM) only extracted a fixed list of video
    sections and 'Storyboard grid' was not in it, so the panel-order instruction was
    silently dropped. ``build_director_prompt``'s ``prompt_rule`` also omitted
    the grid directive from its PRESERVE list, so real LLMs had no instruction to keep it.
    """
    p = _project(tmp_path)
    gen = direct_video_prompt(
        Providers(llm=FakeLLM()),
        p,
        brief=VIDEO_BRIEF_WITH_GRID,
        style={"look": "cinematic"},
    )
    text = directed_prompt_text(gen.content)
    assert "grid" in text.lower(), f"'grid' missing from directed output:\n{text}"
    assert "panel" in text.lower(), f"'panel' missing from directed output:\n{text}"


VIDEO_BRIEF_WITH_TARGET_STATE = (
    "# Video prompt — sh-transform\n\n"
    "## Subject & action\n"
    "A man transforms into a dragon.\n\n"
    "## State trajectory\n"
    "Man: human → dragon\n\n"
    "## Target state lock\n"
    "Final visible state must match the approved target-state reference: Man: dragon.\n"
    "Do not invent a different dragon design.\n\n"
    "## Motion beats (~6s)\n"
    "- Beat 1 (0-3s): human body begins changing.\n"
    "- Beat 2 (3-6s): the exact approved dragon is revealed.\n\n"
    "## Camera\n"
    "low-angle push-in.\n"
)


def test_direct_video_prompt_preserves_target_state_lock_on_fake(tmp_path):
    p = _project(tmp_path)
    gen = direct_video_prompt(
        Providers(llm=FakeLLM()),
        p,
        brief=VIDEO_BRIEF_WITH_TARGET_STATE,
        style={"look": "cinematic"},
    )
    text = directed_prompt_text(gen.content)
    assert "target-state" in text.lower()
    assert "Man: dragon" in text
    assert "different dragon design" in text


def test_image_director_rule_forbids_motion_but_video_rule_keeps_it():
    # The image rule must explicitly mark the prompt as a STATIC still (one frozen instant,
    # no camera movement / action sequence); the video rule must still preserve motion beats.
    image_prompt = build_director_prompt(brief="a hero at a door", media="image")
    assert "frozen instant" in image_prompt.lower()
    assert "no camera movement" in image_prompt.lower()

    video_prompt = build_director_prompt(brief="a hero at a door", media="video")
    assert "motion beats" in video_prompt.lower()
    assert "frozen instant" not in video_prompt.lower()


def test_enforce_backfills_a_dropped_appearance_lock_verbatim():
    from studio_agent.image_prompt_director import enforce_prompt_constraints

    lock = "Oval face, teardrop mole under left eye; red wool coat."
    out = enforce_prompt_constraints(
        {"prompt": "A woman reads a letter by a window.", "negative": "blurry"},
        verbatim_locks=[lock],
    )
    assert lock in out["prompt"]


def test_enforce_keeps_a_present_lock_without_duplicating():
    from studio_agent.image_prompt_director import enforce_prompt_constraints

    lock = "Oval face, teardrop mole under left eye; red wool coat."
    out = enforce_prompt_constraints(
        {"prompt": f"A woman reads a letter. {lock}", "negative": ""},
        verbatim_locks=[lock],
    )
    assert out["prompt"].count(lock) == 1


def test_enforce_backfills_a_dropped_style_signature_verbatim():
    from studio_agent.image_prompt_director import enforce_prompt_constraints

    signature = (
        "Art style and medium: 2D hand-drawn illustration. Render strictly in this exact "
        "art style and medium; do not turn a 2D illustration into a 3D render."
    )
    out = enforce_prompt_constraints(
        {"prompt": "A woman reads a letter by a window.", "negative": "blurry"},
        style_signature=signature,
    )
    assert signature in out["prompt"]


def test_enforce_keeps_a_present_style_signature_without_duplicating():
    from studio_agent.image_prompt_director import enforce_prompt_constraints

    signature = "Art style and medium: 2D hand-drawn illustration."
    out = enforce_prompt_constraints(
        {"prompt": f"A woman reads a letter. {signature}", "negative": ""},
        style_signature=signature,
    )
    assert out["prompt"].count(signature) == 1


def test_direct_image_prompt_backfills_dropped_style_signature(tmp_path):
    """Style is load-bearing like identity/avoidances: when the model re-renders the brief
    and drops the medium/style lock, the director's net must restore it verbatim so a
    single keyframe cannot drift into a different style."""
    signature = (
        "Art style and medium: 2D hand-drawn illustration. Render strictly in this exact "
        "art style and medium; do not substitute a 3D / CGI / photoreal render."
    )
    p = _project(tmp_path)
    gen = direct_image_prompt(
        Providers(llm=_DroppingLLM()),
        p,
        stage="keyframes",
        brief=BRIEF,
        purpose="keyframe",
        style_signature=signature,
    )
    text = directed_prompt_text(gen.content)
    assert signature in text


def test_build_director_prompt_states_budget_and_locks():
    from studio_agent.image_prompt_director import build_director_prompt

    prompt = build_director_prompt(
        brief="BRIEF",
        verbatim_locks=["Oval face, teardrop mole."],
        max_prompt_chars=1500,
    )
    assert "1500" in prompt
    assert "Oval face, teardrop mole." in prompt
    assert "verbatim" in prompt.lower()


def test_build_director_prompt_grid_branch_enforces_per_panel_coherence():
    from studio_agent.image_prompt_director import build_director_prompt

    prompt = build_director_prompt(brief="BRIEF", media="grid")
    lower = prompt.lower()
    # A grid is many panels: demand each panel be internally coherent (not one global
    # perspective), so no panel comes out broken/illogical.
    assert "physically coherent" in lower
    assert "panel" in lower
