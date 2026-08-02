"""LLM prompt director (the "prompt skill" that makes images and video good).

The deterministic compilers (``keyframe_prompt`` / ``video_prompt`` / the bible image
templates) produce a rich, structured *brief*. Generation models, however, render best
from a single dense, concrete prompt — not multi-section instructional markdown with
whole skill files dumped in. This module adds the missing step: an LLM that rewrites the
brief into the final, model-optimized prompt, guided by a media-specific prompting skill
(``image_prompting`` / ``video_prompting``) and a per-style playbook. Non-experts type
plain intent ("sexy", "realistic face", "3d") and the director turns it into
craft-rich prompt language.

Provider-agnostic (invariant #6): the only model call goes through ``providers.llm``.
Cost is logged per call (invariant #7). The result is a plain ``{prompt, negative}`` dict
the caller folds into the text it sends to the image/video provider; that text is still
saved as a hand-editable artifact by the stage (invariant #8). Fully deterministic on
``FakeLLM`` so the pipeline runs offline and in CI.
"""

from __future__ import annotations

import json
from typing import Any

from .providers.base import Generation
from .runtime_skills import load_prompt_skill

# Per-media task tag (FakeLLM dispatch) + prompting skill.
_MEDIA = {
    "image": {"task": "image_prompt", "skill": "image_prompting", "noun": "image"},
    "video": {"task": "video_prompt", "skill": "video_prompting", "noun": "video"},
    "grid": {"task": "grid_prompt", "skill": "storyboard_grid_prompting", "noun": "storyboard grid"},
}

DEFAULT_QUALITY_BASELINE = (
    "Render at the highest fidelity: sharp focus, coherent anatomy, correct hands (five "
    "fingers per hand) and eyes, natural skin and material texture, physically plausible "
    "objects that are supported and correctly scaled, a single consistent perspective and "
    "light direction, clean edges, controlled cinematic lighting, no text, watermark, or logo."
)


def director_enabled(config: dict[str, Any] | None) -> bool:
    """Whether the LLM prompt director is on (default: on). When off, stages fall back
    to sending the compiled brief directly — today's behavior — so there is always a
    safe path."""
    cfg = (config or {}).get("image_prompt_director")
    if cfg is None:
        return True
    if isinstance(cfg, dict):
        return bool(cfg.get("enabled", True))
    return bool(cfg)


def style_playbook(style: dict[str, Any] | None) -> str:
    """The per-style prompt playbook carried on the resolved style preset, if any."""
    return str((style or {}).get("prompt_playbook") or "").strip()


def quality_baseline(config: dict[str, Any] | None) -> str:
    """Project-level quality booster, defaulting to a sane baseline."""
    return str((config or {}).get("image_quality_baseline") or DEFAULT_QUALITY_BASELINE).strip()


def build_director_prompt(
    *,
    brief: str,
    style: dict[str, Any] | None = None,
    style_name: str = "",
    intent: str = "",
    purpose: str = "keyframe",
    reference_aliases: list[str] | None = None,
    language: str = "en",
    playbook: str = "",
    quality_baseline: str = DEFAULT_QUALITY_BASELINE,
    media: str = "image",
    verbatim_locks: list[str] | None = None,
    max_prompt_chars: int | None = None,
) -> str:
    """Compile the LLM instruction that rewrites a brief into a final prompt.

    ``media`` selects the prompting skill + task tag and the output discipline: an image
    prompt is one dense still description; a video prompt must additionally preserve the
    brief's motion beats, camera move, continuity/last-frame rules, exact dialogue, and
    any provider capability constraints."""
    spec = _MEDIA.get(media, _MEDIA["image"])
    skill = load_prompt_skill(spec["skill"])
    style = style or {}
    aliases = ", ".join(a for a in (reference_aliases or []) if a) or "(none)"
    look = style_name or str(style.get("look") or "cinematic")
    if media == "video":
        role = "AI video generation"
        prompt_rule = (
            "- prompt: vivid, motion-forward direction. PRESERVE from the brief, verbatim "
            "where they are constraints: the time-keyed motion beats, the camera move, the "
            "continuity / last-frame instructions, the exact spoken dialogue lines, any "
            "provider capability limits, any target-state locks for transformations, and, "
            "if present, the storyboard-grid directive "
            "(that the supplied image is an N-panel grid to follow panel-by-panel in order, "
            "producing one continuous shot without showing the grid or gutters). "
            "Do not invent new references the brief withholds. "
            "No markdown headings or skill text.\n"
        )
    elif media == "grid":
        role = "AI storyboard contact-sheet generation"
        prompt_rule = (
            "- prompt: one dense description of a SINGLE composite contact-sheet image. "
            "PRESERVE from the brief, verbatim where they are constraints: the exact panel "
            "count and R x C grid layout, the left-to-right top-to-bottom read order, the "
            "per-panel motion beats in order, and the rule that every panel shows the SAME "
            "character and SAME location with only pose/action/camera changing. "
            "Keep every panel physically coherent on its own — correct anatomy and counts "
            "(two hands, five fingers), supported and correctly-scaled objects, and one coherent "
            "perspective within each panel — so no panel comes out broken or illogical. "
            "Forbid panel numbers, captions, and any non-grid single image. No markdown, no headings.\n"
        )
    else:
        role = "AI image generation"
        prompt_rule = (
            "- prompt: one vivid paragraph of concrete nouns, materials, lighting, lens, "
            "palette, mood. No markdown, no headings, no meta-instructions, no skill text. "
            "This is a STATIC still image: describe ONE frozen instant — the shot's opening "
            "moment — as a held pose, not an action playing out. No camera movement, no motion "
            "blur, no sequence of events, no temporal/'then' progression; if the brief narrates "
            "motion, freeze it into the single most telling frame. "
            "Ground every element physically: state where each subject and object sits, what "
            "supports it, and what it touches or occludes; keep counts and anatomy correct (two "
            "hands, five fingers) and hold one coherent perspective and light direction so the "
            "frame reads as real, not broken.\n"
        )
    locks_block = "\n".join(f"- {lock}" for lock in (verbatim_locks or []) if lock)
    locks_line = (
        "LOCKED_APPEARANCE (reproduce each line VERBATIM, unchanged, somewhere in the prompt):\n"
        + locks_block
        + "\n\n"
        if locks_block
        else ""
    )
    budget_line = (
        f"TARGET_LENGTH: write a rich, exhaustive prompt up to about {max_prompt_chars} "
        "characters — be as detailed and specific as possible without exceeding it.\n"
        if max_prompt_chars
        else ""
    )
    return (
        f"[task:{spec['task']}]\n"
        f"You are a master prompt engineer for {role}. Rewrite the BRIEF into ONE dense, "
        f"concrete, model-ready {spec['noun']} prompt that will render at the highest "
        "quality. The user may not know how to prompt — infer the strongest, most tasteful "
        "interpretation of their intent and make it specific.\n\n"
        f"PROMPTING_SKILL:\n{skill}\n\n"
        f"STYLE_PLAYBOOK ({look}):\n{playbook or '(none — use general knowledge of this look)'}\n\n"
        f"QUALITY_BASELINE:\n{quality_baseline}\n\n"
        f"CREATIVE_INTENT (honor tastefully, stay within provider safety): {intent or '(none)'}\n"
        f"PURPOSE: {purpose}\n"
        f"PRESERVE_IDENTITY (faces/wardrobe must match references): {aliases}\n"
        f"LANGUAGE: {language}\n\n"
        "Output rules:\n"
        '- Return only JSON: {"prompt": "...", "negative": "..."}\n'
        f"{prompt_rule}"
        "- Lean into the style playbook's idiom; keep the named subject and composition "
        "from the brief; preserve identity references.\n"
        "- negative: short comma-separated list of defects to avoid.\n\n"
        f"{locks_line}"
        f"{budget_line}"
        f"BRIEF:\n{brief}\n"
    )


def _normalize_for_match(text: Any) -> str:
    """Casefold + collapse whitespace for tolerant verbatim-presence matching."""
    return " ".join(str(text or "").casefold().split())


def enforce_prompt_constraints(
    content: Any,
    *,
    reference_aliases: list[str] | None = None,
    hard_avoidances: list[str] | None = None,
    verbatim_locks: list[str] | None = None,
    style_signature: str = "",
) -> dict[str, str]:
    """Guarantee the directed prompt keeps every identity anchor and hard avoidance.

    These are the load-bearing constraints: every required identity anchor, the style /
    medium lock, and every hard avoidance must appear *verbatim* in the prompt text.
    Cheap or cross-language models routinely translate the character name, rephrase the
    avoidances (e.g. a 中文 project where 神秘女子 became "mysterious woman"), or dilute the
    art-style/medium instruction — the last of which lets a single keyframe drift into a
    different look. Rather than trust the model, deterministically backfill anything it
    dropped: missing identity anchors and the style signature into the prompt, missing
    avoidances into the negative. This is the director's own safety net, independent of
    any external gate."""
    parsed = parse_director_content(content)
    prompt = parsed["prompt"]
    negative = parsed["negative"]

    combined = _normalize_for_match(f"{prompt} {negative}")
    missing_aliases = [
        alias for alias in (reference_aliases or [])
        if alias and _normalize_for_match(alias) not in combined
    ]
    if missing_aliases:
        prompt = " ".join(p for p in (prompt, ", ".join(missing_aliases)) if p)

    combined = _normalize_for_match(f"{prompt} {negative}")
    if style_signature and _normalize_for_match(style_signature) not in combined:
        prompt = " ".join(p for p in (prompt, style_signature) if p)

    combined = _normalize_for_match(f"{prompt} {negative}")
    for lock in (verbatim_locks or []):
        if lock and _normalize_for_match(lock) not in combined:
            prompt = " ".join(p for p in (prompt, lock) if p)
            combined = _normalize_for_match(f"{prompt} {negative}")

    combined = _normalize_for_match(f"{prompt} {negative}")
    missing_avoidances = [
        rule for rule in (hard_avoidances or [])
        if rule and _normalize_for_match(rule) not in combined
    ]
    if missing_avoidances:
        negative = ", ".join(n for n in ([negative] if negative else []) + missing_avoidances)

    return {"prompt": prompt, "negative": negative}


def direct_image_prompt(
    providers,
    project,
    *,
    brief: str,
    style: dict[str, Any] | None = None,
    style_name: str = "",
    intent: str = "",
    purpose: str = "keyframe",
    reference_aliases: list[str] | None = None,
    hard_avoidances: list[str] | None = None,
    language: str = "en",
    stage: str = "storyboard",
    media: str = "image",
    verbatim_locks: list[str] | None = None,
    max_prompt_chars: int | None = None,
    style_signature: str = "",
) -> Generation:
    """Run the director: brief -> ``Generation`` whose content is ``{prompt, negative}``."""
    config = project.model_config or {}
    prompt = build_director_prompt(
        brief=brief,
        style=style,
        style_name=style_name,
        intent=intent,
        purpose=purpose,
        reference_aliases=reference_aliases,
        language=language,
        playbook=style_playbook(style),
        quality_baseline=quality_baseline(config),
        media=media,
        verbatim_locks=verbatim_locks,
        max_prompt_chars=max_prompt_chars,
    )
    project.assert_budget_available()
    gen = providers.llm.complete_json(prompt)
    project.add_generation_cost(stage=stage, generation=gen)
    gen.content = enforce_prompt_constraints(
        gen.content,
        reference_aliases=reference_aliases,
        hard_avoidances=hard_avoidances,
        verbatim_locks=verbatim_locks,
        style_signature=style_signature,
    )
    return gen


def direct_video_prompt(providers, project, *, brief: str, **kwargs) -> Generation:
    """Convenience wrapper: run the director in video mode."""
    kwargs.setdefault("stage", "video")
    kwargs.setdefault("purpose", "clip")
    return direct_image_prompt(providers, project, brief=brief, media="video", **kwargs)


def direct_grid_prompt(providers, project, *, brief: str, **kwargs) -> Generation:
    """Convenience wrapper: run the director in storyboard-grid mode."""
    kwargs.setdefault("stage", "clip")
    kwargs.setdefault("purpose", "storyboard_grid")
    return direct_image_prompt(providers, project, brief=brief, media="grid", **kwargs)


def directed_prompt_text(content: dict[str, Any] | None) -> str:
    """Fold the director's ``{prompt, negative}`` into the text sent to the image model."""
    content = content or {}
    prompt = str(content.get("prompt") or "").strip()
    negative = str(content.get("negative") or "").strip()
    if negative:
        prompt = f"{prompt}\n\nAvoid: {negative}"
    return prompt.strip()


def parse_director_content(content: Any) -> dict[str, str]:
    """Normalize an LLM response into ``{prompt, negative}`` (tolerant of strings)."""
    if isinstance(content, dict):
        return {
            "prompt": str(content.get("prompt") or "").strip(),
            "negative": str(content.get("negative") or "").strip(),
        }
    if isinstance(content, str):
        try:
            return parse_director_content(json.loads(content))
        except (ValueError, TypeError):
            return {"prompt": content.strip(), "negative": ""}
    return {"prompt": "", "negative": ""}
