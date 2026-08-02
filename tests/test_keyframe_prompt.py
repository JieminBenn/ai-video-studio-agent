"""Tests for the structured keyframe prompt compiler.

Keyframes are *stills*, so the compiler emphasizes composition, lighting, lens, and
identity rather than motion. Deterministic and offline — no provider, no network.
"""

from studio_agent.keyframe_prompt import compile_keyframe_prompt

SHOT = {
    "camera": "medium close-up",
    "action": "Mara grips the lighthouse rail",
    "description": "storm light at dusk",
    "start_frame": "Mara stands still with both hands on the lighthouse rail",
}

STYLE = {
    "look": "cartoon",
    "palette": "bright candy colors",
    "aspect_ratio": "4:3",
    "rendering": "soft rounded shapes",
}


def test_locked_genre_reaches_the_keyframe_story_beat():
    out = compile_keyframe_prompt(SHOT, style=STYLE, context={"genre": "中国古代神话"})
    assert "中国古代神话" in out


def test_style_reference_emits_a_style_only_guard():
    out = compile_keyframe_prompt(SHOT, style=STYLE, has_style_reference=True).lower()
    assert "style reference" in out
    # Style only — match the look, never copy the reference's subject/composition.
    assert "do not copy" in out
    assert "subject" in out and "composition" in out


def test_no_style_reference_means_no_guard():
    out = compile_keyframe_prompt(SHOT, style=STYLE).lower()
    assert "do not copy" not in out


def test_includes_camera_static_opening_description_and_style():
    out = compile_keyframe_prompt(SHOT, style=STYLE)
    assert "medium close-up" in out
    assert "Mara stands still with both hands on the lighthouse rail" in out
    assert "storm light at dusk" in out
    assert "cartoon" in out
    assert "soft rounded shapes" in out


def test_has_composition_and_lighting_and_lens_direction():
    low = compile_keyframe_prompt(SHOT, style=STYLE).lower()
    assert "composition" in low
    assert "lighting" in low
    assert "lens" in low


def test_keyframe_optics_reflect_shot_size_angle_and_lens_intent():
    # shot_size / camera_angle / lens_intent are computed by normalize_shot_design but were
    # previously discarded by the keyframe compiler. They must reach the still now.
    shot = {
        **SHOT,
        "camera": "wide establishing",
        "shot_size": "wide establishing",
        "camera_angle": "low angle",
        "lens_intent": "wide environmental perspective with controlled edge distortion",
    }
    low = compile_keyframe_prompt(shot, style=STYLE).lower()
    assert "low angle" in low
    assert "24-35mm" in low or "wide lens" in low
    assert "environmental perspective" in low


def test_keyframe_extreme_close_up_uses_macro_or_telephoto():
    shot = {**SHOT, "camera": "extreme close-up", "shot_size": "extreme close-up on the eyes"}
    low = compile_keyframe_prompt(shot, style=STYLE).lower()
    assert "macro" in low or "100mm" in low or "telephoto" in low


def _section(out: str, header: str) -> str:
    """Return the body of a '## header' section from a compiled prompt."""
    blocks = out.split("## ")
    for block in blocks:
        if block.startswith(header):
            return block
    return ""


def test_composition_adapts_framing_to_shot_size():
    wide = _section(
        compile_keyframe_prompt({**SHOT, "shot_size": "wide establishing vista"}, style=STYLE),
        "Shot & composition",
    ).lower()
    assert "negative space" in wide or "environment" in wide
    close = _section(
        compile_keyframe_prompt({**SHOT, "shot_size": "extreme close-up"}, style=STYLE),
        "Shot & composition",
    ).lower()
    assert "headroom" in close or "upper third" in close


def test_composition_passes_beat_emotion_through_for_the_model():
    # Invariant #11: code must not infer meaning from the free-text emotion — it passes it
    # through verbatim as a compositional directive for the model to interpret.
    shot = {**SHOT, "emotion": "dread of the approaching storm"}
    section = _section(compile_keyframe_prompt(shot, style=STYLE), "Shot & composition")
    assert "dread of the approaching storm" in section


def test_style_lighting_and_color_grade_reach_the_lighting_section():
    # The style dict carries specific lighting/color_grade; they previously only reached the
    # compact ## Style line, never the lighting engine. Now they must shape the lighting.
    style = {
        **STYLE,
        "lighting": "hard chiaroscuro with deep shadows",
        "color_grade": "bleach-bypass teal-orange",
    }
    lighting_section = _section(compile_keyframe_prompt(SHOT, style=style), "Lighting & lens")
    assert "chiaroscuro" in lighting_section
    assert "bleach-bypass" in lighting_section


def test_avoids_still_image_defects():
    low = compile_keyframe_prompt(SHOT, style=STYLE).lower()
    assert "avoid" in low
    for term in ("text", "watermark", "deformed", "extra"):
        assert term in low


def test_identity_section_lists_aliases_when_referenced():
    out = compile_keyframe_prompt(SHOT, style=STYLE, ref_aliases=["Mara", "the keeper"])
    low = out.lower()
    assert "reference image" in low
    assert "the keeper" in out


def test_no_identity_section_without_references():
    low = compile_keyframe_prompt(SHOT, style=STYLE).lower()
    assert "reference image" not in low


def test_expression_only_when_speaking_with_a_sheet():
    speaking = compile_keyframe_prompt(
        SHOT, style=STYLE, ref_aliases=["Mara"], has_expression_sheet=True, speaking=True
    )
    silent = compile_keyframe_prompt(
        SHOT, style=STYLE, ref_aliases=["Mara"], has_expression_sheet=True, speaking=False
    )
    no_sheet = compile_keyframe_prompt(
        SHOT, style=STYLE, ref_aliases=["Mara"], has_expression_sheet=False, speaking=True
    )
    assert "expression" in speaking.lower()
    assert "expression" not in silent.lower()
    assert "expression" not in no_sheet.lower()


def test_includes_location_references_and_prompt_skill_guidance():
    out = compile_keyframe_prompt(
        SHOT,
        style=STYLE,
        location_aliases=["Clock Shop"],
        prompt_skills={"location_identity": "Keep architecture, props, palette, and light states locked."},
    )

    assert "Clock Shop" in out
    assert "architecture, props, palette" in out


def test_includes_product_format_when_given():
    fmt = {"name": "short_drama_episode", "label": "Short drama episode"}
    out = compile_keyframe_prompt(SHOT, style=STYLE, product_format=fmt)
    assert "Short drama episode" in out


def test_includes_static_world_context_and_adjacent_frame_continuity():
    context = {
        "idea": "A princess leaves a television and enters a lonely home.",
        "story": {"logline": "A fairy-tale princess crosses into a real apartment."},
        "scene": {
            "heading": "INT. HOME - NIGHT",
            "beats": ["The TV glows", "The princess steps onto the carpet"],
        },
        "previous_shot": {
            "id": "sh-001", "camera": "wide", "action": "the TV flares",
            "end_frame": "the television holds a pale blue glow",
        },
        "next_shot": {
            "id": "sh-003", "camera": "close-up", "action": "the homeowner gasps",
            "start_frame": "the homeowner faces the television in close-up",
        },
    }

    out = compile_keyframe_prompt(SHOT, style=STYLE, context=context)

    assert "Static world context" in out
    assert "INT. HOME - NIGHT" in out
    assert "fairy-tale princess crosses" not in out
    assert "The princess steps onto the carpet" not in out
    assert "the television holds a pale blue glow" in out
    assert "the homeowner faces the television in close-up" in out
    assert "the TV flares" not in out
    assert "the homeowner gasps" not in out


def test_context_preserves_chinese_text():
    context = {
        "genre": "东方奇幻",
        "idea": "一个灯塔守望者遇见会说话的海鸥",
        "story": {"logline": "一部关于孤独与勇气的短片。"},
        "scene": {"heading": "EXT. LIGHTHOUSE - DAWN", "beats": ["海鸥发出警告"]},
    }

    out = compile_keyframe_prompt(SHOT, style=STYLE, context=context)

    assert "东方奇幻" in out
    assert "一个灯塔守望者" not in out
    assert "孤独与勇气" not in out
    assert "海鸥发出警告" not in out


def test_retrieved_knowledge_follows_locked_identity_references():
    guidance = "Use foreground occlusion to support unease."
    out = compile_keyframe_prompt(
        SHOT,
        style=STYLE,
        ref_aliases=["Mara"],
        knowledge_guidance=guidance,
    )

    assert guidance in out
    assert out.index("Mara") < out.index(guidance)


def test_character_rules_section_lists_do_dont_and_hero_props():
    # do/dont/hero_props are generated into the identity board but were previously stranded
    # (UI + post-hoc review only). They must now reach the keyframe brief to shape generation.
    rules = {
        "Mara": {
            "do": ["keep the copper scarf visible"],
            "dont": ["never remove her gloves"],
            "hero_props": ["brass telescope"],
            "continuity_priority": ["face", "silhouette"],
        }
    }
    section = _section(
        compile_keyframe_prompt(SHOT, style=STYLE, character_rules=rules), "Character rules"
    )
    assert "keep the copper scarf visible" in section
    assert "never remove her gloves" in section
    assert "brass telescope" in section


def test_no_character_rules_section_when_empty():
    assert "## Character rules" not in compile_keyframe_prompt(SHOT, style=STYLE, character_rules={})


def test_stateful_keyframe_anchors_only_the_opening_state():
    shot = {
        **SHOT,
        "start_frame": "",
        "characters": ["男人"],
        "start_states": {"男人": "human"},
        "end_states": {"男人": "gas-onset"},
        "visual_beats": [
            {"id": "human", "description": "普通男人静立在马路上，没有红色气体"},
            {"id": "gas-onset", "description": "红色气体开始出现"},
        ],
    }

    out = compile_keyframe_prompt(shot, style=STYLE, ref_aliases=["男人"])

    assert "## Opening visual state" in out
    assert "男人: human" in out
    assert "普通男人静立在马路上，没有红色气体" in out
    assert "Do not show later states" in out


def test_keyframe_prompt_demands_a_single_frozen_instant_not_motion():
    # A keyframe is the image-to-video seed (first frame): the still must depict one frozen
    # instant — the shot's opening — not the action playing out or a camera move. The video
    # prompt owns motion; the keyframe must not describe it.
    out = compile_keyframe_prompt(
        {
            "camera": "wide shot",
            "camera_movement": "fast orbit around Mara",
            "action": "Mara runs across the room and grabs the ringing phone",
            "description": "cramped apartment at night",
            "start_frame": "Mara stands beside the sofa, looking toward the phone",
            "end_frame": "Mara holds the phone beside the far wall",
        },
        style=STYLE,
    )
    assert "one sharp still image" in out.lower()
    assert "Mara stands beside the sofa" in out
    assert "runs across the room" not in out
    assert "fast orbit" not in out
    assert "Mara holds the phone beside the far wall" not in out
    assert "camera movement" not in out.lower()


def test_compile_emits_verbatim_character_appearance_lock():
    shot = {
        "id": "sh-001",
        "camera": "medium shot",
        "action": "Mara studies a letter",
        "start_frame": "Mara holds a still pose with the letter",
    }
    out = compile_keyframe_prompt(
        shot,
        ref_aliases=["Mara"],
        character_locks={"Mara": "Oval face, teardrop mole under left eye; red wool coat."},
    )
    assert "## Character appearance lock (verbatim — do not alter)" in out
    assert "Mara: Oval face, teardrop mole under left eye; red wool coat." in out


def test_compile_omits_lock_section_without_locks():
    out = compile_keyframe_prompt(
        {"id": "sh-001", "camera": "medium shot", "start_frame": "a still pose"},
        ref_aliases=["Mara"],
    )
    assert "Character appearance lock" not in out


def test_compile_avoid_line_includes_coherence_negatives():
    out = compile_keyframe_prompt(
        {"id": "sh-001", "camera": "medium shot", "start_frame": "a still pose"},
    )
    lower = out.lower()
    assert "fused fingers" in lower
    assert "floating objects" in lower
    assert "inconsistent perspective" in lower


def test_same_character_lock_is_identical_across_two_shots():
    lock = "Mara: Oval face, teardrop mole under left eye; red wool coat."
    locks = {"Mara": "Oval face, teardrop mole under left eye; red wool coat."}
    shot_a = {"id": "sh-001", "camera": "wide shot", "start_frame": "Mara by the door"}
    shot_b = {"id": "sh-002", "camera": "close-up", "start_frame": "Mara at the desk"}

    out_a = compile_keyframe_prompt(shot_a, ref_aliases=["Mara"], character_locks=locks)
    out_b = compile_keyframe_prompt(shot_b, ref_aliases=["Mara"], character_locks=locks)

    assert lock in out_a
    assert lock in out_b
