"""Tests for the provider-aware video prompt compiler.

These exercise the deterministic prompt-compiler layer only — no provider, no
network, no cost. They lock in that compiled prompts encourage real motion and that
the compiler respects a provider's reference/continuity capability limits (e.g. the
BytePlus ModelArk profile must not mix a first-frame keyframe with extra reference
media).
"""

from studio_agent.providers.base import VideoCapabilities
from studio_agent.video_prompt import compile_video_prompt

SHOT = {
    "id": "sh-001",
    "camera": "wide establishing",
    "action": "Mara steps onto the windswept pier",
    "description": "A lone figure at dawn",
    "dialogue": [{"character": "Mara", "line": "I shouldn't be here."}],
    "duration_s": 4.0,
    "characters": ["Mara"],
    "reference_seed": 11,
    "reference_images": ["bible/characters/mara/reference.png"],
}

STYLE = {
    "look": "anime",
    "palette": "moonlit teal and warm brass",
    "rendering": "clean cel shading",
    "motion": "controlled limited-animation timing",
}


def test_style_reference_emits_a_style_only_guard():
    out = compile_video_prompt(SHOT, style=STYLE, has_style_reference=True).lower()
    assert "style reference" in out
    assert "do not copy" in out and "composition" in out


def test_no_style_reference_means_no_guard():
    out = compile_video_prompt(SHOT, style=STYLE).lower()
    assert "do not copy" not in out


def test_sound_section_uses_exact_line_and_forbids_music():
    shot = {
        **SHOT,
        "dialogue": [{"character": "Mara", "line": "I shouldn't be here."}],
        "reference_locations": ["Windswept Pier"],
    }

    out = compile_video_prompt(shot, style=STYLE)

    assert out.count("## Sound") == 1
    assert "## Dialogue" not in out
    assert "Mara: I shouldn't be here." in out
    for phrase in ("no background music", "no score", "no singing", "no beat"):
        assert phrase in out.lower()


def test_locked_genre_reaches_the_video_prompt_story_beat():
    out = compile_video_prompt(SHOT, style=STYLE, context={"genre": "中国古代神话"})
    assert "中国古代神话" in out


def test_narration_is_off_screen_voiceover_not_lip_synced():
    shot = {
        **SHOT,
        "dialogue": [],
        "narration": "Once, a clockmaker mended more than time.",
    }

    out = compile_video_prompt(shot, style=STYLE)

    assert "Once, a clockmaker mended more than time." in out
    lower = out.lower()
    assert "narrator" in lower
    assert "off-screen" in lower or "off screen" in lower
    # The clip must not lip-sync the narration to any character.
    assert "do not lip-sync" in lower or "not lip-synced" in lower or "do not lip sync" in lower


def test_non_speaking_sound_section_prohibits_voices():
    out = compile_video_prompt({**SHOT, "dialogue": []}, style=STYLE).lower()

    assert "natural ambience" in out
    assert "sound effects" in out
    assert "nobody speaks" in out
    assert "no voices" in out


def test_sound_section_preserves_chinese_dialogue():
    shot = {**SHOT, "dialogue": [{"character": "小雨", "line": "别回头。"}]}

    assert "小雨: 别回头。" in compile_video_prompt(shot, style=STYLE)


def test_prompt_includes_style_action_camera_and_duration():
    out = compile_video_prompt(SHOT, style=STYLE)
    assert "anime" in out
    assert "clean cel shading" in out
    assert "Mara steps onto the windswept pier".lower() in out.lower()
    assert "wide establishing" in out
    assert "4" in out  # the ~4s duration


def test_prompt_has_temporal_motion_beats():
    out = compile_video_prompt(SHOT, style=STYLE)
    assert "Beat 1" in out
    assert "Beat 2" in out


def test_synthesized_middle_beats_are_distinct_and_move_anchored():
    # A longer shot with no explicit visual_beats yields multiple middle beats. They must
    # progress (not be identical boilerplate) and reference the shot's actual camera move.
    shot = {**SHOT, "camera": "wide establishing", "duration_s": 6.0}
    out = compile_video_prompt(shot, style=STYLE)
    beat_lines = [ln for ln in out.splitlines() if ln.startswith("- Beat")]
    # Description only (strip the "- Beat N (t0–t1s): " prefix, which varies by time marker).
    descs = [ln.split(": ", 1)[1] for ln in beat_lines]
    middles = descs[1:-1]  # drop the establish + resolve beats
    assert len(middles) >= 2
    # Middle beat descriptions must progress, not repeat one identical boilerplate string.
    assert len(set(middles)) == len(middles)
    # And they must tie the motion to the shot's chosen camera move.
    assert any("push-in" in d for d in middles)


def test_prompt_uses_exact_ordered_visual_beats_and_state_trajectory():
    shot = {
        **SHOT,
        "duration_s": 15,
        "start_states": {"男人": "gas-onset"},
        "end_states": {"男人": "werewolf"},
        "visual_beats": [
            {"id": "change", "action": "男人在红色气体中逐步变形"},
            {"id": "reveal", "action": "完整狼人从消散的红气中显露"},
        ],
    }

    out = compile_video_prompt(shot, style=STYLE)

    assert "男人在红色气体中逐步变形" in out
    assert "完整狼人从消散的红气中显露" in out
    assert out.index("逐步变形") < out.index("完整狼人")
    assert "男人: gas-onset → werewolf" in out


def test_prompt_locks_approved_target_state_reference():
    shot = {
        **SHOT,
        "duration_s": 15,
        "start_states": {"男人": "human"},
        "end_states": {"男人": "werewolf"},
        "target_state_references": [
            {
                "character": "男人",
                "state": "werewolf",
                "image": "bible/characters/man/states/werewolf/reference.png",
            }
        ],
        "target_state_reference_images": [
            "bible/characters/man/states/werewolf/reference.png"
        ],
    }

    out = compile_video_prompt(shot, style=STYLE)

    assert "## Target state lock" in out
    assert "男人: werewolf" in out
    assert "final visible state" in out.lower()
    assert "match the approved target-state" in out.lower()


def test_restricted_provider_keeps_target_state_text_without_reference_image_mixing():
    caps = VideoCapabilities(supports_reference_images=False, supports_last_frame=False)
    shot = {
        **SHOT,
        "start_states": {"男人": "human"},
        "end_states": {"男人": "werewolf"},
        "target_state_references": [
            {
                "character": "男人",
                "state": "werewolf",
                "image": "bible/characters/man/states/werewolf/reference.png",
            }
        ],
    }

    out = compile_video_prompt(shot, style=STYLE, capabilities=caps)
    low = out.lower()

    assert "男人: werewolf" in out
    assert "provider cannot receive extra target-state media" in low
    assert "reference image" not in low


def test_prompt_discourages_static_imagery():
    low = compile_video_prompt(SHOT, style=STYLE).lower()
    assert "avoid" in low
    for term in ("static", "still image", "slideshow"):
        assert term in low


def test_prompt_adds_camera_movement_for_static_framing():
    shot = {**SHOT, "camera": "wide"}  # no movement verb in the framing
    low = compile_video_prompt(shot, style=STYLE).lower()
    assert any(w in low for w in ("push-in", "push in", "drift", "dolly", "pan", "track"))


def test_neutral_framing_defaults_to_a_motivated_move_not_handheld_drift():
    # Medium framing with no movement/recipe: the silent default must be a motivated,
    # model-safe move, not the jitter-prone "subtle handheld drift" the skill warns against.
    shot = {**SHOT, "camera": "medium two-shot"}
    low = compile_video_prompt(shot, style=STYLE).lower()
    assert "push-in" in low or "push in" in low
    assert "subtle handheld drift" not in low


def test_missing_framing_still_avoids_unmotivated_handheld_default():
    shot = {k: v for k, v in SHOT.items() if k != "camera"}
    low = compile_video_prompt(shot, style=STYLE).lower()
    assert "push-in" in low or "push in" in low
    assert "handheld camera with a subtle drift" not in low


def test_prompt_gives_lighting_and_lens_direction():
    low = compile_video_prompt(SHOT, style=STYLE).lower()
    assert "lighting" in low
    assert "lens" in low


def test_prompt_calls_for_layered_and_acting_motion():
    low = compile_video_prompt(SHOT, style=STYLE).lower()
    assert "foreground" in low and "background" in low
    assert "emotion" in low or "expression" in low


def test_prompt_includes_product_format_pacing():
    fmt = {"name": "short_drama_episode", "label": "Short drama episode",
           "pacing": "dense, dialogue-forward, mobile-friendly"}
    out = compile_video_prompt(SHOT, style=STYLE, product_format=fmt)
    assert "dense, dialogue-forward, mobile-friendly" in out


def test_full_capabilities_use_references_and_continuity():
    out = compile_video_prompt(
        SHOT, style=STYLE, capabilities=VideoCapabilities(), has_previous_shot=True
    )
    low = out.lower()
    assert "previous final frame" in low  # last-frame carry continuity
    assert "reference" in low          # bible reference conditioning


def test_restricted_capabilities_avoid_reference_mixing():
    caps = VideoCapabilities(supports_reference_images=False, supports_last_frame=False)
    out = compile_video_prompt(
        SHOT, style=STYLE, capabilities=caps, has_previous_shot=True
    )
    low = out.lower()
    # Must not instruct the model to consume unsupported reference media or
    # match a separate previous-shot frame.
    assert "previous shot" not in low
    assert "reference image" not in low
    # It must still anchor identity on the starting keyframe and keep real motion.
    assert "keyframe" in low
    assert "Beat 1" in out
    assert "avoid" in low


def test_prompt_includes_story_context_and_performance_objective():
    context = {
        "idea": "A princess leaves a television and enters a lonely home.",
        "story": {"logline": "A fairy-tale princess crosses into a real apartment."},
        "scene": {
            "heading": "INT. HOME - NIGHT",
            "beats": ["The TV glows", "The princess steps onto the carpet"],
            "dialogue": [{"character": "Princess", "line": "Where am I?"}],
        },
        "previous_shot": {"id": "sh-001", "camera": "wide", "action": "the TV flares"},
        "next_shot": {"id": "sh-003", "camera": "close-up", "action": "the homeowner gasps"},
    }

    out = compile_video_prompt(SHOT, style=STYLE, context=context)

    assert "Story beat" in out
    assert "Performance objective" in out
    assert "fairy-tale princess" in out
    assert "INT. HOME - NIGHT" in out
    assert "the TV flares" in out
    assert "the homeowner gasps" in out


def test_scene_dialogue_is_not_duplicated_outside_sound_section():
    context = {
        "scene": {
            "heading": "INT. HOME - NIGHT",
            "dialogue": [{"character": "Other", "line": "Unrelated scene line."}],
        }
    }

    out = compile_video_prompt(SHOT, style=STYLE, context=context)

    assert "Unrelated scene line." not in out
    assert out.count("Mara: I shouldn't be here.") == 1


def test_restricted_provider_keeps_textual_context_but_no_media_reference_mixing():
    caps = VideoCapabilities(supports_reference_images=False, supports_last_frame=False)
    context = {
        "story": {"logline": "A lighthouse keeper trusts a warning gull."},
        "previous_shot": {"id": "sh-001", "camera": "wide", "action": "the door opens"},
        "next_shot": {"id": "sh-003", "camera": "wide", "action": "the beacon turns"},
    }

    out = compile_video_prompt(
        SHOT, style=STYLE, context=context, capabilities=caps, has_previous_shot=True
    )
    low = out.lower()

    assert "lighthouse keeper" in out
    assert "previous shot" not in low
    assert "reference image" not in low
    assert "keyframe" in low


def test_context_preserves_chinese_text():
    context = {
        "idea": "一个灯塔守望者遇见会说话的海鸥",
        "story": {"logline": "一部关于孤独与勇气的短片。"},
        "scene": {"heading": "EXT. LIGHTHOUSE - DAWN", "beats": ["海鸥发出警告"]},
    }

    out = compile_video_prompt(SHOT, style=STYLE, context=context)

    assert "一个灯塔守望者" in out
    assert "孤独与勇气" in out
    assert "海鸥发出警告" in out


def test_retrieved_knowledge_follows_identity_and_provider_constraints():
    guidance = "Use foreground occlusion to support unease."
    out = compile_video_prompt(
        SHOT,
        style=STYLE,
        capabilities=VideoCapabilities(),
        knowledge_guidance=guidance,
    )

    assert guidance in out
    assert out.index("## Identity & references") < out.index(guidance)


def test_prompt_uses_extended_storyboard_direction_fields():
    shot = {
        **SHOT,
        "camera_movement": "slow dolly from TV glow to the princess's bare feet",
        "composition": "television portal in foreground, lonely sofa midground, city window behind",
        "emotion": "wonder turning into fear",
        "continuity_notes": "blue TV light must remain on her dress and the carpet",
    }

    out = compile_video_prompt(shot, style=STYLE)

    assert "slow dolly from TV glow" in out
    assert "television portal in foreground" in out
    assert "wonder turning into fear" in out
    assert "blue TV light" in out


CAMERA_RECIPE = {
    "id": "orbit_push_eyes",
    "name": "360° orbit into rapid push-in",
    "intents": ["villain entrance"],
    "prompt": "360° orbital camera: slow then snap-accelerate, rapid push-in to a close-up of the eyes; 24mm wide lens",
    "motion_strength": "high",
}


def test_camera_recipe_text_is_woven_into_the_prompt():
    out = compile_video_prompt(SHOT, style=STYLE, camera_recipe=CAMERA_RECIPE)
    assert "360° orbital camera" in out
    assert "24mm wide lens" in out
    # The recipe's human-readable name should be attributed so editors can trace it.
    assert "360° orbit into rapid push-in" in out


def test_recipe_name_drives_movement_label_when_none_explicit():
    # SHOT has no camera_movement; the recipe name should fill the base label.
    out = compile_video_prompt(SHOT, style=STYLE, camera_recipe=CAMERA_RECIPE)
    assert "Camera movement: 360° orbit into rapid push-in" in out


def test_explicit_camera_movement_takes_precedence_over_recipe_name():
    shot = {**SHOT, "camera_movement": "slow handheld drift"}
    out = compile_video_prompt(shot, style=STYLE, camera_recipe=CAMERA_RECIPE)
    assert "Camera movement: slow handheld drift" in out
    # The recipe's detailed text is still appended.
    assert "360° orbital camera" in out


def test_free_text_movement_and_recipe_are_reconciled_as_one_move():
    # When a shot carries both a free-text camera_movement and a recipe, the block must
    # reconcile them so the model performs ONE move, not two conflicting ones.
    shot = {**SHOT, "camera_movement": "slow handheld drift"}
    out = compile_video_prompt(shot, style=STYLE, camera_recipe=CAMERA_RECIPE).lower()
    assert "same" in out and "one move" in out


def test_no_camera_recipe_is_byte_identical_to_baseline():
    baseline = compile_video_prompt(SHOT, style=STYLE)
    with_none = compile_video_prompt(SHOT, style=STYLE, camera_recipe=None)
    assert baseline == with_none


BRACKET_RECIPE = {
    "id": "slow_push_tension",
    "name": "slow push-in",
    "intents": ["rising tension"],
    "prompt": "slow eased push-in toward [the subject], medium to close, focus locked on the subject",
    "motion_strength": "low",
}


def test_recipe_bracket_slots_are_filled_with_the_subject():
    out = compile_video_prompt(SHOT, style=STYLE, camera_recipe=BRACKET_RECIPE)
    # The subject slot resolves to the shot's actual subject...
    assert "push-in toward Mara" in out
    # ...and no raw recipe placeholder token survives into the model prompt.
    assert "[the subject]" not in out


def test_recipe_directional_slot_never_leaks_a_bracket():
    recipe = {
        **BRACKET_RECIPE,
        "prompt": "slow lateral dolly to the [left/right], lens facing constant",
    }
    out = compile_video_prompt(SHOT, style=STYLE, camera_recipe=recipe)
    assert "[left/right]" not in out
    assert "to the left" in out


def test_camera_block_emits_single_move_path_from_framing_fields():
    shot = {
        **SHOT,
        "camera": "medium close-up",
        "camera_movement": "slow push-in",
        "start_frame": "Mara mid-frame, hands at her sides",
        "end_frame": "Mara filling frame, eyes to lens",
        "shot_size": "medium close-up",
        "camera_angle": "eye level",
        "lens_intent": "portrait compression",
    }
    out = compile_video_prompt(shot, style=STYLE)
    cam = out.split("## Camera\n", 1)[1].split("\n##", 1)[0]
    assert "Start framing: Mara mid-frame" in cam
    assert "End framing: Mara filling frame" in cam
    assert "eye level" in cam
    assert "single move" in cam.lower() or "one move" in cam.lower()


def test_prompt_includes_location_references_and_micro_expression_skill():
    shot = {
        **SHOT,
        "reference_locations": ["Clock Shop"],
        "emotion": "grief cracking into resolve",
    }

    out = compile_video_prompt(
        shot,
        style=STYLE,
        prompt_skills={
            "micro_expression": "Use temporal facial beats: eyes, brow, mouth, breath, and shoulders.",
            "location_identity": "Keep the location palette, hero props, and light states consistent.",
        },
    )

    assert "Clock Shop" in out
    assert "eyes, brow, mouth" in out
    assert "hero props" in out


def test_video_prompt_includes_grid_directive_when_present():
    shot = {
        "id": "sh-001", "action": "draw a sword", "duration_s": 4,
        "motion_grid": {"layout": "2x2", "rows": 2, "cols": 2, "panel_count": 4},
    }
    text = compile_video_prompt(shot)
    assert "storyboard grid" in text.lower()
    assert "4" in text and "follow" in text.lower()
    assert "do not show the grid" in text.lower()


def test_video_prompt_has_no_grid_directive_without_metadata():
    text = compile_video_prompt({"id": "sh-001", "action": "draw a sword", "duration_s": 4})
    assert "storyboard grid" not in text.lower()


def test_video_prompt_grid_fallback_without_layout_rows_cols():
    """Test that motion_grid with panel_count but no layout/rows/cols degrades gracefully."""
    shot = {
        "id": "sh-001",
        "action": "draw a sword",
        "duration_s": 4,
        "motion_grid": {"panel_count": 4},
    }
    text = compile_video_prompt(shot)
    assert "storyboard grid" in text.lower()
    assert "4-panel" in text
    assert "None" not in text


def test_camera_movement_skill_is_emitted_when_supplied():
    out = compile_video_prompt(
        SHOT, style=STYLE, prompt_skills={"camera_movement": "ONE primary move per shot."}
    )
    assert "## Camera movement skill" in out
    assert "ONE primary move per shot." in out


def test_character_motion_skill_section_emitted():
    out = compile_video_prompt(SHOT, style=STYLE, prompt_skills={"character_motion": "MOTION-CRAFT-XYZ"})
    assert "## Character motion skill\nMOTION-CRAFT-XYZ" in out


def test_no_character_motion_section_without_skill():
    out = compile_video_prompt(SHOT, style=STYLE, prompt_skills={})
    assert "## Character motion skill" not in out


STATIC_RECIPE = {
    "id": "static_subject_moves",
    "name": "locked-off, subject carries the motion",
    "motion_strength": "static",
    "selection": "common",
    "prompt": "locked-off tripod camera holding still; the subject carries all motion",
}


def test_static_recipe_emits_intentional_lockoff_not_a_drift():
    shot = {**SHOT, "camera": "medium close-up", "camera_movement": ""}
    out = compile_video_prompt(shot, style=STYLE, camera_recipe=STATIC_RECIPE)
    cam = out.split("## Camera\n", 1)[1].split("\n##", 1)[0].lower()
    assert "locked-off" in cam or "camera holds still" in cam
    # Must NOT bolt a drift/push onto an intentional static camera.
    assert "never locked off" not in cam
    assert "push-in so the frame" not in cam


def test_static_shot_avoid_block_forbids_frozen_subject_not_still_camera():
    shot = {**SHOT, "camera": "medium close-up"}
    out = compile_video_prompt(shot, style=STYLE, camera_recipe=STATIC_RECIPE)
    avoid = out.split("## Avoid\n", 1)[1].lower()
    assert "frozen subject" in avoid or "subject that does not move" in avoid
    assert "every second must contain visible, intentional motion" not in avoid


def test_moving_shot_still_demands_motion_every_second():
    out = compile_video_prompt(SHOT, style=STYLE)  # no static recipe
    avoid = out.split("## Avoid\n", 1)[1].lower()
    assert "every second must contain visible, intentional motion" in avoid


def test_dialogue_is_colon_form_with_intelligibility_guidance():
    shot = {**SHOT, "dialogue": [{"character": "Mara", "line": "I shouldn't be here."}],
            "duration_s": 4.0}
    out = compile_video_prompt(shot, style=STYLE, language="English")
    sound = out.split("## Sound\n", 1)[1].split("\n##", 1)[0]
    assert "- Mara: I shouldn't be here." in sound          # colon form
    assert '- Mara: "I shouldn\'t be here."' not in sound    # NOT wrapped in quotes
    assert "intelligible in English" in sound
    assert "subtitles" in sound.lower()
    assert "fits the shot" in sound.lower()


def test_dialogue_shot_avoid_block_forbids_gibberish_and_subtitles():
    shot = {**SHOT, "dialogue": [{"character": "Mara", "line": "Hello."}]}
    avoid = compile_video_prompt(shot, style=STYLE).split("## Avoid\n", 1)[1].lower()
    assert "gibberish" in avoid
    assert "subtitles" in avoid


def test_no_dialogue_shot_says_nobody_speaks_and_has_no_dialogue_negatives():
    out = compile_video_prompt({**SHOT, "dialogue": []}, style=STYLE)
    assert "Nobody speaks" in out
    assert "gibberish" not in out.split("## Avoid\n", 1)[1].lower()


def test_no_background_music_line_is_unconditional():
    for shot in ({**SHOT}, {**SHOT, "dialogue": []}):
        assert "No background music" in compile_video_prompt(shot, style=STYLE)


def _beat_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.startswith("- Beat")]


def test_motion_beats_weighted_by_weight():
    shot = {
        **SHOT,
        "duration_s": 8.0,
        "visual_beats": [
            {"action": "she loads her stance", "weight": 1},
            {"action": "she snaps the blade forward", "weight": 1},
            {"action": "she holds the extended pose", "weight": 2},
        ],
    }
    lines = _beat_lines(compile_video_prompt(shot, style=STYLE))
    assert "(0–2s)" in lines[0]
    assert "(2–4s)" in lines[1]
    assert "(4–8s)" in lines[2]


def test_motion_beats_weighted_by_per_beat_duration_s():
    shot = {
        **SHOT,
        "duration_s": 8.0,
        "visual_beats": [
            {"action": "a", "duration_s": 2},
            {"action": "b", "duration_s": 2},
            {"action": "c", "duration_s": 4},
        ],
    }
    lines = _beat_lines(compile_video_prompt(shot, style=STYLE))
    assert "(0–2s)" in lines[0] and "(2–4s)" in lines[1] and "(4–8s)" in lines[2]


def test_motion_beats_even_split_when_no_size_backcompat():
    shot = {**SHOT, "duration_s": 8.0,
            "visual_beats": [{"action": "a"}, {"action": "b"}]}
    lines = _beat_lines(compile_video_prompt(shot, style=STYLE))
    assert "(0–4s)" in lines[0] and "(4–8s)" in lines[1]


def test_motion_beats_no_size_is_byte_identical_even_split():
    from studio_agent.video_prompt import _fmt
    duration = 8.25
    beats = [{"action": f"a{i}"} for i in range(5)]  # no duration_s / weight
    out = compile_video_prompt({**SHOT, "duration_s": duration, "visual_beats": beats}, style=STYLE)
    step = duration / len(beats)
    for i in range(len(beats)):
        assert f"({_fmt(i * step)}–{_fmt((i + 1) * step)}s)" in out


def test_motion_beats_thread_secondary_motion():
    shot = {**SHOT, "duration_s": 4.0,
            "visual_beats": [{"action": "she turns", "secondary_motion": "her cloak lags a beat behind"}]}
    out = compile_video_prompt(shot, style=STYLE)
    assert "her cloak lags a beat behind" in out
    assert "Secondary motion:" in out
