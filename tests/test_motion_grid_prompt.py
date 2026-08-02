from studio_agent.motion_grid_prompt import compile_grid_prompt, panel_beats, resolve_layout
from studio_agent.prompt_validation import validate_grid_prompt


def test_resolve_layout_aliases():
    assert resolve_layout("九宫格") == (3, 3, 9)
    assert resolve_layout("十二宫格") == (3, 4, 12)
    assert resolve_layout("2x2") == (2, 2, 4)
    assert resolve_layout("3x4") == (3, 4, 12)


def test_resolve_layout_auto_scales_with_duration():
    assert resolve_layout("auto", duration_s=4.0) == (2, 2, 4)
    assert resolve_layout("auto", duration_s=12.0) == (3, 3, 9)
    assert resolve_layout("bogus", duration_s=4.0) == (2, 2, 4)


def test_grid_prompt_emits_verbatim_character_appearance_lock():
    out = compile_grid_prompt(
        {"id": "sh-001", "action": "a knight draws a sword", "characters": ["Knight"]},
        rows=2,
        cols=2,
        character_locks=["Knight: scarred jaw, silver pauldrons, ash-grey eyes."],
    )
    assert "## Character appearance lock (verbatim — do not alter)" in out
    assert "Knight: scarred jaw, silver pauldrons, ash-grey eyes." in out


def test_grid_prompt_omits_lock_section_without_locks():
    out = compile_grid_prompt(
        {"id": "sh-001", "action": "x", "characters": ["Knight"]}, rows=2, cols=2
    )
    assert "Character appearance lock" not in out


def test_grid_prompt_avoid_includes_coherence_negatives():
    out = compile_grid_prompt(
        {"id": "sh-001", "action": "x", "characters": ["Knight"]}, rows=2, cols=2
    )
    lower = out.lower()
    assert "fused fingers" in lower
    assert "floating objects" in lower


def test_panel_beats_count_and_order():
    beats = panel_beats("a knight draws a sword", 4)
    assert len(beats) == 4
    assert all("frozen still" in beat.lower() for beat in beats)
    assert "opening pose" in beats[0].lower()
    assert "final pose" in beats[-1].lower()


def test_compile_grid_prompt_encodes_constraints():
    shot = {
        "id": "sh-001", "action": "a knight draws a sword",
        "description": "low-angle hero shot", "duration_s": 4,
        "characters": ["Knight"], "reference_locations": ["Courtyard"],
    }
    text = compile_grid_prompt(shot, rows=2, cols=2)
    assert "exactly 4" in text
    assert "2 rows" in text and "2 columns" in text
    assert "16:9" in text
    assert "SAME character" in text
    assert "Panel 1" in text and "Panel 4" in text
    assert validate_grid_prompt(text, shot, panel_count=4).valid is True
    # negatives that protect slicing/consistency
    assert "panel numbers" in text.lower()
    assert "collage" in text.lower()


def test_grid_uses_ordered_visual_beats_and_allows_intentional_state_change():
    shot = {
        "id": "sh-001",
        "action": "a man transforms",
        "characters": ["男人"],
        "visual_beats": [
            {"id": "human", "description": "普通男人静立在马路上", "end_states": {"男人": "human"}},
            {"id": "gas", "description": "男人周围悬停着红色气体", "end_states": {"男人": "gas-onset"}},
            {"id": "change", "description": "男人保持半狼人形态", "end_states": {"男人": "partial-werewolf"}},
            {"id": "reveal", "description": "完整狼人保持最终姿势", "end_states": {"男人": "werewolf"}},
        ],
        "start_states": {"男人": "human"},
        "end_states": {"男人": "werewolf"},
    }

    beats = panel_beats(shot["action"], 4, visual_beats=shot["visual_beats"])
    text = compile_grid_prompt(shot, rows=2, cols=2)

    assert "普通男人静立在马路上" in beats[0]
    assert "完整狼人保持最终姿势" in beats[-1]
    assert "intentional state changes" in text
    assert "human" in text and "werewolf" in text
    assert validate_grid_prompt(text, shot, panel_count=4).valid is True
