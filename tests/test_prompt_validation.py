from studio_agent.prompt_validation import validate_grid_prompt


SHOT = {
    "id": "sh-001",
    "camera_movement": "slow dolly push",
    "end_frame": "she has crossed the screen",
}


def test_grid_prompt_accepts_ordered_static_panels():
    text = """Grid 2x2, four ordered panels.
Panel 1: frozen wide still, woman faces the monitor.
Panel 2: frozen medium still, fingertips touch the glass.
Panel 3: frozen close still, her shoulders are outside the screen.
Panel 4: frozen full-body still, both feet rest on the desk."""

    result = validate_grid_prompt(text, SHOT, panel_count=4)

    assert result.valid is True
    assert result.errors == []


def test_grid_prompt_requires_every_panel_to_be_enumerated():
    # Structural (not semantic) contract: grid slicing needs each panel named. A prompt that
    # skips Panel 3 must be flagged — this is the one check kept after dropping prose matching.
    text = """Grid 2x2.
Panel 1: frozen wide still. Panel 2: frozen medium still. Panel 4: frozen close still."""

    result = validate_grid_prompt(text, SHOT, panel_count=4)

    assert result.valid is False
    assert "missing panel 3" in result.errors


def test_grid_prompt_accepts_chinese_panel_labels():
    text = "画格 1：定格远景。画格 2：定格近景。"

    result = validate_grid_prompt(text, SHOT, panel_count=2)

    assert result.valid is True


def test_grid_prompt_no_longer_flags_camera_or_temporal_prose():
    # After removing semantic matching, camera/temporal wording inside a well-formed grid is not
    # a validation failure — the director is trusted and a human reviews the prompt at the gate.
    text = """Grid 1x2.
Panel 1: the camera pushes in as she gradually turns.
Panel 2: then a close still with ambient sound."""

    result = validate_grid_prompt(text, SHOT, panel_count=2)

    assert result.valid is True
    assert result.errors == []
