"""The style dict carries explicit medium/idiom/lighting/grade/lens/atmosphere fields."""

from studio_agent.style import (
    STYLE_PROFILE_KEYS,
    format_style_markdown,
    format_style_prompt,
    normalize_style_dict,
)

NEW_KEYS = ("medium", "idiom", "lighting", "color_grade", "lens", "atmosphere")


def test_new_keys_are_part_of_the_profile_shape():
    for key in NEW_KEYS:
        assert key in STYLE_PROFILE_KEYS


def test_normalize_fills_every_new_key_even_when_absent():
    style = normalize_style_dict({"look": "x"})
    for key in NEW_KEYS:
        assert key in style
        assert style[key] == ""  # present but empty, never missing


def test_normalize_keeps_supplied_values():
    style = normalize_style_dict(
        {"look": "x", "medium": "3D CG render", "idiom": "guoman", "lighting": "rim + bloom"}
    )
    assert style["medium"] == "3D CG render"
    assert style["idiom"] == "guoman"
    assert style["lighting"] == "rim + bloom"


def test_markdown_and_prompt_include_new_fields_and_lead_with_medium_idiom():
    style = normalize_style_dict(
        {"look": "x", "medium": "3D", "idiom": "guoman", "rendering": "PBR", "lighting": "rim"}
    )
    md = format_style_markdown(style)
    assert "Medium: 3D" in md
    assert "Idiom: guoman" in md
    # medium/idiom anchor the rendering block: they appear before rendering.
    assert md.index("Medium: 3D") < md.index("Rendering: PBR")
    assert md.index("Idiom: guoman") < md.index("Rendering: PBR")
    # Empty fields are omitted, not rendered blank.
    assert "Lens:" not in md
    prompt = format_style_prompt(style)
    assert "Medium: 3D." in prompt and "Lighting: rim." in prompt
