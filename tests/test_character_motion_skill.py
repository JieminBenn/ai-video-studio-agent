"""The character_motion choreography skill must load and be bilingual (invariant #10)."""

from studio_agent.runtime_skills import load_prompt_skill


def test_character_motion_skill_loads_and_is_bilingual():
    text = load_prompt_skill("character_motion")
    assert text.strip(), "character_motion.md must exist and be non-empty"
    # Core craft concepts are present (English side).
    low = text.lower()
    assert "anticipation" in low
    assert "follow-through" in low or "follow through" in low
    assert "secondary" in low  # overlapping/secondary motion
    # Chinese side is present (invariant #10): at least one CJK character.
    assert any("一" <= ch <= "鿿" for ch in text), "must include 中文"
