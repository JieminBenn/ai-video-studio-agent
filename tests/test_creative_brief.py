from studio_agent.creative_brief import (
    compact_creative_brief,
    normalize_creative_brief,
    resolved_value,
)


def test_user_input_is_locked_and_overrides_director_value():
    brief = normalize_creative_brief(
        {"visual_direction": {"tone": "grim"}},
        idea="A lighthouse keeper meets a gull",
        model_config={"language": "en"},
        user_inputs={"tone": "melancholic wonder"},
    )

    assert brief["visual_direction"]["tone"] == {
        "value": "melancholic wonder",
        "source": "user",
        "confidence": "high",
        "locked": True,
        "reason": "Selected in the kickoff brief",
    }
    assert resolved_value(brief, "tone") == "melancholic wonder"


def test_old_scalar_visual_direction_normalizes_without_paid_work():
    brief = normalize_creative_brief(
        {"visual_direction": {"camera_language": "patient and intimate"}},
        idea="idea",
        model_config={"language": "en"},
    )

    assert brief["version"] == 2
    assert brief["visual_direction"]["camera_language"]["source"] == "director_default"
    assert compact_creative_brief(brief)["camera_language"] == "patient and intimate"


def test_chinese_value_and_hard_avoidance_are_preserved():
    brief = normalize_creative_brief(
        {"hard_avoidances": ["不要手持抖动"]},
        idea="一个雨夜重逢",
        model_config={"language": "zh"},
        user_inputs={"light_texture": "潮湿霓虹与柔和逆光"},
    )

    assert resolved_value(brief, "light_texture") == "潮湿霓虹与柔和逆光"
    assert brief["hard_avoidances"] == ["不要手持抖动"]


def test_user_hard_avoidances_are_locked_into_the_brief():
    brief = normalize_creative_brief(
        {"hard_avoidances": ["flat lighting"]},
        idea="a quiet room",
        model_config={"language": "en"},
        user_inputs={"hard_avoidances": ["unmotivated orbit", "flat lighting"]},
    )

    assert brief["hard_avoidances"] == ["unmotivated orbit", "flat lighting"]
