from studio_agent.shot_design import normalize_shot_design


def test_normalize_shot_design_backfills_legacy_shot():
    shot = normalize_shot_design({
        "camera": "close-up",
        "camera_recipe": "push_in",
        "action": "Mara notices the seal",
        "composition": "the letter in foreground, Mara behind it",
    })

    assert shot["shot_size"] == "close-up"
    assert shot["camera_recipe"] == "push_in"
    assert shot["craft_recipe_ids"] == ["push_in"]
    assert shot["movement_motivation"]
    assert set(shot) >= {
        "dramatic_purpose",
        "start_frame",
        "end_frame",
        "subject_blocking",
        "camera_angle",
        "lens_intent",
        "lighting_state",
        "edit_relationship",
    }


def test_normalize_preserves_explicit_camera_design():
    raw = {
        "camera": "low-angle medium shot",
        "shot_size": "medium shot",
        "camera_movement": "slow push-in",
        "dramatic_purpose": "show the chairman losing control",
        "start_frame": "chairman owns the center of the table",
        "end_frame": "the junior lawyer fills the foreground",
        "subject_blocking": "lawyer steps into the chairman's axis",
        "camera_angle": "low axis becoming eye-level",
        "lens_intent": "natural perspective with readable boardroom depth",
        "movement_motivation": "push only as the document changes the room",
        "lighting_state": "cold window key and warm table practicals",
        "edit_relationship": "cut on the board members turning away",
        "craft_recipe_ids": ["composition.power.blocking.en"],
    }

    assert normalize_shot_design(raw) == {
        **raw,
        "camera_recipe": "",
        "story_advance": "advance the scene's next beat",
    }


def test_normalize_backfills_story_advance_from_action():
    shot = normalize_shot_design({
        "camera": "close-up",
        "action": "Mara notices the seal",
    })
    assert shot["story_advance"] == "Mara notices the seal"


def test_normalize_preserves_explicit_story_advance():
    shot = normalize_shot_design({
        "action": "Mara opens the letter",
        "story_advance": "the audience learns the letter is a summons",
    })
    assert shot["story_advance"] == "the audience learns the letter is a summons"
