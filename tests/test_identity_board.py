from studio_agent.identity_board import (
    coerce_aliases,
    coerce_board_dict,
    humanize_board_value,
    normalize_identity_board,
)


def test_coerce_aliases_keeps_strings_dedupes_and_trims():
    assert coerce_aliases(["A", "A", " B "]) == ["A", "B"]


def test_coerce_aliases_extracts_label_from_alias_objects():
    # Regression: a location-bible LLM returned prompt_aliases as objects
    # ({"alias": "...", "description": "..."}) instead of bare strings, which later crashed
    # ", ".join(location_aliases) with "expected str instance, dict found".
    aliases = coerce_aliases([
        {"alias": "ESTABLISHING_WIDE", "description": "全景建立镜头"},
        {"name": "DOOR_OPEN"},
        {"id": "entry"},
    ])
    assert aliases == ["ESTABLISHING_WIDE", "DOOR_OPEN", "entry"]


def test_coerce_aliases_drops_unusable_entries():
    assert coerce_aliases([{"nope": 1}, 5, None, "", "  "]) == []
    assert coerce_aliases(None) == []


def test_identity_board_adds_rich_invariants_without_breaking_legacy_fields():
    board = normalize_identity_board(
        {
            "canonical_face": "angular face and deep-set grey eyes",
            "canonical_body": "tall with guarded posture",
            "wardrobe": "navy wool coat",
            "prompt_aliases": ["the keeper"],
        },
        name="Mara",
        canonical={"wardrobe": "navy wool coat"},
        references=["bible/characters/mara/reference.png"],
    )

    assert board["wardrobe"] == "navy wool coat"
    assert board["wardrobe_details"]["immutable"] == ["navy wool coat"]
    assert board["identity_signature"]
    assert set(board) >= {
        "face",
        "body",
        "hair",
        "hero_props",
        "continuity_priority",
        "allowed_variation",
        "do",
        "dont",
        "prompt_aliases",
        "reference_bindings",
    }
    assert board["prompt_aliases"] == ["the keeper", "Mara"]
    assert board["reference_bindings"] == [
        "bible/characters/mara/reference.png"
    ]


def test_identity_board_preserves_structured_model_details():
    board = normalize_identity_board(
        {
            "identity_signature": "Mara's narrow face, silver streak, and navy coat",
            "face": {"geometry": "narrow oval", "distinctive_marks": ["left brow scar"]},
            "body": {"silhouette": "long vertical coat line"},
            "hair": {"structure": "black bob with one silver streak"},
            "wardrobe_details": {
                "summary": "navy coat",
                "immutable": ["silver collar pin"],
                "variable": ["coat may be open or closed"],
            },
        },
        name="Mara",
        canonical={},
        references=[],
    )

    assert board["face"]["geometry"] == "narrow oval"
    assert board["body"]["silhouette"] == "long vertical coat line"
    assert board["wardrobe_details"]["immutable"] == ["silver collar pin"]


def test_humanize_board_value_formats_dict_and_repr_string():
    d = {"face_shape": "长窄脸型", "eyes": "单眼皮，眼型细长", "signature_marks": "无"}
    out = humanize_board_value(d)
    assert "{" not in out and "}" not in out
    assert "长窄脸型" in out and "单眼皮，眼型细长" in out
    # A Python-dict-repr string left by older boards is reformatted too.
    assert humanize_board_value(repr(d)) == out


def test_humanize_board_value_passthrough_for_plain_text():
    assert humanize_board_value("标准鹅蛋脸，轮廓柔和。") == "标准鹅蛋脸，轮廓柔和。"
    assert humanize_board_value(["a", "", "b"]) == "a · b"


def test_coerce_board_dict_detects_dict_and_repr():
    assert coerce_board_dict({"k": "v"}) == {"k": "v"}
    assert coerce_board_dict("{'k': 'v'}") == {"k": "v"}
    assert coerce_board_dict("plain text") is None


def test_null_identity_fields_normalize_to_empty_not_the_string_none():
    # Defect C: a VLM that returns JSON null for a field used to be stringified to the
    # literal "None", poisoning the model-sheet prompt (wardrobe="None"). Null must become "".
    board = normalize_identity_board(
        {"wardrobe": None, "palette": None, "hair": None, "canonical_body": None},
        name="美女",
        canonical={},
        references=[],
    )

    assert board["wardrobe"] == ""
    assert board["palette"] == ""
    assert board["canonical_body"] == ""
    assert board["wardrobe_details"]["summary"] == ""


def test_appearance_lock_passthrough_from_vlm():
    board = normalize_identity_board(
        {"appearance_lock": "  A dense grounded description of the exact face.  "},
        name="Mara",
        canonical={},
        references=[],
    )
    assert board["appearance_lock"] == "A dense grounded description of the exact face."


def test_appearance_lock_assembled_when_absent():
    board = normalize_identity_board(
        {
            "canonical_face": "oval face, high cheekbones",
            "canonical_body": "slight, 1.6m",
            "face": {"distinctive_marks": ["teardrop mole under left eye"], "skin_details": "porcelain, fine pores"},
            "hair": "straight black, chin length",
            "wardrobe_details": {"immutable": ["red wool coat"]},
            "palette": "amber and teal, low-contrast film curve",
        },
        name="Mara",
        canonical={},
        references=[],
    )
    lock = board["appearance_lock"]
    assert lock  # non-empty
    assert "teardrop mole under left eye" in lock
    assert "red wool coat" in lock
    assert "amber and teal" in lock
    assert "oval face" in lock
