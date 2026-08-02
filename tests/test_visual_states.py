from studio_agent.visual_states import (
    normalize_visual_state_changes,
    state_plan_for_character,
)


CHARACTERS = [
    {"name": "男人", "role": "主体", "description": "穿完整深色西装的中年男人"},
]


def test_normalize_visual_states_keeps_ordered_states_for_known_character():
    raw = {
        "characters": [
            {
                "character": "男人",
                "initial_state": "Human Form",
                "states": [
                    {
                        "id": "Human Form",
                        "label": "正常人类",
                        "kind": "base",
                        "description": "衣物完整，尚未出现红气",
                        "reference_required": True,
                    },
                    {
                        "id": "Gas Onset",
                        "label": "红气出现",
                        "kind": "transient",
                        "description": "红气出现，人体尚未变形",
                    },
                    {
                        "id": "Full Werewolf",
                        "label": "完整狼人",
                        "kind": "endpoint",
                        "description": "完整狼人，仍可认出本人",
                        "appearance_changes": ["species", "anatomy", "face", "fur"],
                        "reference_required": True,
                    },
                ],
                "transitions": [
                    {
                        "from": "Human Form",
                        "to": "Full Werewolf",
                        "trigger": "红气从身体出现",
                        "ordered_state_ids": [
                            "Human Form",
                            "Gas Onset",
                            "Full Werewolf",
                        ],
                        "preserve": ["eyes", "facial geometry"],
                    }
                ],
            }
        ]
    }

    document = normalize_visual_state_changes(raw, CHARACTERS)

    plan = document["characters"][0]
    assert document["version"] == 1
    assert plan["initial_state"] == "human-form"
    assert [state["id"] for state in plan["states"]] == [
        "human-form",
        "gas-onset",
        "full-werewolf",
    ]
    assert plan["states"][1]["reference_required"] is False
    assert plan["transitions"][0]["ordered_state_ids"] == [
        "human-form",
        "gas-onset",
        "full-werewolf",
    ]


def test_normalize_visual_states_drops_unknown_characters_and_duplicate_states():
    raw = {
        "characters": [
            {
                "character": "男人",
                "states": [
                    {"id": "human", "kind": "base", "description": "baseline"},
                    {"id": "human", "kind": "endpoint", "description": "duplicate"},
                ],
            },
            {
                "character": "不存在的人",
                "states": [{"id": "ghost", "kind": "endpoint"}],
            },
        ]
    }

    document = normalize_visual_state_changes(raw, CHARACTERS)

    assert len(document["characters"]) == 1
    assert [state["id"] for state in document["characters"][0]["states"]] == ["human"]


def test_empty_visual_states_is_versioned_and_lookup_is_safe():
    document = normalize_visual_state_changes(None, CHARACTERS)

    assert document == {"version": 1, "characters": []}
    assert state_plan_for_character(document, "男人") is None


def test_state_plan_lookup_matches_character_name_case_insensitively():
    document = normalize_visual_state_changes(
        {
            "characters": [
                {
                    "character": "Mara",
                    "states": [{"id": "normal", "kind": "base"}],
                }
            ]
        },
        [{"name": "Mara"}],
    )

    assert state_plan_for_character(document, "mara")["character"] == "Mara"


def test_context_only_endpoint_does_not_require_second_reference():
    document = normalize_visual_state_changes(
        {
            "characters": [{
                "character": "男人",
                "initial_state": "human",
                "states": [
                    {"id": "human", "kind": "base", "description": "same man"},
                    {
                        "id": "outside-monitor",
                        "kind": "endpoint",
                        "description": "Same man now stands outside the monitor",
                        "appearance_changes": [],
                        "reference_required": True,
                    },
                ],
            }],
        },
        CHARACTERS,
    )

    state = document["characters"][0]["states"][-1]
    assert state["appearance_changes"] == []
    assert state["reference_required"] is False


def test_material_endpoint_requires_reference_and_normalizes_changes():
    document = normalize_visual_state_changes(
        {
            "characters": [{
                "character": "男人",
                "states": [
                    {"id": "human", "kind": "base"},
                    {
                        "id": "werewolf",
                        "kind": "endpoint",
                        "description": "A complete werewolf form",
                        "appearance_changes": ["species", "body", "face", "species", ""],
                        "reference_required": True,
                    },
                ],
            }],
        },
        CHARACTERS,
    )

    state = document["characters"][0]["states"][-1]
    assert state["appearance_changes"] == ["species", "body", "face"]
    assert state["reference_required"] is True
