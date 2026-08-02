from studio_agent.clip_sequence import (
    balanced_clip_durations,
    normalize_visual_sequence,
    plan_scene_segments,
)


def test_balanced_clip_durations_use_the_fewest_long_clips():
    assert balanced_clip_durations(15, max_clip_s=15) == [15]
    assert balanced_clip_durations(25, max_clip_s=15) == [13, 12]
    assert balanced_clip_durations(30, max_clip_s=15) == [15, 15]
    assert balanced_clip_durations(45, max_clip_s=15) == [15, 15, 15]


def test_auto_duration_expands_a_complex_sequence_to_thirty_seconds():
    sequence = normalize_visual_sequence(
        {
            "intent_class": "transformation",
            "recommended_duration_s": 28,
            "beats": [
                {"id": "human", "action": "A man walks normally", "end_states": {"Man": "human"}},
                {"id": "gas", "action": "Red gas emerges", "start_states": {"Man": "human"}, "end_states": {"Man": "gas-onset"}},
                {"id": "change", "action": "His body transforms", "start_states": {"Man": "gas-onset"}, "end_states": {"Man": "partial-werewolf"}},
                {"id": "reveal", "action": "The werewolf is revealed", "start_states": {"Man": "partial-werewolf"}, "end_states": {"Man": "werewolf"}},
            ],
        },
        requested_duration="auto",
        max_clip_s=15,
    )

    assert sequence["target_duration_s"] == 30
    assert sequence["clip_durations"] == [15, 15]
    assert [[beat["id"] for beat in segment["visual_beats"]] for segment in sequence["segments"]] == [
        ["human", "gas"],
        ["change", "reveal"],
    ]
    assert sequence["segments"][0]["start_states"] == {"Man": "human"}
    assert sequence["segments"][0]["end_states"] == {"Man": "gas-onset"}
    assert sequence["segments"][1]["start_states"] == {"Man": "gas-onset"}
    assert sequence["segments"][1]["end_states"] == {"Man": "werewolf"}


def test_explicit_duration_overrides_the_model_recommendation():
    sequence = normalize_visual_sequence(
        {"recommended_duration_s": 60, "beats": [{"id": "hook", "action": "A reveal"}]},
        requested_duration=15,
        max_clip_s=15,
    )

    assert sequence["target_duration_s"] == 15
    assert sequence["clip_durations"] == [15]


def test_legacy_clip_count_and_seconds_remain_a_compatible_fallback():
    sequence = normalize_visual_sequence(
        {"beats": [{"id": "motion", "action": "Walk through the alley"}]},
        requested_duration=None,
        legacy_count=2,
        legacy_seconds=10,
        max_clip_s=15,
    )

    assert sequence["target_duration_s"] == 20
    assert sequence["clip_durations"] == [10, 10]
    assert len(sequence["segments"]) == 2
    assert all(segment["visual_beats"] for segment in sequence["segments"])


def test_dialogue_beats_stay_atomic_and_narration_rides_action():
    beats = [
        {"id": "b1", "action": "establish room", "narration": "VO opener"},
        {"id": "b2", "action": "hero walks in"},
        {"id": "b3", "action": "hero speaks",
         "dialogue": [{"character": "Hero", "line": "We go now."}]},
        {"id": "b4", "action": "door slams"},
    ]
    segments = plan_scene_segments(beats, max_clip_s=15, min_clip_s=4)
    # b1+b2 merge into one action clip; b3 is its own dialogue clip; b4 its own clip.
    assert len(segments) == 3
    assert [b["id"] for b in segments[0]["visual_beats"]] == ["b1", "b2"]
    assert segments[0]["narration"] == "VO opener"
    assert segments[0]["dialogue"] == []
    assert [b["id"] for b in segments[1]["visual_beats"]] == ["b3"]
    assert segments[1]["dialogue"][0]["line"] == "We go now."
    assert all(4 <= s["duration_s"] <= 15 for s in segments)


def test_long_action_run_splits_into_max_length_clips():
    beats = [{"id": f"b{i}", "action": f"move {i}", "duration_s": 5} for i in range(6)]
    segments = plan_scene_segments(beats, max_clip_s=15, min_clip_s=4)
    assert [s["duration_s"] for s in segments] == [15, 15]
    assert all(s["dialogue"] == [] for s in segments)


def test_small_scene_yields_one_short_clip():
    segments = plan_scene_segments(
        [{"id": "b1", "action": "single glance", "duration_s": 3}],
        max_clip_s=15, min_clip_s=4,
    )
    assert len(segments) == 1
    assert segments[0]["duration_s"] == 4  # clamped up to the minimum


def _raw(beats, **extra):
    return {"intent_class": "showcase", "beats": beats, **extra}


def test_auto_sizes_total_from_per_beat_durations():
    raw = _raw([
        {"id": "a", "action": "x", "duration_s": 4},
        {"id": "b", "action": "y", "duration_s": 4},
    ])
    seq = normalize_visual_sequence(raw, requested_duration="auto", max_clip_s=15)
    assert seq["target_duration_s"] == 8          # 4 + 4, not snapped to 15
    assert seq["clip_durations"] == [8]            # one provider-legal clip


def test_auto_splits_complex_idea_into_multiple_legal_clips():
    beats = [{"id": f"b{i}", "action": "x", "duration_s": 7} for i in range(4)]
    seq = normalize_visual_sequence(_raw(beats), requested_duration="auto", max_clip_s=15)
    assert seq["target_duration_s"] == 28         # 4 * 7
    assert all(4 <= d <= 15 for d in seq["clip_durations"])
    assert sum(seq["clip_durations"]) == 28
    assert len(seq["clip_durations"]) == 2         # ceil(28 / 15)


def test_auto_clamps_total_to_ceiling():
    beats = [{"id": f"b{i}", "action": "x", "duration_s": 15} for i in range(8)]  # 120s
    seq = normalize_visual_sequence(_raw(beats), requested_duration="auto", max_clip_s=15, ceiling_s=60)
    assert seq["target_duration_s"] == 60


def test_auto_without_beat_durations_falls_back_to_intent_bucket():
    raw = _raw([{"id": "a", "action": "x"}], intent_class="transformation")
    seq = normalize_visual_sequence(raw, requested_duration="auto", max_clip_s=15)
    assert seq["target_duration_s"] == 30          # legacy intent-class default preserved


def test_explicit_duration_ignores_beat_durations():
    raw = _raw([{"id": "a", "action": "x", "duration_s": 4}])
    seq = normalize_visual_sequence(raw, requested_duration=30, max_clip_s=15)
    assert seq["target_duration_s"] == 30          # explicit override unchanged


def test_auto_simple_complexity_clamps_to_one_clip():
    # Beats sum to 30s, but the LLM judged the idea simple -> one <=15s clip.
    beats = [{"id": f"b{i}", "action": "x", "duration_s": 10} for i in range(3)]
    seq = normalize_visual_sequence(
        _raw(beats, complexity="simple"), requested_duration="auto", max_clip_s=15
    )
    assert seq["target_duration_s"] == 15
    assert seq["clip_durations"] == [15]


def test_auto_moderate_complexity_clamps_to_two_clip_budget():
    beats = [{"id": f"b{i}", "action": "x", "duration_s": 10} for i in range(5)]  # 50s
    seq = normalize_visual_sequence(
        _raw(beats, complexity="moderate"), requested_duration="auto", max_clip_s=15
    )
    assert seq["target_duration_s"] == 30


def test_auto_complex_complexity_clamps_to_ceiling():
    beats = [{"id": f"b{i}", "action": "x", "duration_s": 15} for i in range(6)]  # 90s
    seq = normalize_visual_sequence(
        _raw(beats, complexity="complex"),
        requested_duration="auto", max_clip_s=15, ceiling_s=60,
    )
    assert seq["target_duration_s"] == 60


def test_auto_unknown_complexity_does_not_clamp():
    # Back-compat: no/garbage complexity -> beat-sum drives, exactly as before.
    beats = [{"id": f"b{i}", "action": "x", "duration_s": 7} for i in range(4)]  # 28s
    seq = normalize_visual_sequence(
        _raw(beats, complexity="banana"), requested_duration="auto", max_clip_s=15
    )
    assert seq["target_duration_s"] == 28


def test_explicit_duration_ignores_complexity():
    beats = [{"id": "a", "action": "x", "duration_s": 4}]
    seq = normalize_visual_sequence(
        _raw(beats, complexity="complex"), requested_duration=30, max_clip_s=15
    )
    assert seq["target_duration_s"] == 30


# --- ClipPlan resolution + fixed-count planning ---------------------------------

from studio_agent.clip_sequence import ClipPlan, clamp_clip_duration, resolve_clip_plan


def test_clamp_clip_duration_uses_provider_band_and_blank_means_default():
    assert clamp_clip_duration(None) is None
    assert clamp_clip_duration("  ") is None
    assert clamp_clip_duration("banana") is None
    assert clamp_clip_duration(2, min_s=4, max_s=15) == 4
    assert clamp_clip_duration(99, min_s=4, max_s=15) == 15
    assert clamp_clip_duration(6, min_s=4, max_s=15) == 6
    assert clamp_clip_duration(10, min_s=4, max_s=8) == 8   # Veo-style band


def test_resolve_clip_plan_new_default_is_one_model_length_clip():
    plan = resolve_clip_plan({"clip_plan_mode": "default"}, {"clip_count": 5})
    assert (plan.mode, plan.count, plan.seconds) == ("default", 1, None)


def test_resolve_clip_plan_manual_reads_count_and_seconds():
    plan = resolve_clip_plan(
        {"clip_plan_mode": "manual", "clip_count": 3, "clip_seconds": 6}, {}
    )
    assert (plan.mode, plan.count, plan.seconds) == ("manual", 3, 6)
    # blank seconds -> per-clip model default
    plan = resolve_clip_plan({"clip_plan_mode": "manual", "clip_count": 2}, {})
    assert (plan.mode, plan.count, plan.seconds) == ("manual", 2, None)


def test_resolve_clip_plan_explicit_mode_wins_over_legacy_keys():
    plan = resolve_clip_plan(
        {"clip_plan_mode": "auto", "clip_count": 3, "clip_target_duration_s": 30}, {}
    )
    assert plan.mode == "auto" and plan.total_s is None


def test_resolve_clip_plan_legacy_auto_stays_llm_planned():
    plan = resolve_clip_plan({"clip_target_duration_s": "auto"}, {})
    assert plan.mode == "auto" and plan.total_s is None


def test_resolve_clip_plan_legacy_numeric_target_pins_the_total():
    plan = resolve_clip_plan({"clip_target_duration_s": 30}, {})
    assert plan.mode == "auto" and plan.total_s == 30


def test_resolve_clip_plan_legacy_count_seconds_is_manual():
    plan = resolve_clip_plan({"clip_count": 2, "clip_seconds": 15}, {})
    assert (plan.mode, plan.count, plan.seconds) == ("manual", 2, 15)


def test_resolve_clip_plan_legacy_ignores_format_spec_auto_target():
    # The format spec's clip_target_duration_s: auto must not override an explicit
    # legacy clip_count/clip_seconds pair (matches the old stage-code precedence).
    plan = resolve_clip_plan(
        {"clip_count": 2, "clip_seconds": 15},
        {"clip_target_duration_s": "auto", "clip_plan_mode": "default"},
    )
    assert (plan.mode, plan.count, plan.seconds) == ("manual", 2, 15)


def test_normalize_with_default_plan_yields_one_unbounded_segment():
    seq = normalize_visual_sequence(
        {"beats": [{"id": "a", "action": "x"}, {"id": "b", "action": "y"}]},
        plan=ClipPlan(mode="default", count=1, seconds=None),
        max_clip_s=15,
    )
    assert len(seq["segments"]) == 1
    assert seq["segments"][0]["duration_s"] is None
    assert seq["clip_durations"] == [None]
    assert seq["target_duration_s"] is None
    # both beats packed into the single clip
    assert [b["id"] for b in seq["segments"][0]["visual_beats"]] == ["a", "b"]


def test_normalize_with_manual_plan_pins_count_and_clamps_seconds():
    seq = normalize_visual_sequence(
        {"beats": [{"id": "a", "action": "x"}]},
        plan=ClipPlan(mode="manual", count=3, seconds=20),
        max_clip_s=15, min_clip_s=4,
    )
    assert [s["duration_s"] for s in seq["segments"]] == [15, 15, 15]
    assert seq["target_duration_s"] == 45


def test_normalize_with_auto_plan_keeps_llm_split_and_pinned_total():
    beats = [{"id": f"b{i}", "action": "x", "duration_s": 7} for i in range(4)]  # 28s
    seq = normalize_visual_sequence(
        {"beats": beats, "complexity": "complex"},
        plan=ClipPlan(mode="auto"),
        max_clip_s=15,
    )
    assert seq["target_duration_s"] == 28
    pinned = normalize_visual_sequence(
        {"beats": beats}, plan=ClipPlan(mode="auto", total_s=30), max_clip_s=15
    )
    assert pinned["target_duration_s"] == 30
