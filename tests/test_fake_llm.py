from studio_agent.providers.fake import FakeLLM


def test_scene_sequence_beats_have_one_atomic_dialogue_beat():
    gen = FakeLLM().complete_json("[task:scene_sequence]\nLANGUAGE: English\nSCENE SCRIPT: {}")
    beats = gen.content["beats"]
    assert len(beats) >= 4
    dialogue_beats = [b for b in beats if b.get("dialogue")]
    assert len(dialogue_beats) == 1
    assert dialogue_beats[0]["dialogue"][0]["line"]
    assert any(str(b.get("narration") or "").strip() for b in beats)
