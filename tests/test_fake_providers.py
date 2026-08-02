"""Tests for fake non-LLM providers (offline, deterministic, zero-cost)."""

import wave

from studio_agent.providers.fake import (
    FakeImageGen, FakeLLM, FakeMusic, FakeReferenceAnalyzer, FakeTTS, FakeVideoGen, FakeVLMCheck,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
WAV_SIGNATURE = b"RIFF"


def test_fake_llm_image_prompt_distills_brief_to_prompt_and_negative():
    brief = (
        "# Keyframe prompt\n\n"
        "## Subject & moment\n"
        "A lighthouse keeper waves at a gull.\n\n"
        "## Style\n"
        "Style: cinematic, natural, aspect 16:9.\n\n"
        "## Avoid\n"
        "text, watermark."
    )
    gen = FakeLLM().complete_json(f"[task:image_prompt]\nBRIEF:\n{brief}\n")

    assert gen.meta["task"] == "image_prompt"
    assert "lighthouse keeper waves at a gull" in gen.content["prompt"]
    assert "cinematic" in gen.content["prompt"]
    assert "##" not in gen.content["prompt"]  # markdown scaffolding dropped
    assert gen.content["negative"] == "text, watermark."


def test_fake_image_gen_writes_valid_png(tmp_path):
    out = tmp_path / "ref.png"
    gen = FakeImageGen().generate("a clockmaker, neutral grey studio", out_path=str(out), seed=7)

    assert out.is_file()
    assert out.read_bytes().startswith(PNG_SIGNATURE)
    assert gen.provider == "fake"
    assert gen.meta["seed"] == 7


def test_fake_image_gen_is_seed_deterministic(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    FakeImageGen().generate("p", out_path=str(a), seed=42)
    FakeImageGen().generate("p", out_path=str(b), seed=42)
    FakeImageGen().generate("p", out_path=str(c), seed=99)

    # Same seed -> identical reference image (locked-seed consistency).
    assert a.read_bytes() == b.read_bytes()
    # Different seed -> different image.
    assert a.read_bytes() != c.read_bytes()


def test_fake_video_gen_writes_a_clip(tmp_path):
    kf = tmp_path / "kf.png"
    FakeImageGen().generate("p", out_path=str(kf), seed=3)
    out = tmp_path / "clip.mp4"

    gen = FakeVideoGen().generate(
        "shot prompt", out_path=str(out), keyframe_path=str(kf), seed=3, duration_s=4.0
    )

    assert out.is_file()
    assert out.stat().st_size > 0
    assert gen.provider == "fake"
    assert gen.meta["duration_s"] == 4.0


def test_fake_video_gen_is_deterministic(tmp_path):
    kf = tmp_path / "kf.png"
    FakeImageGen().generate("p", out_path=str(kf), seed=3)
    a, b, c = (tmp_path / n for n in ["a.mp4", "b.mp4", "c.mp4"])

    FakeVideoGen().generate("p", out_path=str(a), keyframe_path=str(kf), seed=3, duration_s=2.0)
    FakeVideoGen().generate("p", out_path=str(b), keyframe_path=str(kf), seed=3, duration_s=2.0)
    FakeVideoGen().generate("p", out_path=str(c), keyframe_path=str(kf), seed=3, duration_s=5.0)

    # Same keyframe+seed+duration -> identical clip; different duration -> different.
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()


def test_fake_video_gen_writes_deterministic_48khz_stereo_native_audio(tmp_path):
    kf = tmp_path / "kf.png"
    FakeImageGen().generate("p", out_path=str(kf), seed=3)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"

    first_gen = FakeVideoGen().generate(
        "p", out_path=str(first), keyframe_path=str(kf), duration_s=2.0,
        generate_audio=True,
    )
    second_gen = FakeVideoGen().generate(
        "p", out_path=str(second), keyframe_path=str(kf), duration_s=2.0,
        generate_audio=True,
    )

    first_audio = first.with_name("first.native.source.wav")
    second_audio = second.with_name("second.native.source.wav")
    assert first_gen.meta["native_audio_path"] == str(first_audio)
    assert first_gen.meta["generate_audio"] is True
    assert first_audio.read_bytes() == second_audio.read_bytes()
    with wave.open(str(first_audio), "rb") as reader:
        assert reader.getframerate() == 48000
        assert reader.getnchannels() == 2
        assert reader.getsampwidth() == 2
        assert reader.getnframes() == 96000


def test_fake_video_gen_omits_native_audio_when_not_requested(tmp_path):
    kf = tmp_path / "kf.png"
    FakeImageGen().generate("p", out_path=str(kf), seed=3)
    out = tmp_path / "clip.mp4"

    gen = FakeVideoGen().generate(
        "p", out_path=str(out), keyframe_path=str(kf), duration_s=2.0,
        generate_audio=False,
    )

    assert gen.meta["native_audio_path"] is None
    assert gen.meta["generate_audio"] is False
    assert not out.with_name("clip.native.source.wav").exists()


def test_fake_video_gen_carries_last_frame(tmp_path):
    kf1 = tmp_path / "kf1.png"
    kf2 = tmp_path / "kf2.png"
    FakeImageGen().generate("p", out_path=str(kf1), seed=1)
    FakeImageGen().generate("p", out_path=str(kf2), seed=2)
    with_ref = tmp_path / "with.mp4"
    without_ref = tmp_path / "without.mp4"

    FakeVideoGen().generate("p", out_path=str(with_ref), keyframe_path=str(kf2),
                            seed=2, duration_s=2.0, last_frame_ref=str(kf1))
    FakeVideoGen().generate("p", out_path=str(without_ref), keyframe_path=str(kf2),
                            seed=2, duration_s=2.0)

    # Carrying a previous frame forward changes the clip (continuity is conditioned on it).
    assert with_ref.read_bytes() != without_ref.read_bytes()


def test_fake_video_gen_conditions_on_reference_images(tmp_path):
    kf = tmp_path / "kf.png"
    ref = tmp_path / "ref.png"
    FakeImageGen().generate("p", out_path=str(kf), seed=1)
    FakeImageGen().generate("ref", out_path=str(ref), seed=2)
    with_ref = tmp_path / "with-ref.mp4"
    without_ref = tmp_path / "without-ref.mp4"

    gen = FakeVideoGen().generate(
        "p",
        out_path=str(with_ref),
        keyframe_path=str(kf),
        seed=1,
        duration_s=2.0,
        reference_images=[str(ref)],
    )
    FakeVideoGen().generate(
        "p", out_path=str(without_ref), keyframe_path=str(kf), seed=1, duration_s=2.0
    )

    assert gen.meta["reference_images"] == [str(ref)]
    assert with_ref.read_bytes() != without_ref.read_bytes()


def _good_clip(tmp_path):
    kf = tmp_path / "kf.png"
    FakeImageGen().generate("p", out_path=str(kf), seed=1)
    clip = tmp_path / "good.mp4"
    FakeVideoGen().generate("p", out_path=str(clip), keyframe_path=str(kf), seed=1, duration_s=2.0)
    return clip


def test_vlm_passes_a_valid_clip(tmp_path):
    clip = _good_clip(tmp_path)
    gen = FakeVLMCheck().review(str(clip), prompt="a wide shot of Mara")
    report = gen.content

    assert report["overall_pass"] is True
    assert report["recommendation"] == "approve"
    assert {c["dimension"] for c in report["checks"]} >= {
        "story_alignment", "shot_instruction_adherence", "continuity",
        "identity_drift", "motion_anatomy",
        "artifacts", "audio_sync", "safety",
    }


def test_vlm_flags_a_corrupted_clip_with_timestamp(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"FAKECLIP\nQC_FAIL marker\n")
    gen = FakeVLMCheck().review(str(bad), prompt="a wide shot")
    report = gen.content

    assert report["overall_pass"] is False
    assert report["recommendation"] == "regenerate"
    assert report["max_severity"] == "high"
    failed = [c for c in report["checks"] if not c["passed"]]
    assert failed and all(c["timestamp"] for c in failed)


def test_vlm_flags_a_missing_or_unplayable_clip(tmp_path):
    missing = tmp_path / "nope.mp4"
    gen = FakeVLMCheck().review(str(missing), prompt="x")
    assert gen.content["overall_pass"] is False


def test_vlm_accepts_real_mp4_container(tmp_path):
    clip = tmp_path / "real.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\0" * 64)

    gen = FakeVLMCheck().review(str(clip), prompt="x")

    assert gen.content["overall_pass"] is True
    assert gen.content["recommendation"] == "approve"


def test_fake_tts_writes_a_valid_wav(tmp_path):
    out = tmp_path / "line.wav"
    gen = FakeTTS().speak("Another grey morning at the lighthouse.", out_path=str(out))

    assert out.read_bytes().startswith(WAV_SIGNATURE)
    assert gen.provider == "fake"
    assert gen.meta["duration_s"] > 0


def test_fake_tts_duration_follows_text_length(tmp_path):
    short = FakeTTS().speak("Hi.", out_path=str(tmp_path / "s.wav"))
    long = FakeTTS().speak("word " * 40, out_path=str(tmp_path / "l.wav"))
    assert long.meta["duration_s"] > short.meta["duration_s"]


def test_fake_tts_respects_explicit_duration_and_is_deterministic(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    g = FakeTTS().speak("anything", out_path=str(a), duration_s=3.0)
    FakeTTS().speak("totally different text", out_path=str(b), duration_s=3.0)
    assert g.meta["duration_s"] == 3.0
    # Same requested duration -> identical silent audio regardless of text.
    assert a.read_bytes() == b.read_bytes()


def test_fake_music_writes_a_wav_of_requested_duration(tmp_path):
    out = tmp_path / "score.wav"
    gen = FakeMusic().compose("wistful piano", out_path=str(out), duration_s=5.0)
    assert out.read_bytes().startswith(WAV_SIGNATURE)
    assert gen.meta["duration_s"] == 5.0


def test_fake_image_records_reference_images(tmp_path):
    from studio_agent.providers.fake import FakeImageGen
    out = tmp_path / "kf.png"
    gen = FakeImageGen().generate(
        "prompt", out_path=str(out),
        reference_images=[], seed=7,
    )
    assert out.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert gen.meta["reference_images"] == []


def test_fake_image_changes_with_references(tmp_path):
    from studio_agent.providers.fake import FakeImageGen
    # Two real reference files with different bytes.
    ref_a = tmp_path / "a.png"
    ref_b = tmp_path / "b.png"
    FakeImageGen().generate("x", out_path=str(ref_a), seed=1)
    FakeImageGen().generate("y", out_path=str(ref_b), seed=2)

    no_ref = tmp_path / "n.png"
    with_ref = tmp_path / "w.png"
    FakeImageGen().generate("p", out_path=str(no_ref), seed=5, reference_images=[])
    FakeImageGen().generate("p", out_path=str(with_ref), seed=5, reference_images=[str(ref_a)])
    # Same seed, different references => different deterministic output.
    assert no_ref.read_bytes() != with_ref.read_bytes()

    # Same seed + same references => identical output.
    again = tmp_path / "again.png"
    FakeImageGen().generate("p", out_path=str(again), seed=5, reference_images=[str(ref_a)])
    assert again.read_bytes() == with_ref.read_bytes()


def test_fake_reference_analyzer_classifies_without_network(tmp_path):
    image = tmp_path / "clock-shop-location.png"
    FakeImageGen().generate("location", out_path=str(image), seed=42)

    gen = FakeReferenceAnalyzer().analyze(
        str(image),
        aliases=["@photo1", "first image"],
        user_note="",
    )

    assert gen.provider == "fake"
    assert gen.content["target_type"] == "location"
    assert gen.content["target_id"] == "Clock Shop"
    assert gen.content["confidence"] >= 0.75


def test_fake_reference_analyzer_can_classify_style_and_character(tmp_path):
    style = tmp_path / "dreamy-mood-style.png"
    hero = tmp_path / "hero-character-face.png"
    FakeImageGen().generate("style", out_path=str(style), seed=1)
    FakeImageGen().generate("hero", out_path=str(hero), seed=2)

    style_gen = FakeReferenceAnalyzer().analyze(str(style), aliases=["@photo1"], user_note="")
    hero_gen = FakeReferenceAnalyzer().analyze(str(hero), aliases=["@photo2"], user_note="")

    assert style_gen.content["target_type"] == "style"
    assert style_gen.content["target_id"] == "global"
    assert hero_gen.content["target_type"] == "character"
    assert hero_gen.content["target_id"] == "Hero"


def test_build_providers_wires_fake_reference_analyzer():
    from studio_agent.cli import build_providers

    providers = build_providers({"llm": "fake"})

    assert isinstance(providers.reference_analyzer, FakeReferenceAnalyzer)


def test_fake_llm_grid_prompt_is_directed_dict():
    brief = (
        "# Storyboard grid brief\n\n"
        "## Grid layout\nexactly 4 frames in 2 rows by 2 columns.\n\n"
        "## Panel sequence\n- Panel 1: establish.\n- Panel 4: resolve.\n\n"
        "## Style\ncinematic.\n\n## Avoid\ncollage, panel numbers.\n"
    )
    gen = FakeLLM().complete_json(f"[task:grid_prompt]\nBRIEF:\n{brief}")
    assert isinstance(gen.content, dict)
    assert gen.content["prompt"]
    assert "Panel" in gen.content["prompt"]
    assert gen.content["negative"]


def test_fake_reference_analyzer_revise_returns_grounded_json_text():
    import json
    from pathlib import Path
    from studio_agent.providers.fake import FakeReferenceAnalyzer

    analyzer = FakeReferenceAnalyzer()
    prompt = (
        "INSTRUCTION: keep the same outfit\n\n"
        'CURRENT_JSON:\n{"name": "Mara", "wardrobe": ""}\n'
    )
    gen = analyzer.revise(["/tmp/hero.png"], prompt=prompt, language="en")

    assert isinstance(gen.content, str)
    revised = json.loads(gen.content)
    assert revised["name"] == "Mara"
    assert revised["revision_note"] == "keep the same outfit"
    assert revised["grounded_in"] == ["hero"]


def test_fake_video_resolves_model_default_duration_deterministically(tmp_path):
    from studio_agent.providers.fake import FakeVideoGen

    keyframe = tmp_path / "kf.png"
    keyframe.write_bytes(b"kf")
    out = tmp_path / "clip.mp4"
    gen = FakeVideoGen().generate(
        "prompt", out_path=str(out), keyframe_path=str(keyframe), duration_s=None
    )
    # None = "model default": the fake resolves a concrete length and reports it in
    # meta so the video stage can record real seconds into clips.json.
    assert gen.meta["duration_s"] == 4.0
    caps = FakeVideoGen().capabilities
    assert caps.min_duration_s == 1
    assert caps.max_duration_s == 15
