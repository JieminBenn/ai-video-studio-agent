from subprocess import CompletedProcess

import pytest

from studio_agent.image_similarity import (
    ImageSimilarityError,
    SSIM_DUPLICATE_THRESHOLD,
    is_near_duplicate,
    structural_similarity,
)


def _runner(stderr: str, returncode: int = 0, calls=None):
    def run(args, **kwargs):
        if calls is not None:
            calls.append((args, kwargs))
        return CompletedProcess(args, returncode, stdout="", stderr=stderr)
    return run


def test_structural_similarity_parses_ffmpeg_all_score(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls = []

    score = structural_similarity(
        first,
        second,
        runner=_runner("[Parsed_ssim_4] SSIM Y:0.951 All:0.958161 (13.78)", calls=calls),
    )

    assert score == pytest.approx(0.958161)
    args = calls[0][0]
    assert args[0] == "ffmpeg"
    assert str(first) in args and str(second) in args
    assert "shell" not in calls[0][1]


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.939999, False), (SSIM_DUPLICATE_THRESHOLD, True), (0.99, True)],
)
def test_near_duplicate_threshold_is_inclusive(tmp_path, score, expected):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    duplicate, measured = is_near_duplicate(
        first,
        second,
        runner=_runner(f"SSIM Y:{score} All:{score} (1.0)"),
    )
    assert duplicate is expected
    assert measured == pytest.approx(score)


def test_similarity_failure_is_actionable(tmp_path):
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    with pytest.raises(ImageSimilarityError, match="could not compare"):
        structural_similarity(first, second, runner=_runner("decode failed", returncode=1))
