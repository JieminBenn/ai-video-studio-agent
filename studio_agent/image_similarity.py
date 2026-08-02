"""Perceptual similarity for generated identity-board images."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

SSIM_DUPLICATE_THRESHOLD = 0.94
_ALL_SCORE = re.compile(r"\bAll:([0-9]+(?:\.[0-9]+)?)")


class ImageSimilarityError(RuntimeError):
    """An image pair could not be validated through FFmpeg."""


def structural_similarity(
    first: str | Path,
    second: str | Path,
    *,
    runner: Callable = subprocess.run,
) -> float:
    first_path = Path(first)
    second_path = Path(second)
    for path in (first_path, second_path):
        if not path.is_file():
            raise ImageSimilarityError(f"image similarity input not found: {path}")
    command = [
        "ffmpeg", "-hide_banner", "-i", str(first_path), "-i", str(second_path),
        "-filter_complex",
        "[0:v]scale=256:256,format=gray[a];"
        "[1:v]scale=256:256,format=gray[b];[a][b]ssim",
        "-f", "null", "-",
    ]
    try:
        result = runner(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ImageSimilarityError(
            f"could not compare {first_path} and {second_path}: {exc}"
        ) from exc
    output = f"{result.stdout}\n{result.stderr}"
    match = _ALL_SCORE.search(output)
    if result.returncode != 0 or match is None:
        detail = output.strip().splitlines()[-1] if output.strip() else "no SSIM output"
        raise ImageSimilarityError(
            f"could not compare {first_path} and {second_path}: {detail}"
        )
    return float(match.group(1))


def is_near_duplicate(
    first: str | Path,
    second: str | Path,
    *,
    threshold: float = SSIM_DUPLICATE_THRESHOLD,
    runner: Callable = subprocess.run,
) -> tuple[bool, float]:
    score = structural_similarity(first, second, runner=runner)
    return score >= threshold, score
