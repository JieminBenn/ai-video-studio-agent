"""Manual Midjourney import provider tests.

Midjourney is intentionally not automated. The provider writes a human-readable
request file and raises a manual-import signal; once the user saves an image at the
expected path, the provider reports success without contacting any service.
"""

from pathlib import Path

import pytest

from studio_agent.cli import build_providers
from studio_agent.providers.base import ManualImportRequired
from studio_agent.providers.manual_midjourney import ManualMidjourneyImageGen


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_manual_midjourney_writes_request_and_requires_import(tmp_path):
    out = tmp_path / "sh-001.png"
    ref = tmp_path / "mara-reference.png"
    ref.write_bytes(PNG_SIGNATURE + b"ref")

    gen = ManualMidjourneyImageGen(aspect_ratio="16:9")

    with pytest.raises(ManualImportRequired) as excinfo:
        gen.generate(
            "cinematic close-up of Mara holding a lantern",
            out_path=str(out),
            reference_images=[str(ref)],
            seed=123,
        )

    assert not out.exists()
    exc = excinfo.value
    assert exc.provider == "manual-midjourney"
    assert exc.out_path == str(out)
    request = Path(exc.request_path)
    assert request.is_file()

    text = request.read_text()
    assert "/imagine" in text
    assert "--ar 16:9" in text
    assert "cinematic close-up of Mara" in text
    assert str(ref) in text
    assert str(out) in text
    assert "Save the chosen/upscaled image exactly here" in text
    assert "123" in text


def test_manual_midjourney_returns_success_when_import_exists(tmp_path):
    out = tmp_path / "reference.png"
    out.write_bytes(PNG_SIGNATURE + b"imported")

    result = ManualMidjourneyImageGen().generate(
        "portrait of Mara",
        out_path=str(out),
    )

    assert result.content == str(out)
    assert result.provider == "manual-midjourney"
    assert result.cost_usd == 0.0
    assert result.meta["manual_import"] == "already_present"


def test_build_providers_selects_manual_midjourney_image():
    providers = build_providers({
        "llm": "fake",
        "image": "manual-midjourney",
        "image_aspect_ratio": "4:3",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
    })

    assert isinstance(providers.image, ManualMidjourneyImageGen)
    assert providers.image.aspect_ratio == "4:3"
