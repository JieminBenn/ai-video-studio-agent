import pytest

from studio_agent.providers.base import ImageCapabilities, ImageGen, VideoCapabilities


def test_video_capabilities_has_storyboard_grid_default_false():
    assert VideoCapabilities().supports_storyboard_grid is False
    assert VideoCapabilities(supports_storyboard_grid=True).supports_storyboard_grid is True


def test_image_capabilities_default_false():
    assert ImageCapabilities().supports_storyboard_grid is False


def test_image_capabilities_default_to_no_prompt_limit():
    capabilities = ImageCapabilities()

    assert capabilities.max_prompt_length is None
    assert capabilities.prompt_length_unit == "characters"


@pytest.mark.parametrize("unit", ["characters", "utf8_bytes"])
def test_image_capabilities_accept_supported_prompt_length_units(unit):
    capabilities = ImageCapabilities(
        max_prompt_length=8000,
        prompt_length_unit=unit,
    )

    assert capabilities.max_prompt_length == 8000
    assert capabilities.prompt_length_unit == unit


@pytest.mark.parametrize(
    ("length", "unit"),
    [(0, "characters"), (-1, "utf8_bytes"), (8, "tokens")],
)
def test_image_capabilities_reject_invalid_prompt_limits(length, unit):
    with pytest.raises(ValueError):
        ImageCapabilities(max_prompt_length=length, prompt_length_unit=unit)


def test_imagegen_capabilities_reads_instance_flag():
    class _Img(ImageGen):
        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            return None

    img = _Img()
    assert img.capabilities.supports_storyboard_grid is False
    img._supports_storyboard_grid = True
    assert img.capabilities.supports_storyboard_grid is True


def test_imagegen_capabilities_reads_prompt_limit_attributes():
    class _Img(ImageGen):
        max_prompt_length = 1234
        prompt_length_unit = "utf8_bytes"

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            return None

    capabilities = _Img().capabilities

    assert capabilities.max_prompt_length == 1234
    assert capabilities.prompt_length_unit == "utf8_bytes"


from studio_agent.cli import build_providers


def test_build_providers_reads_grid_flags():
    profile = {
        "llm": "fake", "image": "fake", "video": "fake",
        "tts": "fake", "music": "fake", "vlm": "fake",
        "image_supports_storyboard_grid": True,
        "video_supports_storyboard_grid": True,
    }
    providers = build_providers(profile)
    assert providers.image.capabilities.supports_storyboard_grid is True
    assert providers.video.capabilities.supports_storyboard_grid is True


def test_build_providers_grid_flags_default_false_for_fakes_when_unset():
    profile = {
        "llm": "fake", "image": "fake", "video": "fake",
        "tts": "fake", "music": "fake", "vlm": "fake",
        "image_supports_storyboard_grid": False,
        "video_supports_storyboard_grid": False,
    }
    providers = build_providers(profile)
    assert providers.image.capabilities.supports_storyboard_grid is False
    assert providers.video.capabilities.supports_storyboard_grid is False


def test_build_providers_honors_image_prompt_limit_override():
    profile = {
        "llm": "fake",
        "image": "fake",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
        "image_max_prompt_length": 1234,
        "image_prompt_length_unit": "utf8_bytes",
    }

    providers = build_providers(profile)

    assert providers.image.capabilities.max_prompt_length == 1234
    assert providers.image.capabilities.prompt_length_unit == "utf8_bytes"


def test_build_providers_rejects_invalid_image_prompt_limit_override():
    profile = {
        "llm": "fake",
        "image": "fake",
        "video": "fake",
        "tts": "fake",
        "music": "fake",
        "vlm": "fake",
        "image_max_prompt_length": 8000,
        "image_prompt_length_unit": "tokens",
    }

    with pytest.raises(ValueError, match="prompt length unit"):
        build_providers(profile)
