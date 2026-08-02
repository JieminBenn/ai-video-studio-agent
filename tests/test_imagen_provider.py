"""Wiring test for the renamed Imagen text-to-image provider (no network, no key)."""

from studio_agent.cli import build_providers
from studio_agent.providers.imagen import ImagenImageGen, DEFAULT_MODEL


def test_build_providers_selects_imagen():
    providers = build_providers({
        "llm": "fake",
        "image": "imagen",
        "image_model": "imagen-4.0-fast-generate-001",
    })
    assert isinstance(providers.image, ImagenImageGen)
    assert providers.image.name == "imagen"
    assert providers.image.model == "imagen-4.0-fast-generate-001"


def test_imagen_constructs_without_network_or_key():
    gen = ImagenImageGen()
    assert gen.model == DEFAULT_MODEL
    assert gen.name == "imagen"
