"""Text-to-image provider backed by Google's Imagen (Gemini API ``generate_images``).

Imagen has no reference-image input, so ``reference_images`` is accepted (to satisfy the
ImageGen contract) but ignored. Kept as a selectable alternate; the default image provider
is the native reference-conditioned ``gemini`` provider. Lazy client: no SDK import / no
key until ``generate()`` is called. Install the SDK with ``pip install -e ".[real]"``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from .base import Generation, ImageGen

DEFAULT_MODEL = "imagen-4.0-fast-generate-001"


class ImagenImageGen(ImageGen):
    name = "imagen"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        aspect_ratio: str = "16:9",
        cost_per_image: float = 0.02,
        api_key_env: str = "GEMINI_API_KEY",
    ):
        self.model = model
        self.aspect_ratio = aspect_ratio
        self.cost_per_image = cost_per_image
        self.api_key_env = api_key_env
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "google-genai not installed — run `pip install -e \".[real]\"`."
                ) from exc
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self.api_key_env} not set — add it to .env (see `cli doctor`)."
                )
            self._client = genai.Client(api_key=key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        out_path: str,
        reference_images: list[str] | None = None,  # ignored: Imagen is text-to-image
        **kwargs,
    ) -> Generation:
        from google.genai import types

        seed = kwargs.get("seed")
        aspect_ratio = kwargs.get("aspect_ratio", self.aspect_ratio)
        config = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            output_mime_type="image/png",
        )
        start = time.time()
        resp = self._client_lazy().models.generate_images(
            model=self.model, prompt=prompt, config=config
        )
        elapsed = round(time.time() - start, 2)

        images = resp.generated_images or []
        if not images or not images[0].image or not images[0].image.image_bytes:
            reason = images[0].rai_filtered_reason if images else "no image returned"
            raise RuntimeError(f"Imagen returned no image: {reason}")

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(images[0].image.image_bytes)
        return Generation(
            content=str(path),
            provider=self.name,
            model=self.model,
            cost_usd=self.cost_per_image,
            seconds=elapsed,
            meta={"seed": seed, "aspect_ratio": aspect_ratio},
        )
