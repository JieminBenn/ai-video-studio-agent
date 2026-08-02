"""Manual Midjourney image import provider.

Midjourney is not automated here. The provider writes a human-readable request file
beside the expected output image, then raises :class:`ManualImportRequired`. A human
uses Midjourney through the official website/Discord flow, downloads the chosen image,
saves it exactly to ``out_path``, and resumes the pipeline. If the image is already on
disk, the provider returns success without contacting any service.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .base import Generation, ImageGen, ManualImportRequired


class ManualMidjourneyImageGen(ImageGen):
    name = "manual-midjourney"
    model = "manual-midjourney-import"

    def __init__(
        self,
        *,
        aspect_ratio: str = "16:9",
        model: str = "manual-midjourney-import",
        cost_per_image: float = 0.0,
    ):
        self.aspect_ratio = aspect_ratio
        self.model = model
        self.cost_per_image = cost_per_image

    def generate(
        self,
        prompt: str,
        *,
        out_path: str,
        reference_images: list[str] | None = None,
        **kwargs: Any,
    ) -> Generation:
        out = Path(out_path)
        request_path = _request_path(out)
        if out.is_file() and out.stat().st_size > 0:
            return Generation(
                content=str(out),
                provider=self.name,
                model=self.model,
                cost_usd=0.0,
                seconds=0.0,
                meta={
                    "manual_import": "already_present",
                    "request_path": str(request_path),
                    "reference_images": list(reference_images or []),
                },
            )

        out.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(
            _request_text(
                prompt,
                out_path=out,
                request_path=request_path,
                reference_images=list(reference_images or []),
                aspect_ratio=self.aspect_ratio,
                seed=kwargs.get("seed"),
            )
        )
        raise ManualImportRequired(
            provider=self.name,
            out_path=str(out),
            request_path=str(request_path),
        )


def _request_path(out: Path) -> Path:
    return Path(str(out) + ".midjourney.md")


def _request_text(
    prompt: str,
    *,
    out_path: Path,
    request_path: Path,
    reference_images: list[str],
    aspect_ratio: str,
    seed: Any = None,
) -> str:
    refs = "\n".join(f"{idx}. `{path}`" for idx, path in enumerate(reference_images, 1))
    refs = refs or "None."
    seed_line = f"\n- Studio Agent seed hint: `{seed}`" if seed is not None else ""
    command = _midjourney_command(prompt, aspect_ratio=aspect_ratio)
    return (
        "# Manual Midjourney Image Request\n\n"
        "Studio Agent needs this image before the pipeline can continue. Use "
        "Midjourney manually, choose/upscale one result, download it, and save it to "
        "the exact output path below.\n\n"
        "## Output\n"
        f"- Save the chosen/upscaled image exactly here: `{out_path}`\n"
        f"- Request file: `{request_path}`\n"
        f"{seed_line}\n\n"
        "## Reference Images\n"
        "Use these local project images as visual references in Midjourney. Upload them "
        "to Midjourney or otherwise make them available in the official UI, then use "
        "them as image prompts, character references, or style references as appropriate.\n\n"
        f"{refs}\n\n"
        "## Prompt\n"
        "```text\n"
        f"{prompt.strip()}\n"
        "```\n\n"
        "## Midjourney Command Draft\n"
        "Paste uploaded reference image handles/URLs before the text if needed, then run:\n\n"
        "```text\n"
        f"{command}\n"
        "```\n\n"
        "## Resume\n"
        "After saving the image at the output path, run `resume <project-id>` or click "
        "Resume in the dashboard. Do not approve the failed stage until the image exists.\n"
        f"\nGenerated request at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}.\n"
    )


def _midjourney_command(prompt: str, *, aspect_ratio: str) -> str:
    clean = " ".join(prompt.strip().split())
    suffix = f" --ar {aspect_ratio}" if aspect_ratio else ""
    return f"/imagine prompt: {clean}{suffix}"
