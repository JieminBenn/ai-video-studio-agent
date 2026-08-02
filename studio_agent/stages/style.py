"""Style stage — resolve the project-wide look and gate it before anything is drawn.

The user defines one style for the whole film/series: a label, a sentence, or an
uploaded reference image whose *style only* is extracted. This stage runs first so a
human reviews the resolved look (``bible/style.md`` + a neutral sample frame) and can
approve / edit / regenerate it before any character or keyframe is generated in it.

The resolved dict is the same shape the fixed presets used, so every downstream prompt
threads it with no other change (``style.py`` / ``format_style_*``).
"""

from __future__ import annotations

from ..image_generation import generate_project_image
from ..reference_assets import style_reference_paths
from ..style import format_style_markdown, format_style_prompt
from .base import Providers, Stage, StageResult

_SAMPLE_REL = ("bible", "style_sample.png")


class StyleStage(Stage):
    name = "style"

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="style already complete")

        style_input = dict(project.model_config.get("style_input") or {})
        description = str(style_input.get("description") or "").strip()
        feedback = str(style_input.get("feedback") or "").strip()
        style_refs = style_reference_paths(project)
        # Reference paths are stored project-relative; vision profilers resolve them against
        # the cwd and require the file to exist, so make the path absolute (same as the image
        # provider path below).
        image_path = str(project.dir / style_refs[0]) if style_refs else None

        label = str(project.model_config.get("style_label") or "").strip()
        style_blind = False
        # Profile when the user has given ANY style input — a description, an uploaded
        # reference, OR refinement feedback. Gating on description/image alone made
        # "Refine style" a silent no-op for a blank/preset style (the common clip-mode
        # path), since the refine gate only ever supplies feedback. When there is no base
        # description or image, the feedback IS the style direction, so route it as the
        # description — real profilers reject an empty description with no image, and there
        # is nothing for a standalone "adjustment" to adjust.
        if not description and not image_path and feedback:
            profile_description, profile_feedback = feedback, ""
        else:
            profile_description, profile_feedback = description, feedback
        if profile_description or image_path:
            label, style_blind = self._profile(
                project, providers, profile_description, image_path, profile_feedback
            )

        self._write_style_md(project, label, style_blind=style_blind)
        self._render_sample(project, providers)

        return StageResult(status="complete", message=f"style: {label or 'preset'}")

    def _profile(self, project, providers, description: str, image_path: str | None, feedback: str = "") -> tuple[str, bool]:
        """Profile text/image into the project style dict; return (label, blind)."""
        if providers.style_profiler is None:
            raise RuntimeError("style profiling requested but no style_profiler is configured")
        project.assert_budget_available()
        gen = providers.style_profiler.profile(
            description=description,
            image_path=image_path,
            language=str(project.model_config.get("language") or "en"),
            feedback=feedback,
        )
        project.add_generation_cost(stage=self.name, generation=gen)
        style = dict(gen.content or {})
        # The label is a human-facing handle for the gate/bible — keep it off the style
        # dict so it never leaks into every compiled prompt.
        label = str(style.pop("label", "") or "").strip()
        project.model_config["style"] = style
        project.model_config["style_name"] = "custom"
        project.model_config["style_label"] = label
        project.save()
        blind = bool(image_path) and not bool((gen.meta or {}).get("saw_image"))
        return label, blind

    def _write_style_md(self, project, label: str, *, style_blind: bool = False) -> None:
        style = dict(project.model_config.get("style") or {})
        style_doc = {"look": "cinematic", "palette": "natural", "aspect_ratio": "16:9", **style}
        lines = format_style_markdown(style_doc)
        header = "# Show Bible — Visual Style\n\n"
        if label:
            header += f"_Custom style: {label}_\n\n"
        disclosure = ""
        if style_blind:
            disclosure = (
                "> ⚠️ This look was inferred from your text description only — no vision "
                "model was configured to read the uploaded style image. Choose a vision "
                "model (reference & style) to extract the image's actual style.\n\n"
            )
        text = (
            header
            + disclosure
            + f"{lines}\n\n"
            + "This look is locked for the whole project. Keep palette, rendering, and "
            "texture identical across every scene.\n"
        )
        project.path("bible", "style.md").write_text(text)

    def _render_sample(self, project, providers) -> None:
        out_path = project.path(*_SAMPLE_REL)
        if out_path.is_file() or providers.image is None:
            return
        style = dict(project.model_config.get("style") or {})
        look = str(style.get("look") or "cinematic")
        style_prompt = format_style_prompt(style)
        prompt = (
            f"Style sample frame for the project look: {look}. A neutral establishing "
            f"environment with no named characters, purely to show the visual style "
            f"(palette, lighting, rendering, texture). {style_prompt} "
            "Do not introduce any subjects, characters, or composition from the uploaded "
            "style reference; use only the resolved style guide."
        )
        gen = generate_project_image(
            project,
            providers.image,
            prompt,
            out_path=str(out_path),
            reference_images=None,
            seed=7,
        )
        project.add_generation_cost(stage=self.name, generation=gen)
