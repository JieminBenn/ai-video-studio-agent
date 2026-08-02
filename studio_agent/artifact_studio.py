"""Artifact Studio helpers for readable review and scoped LLM revisions.

The web UI stays file-first: every adjustment reads and writes a project artifact,
records a history file, and marks the right downstream stages pending without
silently deleting paid outputs.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .invalidation import invalidate_artifact_revision
from .orchestrator.project import Project
from .reference_assets import style_reference_paths
from .stages.base import Providers

TEXT_EXTS = {".json", ".md", ".txt", ".yaml", ".yml"}


@dataclass
class ArtifactRevisionResult:
    artifact_path: str
    instruction: str
    applied: bool
    invalidated: list[str]
    provider: str = ""
    model: str = ""
    error: str = ""
    archived: int = 0
    archive_dir: str = ""


def safe_artifact_path(project: Project, rel_path: str) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("artifact path must stay inside the project")
    path = (project.dir / rel).resolve()
    root = project.dir.resolve()
    if root != path and root not in path.parents:
        raise ValueError("artifact path must stay inside the project")
    return path


def stage_for_artifact(rel_path: str) -> str | None:
    rel = rel_path.replace("\\", "/")
    if rel == "story/plot.json":
        return "plot"
    if rel == "story/script.json":
        return "script"
    if rel.startswith("bible/characters/") or rel.startswith("bible/locations/") or rel == "bible/style.md":
        return "bible"
    if rel == "storyboard/shots.json":
        return "storyboard"
    return None


def downstream_stages_for_artifact(rel_path: str) -> list[str]:
    rel = rel_path.replace("\\", "/")
    if rel == "story/plot.json":
        return [
            "script", "bible", "storyboard", "keyframes", "video_prompts",
            "video", "review", "audio", "assemble",
        ]
    if rel == "story/script.json":
        return [
            "storyboard", "keyframes", "video_prompts", "video", "review",
            "audio", "assemble",
        ]
    if rel.startswith("bible/characters/") or rel.startswith("bible/locations/") or rel == "bible/style.md":
        return [
            "storyboard", "keyframes", "video_prompts", "video", "review",
            "audio", "assemble",
        ]
    if rel.startswith("references/"):
        return [
            "storyboard", "keyframes", "video_prompts", "video", "review",
            "audio", "assemble",
        ]
    if rel == "storyboard/shots.json":
        return ["keyframes", "video_prompts", "video", "review", "audio", "assemble"]
    if rel.startswith("storyboard/prompts/"):
        name = Path(rel).name
        if name.endswith(".audio.md") or name == "music.audio.md":
            return ["audio", "assemble"]
        if name.endswith((".keyframe.md", ".grid.md")):
            return ["keyframes", "video_prompts", "video", "review", "audio", "assemble"]
        return ["video_prompts", "video", "review", "audio", "assemble"]
    return []


def mark_artifact_revised(project: Project, rel_path: str) -> list[str]:
    """Archive stale derivatives and return the downstream stages made pending."""
    result = invalidate_artifact_revision(project, rel_path)
    owner = stage_for_artifact(rel_path)
    if owner in {"plot", "script"} and owner in project.stages:
        project.set_stage_status(owner, "complete")
        project.current_stage = owner
        project.save()
    return result.invalidated_stages


def apply_artifact_revision(
    project: Project,
    rel_path: str,
    instruction: str,
    providers: Providers,
    *,
    invalidate: bool = True,
    image_paths: list[str] | None = None,
) -> ArtifactRevisionResult:
    """Revise one artifact with the configured LLM and record durable history."""
    path = safe_artifact_path(project, rel_path)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    if path.suffix.lower() not in TEXT_EXTS:
        raise ValueError("only text artifacts can be revised")
    if providers.llm is None:
        raise ValueError("this project profile does not include an LLM provider")

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("add an instruction before asking the LLM to revise an artifact")

    current = path.read_text()
    json_mode = path.suffix.lower() == ".json"
    gen = None
    result = ArtifactRevisionResult(
        artifact_path=rel_path,
        instruction=instruction,
        applied=False,
        invalidated=[],
    )
    try:
        project.assert_budget_available()
        prompt = _revision_prompt(project, rel_path, current, instruction, json_mode=json_mode)

        images = [str(image_path) for image_path in (image_paths or [])]
        if images:
            for image_path in images:
                if not Path(image_path).is_file():
                    raise FileNotFoundError(image_path)

            analyzer = getattr(providers, "reference_analyzer", None)
            if analyzer is None:
                raise ValueError(
                    "reference images were requested, but this project profile does not "
                    "include a reference analyzer"
                )

            cap = getattr(
                getattr(getattr(providers, "image", None), "capabilities", None),
                "max_reference_images",
                0,
            )
            if cap and cap > 0:
                images = images[:cap]
            style_anchors = {str(project.dir / rel) for rel in style_reference_paths(project)}
            vision_prompt = _vision_revision_prompt(
                prompt, style_reference=any(p in style_anchors for p in images)
            )
            language = str((project.model_config or {}).get("language") or "en")
            try:
                gen = analyzer.revise(images, prompt=vision_prompt, language=language)
            except NotImplementedError as exc:
                raise ValueError(
                    "the configured reference analyzer does not support vision-grounded "
                    "artifact revision"
                ) from exc
        else:
            gen = providers.llm.complete_json(prompt) if json_mode else providers.llm.complete(prompt)

        project.add_generation_cost(stage="adjustment", generation=gen)

        if json_mode:
            original = json.loads(current)
            revised = gen.content if isinstance(gen.content, (dict, list)) else json.loads(str(gen.content))
            _validate_json_revision(original, revised)
            new_text = json.dumps(revised, ensure_ascii=False, indent=2) + "\n"
        else:
            new_text = (gen.content if isinstance(gen.content, str) else str(gen.content)).strip() + "\n"

        path.write_text(new_text)
        if invalidate:
            invalidation = invalidate_artifact_revision(project, rel_path)
            owner = stage_for_artifact(rel_path)
            if owner in {"plot", "script"} and owner in project.stages:
                project.set_stage_status(owner, "complete")
                project.current_stage = owner
                project.save()
            result = ArtifactRevisionResult(
                artifact_path=rel_path,
                instruction=instruction,
                applied=True,
                invalidated=invalidation.invalidated_stages,
                provider=gen.provider,
                model=gen.model,
                archived=invalidation.archived,
                archive_dir=invalidation.archive_dir,
            )
        else:
            result = ArtifactRevisionResult(
                artifact_path=rel_path,
                instruction=instruction,
                applied=True,
                invalidated=[],
                provider=gen.provider,
                model=gen.model,
                archived=0,
                archive_dir="",
            )
        _write_history(project, result)
        return result
    except Exception as exc:
        result.error = str(exc)
        if gen is not None:
            result.provider = gen.provider
            result.model = gen.model
        _write_history(project, result)
        raise


def _validate_json_revision(original: Any, revised: Any) -> None:
    if isinstance(original, dict) and not isinstance(revised, dict):
        raise ValueError("LLM revision must preserve the same JSON shape: object expected")
    if isinstance(original, list) and not isinstance(revised, list):
        raise ValueError("LLM revision must preserve the same JSON shape: list expected")


def _revision_prompt(
    project: Project,
    rel_path: str,
    current: str,
    instruction: str,
    *,
    json_mode: bool,
) -> str:
    config = project.model_config or {}
    if json_mode:
        output_rule = (
            "Return only the revised artifact as valid JSON. Preserve the same top-level "
            "JSON shape and preserve existing IDs/names whenever possible."
        )
        label = "CURRENT_JSON"
    else:
        output_rule = "Return only the revised artifact text, with no markdown wrapper."
        label = "CURRENT_TEXT"

    return (
        "[task:artifact_revision]\n"
        "You are revising one Studio Agent project artifact. Keep the edit scoped to "
        "the selected file and do not invent downstream generated media.\n\n"
        f"PROJECT_ID: {project.project_id}\n"
        f"IDEA: {project.idea}\n"
        f"LANGUAGE: {config.get('language', 'auto')}\n"
        f"STYLE: {json.dumps(config.get('style', {}), ensure_ascii=False)}\n"
        f"FORMAT: {json.dumps(config.get('product_format', {}), ensure_ascii=False)}\n"
        f"ARTIFACT_PATH: {rel_path}\n"
        f"INSTRUCTION: {instruction}\n\n"
        f"{output_rule}\n\n"
        f"{label}:\n{current}\n"
    )


def _vision_revision_prompt(base_prompt: str, *, style_reference: bool) -> str:
    """Prepend image-grounding discipline; keep CURRENT_JSON/CURRENT_TEXT at the end intact."""
    preamble = (
        "You can SEE the reference image(s) of this subject. They are the ground truth for "
        "appearance — face, hair, wardrobe/outfit, accessories, and colors. Apply the "
        "INSTRUCTION while staying faithful to the image: where the image and the current text "
        "disagree, the IMAGE wins; transcribe the concrete visible details (the exact garments "
        "and colors) instead of deferring to an alias like '参考image1'."
    )
    if style_reference:
        preamble += (
            " One image is a STYLE reference (aliased @style): use ONLY its look — palette, "
            "lighting, rendering medium, texture — never its subject, characters, or composition."
        )
    return preamble + "\n\n" + base_prompt


def _write_history(project: Project, result: ArtifactRevisionResult) -> Path:
    history_dir = project.path("adjustments")
    history_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", result.artifact_path).strip("-") or "artifact"
    filename = f"{int(time.time() * 1000)}-{slug[:80]}.json"
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_path": result.artifact_path,
        "stage": stage_for_artifact(result.artifact_path),
        "instruction": result.instruction,
        "provider": result.provider,
        "model": result.model,
        "applied": result.applied,
        "invalidated": result.invalidated,
        "error": result.error,
    }
    path = history_dir / filename
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return path
