"""Prepare editable motion prompts and reference bindings without generating video."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .base import Providers, Stage, StageResult
from ..cinematography_recipes import load_camera_recipes, resolve_recipe
from ..language import language_label
from ..creative_brief import load_creative_brief, resolved_value
from ..formats import project_format
from ..image_prompt_director import direct_video_prompt, directed_prompt_text, director_enabled
from ..knowledge import (
    KnowledgeQuery,
    get_or_create_packet,
    load_packaged_core,
    packet_guidance,
)
from ..prompt_approvals import confirm_prompt_batch
from ..prompt_context import project_prompt_context
from ..providers.base import Generation, VideoCapabilities
from ..reference_assets import cap_reference_paths, live_reference_paths_for_shot
from ..runtime_skills import load_prompt_skills
from ..video_prompt import compile_video_prompt


class VideoPromptsStage(Stage):
    """Compile all provider-ready video prompts, then stop at a human gate."""

    name = "video_prompts"

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="video prompts already complete")

        shots_path = project.path("storyboard", "shots.json")
        shots = json.loads(shots_path.read_text()).get("shots") or []
        capabilities = getattr(providers.video, "capabilities", VideoCapabilities())

        native_audio = str(project.model_config.get("audio_mode") or "") == "native_video"
        if native_audio and not capabilities.supports_native_audio:
            raise ValueError(
                f"video provider '{getattr(providers.video, 'name', 'unknown')}' does not "
                "support the project's native audio mode"
            )

        for index, shot in enumerate(shots):
            effective_shot, effective_capabilities, binding = prepare_video_binding(
                project,
                shot,
                capabilities,
                has_expected_carry=(
                    index > 0
                    and capabilities.supports_last_frame
                    and int(capabilities.max_image_inputs) > 1
                ),
            )
            shot["video_prompt_binding"] = binding
            self._ensure_prompt_file(
                project,
                effective_shot,
                shots,
                binding["expects_previous_last_frame"],
                effective_capabilities,
                providers,
            )

        shots_path.write_text(json.dumps({"shots": shots}, ensure_ascii=False, indent=2) + "\n")
        return StageResult(status="complete", message=f"video prompts: {len(shots)} shot(s)")

    def on_approve(self, project, *, auto: bool = False) -> None:
        confirm_prompt_batch(
            project,
            "videos",
            confirmer="auto" if auto else "human",
        )

    def _ensure_prompt_file(
        self,
        project,
        shot: dict,
        shots: list[dict],
        has_prev: bool,
        capabilities: VideoCapabilities,
        providers: Providers,
    ) -> str:
        prompt_path = project.path("storyboard", "prompts", f"{shot['id']}.video.md")
        packet = self._shot_packet(project, shot)
        knowledge_path = project.path(
            "storyboard", "prompts", f"{shot['id']}.knowledge.json"
        )
        if not knowledge_path.is_file():
            knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            knowledge_path.write_text(
                json.dumps(packet, ensure_ascii=False, indent=2) + "\n"
            )

        if prompt_path.is_file():
            return prompt_path.read_text()

        language = project.model_config.get("language") or "en"
        camera_recipe = resolve_recipe(
            load_camera_recipes(), shot.get("camera_recipe"), language=language
        )
        brief = compile_video_prompt(
            shot,
            style=dict(project.model_config.get("style") or {}),
            product_format=project_format(project),
            capabilities=capabilities,
            has_previous_shot=has_prev,
            context=project_prompt_context(project, shot, shots=shots),
            prompt_skills=load_prompt_skills([
                "location_identity",
                "micro_expression",
                "cinematography",
                "camera_movement",
                "character_motion",
            ]),
            camera_recipe=camera_recipe,
            knowledge_guidance=packet_guidance(packet),
            hard_avoidances=list(load_creative_brief(project).get("hard_avoidances") or []),
            has_style_reference=bool(shot.get("reference_style_images")),
            language=language_label(language),
        )
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        brief_path = project.path("storyboard", "prompts", f"{shot['id']}.video.brief.md")
        if not brief_path.is_file():
            brief_path.write_text(brief)
        final = self._direct_prompt(project, shot, providers, brief=brief)
        prompt_path.write_text(final)
        return final

    def _direct_generation(
        self,
        project,
        shot: dict,
        providers: Providers,
        *,
        brief: str,
    ) -> Generation:
        config = project.model_config or {}
        if not director_enabled(config) or getattr(providers, "llm", None) is None:
            return Generation(
                content={"prompt": brief, "negative": ""},
                provider="compiled",
                model="deterministic-compiler",
            )
        style = dict(config.get("style") or {})
        return direct_video_prompt(
            providers,
            project,
            brief=brief,
            style=style,
            style_name=str(config.get("style_name") or style.get("look") or ""),
            intent=str(config.get("creative_intent") or project.idea or ""),
            reference_aliases=list(shot.get("characters") or []),
            hard_avoidances=list(load_creative_brief(project).get("hard_avoidances") or []),
            language=str(config.get("language") or "en"),
            stage=self.name,
        )

    def _direct_prompt(self, project, shot: dict, providers: Providers, *, brief: str) -> str:
        gen = self._direct_generation(project, shot, providers, brief=brief)
        final = directed_prompt_text(gen.content if isinstance(gen.content, dict) else {})
        return final or brief

    def _shot_packet(self, project, shot: dict) -> dict:
        config = project.model_config or {}
        style = dict(config.get("style") or {})
        creative = load_creative_brief(project)
        query = KnowledgeQuery(
            stage="storyboard",
            domains=(
                "camera_movement",
                "lens_angle",
                "lighting",
                "composition",
                "performance",
                "editing",
            ),
            intent_text=" ".join(filter(None, [
                project.idea,
                resolved_value(creative, "tone"),
                resolved_value(creative, "camera_language"),
                *(
                    str(shot.get(key) or "")
                    for key in (
                        "dramatic_purpose",
                        "action",
                        "emotion",
                        "camera",
                        "movement_motivation",
                        "lighting_state",
                    )
                ),
            ])),
            intents=("continuity", "pacing"),
            style=str(config.get("style_name") or style.get("look") or ""),
            format_name=str(
                config.get("format_name")
                or (config.get("product_format") or {}).get("name")
                or ""
            ),
            language=str(config.get("language") or "en"),
            capabilities=frozenset({"camera_motion"}),
            hard_avoidances=tuple(creative.get("hard_avoidances") or []),
        )
        retrieval = config.get("knowledge_retrieval") or {}
        return get_or_create_packet(
            project,
            purpose="shot",
            target=str(shot.get("id") or "unknown"),
            query=query,
            entries=load_packaged_core(),
            limit=int(retrieval.get("max_entries", 6)),
        )


def capability_signature(capabilities: VideoCapabilities) -> dict[str, object]:
    return {
        "supports_reference_images": bool(capabilities.supports_reference_images),
        "supports_last_frame": bool(capabilities.supports_last_frame),
        "supports_native_audio": bool(capabilities.supports_native_audio),
        "max_image_inputs": int(capabilities.max_image_inputs),
        "supports_storyboard_grid": bool(capabilities.supports_storyboard_grid),
    }


def prepare_video_binding(
    project,
    shot: dict,
    capabilities: VideoCapabilities,
    *,
    has_expected_carry: bool,
) -> tuple[dict, VideoCapabilities, dict]:
    """Select the exact reference slots shared by prompt preparation and generation."""
    extra_capacity = max(0, int(capabilities.max_image_inputs) - 1)
    expects_carry = bool(has_expected_carry and extra_capacity > 0)
    reference_capacity = extra_capacity - int(expects_carry)
    reference_images: list[str] = []
    target_state_reference_images: list[str] = []
    if capabilities.supports_reference_images and reference_capacity > 0:
        requested_targets = _existing_target_state_reference_images(project, shot)
        requested = _dedupe(requested_targets + _existing_reference_images(project, shot))
        named = _existing_project_paths(
            project, list(shot.get("named_reference_images") or [])
        )
        reference_images = cap_reference_paths(
            project,
            requested,
            reference_capacity,
            named_refs=named,
        )
        selected = set(reference_images)
        target_state_reference_images = [
            ref for ref in requested_targets if ref in selected
        ]

    selected = set(reference_images)
    effective_shot = dict(shot)
    effective_shot["reference_images"] = reference_images
    effective_shot["named_reference_images"] = [
        ref
        for ref in _existing_project_paths(
            project, list(shot.get("named_reference_images") or [])
        )
        if ref in selected
    ]
    effective_shot["target_state_reference_images"] = target_state_reference_images
    effective_shot["reference_style_images"] = [
        ref
        for ref in _existing_project_paths(
            project, list(shot.get("reference_style_images") or [])
        )
        if ref in selected
    ]
    effective_capabilities = replace(
        capabilities,
        supports_reference_images=bool(reference_images),
        supports_last_frame=expects_carry,
    )
    binding = {
        "capabilities": capability_signature(capabilities),
        "expects_previous_last_frame": expects_carry,
        "reference_images": [_project_relative(project, ref) for ref in reference_images],
        "target_state_reference_images": [
            _project_relative(project, ref) for ref in target_state_reference_images
        ],
    }
    return effective_shot, effective_capabilities, binding


def resolve_video_binding(project, binding: dict) -> tuple[list[str], list[str]]:
    refs = _existing_project_paths(project, list(binding.get("reference_images") or []))
    targets = _existing_project_paths(
        project, list(binding.get("target_state_reference_images") or [])
    )
    return refs, [ref for ref in targets if ref in set(refs)]


def _existing_reference_images(project, shot: dict) -> list[str]:
    refs = live_reference_paths_for_shot(
        project,
        shot,
        ("reference_images", "target_state_reference_images"),
    )
    return _existing_project_paths(project, refs)


def _existing_target_state_reference_images(project, shot: dict) -> list[str]:
    return _existing_project_paths(
        project, list(shot.get("target_state_reference_images") or [])
    )


def _existing_project_paths(project, refs: list[str]) -> list[str]:
    paths = []
    seen = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        path = Path(ref)
        if not path.is_absolute():
            path = project.dir / ref
        if path.is_file():
            paths.append(str(path.resolve()))
    return paths


def _project_relative(project, ref: str) -> str:
    path = Path(ref)
    if not path.is_absolute():
        return path.as_posix()
    return path.resolve().relative_to(project.dir.resolve()).as_posix()


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
