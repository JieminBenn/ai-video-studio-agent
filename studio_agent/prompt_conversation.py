"""Persistent, project-aware conversation for revising provider-bound prompts."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from .orchestrator.project import Project
from .prompt_approvals import (
    PromptApprovalError,
    invalidate_prompt_approvals,
    prompt_path,
)
from .prompt_validation import validate_grid_prompt
from .providers.base import Generation
from .reference_assets import load_reference_manifest
from .stages.base import Providers


CONVERSATION_REL = ("storyboard", "conversations", "prompt-review.jsonl")
MAX_CONTEXT_HISTORY_BYTES = 64 * 1024
_GATES = {"keyframes", "videos"}


@dataclass(frozen=True)
class PromptTurnResult:
    gate: str
    shot_ids: list[str]
    changed_paths: list[str]
    assistant_message: str
    vision_mode: str
    disclosure: str


def conversation_records(project: Project) -> list[dict[str, Any]]:
    path = project.path(*CONVERSATION_REL)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def build_prompt_context(
    project: Project,
    *,
    gate: str,
    shot_id: str,
    message: str,
    providers: Providers | None = None,
) -> tuple[str, list[str]]:
    """Read current project artifacts and compile one revision request."""
    gate = _gate(gate)
    shots = _shots(project)
    selected_index = next(
        (index for index, shot in enumerate(shots) if str(shot.get("id")) == shot_id),
        None,
    )
    if selected_index is None:
        raise FileNotFoundError(f"shot not found: {shot_id}")
    shot = shots[selected_index]
    selected_prompt_path = prompt_path(project, gate, shot)
    if not selected_prompt_path.is_file():
        raise FileNotFoundError(selected_prompt_path.relative_to(project.dir).as_posix())
    current_prompt = selected_prompt_path.read_text()

    neighbors = shots[max(0, selected_index - 1): selected_index + 2]
    neighbor_prompts = []
    for neighbor in neighbors:
        path = prompt_path(project, gate, neighbor)
        neighbor_prompts.append({
            "id": neighbor.get("id"),
            "shot": neighbor,
            "prompt": path.read_text() if path.is_file() else "",
        })

    identity_blocks = _identity_blocks(project)
    image_paths = _reference_image_paths(project, shot, gate=gate)
    history = _recent_records(conversation_records(project))
    capabilities = _provider_capabilities(providers)
    project_sources = _project_sources(project)
    reference_records = load_reference_manifest(project)
    prompt_kind = "grid" if (
        gate == "keyframes" and isinstance(shot.get("motion_grid"), dict)
    ) else ("static keyframe" if gate == "keyframes" else "motion video")
    vision_available = bool(
        image_paths and providers is not None and providers.reference_analyzer is not None
    )

    context = (
        "[task:prompt_conversation]\n"
        "You are revising one exact provider-bound prompt inside a persistent film project. "
        "Use the project memory, identity constraints, neighboring shots, and prior turns. "
        "Apply the user's requested change without weakening locked continuity.\n\n"
        f"GATE: {gate}\n"
        f"PROMPT TYPE: {prompt_kind}\n"
        f"SELECTED SHOT: {shot_id}\n"
        f"PROJECT IDEA: {_idea(project)}\n"
        f"LANGUAGE: {project.model_config.get('language', 'en')}\n"
        f"SPEND: ${project.total_cost():.4f}\n"
        f"COST CAP: {project.cost_cap if project.cost_cap is not None else 'none'}\n"
        f"PROVIDER CAPABILITIES: {json.dumps(capabilities, ensure_ascii=False, sort_keys=True)}\n\n"
        "PROJECT SOURCE MEMORY:\n"
        f"{json.dumps(project_sources, ensure_ascii=False, sort_keys=True)}\n\n"
        "LOCKED IDENTITY BOARD MEMORY (never truncate or contradict):\n"
        f"{identity_blocks or '(none)'}\n\n"
        "REFERENCE RECORDS AND DURABLE DESCRIPTIONS:\n"
        f"{json.dumps(reference_records, ensure_ascii=False, sort_keys=True)}\n"
        f"REFERENCE IMAGE PATHS: {json.dumps(image_paths, ensure_ascii=False)}\n"
        f"VISION AVAILABLE FOR THIS TURN: {vision_available}\n\n"
        "SELECTED + NEIGHBOR SHOTS AND CURRENT PROMPTS:\n"
        f"{json.dumps(neighbor_prompts, ensure_ascii=False, sort_keys=True)}\n\n"
        "PRIOR PROJECT CONVERSATION (newest records within 64 KiB):\n"
        f"{json.dumps(history, ensure_ascii=False, sort_keys=True)}\n\n"
        f"USER DIRECTION: {message.strip()}\n\n"
        f"CURRENT PROMPT:\n{current_prompt}\n\n"
        "RETURN ONLY the complete revised provider prompt as raw text. Do not explain it. "
        + _gate_contract(gate, shot)
    )
    return context, image_paths


def revise_prompt_turn(
    project: Project,
    *,
    gate: str,
    shot_id: str,
    message: str,
    apply_to_all: bool,
    providers: Providers,
) -> PromptTurnResult:
    gate = _gate(gate)
    message = str(message or "").strip()
    if not message:
        raise ValueError("add a message before revising prompts")
    shots = _shots(project)
    if not any(str(shot.get("id")) == shot_id for shot in shots):
        raise FileNotFoundError(f"shot not found: {shot_id}")
    targets = shots if apply_to_all else [
        shot for shot in shots if str(shot.get("id")) == shot_id
    ]

    contexts: list[tuple[dict, str, list[str]]] = []
    attachments: list[str] = []
    for shot in targets:
        sid = str(shot["id"])
        context, images = build_prompt_context(
            project,
            gate=gate,
            shot_id=sid,
            message=message,
            providers=providers,
        )
        contexts.append((shot, context, images))
        attachments.extend(images)

    user_record = _record(
        gate=gate,
        shot_id=shot_id,
        scope="all" if apply_to_all else "selected",
        role="user",
        text=message,
        attachments=_dedupe(attachments),
    )
    _append_records(project, [user_record])

    candidates: dict[Path, str] = {}
    generations: list[Generation] = []
    modes: list[str] = []
    try:
        for shot, context, images in contexts:
            generation, mode = _revise_with_available_model(
                project,
                providers,
                context=context,
                image_paths=images,
            )
            candidate = _generation_text(generation)
            generations.append(generation)
            modes.append(mode)
            _validate_candidate(gate, shot, candidate)
            candidates[prompt_path(project, gate, shot)] = candidate
    except Exception as exc:
        _append_records(project, [
            _record(
                gate=gate,
                shot_id=shot_id,
                scope="all" if apply_to_all else "selected",
                role="assistant",
                text=f"Prompt revision failed: {exc}",
                attachments=[],
                vision_mode=_combined_mode(modes),
                provider=", ".join(_dedupe([item.provider for item in generations])),
                model=", ".join(_dedupe([item.model for item in generations])),
                error=str(exc),
            )
        ])
        raise

    _write_candidates_atomically(candidates)
    changed_paths = [
        path.relative_to(project.dir).as_posix() for path in candidates
    ]
    changed_ids = [str(shot["id"]) for shot in targets]
    invalidate_prompt_approvals(project, gate, shot_ids=changed_ids)
    hashes = {
        path.relative_to(project.dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in candidates
    }
    mode = _combined_mode(modes)
    disclosure = _disclosure(mode, bool(attachments))
    assistant_message = (
        f"Updated {len(changed_paths)} {gate} prompt"
        + ("s." if len(changed_paths) != 1 else ".")
    )
    provider_names = _dedupe([generation.provider for generation in generations])
    model_names = _dedupe([generation.model for generation in generations])
    _append_records(project, [
        _record(
            gate=gate,
            shot_id=shot_id,
            scope="all" if apply_to_all else "selected",
            role="assistant",
            text=assistant_message,
            attachments=_dedupe(attachments),
            changed_paths=changed_paths,
            resulting_hashes=hashes,
            provider=", ".join(provider_names),
            model=", ".join(model_names),
            vision_mode=mode,
        )
    ])
    _write_adjustment(
        project,
        gate=gate,
        message=message,
        changed_paths=changed_paths,
        provider=", ".join(provider_names),
        model=", ".join(model_names),
    )
    return PromptTurnResult(
        gate=gate,
        shot_ids=changed_ids,
        changed_paths=changed_paths,
        assistant_message=assistant_message,
        vision_mode=mode,
        disclosure=disclosure,
    )


def _revise_with_available_model(
    project: Project,
    providers: Providers,
    *,
    context: str,
    image_paths: list[str],
) -> tuple[Generation, str]:
    generation = None
    if image_paths and providers.reference_analyzer is not None:
        try:
            generation = providers.reference_analyzer.revise(
                image_paths,
                prompt=context,
                language=str(project.model_config.get("language") or "en"),
            )
        except NotImplementedError:
            generation = None
        if generation is not None:
            project.add_generation_cost(
                stage="prompt_conversation", generation=generation
            )
            return generation, "vision"
    if providers.llm is None:
        raise ValueError("this project profile does not include a prompt-revision LLM")
    text_context = context.replace(
        "VISION AVAILABLE FOR THIS TURN: True",
        "VISION AVAILABLE FOR THIS TURN: False\n"
        "VISION FALLBACK: the configured analyzer could not inspect these images; use only "
        "the saved identity boards and durable descriptions.",
    )
    generation = providers.llm.complete(text_context)
    project.add_generation_cost(
        stage="prompt_conversation", generation=generation
    )
    return generation, "text_only"


def _validate_candidate(gate: str, shot: dict, text: str) -> None:
    if not text.strip():
        raise PromptApprovalError(f"empty revised {gate} prompt for {shot.get('id')}")
    if gate == "videos":
        return
    grid = shot.get("motion_grid")
    if not isinstance(grid, dict):
        # Plain keyframe prompts are trusted as written — no prose motion-word matching.
        return
    validation = validate_grid_prompt(
        text,
        shot,
        panel_count=int(grid.get("panel_count") or 0),
    )
    if not validation.valid:
        raise PromptApprovalError(
            f"invalid keyframes prompt for {shot.get('id')}: "
            + ", ".join(validation.errors)
        )


def _write_candidates_atomically(candidates: dict[Path, str]) -> None:
    originals = {path: path.read_bytes() for path in candidates}
    temp_paths: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for path, text in candidates.items():
            temp = path.with_suffix(path.suffix + ".conversation.tmp")
            temp.write_text(text)
            temp_paths[path] = temp
        for path, temp in temp_paths.items():
            temp.replace(path)
            replaced.append(path)
    except Exception:
        for path in replaced:
            path.write_bytes(originals[path])
        raise
    finally:
        for temp in temp_paths.values():
            if temp.exists():
                temp.unlink()


def _generation_text(generation: Generation) -> str:
    content = generation.content
    if isinstance(content, dict):
        return str(content.get("prompt") or content.get("text") or "").strip()
    return str(content or "").strip()


def _shots(project: Project) -> list[dict]:
    path = project.path("storyboard", "shots.json")
    if not path.is_file():
        return []
    return [
        shot for shot in json.loads(path.read_text()).get("shots", [])
        if isinstance(shot, dict)
    ]


def _idea(project: Project) -> str:
    path = project.path("story", "idea.md")
    return path.read_text().strip() if path.is_file() else project.idea


def _project_sources(project: Project) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for rel in (
        "story/creative_brief.json",
        "story/plot.json",
        "story/script.json",
        "storyboard/visual_sequence.json",
    ):
        path = project.path(*rel.split("/"))
        if not path.is_file():
            continue
        try:
            sources[rel] = json.loads(path.read_text())
        except json.JSONDecodeError:
            sources[rel] = path.read_text()
    return sources


def _identity_blocks(project: Project) -> str:
    blocks = []
    for pattern in (
        "bible/characters/*/identity_board.json",
        "bible/characters/*/character.json",
        "bible/locations/*/location.json",
    ):
        for path in sorted(project.dir.glob(pattern)):
            blocks.append(
                f"IDENTITY BOARD FILE {path.relative_to(project.dir).as_posix()}:\n"
                + path.read_text()
            )
    return "\n\n".join(blocks)


def _reference_image_paths(project: Project, shot: dict, *, gate: str) -> list[str]:
    refs: list[str] = []
    if gate == "videos":
        keyframe = str(shot.get("keyframe") or "")
        if keyframe:
            refs.append(f"storyboard/keyframes/{keyframe}")
    for field in (
        "keyframe_reference_images",
        "reference_images",
        "named_reference_images",
        "target_state_reference_images",
        "reference_style_images",
    ):
        refs.extend(str(value) for value in (shot.get(field) or []) if value)
    characters = {str(value).casefold() for value in shot.get("characters") or []}
    locations = {str(value).casefold() for value in shot.get("reference_locations") or []}
    for record in load_reference_manifest(project):
        if record.get("status") != "resolved":
            continue
        target_type = str(record.get("target_type") or "")
        target_id = str(record.get("target_id") or "").casefold()
        if (
            target_type == "style"
            or (target_type == "character" and target_id in characters)
            or (target_type == "location" and target_id in locations)
        ):
            refs.append(str(record.get("path") or ""))
    paths = []
    for ref in _dedupe(refs):
        path = Path(ref)
        if not path.is_absolute():
            path = project.dir / path
        if path.is_file():
            paths.append(str(path.resolve()))
    return paths


def _recent_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total = 0
    for record in reversed(records):
        size = len(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        if total + size > MAX_CONTEXT_HISTORY_BYTES:
            break
        selected.append(record)
        total += size
    return list(reversed(selected))


def _provider_capabilities(providers: Providers | None) -> dict[str, Any]:
    if providers is None:
        return {}
    result = {}
    for name in ("image", "video"):
        provider = getattr(providers, name, None)
        capabilities = getattr(provider, "capabilities", None)
        if capabilities is None:
            continue
        if is_dataclass(capabilities):
            result[name] = asdict(capabilities)
        else:
            result[name] = {
                key: value
                for key, value in vars(capabilities).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            }
    return result


def _gate_contract(gate: str, shot: dict) -> str:
    if gate == "videos":
        return (
            "Keep explicit subject motion, camera behavior, timing, continuity, and exact "
            "dialogue/audio constraints suitable for the video provider."
        )
    if isinstance(shot.get("motion_grid"), dict):
        return (
            "Return one composite ordered panel grid; every panel is a frozen still. Keep "
            "all panel labels/counts and do not describe a continuous camera path or audio."
        )
    return (
        "Return one detailed frozen opening instant. Static framing, viewpoint, angle, and "
        "lens are allowed; camera movement, timing, sequential action, future states, and "
        "audio are forbidden."
    )


def _record(
    *,
    gate: str,
    shot_id: str,
    scope: str,
    role: str,
    text: str,
    attachments: list[str],
    changed_paths: list[str] | None = None,
    resulting_hashes: dict[str, str] | None = None,
    provider: str = "",
    model: str = "",
    vision_mode: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": gate,
        "shot_id": shot_id,
        "scope": scope,
        "role": role,
        "text": text,
        "attachments": attachments,
        "changed_paths": list(changed_paths or []),
        "resulting_hashes": dict(resulting_hashes or {}),
        "provider": provider,
        "model": model,
        "vision_mode": vision_mode,
        "error": error,
    }


def _append_records(project: Project, new_records: list[dict[str, Any]]) -> None:
    path = project.path(*CONVERSATION_REL)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = conversation_records(project) + new_records
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    temp.replace(path)


def _write_adjustment(
    project: Project,
    *,
    gate: str,
    message: str,
    changed_paths: list[str],
    provider: str,
    model: str,
) -> None:
    root = project.path("adjustments")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{time.time_ns()}-prompt-conversation-{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": gate,
        "instruction": message,
        "changed_paths": changed_paths,
        "provider": provider,
        "model": model,
        "applied": True,
    }, ensure_ascii=False, indent=2) + "\n")


def _combined_mode(modes: list[str]) -> str:
    unique = set(modes)
    if unique == {"vision"}:
        return "vision"
    if unique == {"text_only"} or not unique:
        return "text_only"
    return "mixed"


def _disclosure(mode: str, had_images: bool) -> str:
    if mode == "vision":
        return "The revision model directly inspected the supplied project images."
    if mode == "mixed":
        return "Some revisions inspected images; others used saved project descriptions."
    if had_images:
        return (
            "The text-only model did not directly inspect images; it used saved identity "
            "boards, reference records, and project descriptions."
        )
    return "No project images were available; the revision used saved project text."


def _gate(gate: str) -> str:
    value = str(gate or "").strip()
    if value not in _GATES:
        raise ValueError(f"unsupported prompt conversation gate: {gate}")
    return value


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
