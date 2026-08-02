"""Scoped deletion helpers for local, file-backed asset regeneration.

These helpers do not call paid providers. They delete only the selected artifact family
and mark the owning/downstream stages pending so the next resume can backfill through the
normal idempotent stage implementations.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .artifact_studio import apply_artifact_revision
from .clip_sequence import clamp_clip_duration
from .invalidation import (
    archive_paths,
    invalidate_bible_asset,
    invalidate_bible_text,
    invalidate_shots,
)
from .orchestrator.project import Project
from .prompt_approvals import (
    confirm_prompt,
    invalidate_prompt_approvals,
    rebase_prompt_approvals,
)
from .reference_assets import (
    reference_paths_for_target,
    reference_paths_in_text,
    reference_upload_paths_for_shot,
    regeneration_reference_images,
)
from .stages.base import Providers


@dataclass
class RegenerationResult:
    kind: str
    removed: int
    invalidated: list[str]
    current_stage: str
    archive_dir: str = ""


@dataclass
class KeyframeFeedbackResult:
    shot_id: str
    prompt_path: str
    feedback: str
    applied: bool
    removed: int
    invalidated: list[str]
    provider: str = ""
    model: str = ""
    error: str = ""


@dataclass
class BibleTextResult:
    kind: str
    slug: str
    rel: str
    applied: bool
    needs_resume: bool


_REGEN_KINDS = {"keyframe", "video"}


def _regeneration_request_path(project: Project, shot_id: str, kind: str) -> Path:
    if kind not in _REGEN_KINDS:
        raise ValueError(f"unsupported regeneration kind: {kind}")
    safe_id = str(shot_id or "").strip()
    if not safe_id or Path(safe_id).name != safe_id:
        raise ValueError("invalid shot id")
    return project.path("storyboard", "regeneration", f"{safe_id}.{kind}.json")


def pending_regeneration(project: Project, shot_id: str, kind: str) -> dict | None:
    """Return one persisted regeneration request, if present."""
    path = _regeneration_request_path(project, shot_id, kind)
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"invalid regeneration request: {path}")
    return value


def _write_regeneration_request(project: Project, request: dict) -> Path:
    path = _regeneration_request_path(
        project, str(request["shot_id"]), str(request["kind"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
    return path


def record_regeneration_error(
    project: Project,
    shot_id: str,
    kind: str,
    error: Exception,
) -> None:
    request = pending_regeneration(project, shot_id, kind)
    if request is None:
        return
    request["error"] = str(error)
    request["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_regeneration_request(project, request)


def _pending_stages(project: Project, from_stage: str) -> list[str]:
    if from_stage not in project.stages:
        return []
    stages = project.stages[project.stages.index(from_stage):]
    for stage in stages:
        project.set_stage_status(stage, "pending")
    project.current_stage = from_stage
    project.status = "in_progress"
    project.save()
    return stages


def _queue_regeneration(project: Project, shot_id: str, kind: str) -> RegenerationResult:
    project.assert_budget_available()
    approval_kind = "keyframes" if kind == "keyframe" else "videos"
    confirm_prompt(project, approval_kind, shot_id, confirmer="regeneration")
    request = {
        "version": 1,
        "request_id": uuid.uuid4().hex,
        "shot_id": shot_id,
        "kind": kind,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "error": "",
    }
    _write_regeneration_request(project, request)
    stage = "keyframes" if kind == "keyframe" else "video"
    invalidated = _pending_stages(project, stage)
    return RegenerationResult(kind, 0, invalidated, stage)


def queue_keyframe_regeneration(project: Project, shot_id: str) -> RegenerationResult:
    return _queue_regeneration(project, shot_id, "keyframe")


def promote_candidate(
    project: Project,
    request: dict,
    *,
    candidate_to_live: dict[Path, str],
    live_family: list[str],
) -> str:
    """Archive the accepted family and atomically promote complete candidate files."""
    required = list(candidate_to_live)
    if not required or any(not path.is_file() for path in required):
        raise FileNotFoundError("regeneration candidate is incomplete")
    shot_id = str(request["shot_id"])
    kind = str(request["kind"])
    archive = archive_paths(
        project,
        live_family,
        reason=f"{kind}-candidate-promoted",
        source_path=_regeneration_request_path(project, shot_id, kind)
        .relative_to(project.dir)
        .as_posix(),
        affected_shot_ids=[shot_id],
    )
    promoted: list[tuple[Path, Path]] = []
    try:
        for candidate, live_rel in candidate_to_live.items():
            live = _safe_project_path(project, live_rel)
            live.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(live)
            promoted.append((candidate, live))
    except Exception:
        for candidate, live in reversed(promoted):
            if live.is_file() and not candidate.exists():
                candidate.parent.mkdir(parents=True, exist_ok=True)
                live.replace(candidate)
        if archive.archive_dir:
            restore_archived_shot_family(project, archive.archive_dir, shot_id)
        raise
    _regeneration_request_path(project, shot_id, kind).unlink(missing_ok=True)
    return archive.archive_dir


def restore_archived_shot_family(
    project: Project,
    archive_dir: str,
    shot_id: str,
) -> list[str]:
    """Restore only missing shot-specific files from a project history snapshot."""
    rel = Path(str(archive_dir or ""))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "history":
        raise ValueError("archive must be inside project history")
    source_root = (project.dir / rel).resolve()
    history_root = project.path("history").resolve()
    if history_root not in source_root.parents or not source_root.is_dir():
        raise ValueError("archive must be inside project history")
    keyframe = ""
    shot = _find_shot(project, shot_id)
    if shot is not None:
        keyframe = str(shot.get("keyframe") or "")
    manifest_path = source_root / "manifest.json"
    aggregate_paths: set[str] = set()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        affected = {str(value) for value in manifest.get("affected_shot_ids") or []}
        if shot_id in affected:
            aggregate_paths = {
                str(value)
                for value in manifest.get("archived_paths") or []
                if str(value) == "assets/clips/clips.json"
            }
    restored: list[str] = []
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.name == "manifest.json":
            continue
        project_rel = source.relative_to(source_root).as_posix()
        name = source.name
        if not (
            name.startswith(f"{shot_id}.")
            or (keyframe and name == keyframe)
            or project_rel in aggregate_paths
        ):
            continue
        destination = _safe_project_path(project, project_rel)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        restored.append(project_rel)
    return restored


def regenerate_bible_text(
    project: Project,
    kind: str,
    slug: str,
    instruction: str,
    providers: Providers,
) -> BibleTextResult:
    """Revise a character/location's text (LLM) without re-rendering its image."""
    kind = str(kind or "").strip().lower()
    slug = str(slug or "").strip()
    instruction = str(instruction or "").strip()
    if not instruction:
        raise ValueError("add a comment before regenerating the text")
    if kind not in {"character", "location"}:
        raise ValueError(f"unknown bible entity kind: {kind}")
    folder = "characters" if kind == "character" else "locations"
    filename = "character.json" if kind == "character" else "location.json"
    rel = f"bible/{folder}/{slug}/{filename}"
    if not _safe_project_path(project, rel).is_file():
        raise FileNotFoundError(rel)

    try:
        name = str(json.loads(_safe_project_path(project, rel).read_text()).get("name") or "")
    except (ValueError, OSError):
        name = ""
    target_ids = [name, slug] if name else [slug]
    subject_refs = [
        str(project.dir / r)
        for r in reference_paths_for_target(project, kind, *target_ids)
    ]
    image_paths = regeneration_reference_images(
        project, comment=instruction, subject_refs=subject_refs
    )
    apply_artifact_revision(
        project, rel, instruction, providers, invalidate=False, image_paths=image_paths
    )
    invalidate_bible_text(project, kind, slug)
    return BibleTextResult(
        kind=kind,
        slug=slug,
        rel=rel,
        applied=True,
        needs_resume=(kind == "character"),
    )


def regenerate_bible_asset(project: Project, rel_path: str) -> RegenerationResult:
    """Delete one generated bible image and resume at the bible stage."""
    rel = _validate_bible_asset_rel(rel_path)
    path = _safe_project_path(project, rel)
    if not path.is_file():
        raise FileNotFoundError(rel)

    _rotate_character_seed(project, rel)
    result = invalidate_bible_asset(project, rel)
    return RegenerationResult(
        "bible_asset",
        result.archived,
        result.invalidated_stages,
        result.current_stage,
        result.archive_dir,
    )


def regenerate_keyframe(project: Project, shot_id: str) -> RegenerationResult:
    """Queue a candidate keyframe while keeping the accepted image live."""
    shot = _find_shot(project, shot_id)
    if shot is None:
        raise FileNotFoundError(f"shot not found: {shot_id}")
    keyframe = str(shot.get("keyframe") or "")
    if not keyframe:
        raise FileNotFoundError(f"shot has no keyframe: {shot_id}")

    keyframe_path = _safe_project_path(project, f"storyboard/keyframes/{keyframe}")
    if not keyframe_path.is_file():
        raise FileNotFoundError(keyframe_path.relative_to(project.dir).as_posix())

    return queue_keyframe_regeneration(project, shot_id)


def _rechain_shots(project: Project, shots: list[dict]) -> None:
    """Rebuild shot-to-shot links after a structural edit (delete/insert), mode-aware.

    Clip mode replays ClipStage's previous-keyframe continuity anchoring. Story mode
    must NOT get those clip-continuity fields — its continuity is the last-frame carry
    that ``prepare_video_binding`` recomputes when the re-pended stages run — so it only
    rebuilds the ``deps`` chain. Both modes drop keyframe references that point at
    shots which no longer exist.
    """
    if "clip" in project.stages:
        from .stages.clip import ClipStage

        for shot in shots:
            shot["continuity_reference"] = ""
            shot["continuity_notes"] = ""
        ClipStage()._chain_continuity(shots)
    valid = {f"storyboard/keyframes/{shot.get('keyframe')}" for shot in shots}
    prev_id = None
    for shot in shots:
        refs = [
            ref for ref in (shot.get("keyframe_reference_images") or [])
            if not str(ref).startswith("storyboard/keyframes/") or ref in valid
        ]
        shot["keyframe_reference_images"] = refs
        shot["deps"] = [prev_id] if prev_id else []
        prev_id = shot.get("id")


def _carry_narration(deleted: dict, remaining: list[dict], deleted_index: int) -> str | None:
    """Move a deleted story-mode shot's narration onto a surviving same-scene shot.

    The storyboard places each scene's 旁白 on the scene's first shot; deleting that
    shot must not silently drop the voiceover. The text moves verbatim (a pure field
    move — no language interpretation, invariant #11) to the nearest surviving shot of
    the same scene: the next one, else the previous. Returns the receiving shot's id,
    or ``None`` when nothing carried.
    """
    narration = str(deleted.get("narration") or "").strip()
    if not narration:
        return None
    scene = deleted.get("scene")
    after = [s for s in remaining[deleted_index:] if s.get("scene") == scene]
    before = [s for s in reversed(remaining[:deleted_index]) if s.get("scene") == scene]
    receiver = after[0] if after else (before[0] if before else None)
    if receiver is None:
        return None
    existing = str(receiver.get("narration") or "").strip()
    if not existing:
        receiver["narration"] = narration
    elif after:
        receiver["narration"] = f"{narration} {existing}"  # deleted shot came first
    else:
        receiver["narration"] = f"{existing} {narration}"
    return str(receiver.get("id"))


def delete_clip_shot(project: Project, shot_id: str) -> RegenerationResult:
    """Remove one shot from shots.json at its gate, archiving its artifacts.

    Works in both clip and story mode. Archives the deleted shot's keyframe/prompt
    family (reversible) via the shared invalidation machinery, re-chains the survivors
    mode-aware (clip continuity vs deps only), carries story-mode narration to a
    surviving same-scene shot, and leaves the planning stage at its human gate (no
    auto-run). Refuses to delete the last shot.
    """
    shots_path = project.path("storyboard", "shots.json")
    if not shots_path.is_file():
        raise FileNotFoundError("storyboard/shots.json")
    shots = [s for s in json.loads(shots_path.read_text()).get("shots", []) if isinstance(s, dict)]
    if not any(s.get("id") == shot_id for s in shots):
        raise FileNotFoundError(f"shot not found: {shot_id}")
    if len(shots) <= 1:
        raise ValueError("cannot delete the last remaining clip")

    deleted_index = next(i for i, shot in enumerate(shots) if shot.get("id") == shot_id)
    deleted = shots[deleted_index]
    remaining = [s for s in shots if s.get("id") != shot_id]

    # In story mode the deleted shot may carry its scene's narration; the receiving
    # neighbor's stale narration stem (if any) must re-synthesize with the moved text.
    narration_receiver = None
    if "clip" not in project.stages:
        narration_receiver = _carry_narration(deleted, remaining, deleted_index)
    extra_paths = []
    if narration_receiver:
        extra_paths = [
            f"assets/audio/{narration_receiver}.narration.wav",
            "assets/audio/audio.json",
        ]

    # Archive the deleted shot's keyframe + prompt family (+ any downstream) — reversible.
    planning_stage = "clip" if "clip" in project.stages else "storyboard"
    result = invalidate_shots(
        project,
        [shot_id],
        from_stage=planning_stage,
        reason="clip-deleted",
        include_keyframes=True,
        include_prompts=True,
        extra_paths=extra_paths,
    )

    _rechain_shots(project, remaining)
    shots_path.write_text(json.dumps({"shots": remaining}, indent=2, ensure_ascii=False))

    sequence_path = project.path("storyboard", "visual_sequence.json")
    if "clip" in project.stages and sequence_path.is_file():
        sequence = json.loads(sequence_path.read_text())
        segments = list(sequence.get("segments") or [])
        if deleted_index < len(segments):
            segments.pop(deleted_index)
        sequence["segments"] = segments
        sequence["clip_durations"] = [
            segment.get("duration_s") for segment in segments
        ]
        sequence["target_duration_s"] = sum(
            float(value or 0) for value in sequence["clip_durations"]
        )
        sequence["beats"] = [
            beat for segment in segments for beat in (segment.get("visual_beats") or [])
        ]
        sequence_path.write_text(json.dumps(sequence, indent=2, ensure_ascii=False))

    successor_ids = (
        [str(remaining[deleted_index]["id"])] if deleted_index < len(remaining) else []
    )
    for kind in ("keyframes", "videos"):
        rebase_prompt_approvals(
            project,
            kind,
            invalidated_shot_ids=[shot_id, *successor_ids],
        )

    # Keep the clip stage at its gate so the user can keep editing, then Approve. No auto-run.
    project.set_stage_status(planning_stage, "complete")
    project.current_stage = planning_stage
    project.save()
    return RegenerationResult(
        "clip_deleted",
        result.archived,
        result.invalidated_stages,
        planning_stage,
        result.archive_dir,
    )


def project_video_capabilities(project: Project):
    """The configured video provider's capabilities (duration bounds, references).

    Constructor-only — building the provider never hits the network. Falls back to the
    ``VideoCapabilities`` defaults when the profile cannot be built locally, so every
    surface (dashboard forms, CLI validation, duration edits) reads the same numbers.
    """
    from .providers.base import VideoCapabilities

    try:
        from . import cli

        providers = cli.build_providers(project.model_config or {})
        capabilities = getattr(getattr(providers, "video", None), "capabilities", None)
        if capabilities is not None:
            return capabilities
    except Exception:
        pass
    return VideoCapabilities()


def set_clip_duration(project: Project, shot_id: str, seconds) -> RegenerationResult:
    """Set one clip's duration at the clip gate, clamped to the provider's legal range.

    Blank/``None`` seconds stores ``null`` — "let the video model's default length
    govern". Writes duration_s to shots.json. If a paid clip already exists for the
    shot, marks it stale so it re-renders later; at the clip gate nothing downstream
    exists yet.
    """
    capabilities = project_video_capabilities(project)

    shots_path = project.path("storyboard", "shots.json")
    if not shots_path.is_file():
        raise FileNotFoundError("storyboard/shots.json")
    shots = [s for s in json.loads(shots_path.read_text()).get("shots", []) if isinstance(s, dict)]
    target = next((s for s in shots if s.get("id") == shot_id), None)
    if target is None:
        raise FileNotFoundError(f"shot not found: {shot_id}")
    target["duration_s"] = clamp_clip_duration(
        seconds,
        min_s=capabilities.min_duration_s,
        max_s=capabilities.max_duration_s,
    )
    shots_path.write_text(json.dumps({"shots": shots}, indent=2, ensure_ascii=False))

    archived = 0
    invalidated: list[str] = []
    archive_dir = ""
    if _safe_project_path(project, f"assets/clips/{shot_id}.mp4").is_file():
        result = invalidate_shots(
            project, [shot_id], from_stage="video", reason="clip-duration-changed",
        )
        archived = result.archived
        invalidated = result.invalidated_stages
        archive_dir = result.archive_dir
    return RegenerationResult(
        "clip_duration", archived, invalidated, project.current_stage or "", archive_dir,
    )


def revise_keyframe_from_feedback(
    project: Project,
    shot_id: str,
    feedback: str,
    providers: Providers,
) -> KeyframeFeedbackResult:
    """Revise one shot's keyframe prompt from visual feedback, then rerender locally.

    This keeps prompts as the editable artifact: the LLM changes
    ``storyboard/prompts/<shot>.keyframe.md`` first, then only the selected
    keyframe/output family is removed so the normal storyboard resume path can
    generate a new image from the corrected instructions.
    """
    shot = _find_shot(project, shot_id)
    if shot is None:
        raise FileNotFoundError(f"shot not found: {shot_id}")
    feedback = feedback.strip()
    if not feedback:
        raise ValueError("add feedback before revising the keyframe prompt")
    if providers.llm is None:
        raise ValueError("this project profile does not include an LLM provider")

    _validate_keyframe_present(project, shot)
    prompt_rel = _ensure_keyframe_prompt(project, shot)
    instruction = _keyframe_feedback_instruction(shot, feedback)
    shot_uploads = [
        str(project.dir / rel)
        for rel in reference_upload_paths_for_shot(project, shot)
    ]
    image_paths = regeneration_reference_images(
        project, comment=feedback, subject_refs=shot_uploads
    )
    revision = apply_artifact_revision(
        project,
        prompt_rel,
        instruction,
        providers,
        invalidate=False,
        image_paths=image_paths,
    )
    prompt_text = _safe_project_path(project, prompt_rel).read_text().strip()
    # If the feedback names an uploaded reference ("make it look like @image1"), attach that
    # image so the re-render is actually conditioned on it — editing the prompt text alone
    # leaves the image model with no image to match.
    named_refs = reference_paths_in_text(
        project, feedback, target_types={"character", "location"}
    )
    _sync_shot_keyframe_prompt(project, shot_id, prompt_text, add_reference_images=named_refs)
    if revision.applied:
        queue_keyframe_regeneration(project, shot_id)
    return KeyframeFeedbackResult(
        shot_id=shot_id,
        prompt_path=prompt_rel,
        feedback=feedback,
        applied=revision.applied,
        removed=revision.archived,
        invalidated=revision.invalidated,
        provider=revision.provider,
        model=revision.model,
        error=revision.error,
    )


def revise_video_from_feedback(
    project: Project,
    shot_id: str,
    feedback: str,
    providers: Providers,
) -> KeyframeFeedbackResult:
    """Revise one shot's video/motion prompt from a comment — without re-rendering.

    Mirrors ``revise_keyframe_from_feedback`` for the motion prompt
    ``storyboard/prompts/<shot>.video.md``, but with two deliberate differences:
    the LLM edit is applied with ``invalidate=False`` so the **paid** clip is NOT
    re-rendered (invariant #7 — the user spends only when they click Regenerate),
    and the video guardrails (motion beats, camera move, continuity/last-frame, and
    exact dialogue) are preserved so the comment changes only what was asked.
    """
    shot = _find_shot(project, shot_id)
    if shot is None:
        raise FileNotFoundError(f"shot not found: {shot_id}")
    feedback = feedback.strip()
    if not feedback:
        raise ValueError("add feedback before revising the video prompt")
    if providers.llm is None:
        raise ValueError("this project profile does not include an LLM provider")

    prompt_rel = _ensure_video_prompt(project, shot)
    instruction = _video_feedback_instruction(shot, feedback)
    # Resolve aliases before spending the revision call. A manifest-bound upload that
    # has been deleted is an actionable project error, not permission to revise blindly.
    named_refs = reference_paths_in_text(
        project, feedback, target_types={"character", "location"}
    )
    revision = apply_artifact_revision(
        project, prompt_rel, instruction, providers, invalidate=False
    )
    if revision.applied:
        invalidate_prompt_approvals(project, "videos", shot_ids=[shot_id])
    if revision.applied and named_refs:
        _add_shot_reference_images(project, shot_id, named_refs)
    return KeyframeFeedbackResult(
        shot_id=shot_id,
        prompt_path=prompt_rel,
        feedback=feedback,
        applied=revision.applied,
        removed=revision.archived,
        invalidated=revision.invalidated,
        provider=revision.provider,
        model=revision.model,
        error=revision.error,
    )


def regenerate_shot_video(project: Project, shot_id: str) -> RegenerationResult:
    """Queue a video candidate while keeping the accepted clip live."""
    shot = _find_shot(project, shot_id)
    if shot is None:
        raise FileNotFoundError(f"shot not found: {shot_id}")

    clip = project.path("assets", "clips", f"{shot_id}.mp4")
    if not clip.is_file():
        raise FileNotFoundError(clip.relative_to(project.dir).as_posix())
    result = _queue_regeneration(project, shot_id, "video")
    result.kind = "shot_video"
    return result


def _rotate_character_seed(project: Project, rel: str) -> None:
    """Give an explicitly regenerated character sheet a fresh seed so the resume renders a
    genuinely new image that *replaces* the old one.

    The first build locks ``seed = seed_for(name)`` for consistency (invariant #4). But that
    makes regeneration deterministic — same seed + prompt + reference yields the same sheet,
    so a user's regenerate "doesn't take". On each explicit regeneration we bump a counter and
    derive a new seed from it; normal (non-regen) runs stay deterministic. Only the top-level
    character sheet (``bible/characters/<slug>/reference.png``) rotates — state sheets and
    locations are left untouched.
    """
    from .stages.bible import seed_for

    parts = rel.split("/")
    if len(parts) != 4 or parts[1] != "characters" or parts[3] != "reference.png":
        return
    slug = parts[2]
    cpath = _safe_project_path(project, f"bible/characters/{slug}/character.json")
    if not cpath.is_file():
        return
    data = json.loads(cpath.read_text())
    count = int(data.get("regen_count") or 0) + 1
    name = str(data.get("name") or slug)
    data["regen_count"] = count
    data["seed"] = seed_for(f"{name}#regen{count}")
    cpath.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _validate_bible_asset_rel(rel_path: str) -> str:
    rel = str(rel_path or "").replace("\\", "/").strip("/")
    parts = rel.split("/")
    if (
        len(parts) == 6
        and parts[0:2] == ["bible", "characters"]
        and parts[3] == "states"
        and parts[4]
        and parts[5] == "reference.png"
    ):
        return rel
    if len(parts) != 4 or parts[0] != "bible" or parts[1] not in {"characters", "locations"}:
        raise ValueError("bible asset must be under bible/characters or bible/locations")
    allowed = {
        "characters": {"reference.png"},
        "locations": {"reference.png"},
    }
    if parts[3] not in allowed[parts[1]]:
        raise ValueError("unsupported bible generated asset")
    return rel


def _safe_project_path(project: Project, rel_path: str) -> Path:
    rel = Path(rel_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("asset path must stay inside the project")
    path = (project.dir / rel).resolve()
    root = project.dir.resolve()
    if root != path and root not in path.parents:
        raise ValueError("asset path must stay inside the project")
    return path


def _find_shot(project: Project, shot_id: str) -> dict | None:
    shots_path = project.path("storyboard", "shots.json")
    if not shots_path.is_file():
        return None
    data = json.loads(shots_path.read_text())
    for shot in data.get("shots", []):
        if isinstance(shot, dict) and shot.get("id") == shot_id:
            return shot
    return None


def _validate_keyframe_present(project: Project, shot: dict) -> None:
    keyframe = str(shot.get("keyframe") or "")
    if not keyframe:
        raise FileNotFoundError(f"shot has no keyframe: {shot.get('id')}")
    keyframe_path = _safe_project_path(project, f"storyboard/keyframes/{keyframe}")
    if not keyframe_path.is_file():
        raise FileNotFoundError(keyframe_path.relative_to(project.dir).as_posix())


def _ensure_keyframe_prompt(project: Project, shot: dict) -> str:
    shot_id = str(shot.get("id") or "")
    rel = f"storyboard/prompts/{shot_id}.keyframe.md"
    path = _safe_project_path(project, rel)
    if path.is_file():
        return rel
    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = str(shot.get("keyframe_prompt") or "").strip()
    if not prompt:
        prompt = (
            f"{shot.get('camera', '')}. {shot.get('action', '')}. "
            f"{shot.get('description', '')}"
        ).strip()
    path.write_text(prompt)
    return rel


def _ensure_video_prompt(project: Project, shot: dict) -> str:
    """Return the rel path of the shot's motion prompt, requiring it to exist.

    Unlike the keyframe helper, this does NOT backfill: a ``<id>.video.md`` only exists
    once the video stage has run, and revising motion before there is a clip to react to
    is meaningless. Missing → ``FileNotFoundError`` so the gate surfaces a clear message.
    """
    shot_id = str(shot.get("id") or "")
    rel = f"storyboard/prompts/{shot_id}.video.md"
    if not _safe_project_path(project, rel).is_file():
        raise FileNotFoundError(rel)
    return rel


def _video_feedback_instruction(shot: dict, feedback: str) -> str:
    shot_id = str(shot.get("id") or "")
    action = str(shot.get("action") or "").strip()
    camera = str(shot.get("camera") or "").strip()
    camera_movement = str(shot.get("camera_movement") or "").strip()
    context = "; ".join(
        part for part in [
            f"camera: {camera}" if camera else "",
            f"camera movement: {camera_movement}" if camera_movement else "",
            f"action: {action}" if action else "",
        ]
        if part
    )
    context = f" Shot context: {context}." if context else ""
    one_line_feedback = " ".join(feedback.split())
    return (
        f"Revise this video/motion generation prompt for shot {shot_id}.{context} "
        "Keep the motion beats, camera move, shot-to-shot continuity / last-frame anchor, "
        "and any spoken dialogue exactly as written; change only what this user feedback "
        f"asks for, in the prompt's existing language: {one_line_feedback}"
    )


def _keyframe_feedback_instruction(shot: dict, feedback: str) -> str:
    shot_id = str(shot.get("id") or "")
    action = str(shot.get("action") or "").strip()
    camera = str(shot.get("camera") or "").strip()
    description = str(shot.get("description") or "").strip()
    context = "; ".join(
        part for part in [
            f"camera: {camera}" if camera else "",
            f"action: {action}" if action else "",
            f"description: {description}" if description else "",
        ]
        if part
    )
    context = f" Shot context: {context}." if context else ""
    one_line_feedback = " ".join(feedback.split())
    return (
        f"Revise this keyframe image-generation prompt for shot {shot_id}.{context} "
        "Keep the approved character, location, style, and story continuity anchors, "
        "but make the next generated image directly address this user feedback: "
        f"{one_line_feedback}"
    )


def _sync_shot_keyframe_prompt(
    project: Project,
    shot_id: str,
    prompt: str,
    *,
    add_reference_images: list[str] | None = None,
) -> None:
    shots_path = project.path("storyboard", "shots.json")
    data = json.loads(shots_path.read_text())
    found = False
    for shot in data.get("shots", []):
        if isinstance(shot, dict) and shot.get("id") == shot_id:
            shot["keyframe_prompt"] = prompt
            if add_reference_images:
                existing = list(shot.get("reference_images") or [])
                named = list(shot.get("named_reference_images") or [])
                for ref in add_reference_images:
                    if ref and ref not in existing:
                        existing.append(ref)
                    if ref and ref not in named:
                        named.append(ref)
                shot["reference_images"] = existing
                shot["named_reference_images"] = named
            found = True
            break
    if not found:
        raise FileNotFoundError(f"shot not found: {shot_id}")
    shots_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _add_shot_reference_images(
    project: Project,
    shot_id: str,
    refs: list[str],
) -> None:
    shots_path = project.path("storyboard", "shots.json")
    data = json.loads(shots_path.read_text())
    for shot in data.get("shots", []):
        if isinstance(shot, dict) and shot.get("id") == shot_id:
            shot["reference_images"] = list(dict.fromkeys(
                list(shot.get("reference_images") or []) + refs
            ))
            shot["named_reference_images"] = list(dict.fromkeys(
                list(shot.get("named_reference_images") or []) + refs
            ))
            shots_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            )
            return
    raise FileNotFoundError(f"shot not found: {shot_id}")
