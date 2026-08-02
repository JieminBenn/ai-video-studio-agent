"""File-backed approval manifests for provider-bound media prompts."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

from .prompt_validation import validate_grid_prompt


_KINDS = {
    "keyframes": ("keyframes.json", ".keyframe.md"),
    "videos": ("videos.json", ".video.md"),
}
_REFERENCE_FIELDS = (
    "keyframe_reference_images",
    "reference_images",
    "target_state_reference_images",
    "named_reference_images",
    "reference_style_images",
)


class PromptApprovalError(ValueError):
    """A prompt batch is absent, invalid, or no longer matches its approved bytes."""


def prompt_path(project, kind: str, shot: dict) -> Path:
    _kind_config(kind)
    shot_id = str(shot.get("id") or "").strip()
    if not shot_id:
        raise PromptApprovalError("shot is missing an id")
    if kind == "keyframes" and isinstance(shot.get("motion_grid"), dict):
        return project.path("storyboard", "prompts", f"{shot_id}.grid.md")
    suffix = _KINDS[kind][1]
    return project.path("storyboard", "prompts", f"{shot_id}{suffix}")


def confirm_prompt_batch(project, kind: str, *, confirmer: str) -> dict:
    shots = _shots(project)
    if not shots:
        raise PromptApprovalError("no live shots are available for prompt approval")
    prompts = {
        str(shot["id"]): _prompt_record(project, kind, shot)
        for shot in shots
    }
    manifest = _manifest(
        project,
        kind,
        confirmer=confirmer,
        prompts=prompts,
    )
    _write_manifest(project, kind, manifest)
    return manifest


def confirm_prompt(project, kind: str, shot_id: str, *, confirmer: str) -> dict:
    shots = _shots(project)
    shot = next(
        (item for item in shots if str(item.get("id") or "") == str(shot_id)),
        None,
    )
    if shot is None:
        raise PromptApprovalError(f"shot not found: {shot_id}")

    context_hash = _generation_context_sha256(project, kind, shots)
    existing = _read_manifest(project, kind)
    prompts = dict(existing.get("prompts") or {})
    if existing.get("generation_context_sha256") != context_hash:
        prompts = {}
    prompts[str(shot_id)] = _prompt_record(project, kind, shot)
    manifest = _manifest(
        project,
        kind,
        confirmer=confirmer,
        prompts=prompts,
        context_hash=context_hash,
    )
    _write_manifest(project, kind, manifest)
    return manifest


def require_prompt_batch_approval(project, kind: str) -> dict:
    shots = _shots(project)
    manifest = _read_manifest(project, kind)
    if not manifest:
        raise PromptApprovalError(f"{kind} prompt batch has not been confirmed")

    expected_context = _generation_context_sha256(project, kind, shots)
    if manifest.get("generation_context_sha256") != expected_context:
        raise PromptApprovalError(f"{kind} generation context changed after confirmation")

    records = manifest.get("prompts") or {}
    for shot in shots:
        shot_id = str(shot.get("id") or "")
        expected = _prompt_record(project, kind, shot)
        actual = records.get(shot_id)
        if actual != expected:
            raise PromptApprovalError(
                f"{kind} prompt for {shot_id} changed or is not confirmed"
            )
    return manifest


def invalidate_prompt_approvals(
    project,
    kind: str,
    *,
    shot_ids: Iterable[str] | None = None,
) -> None:
    path = _manifest_path(project, kind)
    if not path.is_file():
        return
    if shot_ids is None:
        path.unlink()
        return
    blocked = {str(value) for value in shot_ids}
    manifest = _read_manifest(project, kind)
    prompts = {
        shot_id: record
        for shot_id, record in (manifest.get("prompts") or {}).items()
        if shot_id not in blocked
    }
    if not prompts:
        path.unlink()
        return
    manifest["prompts"] = prompts
    _write_manifest(project, kind, manifest)


def rebase_prompt_approvals(
    project,
    kind: str,
    *,
    invalidated_shot_ids: Iterable[str] = (),
) -> dict:
    """Carry unchanged per-shot approvals onto the current live-shot context.

    Structural edits such as deleting one clip change the batch context hash.  This
    preserves approvals whose exact prompt records are still byte-for-byte identical,
    while explicitly dropping the deleted/continuity-affected shots.
    """
    existing = _read_manifest(project, kind)
    if not existing:
        return {}
    blocked = {str(value) for value in invalidated_shot_ids}
    old_records = dict(existing.get("prompts") or {})
    prompts: dict[str, dict[str, str]] = {}
    for shot in _shots(project):
        shot_id = str(shot.get("id") or "")
        if not shot_id or shot_id in blocked or shot_id not in old_records:
            continue
        try:
            current = _prompt_record(project, kind, shot)
        except PromptApprovalError:
            continue
        if current == old_records[shot_id]:
            prompts[shot_id] = current
    if not prompts:
        _manifest_path(project, kind).unlink(missing_ok=True)
        return {}
    manifest = _manifest(
        project,
        kind,
        confirmer=str(existing.get("confirmer") or "human"),
        prompts=prompts,
    )
    _write_manifest(project, kind, manifest)
    return manifest


def _prompt_record(project, kind: str, shot: dict) -> dict[str, str]:
    path = prompt_path(project, kind, shot)
    if not path.is_file():
        raise PromptApprovalError(
            f"missing {kind} prompt for {shot.get('id')}: "
            f"{path.relative_to(project.dir).as_posix()}"
        )
    text = path.read_text()
    if not text.strip():
        raise PromptApprovalError(f"empty {kind} prompt for {shot.get('id')}")
    grid = shot.get("motion_grid")
    if kind == "keyframes" and isinstance(grid, dict):
        # Structural gate only: a motion grid must enumerate every panel it is sliced into.
        # Plain keyframe prompts are trusted as written (no prose motion-word matching).
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
    return {
        "path": path.relative_to(project.dir).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _manifest(
    project,
    kind: str,
    *,
    confirmer: str,
    prompts: dict,
    context_hash: str | None = None,
) -> dict:
    return {
        "version": 1,
        "kind": kind,
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "confirmer": str(confirmer),
        "generation_context_sha256": context_hash
        or _generation_context_sha256(project, kind, _shots(project)),
        "prompts": prompts,
    }


def _generation_context_sha256(project, kind: str, shots: list[dict]) -> str:
    prefix = "image" if kind == "keyframes" else "video"
    model_config = {
        key: value
        for key, value in (project.model_config or {}).items()
        if key == prefix
        or key.startswith(prefix + "_")
        or (kind == "keyframes" and key == "motion_grid")
        or (kind == "videos" and key == "audio_mode")
    }
    shot_context = []
    for shot in shots:
        refs = {
            field: list(shot.get(field) or [])
            for field in _REFERENCE_FIELDS
            if shot.get(field)
        }
        shot_context.append({
            "id": str(shot.get("id") or ""),
            "motion_grid": shot.get("motion_grid") if kind == "keyframes" else None,
            "video_prompt_binding": (
                shot.get("video_prompt_binding") if kind == "videos" else None
            ),
            "references": refs,
        })
    payload = json.dumps(
        {"kind": kind, "model_config": model_config, "shots": shot_context},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shots(project) -> list[dict]:
    path = project.path("storyboard", "shots.json")
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return [item for item in data.get("shots", []) if isinstance(item, dict)]


def _kind_config(kind: str) -> tuple[str, str]:
    try:
        return _KINDS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported prompt approval kind: {kind}") from exc


def _manifest_path(project, kind: str) -> Path:
    filename = _kind_config(kind)[0]
    return project.path("storyboard", "prompt_approvals", filename)


def _read_manifest(project, kind: str) -> dict:
    path = _manifest_path(project, kind)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _write_manifest(project, kind: str, manifest: dict) -> None:
    path = _manifest_path(project, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
