"""Archive-backed dependency invalidation for file-cached pipeline artifacts."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .orchestrator.project import Project
from .prompt_approvals import invalidate_prompt_approvals


@dataclass
class ArchiveResult:
    archive_dir: str
    archived_paths: list[str]


@dataclass
class InvalidationResult:
    archived: int
    archive_dir: str
    archived_paths: list[str]
    affected_shot_ids: list[str]
    invalidated_stages: list[str]
    current_stage: str


def archive_paths(
    project: Project,
    rel_paths: Iterable[str],
    *,
    reason: str,
    source_path: str = "",
    affected_shot_ids: Iterable[str] | None = None,
) -> ArchiveResult:
    """Move existing project files into one reversible history snapshot."""
    existing: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in rel_paths:
        rel = _safe_rel(value)
        if rel in seen:
            continue
        seen.add(rel)
        path = _project_path(project, rel)
        if path.is_file():
            existing.append((rel, path))

    if not existing:
        return ArchiveResult("", [])

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_root = project.path(
        "history", f"{stamp}-{_reason_slug(reason)}-{uuid.uuid4().hex[:6]}"
    )
    moved: list[tuple[Path, Path]] = []
    try:
        for rel, source in existing:
            destination = archive_root.joinpath(*rel.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved.append((source, destination))
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
            "source_path": source_path,
            "affected_shot_ids": list(affected_shot_ids or []),
            "archived_paths": [rel for rel, _path in existing],
        }
        archive_root.joinpath("manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
    except Exception:
        for source, destination in reversed(moved):
            if destination.is_file() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(source)
        raise

    return ArchiveResult(
        archive_dir=archive_root.relative_to(project.dir).as_posix(),
        archived_paths=[rel for rel, _path in existing],
    )


def invalidate_shots(
    project: Project,
    shot_ids: Iterable[str],
    *,
    from_stage: str,
    reason: str,
    source_path: str = "",
    include_keyframes: bool = False,
    include_prompts: bool = False,
    extra_paths: Iterable[str] | None = None,
    preserve_paths: Iterable[str] | None = None,
) -> InvalidationResult:
    ordered_ids = _ordered_shot_ids(project, shot_ids)
    _invalidate_prompt_approvals_from_stage(project, from_stage, ordered_ids)
    paths = list(extra_paths or [])
    paths.extend(
        _shot_paths(
            project,
            ordered_ids,
            from_stage=from_stage,
            include_keyframes=include_keyframes,
            include_prompts=include_prompts,
        )
    )
    preserved = {_safe_rel(value) for value in (preserve_paths or [])}
    if preserved:
        paths = [_safe_rel(value) for value in paths if _safe_rel(value) not in preserved]
    return _invalidate(
        project,
        paths,
        from_stage=from_stage,
        reason=reason,
        source_path=source_path,
        shot_ids=ordered_ids,
    )


def invalidate_artifact_revision(project: Project, rel_path: str) -> InvalidationResult:
    rel = _safe_rel(rel_path)
    if rel == "storyboard/prompts/music.audio.md":
        if _native_audio(project):
            return _unchanged_revision(project)
        return invalidate_shots(
            project,
            [],
            from_stage="audio",
            reason="music-prompt-revised",
            source_path=rel,
            extra_paths=["assets/audio/music.wav"],
        )
    prompt_match = re.fullmatch(
        r"storyboard/prompts/(.+)\.(keyframe|video|audio|grid)\.md", rel
    )
    if prompt_match:
        shot_id, prompt_kind = prompt_match.groups()
        if shot_id not in set(_all_shot_ids(project)):
            return _unchanged_revision(project)
        if prompt_kind == "audio" and _native_audio(project):
            return _unchanged_revision(project)
        if prompt_kind == "video":
            invalidate_prompt_approvals(
                project, "videos", shot_ids=[shot_id]
            )
            return _invalidate(
                project,
                [],
                from_stage="video_prompts",
                reason="video-prompt-revised",
                source_path=rel,
                shot_ids=[shot_id],
            )
        stage = {
            "keyframe": "keyframes",
            "audio": "audio",
            "grid": "keyframes",
        }[prompt_kind]
        return invalidate_shots(
            project,
            [shot_id],
            from_stage=stage,
            reason=f"{prompt_kind}-prompt-revised",
            source_path=rel,
            include_keyframes=prompt_kind in {"keyframe", "grid"},
        )

    if rel == "storyboard/shots.json":
        return invalidate_shots(
            project,
            _all_shot_ids(project),
            from_stage="storyboard",
            reason="storyboard-revised",
            source_path=rel,
            include_keyframes=True,
            include_prompts=True,
        )

    if rel.startswith("bible/characters/") or rel.startswith("bible/locations/"):
        return _invalidate_bible_source(project, rel)

    if rel == "bible/style.md":
        return invalidate_shots(
            project,
            _all_shot_ids(project),
            from_stage="storyboard",
            reason="bible-style-revised",
            source_path=rel,
            include_keyframes=True,
            include_prompts=True,
        )

    if rel == "story/script.json":
        extra = _tree_file_paths(project, "storyboard")
        return invalidate_shots(
            project,
            _all_shot_ids(project),
            from_stage="storyboard",
            reason="script-revised",
            source_path=rel,
            include_keyframes=True,
            include_prompts=True,
            extra_paths=extra,
        )

    if rel == "story/plot.json":
        extra = ["story/script.json"]
        extra.extend(_tree_file_paths(project, "bible"))
        extra.extend(_tree_file_paths(project, "storyboard"))
        return invalidate_shots(
            project,
            _all_shot_ids(project),
            from_stage="script",
            reason="plot-revised",
            source_path=rel,
            include_keyframes=True,
            include_prompts=True,
            extra_paths=extra,
        )

    return _invalidate(
        project,
        [],
        from_stage=project.current_stage or project.stages[0],
        reason="artifact-revised",
        source_path=rel,
        shot_ids=[],
    )


def _unchanged_revision(project: Project) -> InvalidationResult:
    """Ignore obsolete legacy-audio targets in native-audio projects."""
    return InvalidationResult(
        archived=0,
        archive_dir="",
        archived_paths=[],
        affected_shot_ids=[],
        invalidated_stages=[],
        current_stage=project.current_stage or "",
    )


def invalidate_bible_text(project: Project, kind: str, slug: str) -> InvalidationResult:
    """Archive a character's derived text and reset the bible stage only.

    Unlike :func:`invalidate_bible_asset`, this never touches ``reference.png`` or
    any downstream shot artifact — text edits stay decoupled from image regen.
    """
    kind = str(kind or "").strip().lower()
    slug = str(slug or "").strip()
    archived = 0
    archive_dir = ""
    archived_paths: list[str] = []
    if kind == "character":
        board_rel = f"bible/characters/{slug}/identity_board.json"
        if (project.dir / board_rel).is_file():
            archive = archive_paths(
                project,
                [board_rel],
                reason="bible-text-revised",
                source_path=board_rel,
                affected_shot_ids=[],
            )
            archived = len(archive.archived_paths)
            archive_dir = archive.archive_dir
            archived_paths = list(archive.archived_paths)
    project.set_stage_status("bible", "pending")
    project.current_stage = "bible"
    project.status = "in_progress"
    project.save()
    return InvalidationResult(
        archived=archived,
        archive_dir=archive_dir,
        archived_paths=archived_paths,
        affected_shot_ids=[],
        invalidated_stages=["bible"],
        current_stage="bible",
    )


def invalidate_bible_asset(project: Project, rel_path: str) -> InvalidationResult:
    rel = _safe_rel(rel_path)
    parts = rel.split("/")
    if (
        len(parts) == 6
        and parts[0:2] == ["bible", "characters"]
        and parts[3] == "states"
        and parts[4]
        and parts[5] == "reference.png"
    ):
        slug = parts[2]
        shot_ids = _shots_for_target(project, "character", slug)
        return invalidate_shots(
            project,
            shot_ids,
            from_stage="bible",
            reason="bible-state-asset-regenerated",
            source_path=rel,
            include_keyframes=True,
            include_prompts=True,
            extra_paths=[rel, *_provider_prompt_sidecar_paths(rel)],
        )
    if len(parts) != 4 or parts[0] != "bible" or parts[1] not in {
        "characters",
        "locations",
    }:
        raise ValueError("bible asset must be under bible/characters or bible/locations")
    kind, slug, filename = parts[1], parts[2], parts[3]
    allowed = {
        "characters": {"reference.png"},
        "locations": {"reference.png"},
    }
    if filename not in allowed[kind]:
        raise ValueError("unsupported bible generated asset")

    bible_paths = [rel, *_provider_prompt_sidecar_paths(rel)]
    # When this subject's identity came from an uploaded reference, also archive the identity
    # board so it re-derives from the image (the new sheet + board agree instead of the sheet
    # fighting a stale, image-blind board). Deferred import avoids a cycle with reference_assets.
    if kind == "characters":
        from .reference_assets import reference_records_for_target

        bible_paths.extend(_character_state_anchor_paths(project, slug))
        if reference_records_for_target(project, "character", slug):
            board_rel = f"bible/characters/{slug}/identity_board.json"
            if (project.dir / board_rel).is_file():
                bible_paths.append(board_rel)
    shot_ids = _shots_for_target(project, kind[:-1], slug)
    return invalidate_shots(
        project,
        shot_ids,
        from_stage="bible",
        reason="bible-asset-regenerated",
        source_path=rel,
        include_keyframes=True,
        include_prompts=True,
        extra_paths=bible_paths,
    )


def invalidate_reference_change(
    project: Project,
    *,
    target_type: str = "",
    target_id: str = "",
) -> InvalidationResult:
    target_type = str(target_type or "").strip().lower()
    target_id = str(target_id or "").strip()
    all_ids = _all_shot_ids(project)
    if target_type == "style":
        shot_ids = all_ids
    elif target_type in {"character", "location"} and target_id:
        shot_ids = _shots_for_target(project, target_type, target_id)
    else:
        shot_ids = all_ids

    bible_slug = ""
    if target_type in {"character", "location"} and target_id:
        explicit_matches = _matching_shot_ids(project, target_type, target_id)
        bible_slug = _bible_target_slug(
            project, target_type, target_id
        )
        matched = bool(explicit_matches) or bool(bible_slug)
        shot_ids = explicit_matches or all_ids
    else:
        matched = False

    extra: list[str] = []
    # Unresolved/automatic uploads must revisit Bible so its resolver can bind them to
    # the generated cast and location roster. A named-but-unknown target retains the
    # conservative storyboard-wide fallback used by legacy callers.
    from_stage = "bible" if not target_type and "bible" in project.stages else "storyboard"
    if matched:
        slug = bible_slug or _slug(target_id)
        if target_type == "character":
            extra.extend([
                f"bible/characters/{slug}/identity_board.json",
                f"bible/characters/{slug}/reference.png",
            ])
            extra.extend(_character_state_anchor_paths(project, slug))
        else:
            extra.append(f"bible/locations/{slug}/reference.png")
        from_stage = "bible"

    return invalidate_shots(
        project,
        shot_ids,
        from_stage=from_stage,
        reason="reference-changed",
        source_path="references/references.json",
        include_keyframes=True,
        include_prompts=True,
        extra_paths=extra,
    )


def refresh_knowledge_target(
    project: Project,
    *,
    purpose: str,
    target: str,
) -> InvalidationResult:
    """Archive one frozen packet and only the artifacts that consumed that target."""
    purpose = str(purpose or "").strip()
    target = str(target or "").strip()
    if purpose not in {"identity", "scene", "shot"}:
        raise ValueError(f"unsupported knowledge packet purpose: {purpose}")
    if not target or not re.fullmatch(r"[A-Za-z0-9_.-]+", target):
        raise ValueError("invalid knowledge packet target")
    packet = f"knowledge/packets/{purpose}-{target}.json"

    if purpose == "shot":
        return invalidate_shots(
            project,
            [target],
            from_stage="storyboard",
            reason="knowledge-packet-refreshed",
            source_path=packet,
            include_keyframes=True,
            include_prompts=True,
            extra_paths=[packet],
        )

    if purpose == "scene":
        shot_ids = [
            str(shot.get("id"))
            for shot in _shots(project)
            if shot.get("id") and str(shot.get("scene")) == target
        ]
        return invalidate_shots(
            project,
            shot_ids,
            from_stage="storyboard",
            reason="knowledge-packet-refreshed",
            source_path=packet,
            include_keyframes=True,
            include_prompts=True,
            extra_paths=[packet],
        )

    shot_ids = _shots_for_target(project, "character", target)
    base = f"bible/characters/{target}"
    reference_rel = f"{base}/reference.png"
    return invalidate_shots(
        project,
        shot_ids,
        from_stage="bible",
        reason="knowledge-packet-refreshed",
        source_path=packet,
        include_keyframes=True,
        include_prompts=True,
        extra_paths=[
            packet,
            f"{base}/identity_board.json",
            f"{base}/identity.rationale.md",
            f"{base}/reference.prompt.md",
            f"{base}/reference.knowledge.json",
            reference_rel,
            *_provider_prompt_sidecar_paths(reference_rel),
        ],
    )


def _invalidate_bible_source(project: Project, rel: str) -> InvalidationResult:
    parts = rel.split("/")
    kind, slug = parts[1], parts[2]
    if kind == "characters":
        reference_rel = f"bible/characters/{slug}/reference.png"
    else:
        reference_rel = f"bible/locations/{slug}/reference.png"
    extra = [reference_rel, *_provider_prompt_sidecar_paths(reference_rel)]
    return invalidate_shots(
        project,
        _shots_for_target(project, kind[:-1], slug),
        from_stage="bible",
        reason="bible-source-revised",
        source_path=rel,
        include_keyframes=True,
        include_prompts=True,
        extra_paths=extra,
    )


def _invalidate(
    project: Project,
    rel_paths: Iterable[str],
    *,
    from_stage: str,
    reason: str,
    source_path: str,
    shot_ids: list[str],
) -> InvalidationResult:
    archive = archive_paths(
        project,
        rel_paths,
        reason=reason,
        source_path=source_path,
        affected_shot_ids=shot_ids,
    )
    invalidated = _stages_from(project, from_stage)
    for stage in invalidated:
        project.set_stage_status(stage, "pending")
    if invalidated:
        project.current_stage = invalidated[0]
    project.status = "in_progress"
    project.save()
    return InvalidationResult(
        archived=len(archive.archived_paths),
        archive_dir=archive.archive_dir,
        archived_paths=archive.archived_paths,
        affected_shot_ids=shot_ids,
        invalidated_stages=invalidated,
        current_stage=project.current_stage or "",
    )


def _invalidate_prompt_approvals_from_stage(
    project: Project,
    from_stage: str,
    shot_ids: list[str],
) -> None:
    """Remove only affected hashes when their generation inputs became stale."""
    stage = _resolve_stage(project, from_stage)
    if stage not in project.stages:
        return
    start = project.stages.index(stage)
    if "keyframes" in project.stages and start <= project.stages.index("keyframes"):
        invalidate_prompt_approvals(project, "keyframes", shot_ids=shot_ids)
        invalidate_prompt_approvals(project, "videos", shot_ids=shot_ids)
    elif (
        "video_prompts" in project.stages
        and start <= project.stages.index("video_prompts")
    ):
        invalidate_prompt_approvals(project, "videos", shot_ids=shot_ids)


def _shot_paths(
    project: Project,
    shot_ids: list[str],
    *,
    from_stage: str,
    include_keyframes: bool,
    include_prompts: bool,
) -> list[str]:
    paths: list[str] = []
    stages = project.stages
    native_audio = _native_audio(project)
    from_stage = _resolve_stage(project, from_stage)
    start = stages.index(from_stage) if from_stage in stages else 0

    for shot_id in shot_ids:
        if include_keyframes:
            keyframe = _keyframe_for_shot(project, shot_id)
            if keyframe:
                keyframe_rel = f"storyboard/keyframes/{keyframe}"
                paths.append(keyframe_rel)
                paths.extend(_provider_prompt_sidecar_paths(keyframe_rel))
        if include_prompts:
            paths.extend([
                f"storyboard/prompts/{shot_id}.knowledge.json",
                f"storyboard/prompts/{shot_id}.brief.md",
                f"storyboard/prompts/{shot_id}.keyframe.md",
                f"storyboard/prompts/{shot_id}.video.brief.md",
                f"storyboard/prompts/{shot_id}.video.md",
                f"storyboard/prompts/{shot_id}.grid.brief.md",
                f"storyboard/prompts/{shot_id}.grid.md",
            ])
            if not native_audio:
                paths.append(f"storyboard/prompts/{shot_id}.audio.md")
        if _stage_at_or_after(stages, "video", start):
            paths.extend(
                [
                    f"assets/clips/{shot_id}.mp4",
                    f"assets/clips/{shot_id}.last_frame.png",
                ]
            )
            if native_audio:
                paths.append(f"assets/clips/{shot_id}.native.source.wav")
                declared_source = _native_audio_source_for_shot(project, shot_id)
                if declared_source:
                    paths.append(f"assets/clips/{declared_source}")
        if _stage_at_or_after(stages, "review", start):
            paths.append(f"assets/qc/{shot_id}.json")
        if _stage_at_or_after(stages, "audio", start):
            suffix = "native.wav" if native_audio else "dialogue.wav"
            paths.append(f"assets/audio/{shot_id}.{suffix}")
            # 旁白 VO is a separate stem in both modes; re-synthesize it on regeneration.
            paths.append(f"assets/audio/{shot_id}.narration.wav")

    if _stage_at_or_after(stages, "video", start):
        paths.append("assets/clips/clips.json")
    if _stage_at_or_after(stages, "review", start):
        paths.append("assets/qc/summary.json")
    if _stage_at_or_after(stages, "audio", start):
        paths.append("assets/audio/audio.json")
    if _stage_at_or_after(stages, "assemble", start):
        paths.extend(["edit/timeline.json", f"output/{project.project_id}.mp4"])
    return paths


def _native_audio(project: Project) -> bool:
    return project.model_config.get("audio_mode") == "native_video"


def _native_audio_source_for_shot(project: Project, shot_id: str) -> str:
    manifest = project.path("assets", "clips", "clips.json")
    if not manifest.is_file():
        return ""
    try:
        clips = json.loads(manifest.read_text()).get("clips", [])
    except (json.JSONDecodeError, AttributeError):
        return ""
    for entry in clips:
        if not isinstance(entry, dict) or str(entry.get("id") or "") != shot_id:
            continue
        source = str(entry.get("native_audio") or "")
        return source if source and Path(source).name == source else ""
    return ""


def _resolve_stage(project: Project, stage: str) -> str:
    """Map a conceptual story-mode stage name to the project's actual stage.
    Clip mode produces storyboard/* artifacts from the ``clip`` stage, not ``storyboard``."""
    if stage in project.stages:
        return stage
    if stage == "storyboard" and "clip" in project.stages:
        return "clip"
    return stage


def _stage_at_or_after(stages: list[str], stage: str, start: int) -> bool:
    return stage in stages and stages.index(stage) >= start


def _stages_from(project: Project, stage: str) -> list[str]:
    stage = _resolve_stage(project, stage)
    if stage not in project.stages:
        return []
    return project.stages[project.stages.index(stage) :]


def _all_shot_ids(project: Project) -> list[str]:
    return [str(shot.get("id")) for shot in _shots(project) if shot.get("id")]


def _ordered_shot_ids(project: Project, shot_ids: Iterable[str]) -> list[str]:
    wanted = {str(value) for value in shot_ids}
    ordered = [shot_id for shot_id in _all_shot_ids(project) if shot_id in wanted]
    ordered.extend(sorted(wanted - set(ordered)))
    return ordered


def _shots(project: Project) -> list[dict]:
    path = project.path("storyboard", "shots.json")
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return [shot for shot in data.get("shots", []) if isinstance(shot, dict)]


def _matching_shot_ids(project: Project, target_type: str, target_id: str) -> list[str]:
    # Join on the STABLE bible-folder slug, not just the ASCII name-slug. ``target_id`` may be
    # either a bible folder slug or a display name; the shot carries the plot name. The ASCII
    # ``_slug`` collapses every CJK name to "unknown" (invariant #10/#11) and the display name
    # can drift after a text revision, so name/path matches alone silently miss a character and
    # fall back to invalidating EVERY shot. ``char_slug``/``location_slug`` map both the target
    # and each shot name to the same immutable folder slug, targeting exactly the featuring shots.
    from .stages.bible import char_slug, location_slug

    stable_slug = char_slug if target_type == "character" else location_slug
    target_slug = _slug(target_id)
    target_folder = stable_slug(target_id)
    target_name = target_id.casefold()
    field = "reference_characters" if target_type == "character" else "reference_locations"
    matches: list[str] = []
    for shot in _shots(project):
        names = [str(value) for value in shot.get(field, [])]
        if target_type == "character":
            names.extend(str(value) for value in shot.get("characters", []))
        name_match = any(
            value.casefold() == target_name
            or _slug(value) == target_slug
            or stable_slug(value) == target_folder
            for value in names
        )
        normalized_paths = [
            str(value).replace("\\", "/") for value in shot.get("reference_images", [])
        ]
        path_match = any(
            f"/{target_slug}/" in f"/{value}" or f"/{target_folder}/" in f"/{value}"
            for value in normalized_paths
        )
        if (name_match or path_match) and shot.get("id"):
            matches.append(str(shot["id"]))
    return matches


def _shots_for_target(project: Project, target_type: str, target_id: str) -> list[str]:
    matches = _matching_shot_ids(project, target_type, target_id)
    return matches or _all_shot_ids(project)


def _bible_target_slug(project: Project, target_type: str, target_id: str) -> str:
    plural = "characters" if target_type == "character" else "locations"
    root = project.path("bible", plural)
    if not root.is_dir():
        return ""
    target_slug = _slug(target_id)
    target_name = target_id.casefold()
    json_name = "character.json" if target_type == "character" else "location.json"
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if _slug(child.name) == target_slug:
            return child.name
        data_path = child / json_name
        if not data_path.is_file():
            continue
        try:
            name = str(json.loads(data_path.read_text()).get("name") or "")
        except (json.JSONDecodeError, OSError):
            continue
        if name.casefold() == target_name or _slug(name) == target_slug:
            return child.name
    return ""


def _keyframe_for_shot(project: Project, shot_id: str) -> str:
    for shot in _shots(project):
        if str(shot.get("id")) == shot_id:
            return str(shot.get("keyframe") or "")
    return ""


def _tree_file_paths(project: Project, rel_root: str) -> list[str]:
    root = _project_path(project, rel_root)
    if not root.is_dir():
        return []
    return [path.relative_to(project.dir).as_posix() for path in root.rglob("*") if path.is_file()]


def _provider_prompt_sidecar_paths(image_rel: str) -> list[str]:
    image_path = Path(image_rel)
    stem = image_path.with_suffix("").as_posix()
    return [
        f"{stem}.provider-prompt.md",
        f"{stem}.provider-prompt.json",
    ]


def _character_state_anchor_paths(project: Project, slug: str) -> list[str]:
    """Return generated state images and their editable prompts for one character."""
    states_root = project.path("bible", "characters", slug, "states")
    if not states_root.is_dir():
        return []
    paths: list[str] = []
    for state_dir in sorted(states_root.iterdir()):
        if not state_dir.is_dir():
            continue
        for filename in (
            "reference.png",
            "reference.prompt.md",
            "reference.provider-prompt.md",
            "reference.provider-prompt.json",
        ):
            path = state_dir / filename
            if path.is_file():
                paths.append(path.relative_to(project.dir).as_posix())
    return paths


def _safe_rel(value: str) -> str:
    rel = Path(str(value or "").replace("\\", "/"))
    if not str(rel) or rel.is_absolute() or ".." in rel.parts:
        raise ValueError("artifact path must stay inside the project")
    return rel.as_posix().strip("/")


def _project_path(project: Project, rel: str) -> Path:
    path = (project.dir / rel).resolve()
    root = project.dir.resolve()
    if path != root and root not in path.parents:
        raise ValueError("artifact path must stay inside the project")
    return path


def _reason_slug(reason: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", reason.lower()).strip("-")[:48] or "change"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "unknown"
