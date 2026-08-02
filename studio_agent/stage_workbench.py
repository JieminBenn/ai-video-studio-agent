"""Focused Stage Workbench context for the local review UI.

The web app can render this plain data without knowing stage internals. Helpers stay
file-first: they inspect the current project tree, choose sensible editable artifacts,
and expose only scoped regeneration actions for the active gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .orchestrator.project import Project
from .prompt_approvals import prompt_path
from .prompt_conversation import conversation_records
from .prompt_validation import validate_grid_prompt


STAGE_LABELS = {
    "style": "Visual Style",
    "plot": "Plot",
    "script": "Script",
    "bible": "Show Bible",
    "storyboard": "Storyboard",
    "clip": "Clip",
    "keyframes": "Clip / Storyboard — Keyframes",
    "video_prompts": "Video — Prompt review",
    "video": "Video",
    "review": "QC Review",
    "audio": "Audio",
    "assemble": "Assembly",
    "done": "Done",
}


def resolve_stage(project: Project, requested: str | None = None) -> str:
    """Resolve the stage the workspace should show: requested if valid, else the gate, else last."""
    stages = list(project.stages)
    if requested and requested in stages:
        return requested
    if project.current_stage and project.current_stage in stages:
        return project.current_stage
    return stages[-1] if stages else "done"


def workbench_context(project: Project, stage: str | None = None) -> dict[str, Any]:
    """Return all data needed to render the workspace for the selected stage."""
    selected = resolve_stage(project, stage)
    target = revision_target_for_stage(project, selected)
    payload = stage_payload(project, selected)
    status = project.stage_status(selected) if selected in project.stages else project.status
    is_gate = bool(project.current_stage) and selected == project.current_stage
    return {
        "stage": selected,
        "selected": selected,
        "is_gate": is_gate,
        "stage_label": STAGE_LABELS.get(selected, selected.replace("_", " ").title()),
        "status": status,
        "summary": _stage_summary(selected, payload),
        "revision_target": target,
        "revision_targets": revision_targets_for_stage(project, selected),
        "asset_actions": stage_asset_actions(project, selected),
        "decision": _pending_decision(project, selected),
        "knowledge_packets": _knowledge_packets(project, selected),
        "payload": payload,
        "can_approve": is_gate and project.stage_status(selected) in {"complete", "failed"},
    }


def _pending_decision(project: Project, stage: str) -> dict[str, Any]:
    path = project.path("story", "decisions", f"{stage}.json")
    data = _read_json(path) or {}
    return data if data and not data.get("resolution") else {}


def _knowledge_packets(project: Project, stage: str) -> list[dict[str, Any]]:
    root = project.path("knowledge", "packets")
    if not root.is_dir():
        return []
    allowed = {
        "bible": {"identity"},
        "storyboard": {"scene", "shot"},
        "clip": {"scene", "shot"},
        "video": {"shot"},
        "review": {"shot"},
    }.get(stage, set())
    packets = []
    for path in sorted(root.glob("*.json")):
        data = _read_json(path) or {}
        purpose = str(data.get("purpose") or "")
        if purpose not in allowed:
            continue
        entries = []
        for entry in data.get("selected_entries") or []:
            if not isinstance(entry, dict):
                continue
            entries.append({
                "id": str(entry.get("id") or ""),
                "title": str(entry.get("title") or entry.get("id") or "Knowledge entry"),
                "rationale": ", ".join(str(item) for item in entry.get("reasons") or []),
                "source_path": str((entry.get("source") or {}).get("path") or ""),
            })
        packets.append({
            "purpose": purpose,
            "target": str(data.get("target") or ""),
            "path": path.relative_to(project.dir).as_posix(),
            "entries": entries,
        })
    return packets


def revision_target_for_stage(
    project: Project,
    stage: str | None,
    selected: str | None = None,
) -> dict[str, str] | None:
    """Choose the selected target when valid, else the stage's default target."""
    targets = revision_targets_for_stage(project, stage)
    if selected:
        for target in targets:
            if target["path"] == selected:
                return target
    return targets[0] if targets else None


def revision_targets_for_stage(project: Project, stage: str | None) -> list[dict[str, str]]:
    """Editable text artifacts that make sense for the focused stage."""
    if stage == "plot":
        return _existing_targets(project, [("story/plot.json", "Plot JSON")])
    if stage == "script":
        return _existing_targets(project, [
            ("story/script.json", "Script JSON"),
            ("story/script.review.md", "Structure Review (起承转合)"),
        ])
    if stage == "bible":
        candidates: list[tuple[str, str]] = []
        chars = project.path("bible", "characters")
        if chars.is_dir():
            for child in sorted(chars.iterdir()):
                candidates.extend([
                    (f"bible/characters/{child.name}/character.json", f"Character: {child.name}"),
                    (f"bible/characters/{child.name}/identity_board.json", f"Identity Board: {child.name}"),
                    (f"bible/characters/{child.name}/states.json", f"Visual States: {child.name}"),
                ])
        locs = project.path("bible", "locations")
        if locs.is_dir():
            for child in sorted(locs.iterdir()):
                candidates.append((f"bible/locations/{child.name}/location.json", f"Location: {child.name}"))
        candidates.append(("bible/style.md", "Visual Style"))
        return _existing_targets(project, candidates)
    if stage in {"storyboard", "clip"}:
        return _existing_targets(project, [
            ("storyboard/shots.json", "Storyboard Shots"),
            ("storyboard/shots.review.md", "Flow Review (连贯性)"),
        ])
    if stage == "video":
        return _prompt_targets(project, ".video.md", "Video Prompt")
    if stage == "audio":
        if _native_audio(project):
            return _prompt_targets(project, ".video.md", "Audiovisual Prompt")
        return _prompt_targets(
            project, ".audio.md", "Audio Prompt", include_global_music=True
        )
    if stage == "assemble":
        return _existing_targets(project, [("edit/timeline.json", "Edit Timeline")])
    return []


def stage_payload(project: Project, stage: str | None) -> dict[str, Any]:
    """Readable data for the stage's center panel."""
    if stage == "style":
        md = project.path("bible", "style.md")
        sample = project.path("bible", "style_sample.png")
        return {
            "kind": "style",
            "style_md": md.read_text() if md.is_file() else "",
            "sample_rel": "bible/style_sample.png" if sample.is_file() else "",
            "label": str(project.model_config.get("style_label") or ""),
        }
    if stage == "plot":
        return {"kind": "plot", "plot": _read_json(project.path("story", "plot.json")) or {}}
    if stage == "script":
        script = _read_json(project.path("story", "script.json")) or {}
        return {"kind": "script", "script": script, "scenes": _script_scenes(script)}
    if stage == "bible":
        return {
            "kind": "bible",
            "characters": _character_payloads(project),
            "locations": _location_payloads(project),
            "style_exists": project.path("bible", "style.md").is_file(),
        }
    if stage in {"storyboard", "clip"}:
        return _prompt_review_payload(project, "keyframes")
    if stage == "keyframes":
        return {"kind": "keyframe_review", "shots": _shot_payloads(project)}
    if stage == "video_prompts":
        return _prompt_review_payload(project, "videos")
    if stage in {"video", "review"}:
        return {"kind": "video_review", "shots": _shot_payloads(project)}
    if stage == "audio":
        native = _native_audio(project)
        return {
            "kind": "audio",
            "mode": "native_video" if native else "legacy",
            "prompts": _prompt_paths(
                project,
                ".video.md" if native else ".audio.md",
                include_global_music=not native,
            ),
            "audio_files": _rel_glob(project, "assets/audio", "*"),
            "source_clips": (
                _rel_glob(project, "assets/clips", "*.mp4") if native else []
            ),
        }
    if stage == "assemble":
        output = project.path("output", f"{project.project_id}.mp4")
        return {
            "kind": "assemble",
            "timeline": _read_json(project.path("edit", "timeline.json")) or {},
            "output": output.relative_to(project.dir).as_posix() if output.is_file() else "",
        }
    return {"kind": "done"}


def stage_asset_actions(project: Project, stage: str | None) -> list[dict[str, str]]:
    """Regeneration buttons scoped to the active stage."""
    if stage == "bible":
        actions: list[dict[str, str]] = []
        for rel in _generated_bible_asset_paths(project):
            label = f"Regenerate {Path(rel).parent.name} {Path(rel).name}"
            actions.append({"kind": "bible_asset", "shot_id": "", "rel_path": rel, "label": label})
        return actions
    if stage == "keyframes":
        actions = []
        for shot in _shots(project):
            rel = _shot_keyframe_rel(project, shot)
            if rel:
                sid = str(shot.get("id") or "")
                actions.append({
                    "kind": "keyframe",
                    "shot_id": sid,
                    "rel_path": rel,
                    "label": f"Regenerate keyframe {sid}",
                })
        return actions
    if stage == "video":
        actions = []
        for shot in _shots(project):
            sid = str(shot.get("id") or "")
            rel = f"assets/clips/{sid}.mp4"
            if sid and project.path(*rel.split("/")).is_file():
                actions.append({
                    "kind": "shot_video",
                    "shot_id": sid,
                    "rel_path": rel,
                    "label": f"Regenerate video {sid}",
                })
        return actions
    return []


def _prompt_review_payload(project: Project, gate: str) -> dict[str, Any]:
    records = conversation_records(project)
    revision_counts: dict[str, int] = {}
    for record in records:
        if record.get("role") != "assistant" or record.get("gate") != gate:
            continue
        for rel in record.get("changed_paths") or []:
            revision_counts[rel] = revision_counts.get(rel, 0) + 1

    payloads = []
    for shot, media_payload in zip(_shots(project), _shot_payloads(project)):
        sid = str(shot.get("id") or "")
        path = prompt_path(project, gate, shot)
        rel = path.relative_to(project.dir).as_posix()
        text = path.read_text() if path.is_file() else ""
        grid = shot.get("motion_grid") if gate == "keyframes" else None
        if not text:
            errors = ["missing prompt"]
        elif gate == "keyframes" and isinstance(grid, dict):
            # Only motion grids are validated, and only structurally (every panel enumerated).
            # Plain keyframe prompts are trusted as written — no prose motion-word matching.
            errors = validate_grid_prompt(
                text, shot, panel_count=int(grid.get("panel_count") or 0)
            ).errors
        else:
            errors = []
        refs = []
        for field in (
            "keyframe_reference_images",
            "reference_images",
            "target_state_reference_images",
            "reference_style_images",
        ):
            refs.extend(str(value) for value in (shot.get(field) or []) if value)
        item = dict(media_payload)
        item.update({
            "prompt_kind": "grid" if isinstance(grid, dict) else (
                "static" if gate == "keyframes" else "video"
            ),
            "prompt_rel": rel,
            "prompt_text": text,
            "validation_errors": errors,
            "prompt_status": "invalid" if errors else "ready",
            "revision_count": revision_counts.get(rel, 0),
            "references": list(dict.fromkeys(refs)),
            "vision_revision_available": bool(refs),
            "panel_count": int((grid or {}).get("panel_count") or 0),
            "panel_plan": list((grid or {}).get("panel_beats") or []),
            "later_video_intent": {
                "action": shot.get("action") or "",
                "camera_movement": shot.get("camera_movement") or "",
                "duration_s": shot.get("duration_s"),
                "visual_beats": shot.get("visual_beats") or [],
            },
        })
        payloads.append(item)
    return {
        "kind": "keyframe_prompt_review" if gate == "keyframes" else "video_prompt_review",
        "gate": gate,
        "shots": payloads,
        "conversation": records,
        "batch_ready": bool(payloads) and all(
            not shot["validation_errors"] for shot in payloads
        ),
        "cost": project.total_cost(),
        "cost_cap": project.cost_cap,
    }


def _existing_targets(project: Project, candidates: list[tuple[str, str]]) -> list[dict[str, str]]:
    targets = []
    for rel, label in candidates:
        path = project.path(*rel.split("/"))
        if path.is_file():
            targets.append({"path": rel, "label": label, "kind": _target_kind(path)})
    return targets


def _prompt_targets(
    project: Project,
    suffix: str,
    label: str,
    *,
    include_global_music: bool = False,
) -> list[dict[str, str]]:
    targets = []
    for rel in _prompt_paths(
        project, suffix, include_global_music=include_global_music
    ):
        path = project.path(*rel.split("/"))
        shot_id = path.name.removesuffix(suffix)
        targets.append({"path": rel, "label": f"{label}: {shot_id}", "kind": _target_kind(path)})
    return targets


def _prompt_paths(
    project: Project,
    suffix: str,
    *,
    include_global_music: bool = False,
) -> list[str]:
    prompts = project.path("storyboard", "prompts")
    if not prompts.is_dir():
        return []
    live_shot_ids = {
        str(shot.get("id") or "") for shot in _shots(project) if shot.get("id")
    }
    paths = []
    for path in sorted(prompts.glob(f"*{suffix}")):
        prompt_id = path.name.removesuffix(suffix)
        if prompt_id not in live_shot_ids and not (
            include_global_music and path.name == "music.audio.md"
        ):
            continue
        paths.append(path.relative_to(project.dir).as_posix())
    return paths


def _target_kind(path: Path) -> str:
    return "json" if path.suffix.lower() == ".json" else "text"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _script_scenes(script: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = []
    for episode in script.get("episodes", []):
        if isinstance(episode, dict):
            scenes.extend(s for s in episode.get("scenes", []) if isinstance(s, dict))
    return scenes


def _bible_text_ahead(entity_dir: Path, text_names: list[str]) -> bool:
    try:
        ref_mtime = (entity_dir / "reference.png").stat().st_mtime
    except FileNotFoundError:
        return True
    for name in text_names:
        try:
            if (entity_dir / name).stat().st_mtime > ref_mtime:
                return True
        except FileNotFoundError:
            continue
    return False


def _character_payloads(project: Project) -> list[dict[str, Any]]:
    chars = project.path("bible", "characters")
    if not chars.is_dir():
        return []
    out = []
    for child in sorted(chars.iterdir()):
        if not child.is_dir():
            continue
        data = _read_json(child / "character.json") or {}
        board = _read_json(child / "identity_board.json") or {}
        states = _read_json(child / "states.json") or {}
        base = f"bible/characters/{child.name}"
        images = [
            {"rel": f"{base}/{name}", "label": "Identity reference"}
            for name in ("reference.png",)
            if (child / name).is_file()
        ]
        for state in states.get("states") or []:
            if not isinstance(state, dict):
                continue
            rel = str(state.get("reference_image") or "").strip()
            if not rel or rel == "reference.png" or not (child / rel).is_file():
                continue
            images.append({
                "rel": f"{base}/{rel}",
                "label": (
                    "Appearance state reference: "
                    f"{state.get('label') or state.get('id')}"
                ),
            })
        out.append({
            "slug": child.name,
            "name": data.get("name") or child.name,
            "character": data,
            "identity_board": board,
            "states": states,
            "images": images,
            "text_ahead": _bible_text_ahead(child, ["character.json", "identity_board.json"]),
        })
    return out


def _location_payloads(project: Project) -> list[dict[str, Any]]:
    locs = project.path("bible", "locations")
    if not locs.is_dir():
        return []
    out = []
    for child in sorted(locs.iterdir()):
        if not child.is_dir():
            continue
        data = _read_json(child / "location.json") or {}
        base = f"bible/locations/{child.name}"
        out.append({
            "slug": child.name,
            "name": data.get("name") or child.name,
            "location": data,
            "images": [
                {"rel": f"{base}/{name}", "label": name}
                for name in ("reference.png",)
                if (child / name).is_file()
            ],
            "text_ahead": _bible_text_ahead(child, ["location.json"]),
        })
    return out


def _shot_payloads(project: Project) -> list[dict[str, Any]]:
    payloads = []
    for shot in _shots(project):
        sid = str(shot.get("id") or "")
        item = dict(shot)
        item["keyframe_rel"] = _shot_keyframe_rel(project, shot) or ""
        keyframe_prompt = f"storyboard/prompts/{sid}.keyframe.md"
        item["keyframe_prompt_rel"] = (
            keyframe_prompt if sid and project.path(*keyframe_prompt.split("/")).is_file() else ""
        )
        clip = f"assets/clips/{sid}.mp4"
        item["clip_rel"] = clip if sid and project.path(*clip.split("/")).is_file() else ""
        prompt = f"storyboard/prompts/{sid}.video.md"
        item["video_prompt_rel"] = prompt if sid and project.path(*prompt.split("/")).is_file() else ""
        audio_prompt = f"storyboard/prompts/{sid}.audio.md"
        item["audio_prompt_rel"] = (
            audio_prompt if sid and project.path(*audio_prompt.split("/")).is_file() else ""
        )
        native_audio = f"assets/audio/{sid}.native.wav"
        item["native_audio_rel"] = (
            native_audio
            if _native_audio(project)
            and sid
            and project.path(*native_audio.split("/")).is_file()
            else ""
        )
        payloads.append(item)
    return payloads


def _shot_keyframe_rel(project: Project, shot: dict[str, Any]) -> str:
    keyframe = str(shot.get("keyframe") or "")
    if not keyframe:
        return ""
    rel = f"storyboard/keyframes/{keyframe}"
    return rel if project.path(*rel.split("/")).is_file() else ""


def _shots(project: Project) -> list[dict[str, Any]]:
    data = _read_json(project.path("storyboard", "shots.json")) or {}
    return [shot for shot in data.get("shots", []) if isinstance(shot, dict)]


def _generated_bible_asset_paths(project: Project) -> list[str]:
    rels: list[str] = []
    for base, names in (
        ("bible/characters", ("reference.png",)),
        ("bible/locations", ("reference.png",)),
    ):
        root = project.path(*base.split("/"))
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            for name in names:
                rel = f"{base}/{child.name}/{name}"
                if project.path(*rel.split("/")).is_file():
                    rels.append(rel)
    states_root = project.path("bible", "characters")
    if states_root.is_dir():
        rels.extend(
            path.relative_to(project.dir).as_posix()
            for path in sorted(states_root.glob("*/states/*/reference.png"))
            if path.is_file()
        )
    return rels


def _rel_glob(project: Project, *parts: str) -> list[str]:
    base = project.path(*parts[:-1])
    if not base.is_dir():
        return []
    return [path.relative_to(project.dir).as_posix() for path in sorted(base.glob(parts[-1])) if path.is_file()]


def _native_audio(project: Project) -> bool:
    return project.model_config.get("audio_mode") == "native_video"


def _stage_summary(stage: str, payload: dict[str, Any]) -> str:
    if stage == "style":
        return payload.get("label") or ("Style ready for review." if payload.get("style_md") else "Waiting for style.")
    if stage == "plot":
        plot = payload.get("plot") or {}
        return str(plot.get("logline") or plot.get("synopsis") or "Waiting for plot output.")
    if stage == "script":
        scenes = payload.get("scenes") or []
        return f"{len(scenes)} scene(s) ready for review." if scenes else "Waiting for script output."
    if stage == "bible":
        characters = payload.get("characters") or []
        locations = payload.get("locations") or []
        if not characters and not locations:
            return "Waiting for bible assets."
        return f"{len(characters)} character(s), {len(locations)} location(s)."
    if stage in {"storyboard", "clip", "keyframes", "video_prompts", "video", "review"}:
        shots = payload.get("shots") or []
        clips = sum(1 for shot in shots if shot.get("clip_rel"))
        return f"{len(shots)} shot(s), {clips} clip(s) on disk."
    if stage == "audio":
        return f"{len(payload.get('audio_files') or [])} audio file(s)."
    if stage == "assemble":
        return "Final output exists." if payload.get("output") else "Timeline ready for final assembly."
    return "All stages approved."
