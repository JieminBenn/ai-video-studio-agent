"""Uploaded reference images for file-backed project consistency.

Reference uploads are durable project artifacts, not transient UI state. They live
under ``references/uploads/`` with a manifest at ``references/references.json`` so
humans can inspect, move, or delete them like every other Studio Agent artifact.
"""

from __future__ import annotations

import json
import mimetypes
import re
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Iterable

from .invalidation import invalidate_reference_change
from .orchestrator.project import Project, slugify

ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
TARGET_TYPES = {"character", "location", "style"}
MANIFEST_REL = ("references", "references.json")
INFERENCE_CONFIDENCE_THRESHOLD = 0.75

_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
}

# Role words map to the lead character. English + 中文 (invariant #10): the user names a
# role at intake before the character has a generated name, so every role term resolves to
# the lead during resolve_pending_reference_targets.
_ROLE_WORDS = {
    "protagonist": "protagonist",
    "main character": "protagonist",
    "lead character": "protagonist",
    "lead": "protagonist",
    "hero": "protagonist",
    "heroine": "protagonist",
    "主角": "protagonist",
    "主人公": "protagonist",
    "主人翁": "protagonist",
    "男主": "protagonist",
    "女主": "protagonist",
    "男主角": "protagonist",
    "女主角": "protagonist",
}
# Generic "this is the character" words (no specific name / role) — also resolve to the lead
# so a single-subject reference attaches without the user knowing the generated name.
_CHARACTER_WORDS = {
    "character", "this character", "the character", "person", "the person", "subject",
    "角色", "人物", "女子", "男子", "女人", "男人", "女孩", "男孩", "女生", "男生",
}
_STYLE_WORDS = {
    "style", "mood", "look", "palette", "visual style", "vibe",
    "风格", "画风", "色调", "视觉风格", "氛围", "调色", "风格参考",
}
# Markers classify a clause as a location; places are concrete enough to be a target name.
_LOCATION_MARKERS = {
    "location", "place", "background", "scene", "setting",
    "场景", "地点", "背景", "场所", "环境", "室内", "室外",
}
_LOCATION_PLACES = {
    "room", "apartment", "house", "home", "school", "shop", "street",
    "forest", "city", "beach", "building",
    "房间", "屋子", "公寓", "房子", "街道", "森林", "城市", "海滩", "学校", "商店", "建筑",
}
_LOCATION_WORDS = _LOCATION_MARKERS | _LOCATION_PLACES


def load_reference_manifest(project: Project) -> list[dict[str, Any]]:
    data = _read_manifest_doc(project)
    if isinstance(data, dict):
        data = data.get("references", [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def write_reference_manifest(project: Project, records: list[dict[str, Any]]) -> None:
    doc = _read_manifest_doc(project)
    doc["references"] = records
    _write_manifest_doc(project, doc)


def reference_intent_context(project: Project) -> dict[str, Any]:
    doc = _read_manifest_doc(project)
    records = []
    for record in doc.get("references", []):
        if not isinstance(record, dict):
            continue
        records.append({
            "alias": record.get("alias"),
            "aliases": list(record.get("aliases") or []),
            "status": record.get("status", "resolved"),
            "target_type": record.get("target_type") or "",
            "target_id": record.get("target_id") or "",
            "intent": record.get("intent") or record.get("note") or "",
            "inference": record.get("inference") or {},
        })
    context = {
        "references": records,
        "story_constraints": list(doc.get("story_constraints") or []),
        "warnings": list(doc.get("warnings") or []),
    }
    if records or context["story_constraints"] or context["warnings"]:
        return context
    return {}


def reference_intent_prompt(project: Project) -> str:
    context = reference_intent_context(project)
    if not context:
        return ""
    return "REFERENCE_INTENT: " + json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"


def _write_manifest_doc(project: Project, doc: dict[str, Any]) -> None:
    path = project.path(*MANIFEST_REL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")


def save_reference_upload(
    project: Project,
    *,
    filename: str,
    data: bytes,
    target_type: str,
    target_id: str = "",
    label: str = "",
    note: str = "",
    content_type: str | None = None,
) -> dict[str, Any]:
    records = save_reference_intake_uploads(
        project,
        [{"filename": filename, "data": data, "content_type": content_type}],
        note=note,
        target_type=target_type,
        target_id=target_id,
        label=label,
    )
    return records[0]


def save_reference_intake_uploads(
    project: Project,
    uploads: Iterable[Any],
    *,
    note: str = "",
    analyzer: Any | None = None,
    target_type: str = "",
    target_id: str = "",
    label: str = "",
) -> list[dict[str, Any]]:
    """Save a reference batch, assign aliases, and resolve per-image intent.

    Uploads can be dicts or small objects with ``filename``, ``data``, and optional
    ``content_type`` attributes. Explicit user mappings are applied before analyzer
    guesses; low-confidence guesses stay unresolved so they cannot condition media.
    """
    raw_uploads = list(uploads)
    if not raw_uploads:
        return []

    uploads: list[dict[str, Any]] = []
    for upload in raw_uploads:
        prepared = {
            "filename": _upload_attr(upload, "filename", ""),
            "data": _upload_attr(upload, "data", b""),
            "content_type": _upload_attr(upload, "content_type", None),
        }
        _validate_upload(prepared["filename"], prepared["data"])
        uploads.append(prepared)

    manual_target_type = str(target_type or "").strip()
    manual_target_id = str(target_id or "").strip()
    if manual_target_type:
        manual_target_type = _normalize_target_type(manual_target_type)
        if manual_target_type == "style":
            manual_target_id = "global"
        elif not manual_target_id:
            raise ValueError("reference upload needs a character or location target")

    existing = load_reference_manifest(project)
    doc = _read_manifest_doc(project)
    start = _next_reference_number(existing)
    total = len(uploads)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records: list[dict[str, Any]] = []

    # The style/global image is project-wide look, not a numbered subject reference. It must
    # not consume an @imageN slot (else the user's character/location uploads shift by one:
    # shown @image1 at upload but persisted @image2). Give it a stable @style alias instead so
    # subject references always start at @image1, consistently across upload and bible views.
    style_batch = manual_target_type == "style"
    for offset, upload in enumerate(uploads):
        filename = upload["filename"]
        data = upload["data"]
        content_type = upload["content_type"]
        upload_id, rel_path = _write_upload_file(project, filename, data)
        ext = Path(filename).suffix.lower()
        if style_batch:
            alias = "@style" if offset == 0 else f"@style{offset + 1}"
            aliases = _dedupe(
                [alias, "@style", "style", "style image", "style reference", "the style"]
            )
        else:
            number = start + offset
            batch_index = offset + 1
            alias = f"@image{number}"
            aliases = _dedupe(
                [
                    f"@image{number}",
                    f"@photo{number}",
                    f"image{number}",
                    f"image {number}",
                    f"photo{number}",
                    f"photo {number}",
                    *natural_aliases_for_index(batch_index, total),
                ]
            )
        record = {
            "id": upload_id,
            "status": "unresolved",
            "target_type": "",
            "target_id": "",
            "target_slug": "",
            "alias": alias,
            "aliases": aliases,
            "label": label.strip(),
            "note": note.strip(),
            "intent": note.strip(),
            "path": rel_path,
            "original_filename": Path(filename).name,
            "content_type": content_type or mimetypes.guess_type(filename)[0] or _mime_for_ext(ext),
            "created_at": now,
        }
        records.append(record)

    constraints, warnings = _apply_intake_note(records, note)
    for record in records:
        if manual_target_type:
            _apply_target(
                record,
                target_type=manual_target_type,
                target_id=manual_target_id,
                status="resolved",
                pending_target=None,
            )
            continue
        if _has_explicit_target(record):
            continue
        if analyzer is None:
            continue
        if not project.within_cost_cap():
            warnings.append(
                f"Reference analysis skipped for {record.get('alias')}: project cost cap is spent."
            )
            continue
        try:
            gen = analyzer.analyze(
                str(project.dir / record["path"]),
                aliases=list(record.get("aliases") or []),
                user_note=note,
            )
            _apply_inference(record, gen.content)
            project.add_generation_cost(stage="references", generation=gen)
        except Exception as exc:
            warnings.append(
                f"Reference analysis failed for {record.get('alias') or record.get('original_filename')}: "
                f"{exc}"
            )

    all_records = existing + records
    doc["references"] = all_records
    if constraints:
        doc["story_constraints"] = list(doc.get("story_constraints") or []) + constraints
    if warnings:
        doc["warnings"] = list(doc.get("warnings") or []) + warnings
    _write_manifest_doc(project, doc)
    if (
        "bible" in project.stages
        and project.stage_status("bible") != "pending"
        and any(not _record_is_resolved(record) for record in records)
    ):
        # A late automatic upload still needs the Bible resolver to see the now-known
        # cast/location roster. Rewinding only to storyboard would strand it unresolved.
        invalidate_reference_change(project)
    return records


def save_reference_intake_upload(
    project: Project,
    *,
    filename: str,
    data: bytes,
    note: str = "",
    content_type: str | None = None,
    analyzer: Any | None = None,
) -> dict[str, Any]:
    records = save_reference_intake_uploads(
        project,
        [{"filename": filename, "data": data, "content_type": content_type}],
        note=note,
        analyzer=analyzer,
    )
    return records[0]


def retarget_reference(
    project: Project,
    reference_id: str,
    target_type: str,
    target_id: str = "",
) -> dict[str, Any]:
    target_type = _normalize_target_type(target_type)
    target_id = str(target_id or "").strip()
    if target_type == "style":
        target_id = "global"
    elif not target_id:
        raise ValueError("reference upload needs a character or location target")

    records = load_reference_manifest(project)
    for record in records:
        if record.get("id") != reference_id:
            continue
        old_was_resolved = _record_is_resolved(record)
        old_target_type = str(record.get("target_type") or "").strip().lower()
        old_target_id = str(record.get("target_id") or "").strip()
        record["status"] = "resolved"
        record["target_type"] = target_type
        record["target_id"] = target_id
        record["target_slug"] = _target_slug(target_id)
        record.pop("pending_target", None)
        write_reference_manifest(project, records)
        affected_targets: list[tuple[str, str]] = []
        if old_was_resolved and old_target_type in TARGET_TYPES and old_target_id:
            affected_targets.append((old_target_type, old_target_id))
        if (target_type, target_id) not in affected_targets:
            affected_targets.append((target_type, target_id))
        for changed_type, changed_id in affected_targets:
            invalidate_reference_change(
                project,
                target_type=changed_type,
                target_id=changed_id,
            )
        return record
    raise ValueError(f"reference not found: {reference_id}")


def _inference_is_high_confidence(record: dict[str, Any]) -> bool:
    inference = record.get("inference")
    if not isinstance(inference, dict):
        return False
    try:
        return float(inference.get("confidence") or 0.0) >= INFERENCE_CONFIDENCE_THRESHOLD
    except (TypeError, ValueError):
        return False


def resolve_pending_reference_targets(
    project: Project,
    *,
    characters: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    """Map still-pending reference uploads to concrete bible targets.

    A cheap deterministic pass (bilingual keyword/role/slug matching) runs first so the
    common cases resolve offline with no cost. Anything it leaves unresolved is handed to
    ``llm`` — when one is supplied — which reads the user's free-form note plus the real
    character/location names and maps each reference, so arbitrary phrasing in any language
    is understood without hardcoding vocabulary (invariant #10).
    """
    records = load_reference_manifest(project)
    if not records:
        return records

    lead = _lead_character_name(characters)
    character_by_slug = {
        _target_slug(str(character.get("name") or "")): str(character.get("name") or "")
        for character in characters
        if character.get("name")
    }
    location_by_slug = {
        _target_slug(str(location.get("name") or "")): str(location.get("name") or "")
        for location in locations
        if location.get("name")
    }
    changed = False

    for record in records:
        if _record_is_resolved(record):
            continue
        pending = record.get("pending_target") if isinstance(record.get("pending_target"), dict) else {}
        kind = str(pending.get("kind") or "")
        value = str(pending.get("value") or record.get("target_id") or "").strip()

        if kind == "character_role" and lead:
            # A named match wins (user typed a specific character); otherwise every role /
            # generic character reference resolves to the lead (invariant #10 + the
            # auto-attach-to-lead behavior chosen for single/ambiguous subjects).
            named = character_by_slug.get(_target_slug(value))
            _apply_target(
                record,
                target_type="character",
                target_id=named or lead,
                status="resolved",
                pending_target=None,
            )
            changed = True
            continue

        if record.get("target_type") == "character":
            name = character_by_slug.get(_target_slug(value))
            if not name and _inference_is_high_confidence(record) and len(character_by_slug) == 1:
                name = next(iter(character_by_slug.values()))
            if name:
                _apply_target(
                    record,
                    target_type="character",
                    target_id=name,
                    status="resolved",
                    pending_target=None,
                )
                changed = True
            continue

        if record.get("target_type") == "location":
            name = location_by_slug.get(_target_slug(value))
            if not name and _inference_is_high_confidence(record) and len(location_by_slug) == 1:
                name = next(iter(location_by_slug.values()))
            if name:
                _apply_target(
                    record,
                    target_type="location",
                    target_id=name,
                    status="resolved",
                    pending_target=None,
                )
                changed = True
            continue

        if record.get("target_type") == "style" and record.get("target_id") == "global":
            record["status"] = "resolved"
            changed = True

    if llm is not None and any(not _record_is_resolved(record) for record in records):
        changed = _llm_resolve_reference_targets(
            records, characters=characters, locations=locations, llm=llm
        ) or changed

    if changed:
        write_reference_manifest(project, records)
    return records


def _llm_resolve_reference_targets(
    records: list[dict[str, Any]],
    *,
    characters: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    llm: Any,
) -> bool:
    """Map remaining unresolved references with the LLM. Returns True if anything changed."""
    pending = [record for record in records if not _record_is_resolved(record)]
    if not pending:
        return False

    character_names = {
        str(character.get("name") or "").strip(): str(character.get("name") or "").strip()
        for character in characters
        if character.get("name")
    }
    location_names = {
        str(location.get("name") or "").strip(): str(location.get("name") or "").strip()
        for location in locations
        if location.get("name")
    }
    by_alias = {str(record.get("alias") or ""): record for record in pending}

    prompt = _reference_mapping_prompt(pending, characters, locations)
    try:
        generation = llm.complete_json(prompt)
    except Exception:
        return False
    mappings = (getattr(generation, "content", None) or {}).get("mappings")
    if not isinstance(mappings, list):
        return False

    changed = False
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        try:
            confidence = float(mapping.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < INFERENCE_CONFIDENCE_THRESHOLD:
            continue
        record = by_alias.get(str(mapping.get("alias") or ""))
        if record is None or _record_is_resolved(record):
            continue
        has_user_intent = bool(
            str(record.get("note") or record.get("intent") or "").strip()
        )
        if not has_user_intent and not _inference_is_high_confidence(record):
            continue
        target_type = str(mapping.get("target_type") or "").strip().lower()
        target_id = str(mapping.get("target_id") or "").strip()
        if target_type == "character":
            name = character_names.get(target_id) or _ci_lookup(character_names, target_id)
            if not name:
                continue
            _apply_target(record, target_type="character", target_id=name,
                          status="resolved", pending_target=None)
            changed = True
        elif target_type == "location":
            name = location_names.get(target_id) or _ci_lookup(location_names, target_id)
            if not name:
                continue
            _apply_target(record, target_type="location", target_id=name,
                          status="resolved", pending_target=None)
            changed = True
        elif target_type == "style":
            _apply_target(record, target_type="style", target_id="global",
                          status="resolved", pending_target=None)
            changed = True
    return changed


def _ci_lookup(names: dict[str, str], value: str) -> str:
    folded = value.casefold()
    for key, name in names.items():
        if key.casefold() == folded:
            return name
    return ""


_REFERENCE_MAPPING_PROMPT = """[task:reference_mapping]
You assign uploaded reference images to a story's characters and locations using the user's
note. For each reference decide what it is for, matching meaning (any language), not keywords.

Return ONLY JSON: {{"mappings": [{{"alias": "@imageN", "target_type": "...", "target_id": "...", "confidence": 0.0}}]}}
- target_type: one of character, location, style, none.
- target_id: for character/location use the EXACT name from the lists; for style use "global";
  for none use "".
- Use confidence >= 0.75 only when the image evidence and story entity are a strong semantic match.
- Choose character/location when the note or high-confidence vision evidence clearly indicates
  a strong match; otherwise use none.
- Do not invent names that are not in the lists.

USER NOTE / INTENT:
{note}

REFERENCES (alias — label — per-image note):
{references}

CHARACTERS (name — role):
{characters}

LOCATIONS (name):
{locations}
"""


def _reference_mapping_prompt(
    pending: list[dict[str, Any]],
    characters: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> str:
    notes = _dedupe([
        str(record.get("intent") or record.get("note") or "").strip()
        for record in pending
        if str(record.get("intent") or record.get("note") or "").strip()
    ])
    reference_lines = "\n".join(
        f"- {record.get('alias')} — {record.get('label') or '(no label)'} — "
        f"{record.get('note') or record.get('intent') or '(no note)'} — "
        f"vision={json.dumps(record.get('inference') or {}, ensure_ascii=False, sort_keys=True)}"
        for record in pending
    ) or "(none)"
    character_lines = "\n".join(
        f"- {character.get('name')} — {character.get('role') or 'unspecified role'}"
        for character in characters
        if character.get("name")
    ) or "(none)"
    location_lines = "\n".join(
        f"- {location.get('name')}" for location in locations if location.get("name")
    ) or "(none)"
    return _REFERENCE_MAPPING_PROMPT.format(
        note="\n".join(notes) or "(none)",
        references=reference_lines,
        characters=character_lines,
        locations=location_lines,
    )


def next_reference_alias(project: Project) -> str:
    return f"@image{_next_reference_number(load_reference_manifest(project))}"


def natural_aliases_for_index(index: int, total: int) -> list[str]:
    aliases = [
        f"photo{index}",
        f"photo {index}",
        f"image{index}",
        f"image {index}",
    ]
    ordinal = _ORDINALS.get(index)
    if ordinal:
        aliases.extend([
            f"{ordinal} image",
            f"{ordinal} photo",
            f"{ordinal} one",
        ])
    if total == 1 and index == 1:
        aliases.extend(["the image", "this image", "the photo", "this photo"])
    return _dedupe(aliases)


def reference_records_for_target(
    project: Project,
    target_type: str,
    *target_ids: str,
) -> list[dict[str, Any]]:
    target_type = _normalize_target_type(target_type)
    records = load_reference_manifest(project)
    if target_type == "style":
        matched = [
            r for r in records
            if _record_is_resolved(r)
            and r.get("target_type") == "style"
        ]
        for record in matched:
            _require_record_file(project, record)
        return matched

    slugs = {_target_slug(value) for value in target_ids if value}
    names = {str(value).strip().lower() for value in target_ids if value}
    matched = []
    for record in records:
        if not _record_is_resolved(record):
            continue
        if record.get("target_type") != target_type:
            continue
        # Recompute from target_id: legacy manifests stored the colliding "project" slug.
        record_slug = _target_slug(str(record.get("target_id", "")))
        record_name = str(record.get("target_id") or "").strip().lower()
        if record_slug in slugs or record_name in names:
            _require_record_file(project, record)
            matched.append(record)
    return matched


def reference_paths_for_target(project: Project, target_type: str, *target_ids: str) -> list[str]:
    return _dedupe(
        [
            str(record.get("path"))
            for record in reference_records_for_target(project, target_type, *target_ids)
            if record.get("path")
        ]
    )


def reference_paths_in_text(
    project: Project,
    text: str,
    *,
    target_types: set[str] | None = None,
) -> list[str]:
    """Resolve uploaded references the user names by alias in free text.

    When a human says "make it look like @image1" in regeneration feedback, the named upload
    must actually condition the re-render — not just be mentioned in the prompt. Matches the
    structured alias tokens (``@image1``, ``photo 2``, "the image", …) carried on each record,
    never the image's semantic content (invariant #11), and returns existing upload paths.
    """
    if not str(text or "").strip():
        return []
    paths: list[str] = []
    for record in load_reference_manifest(project):
        if not _record_is_resolved(record):
            continue
        if target_types is not None and record.get("target_type") not in target_types:
            continue
        aliases = [record.get("alias"), *(record.get("aliases") or [])]
        if any(_phrase_in_text(alias, text) for alias in aliases if alias):
            _require_record_file(project, record)
            path = str(record.get("path") or "")
            if path:
                paths.append(path)
    return _dedupe(paths)


def reference_upload_paths_for_shot(project: Project, shot: dict[str, Any]) -> list[str]:
    """Return live character/location uploads bound to subjects named by a shot."""
    character_names = _dedupe([
        str(name)
        for name in list(shot.get("characters") or [])
        + list(shot.get("reference_characters") or [])
        if str(name).strip()
    ])
    location_names = _dedupe([
        str(name)
        for name in (shot.get("reference_locations") or [])
        if str(name).strip()
    ])
    paths: list[str] = []
    for name in character_names:
        paths.extend(reference_paths_for_target(project, "character", name))
    for name in location_names:
        paths.extend(reference_paths_for_target(project, "location", name))
    return _dedupe(paths)


def live_reference_paths_for_shot(
    project: Project,
    shot: dict[str, Any],
    fields: tuple[str, ...],
) -> list[str]:
    """Merge live subject uploads ahead of stored refs, excluding raw style uploads."""
    live_uploads = reference_upload_paths_for_shot(project, shot)
    live_set = set(live_uploads)
    manifest_paths = {
        str(record.get("path") or "")
        for record in load_reference_manifest(project)
        if str(record.get("path") or "")
    }
    blocked_styles = {
        str(record.get("path") or "")
        for record in load_reference_manifest(project)
        if record.get("target_type") == "style" and record.get("path")
    }
    blocked_styles |= {str(project.dir / rel) for rel in blocked_styles}

    def live_stored_path(value: Any) -> str:
        path = str(value or "")
        if not path or path in blocked_styles:
            return ""
        rel = _project_relative_path(project, path)
        if rel in manifest_paths:
            # A shot may cache a raw upload path. Keep it only while the live manifest
            # still binds that upload to a subject in this shot.
            return rel if rel in live_set else ""
        return path

    named = [
        live_stored_path(path)
        for path in (shot.get("named_reference_images") or [])
    ]
    stored = [
        live_stored_path(path)
        for field in fields
        for path in (shot.get(field) or [])
    ]
    return _dedupe([path for path in named + live_uploads + stored if path])


def regeneration_reference_images(
    project: Project,
    *,
    comment: str,
    subject_refs: list[str],
    max_n: int = 0,
) -> list[str]:
    """Absolute reference-image paths to show a vision model during regeneration.

    Combines the caller's contextual ``subject_refs`` (already absolute) with every reference
    the user names by alias in ``comment`` — read from the live manifest, so uploads added
    moments ago at the bible gate are included. Subjects first, deduped, existing files only,
    optionally capped to ``max_n``.
    """
    named = [
        str(project.dir / rel)
        for rel in reference_paths_in_text(
            project,
            comment,
            target_types={"character", "location"},
        )
    ]
    ordered: list[str] = []
    for path in [str(p) for p in subject_refs] + named:
        if path and path not in ordered and Path(path).is_file():
            ordered.append(path)
    if max_n and max_n > 0:
        return ordered[:max_n]
    return ordered


def style_reference_paths(project: Project) -> list[str]:
    return reference_paths_for_target(project, "style", "global")


def style_anchor_paths(project: Project) -> list[str]:
    """Style references that condition every downstream generation.

    Only the approved generated ``bible/style_sample.png`` is safe to use as a
    downstream visual reference. Raw uploaded style images are for profiling and
    sample generation only: they may contain people, objects, or composition that
    must not bleed into characters/locations/keyframes/clips. The style stage
    itself must NOT use this — it produces the sample.
    """
    sample = project.path("bible", "style_sample.png")
    return ["bible/style_sample.png"] if sample.is_file() else []


def cap_references(subject_refs: list[str], style_refs: list[str], max_n: int) -> list[str]:
    """Dedupe subject refs then style refs (style last) and cap to ``max_n``.

    Subjects are highest priority; the style anchor is shed first when the total exceeds
    the provider's capacity. ``max_n <= 0`` disables the cap.
    """
    ordered: list[str] = []
    for ref in list(subject_refs) + list(style_refs):
        if ref and ref not in ordered:
            ordered.append(ref)
    if max_n and max_n > 0:
        return ordered[:max_n]
    return ordered


def cap_reference_paths(
    project: Project,
    refs: list[str],
    max_n: int,
    *,
    named_refs: list[str] | None = None,
) -> list[str]:
    """Cap an assembled reference list, dropping style-anchor refs first.

    Matches style anchors in both project-relative and absolute form so it works at call
    sites that pass either (storyboard/clip pass relative, bible passes absolute).
    """
    # ``refs`` is the caller's live, policy-filtered set. Named references only alter
    # priority inside that set; they must never resurrect a stale/retargeted manifest path.
    candidates = _dedupe(list(refs))
    raw_style_paths = {
        str(record.get("path") or "")
        for record in load_reference_manifest(project)
        if record.get("target_type") == "style" and record.get("path")
    }
    anchors = set(style_anchor_paths(project)) | raw_style_paths
    anchors |= {str(project.dir / rel) for rel in anchors}
    named = [
        ref for ref in (named_refs or [])
        if ref in candidates and ref not in anchors
    ]
    raw = [
        ref for ref in candidates
        if ref not in anchors
        and ref not in named
        and "references/uploads/" in ref.replace("\\", "/")
    ]
    derived = [
        ref for ref in candidates
        if ref not in anchors and ref not in named and ref not in raw
    ]
    styles = [ref for ref in candidates if ref in anchors]
    # Derived bible sheets (character/location reference.png + state anchors) are the
    # canonical downstream identity anchor (invariant #4), so they lead ahead of the raw
    # uploads that merely *fed* the bible. Otherwise the raw upload dominates the
    # reference-conditioned model and a regenerated bible character never takes effect
    # downstream. Only an explicitly *named* upload (user feedback) outranks the sheet.
    return cap_references(_dedupe(named + derived + raw), styles, max_n)


def mark_reference_upload_revised(
    project: Project,
    *,
    target_type: str = "",
    target_id: str = "",
) -> list[str]:
    """Mark downstream media stages pending after reference uploads change.

    Stale generated media is archived under ``history/`` for comparison. Re-running starts
    at the earliest affected stage so the new reference reaches Bible assets, keyframes,
    video, QC, audio, and assembly instead of being hidden by file-cache reuse.
    """
    result = invalidate_reference_change(
        project,
        target_type=target_type,
        target_id=target_id,
    )
    return result.invalidated_stages


def _validate_upload(filename: str, data: bytes) -> None:
    if not filename:
        raise ValueError("reference upload needs a filename")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTS))
        raise ValueError(f"reference upload must be an image file: {allowed}")
    if not data:
        raise ValueError("reference upload is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("reference upload is larger than 20 MB")
    if not _looks_like_image(data, ext):
        raise ValueError("reference upload does not look like a valid image")


def _write_upload_file(project: Project, filename: str, data: bytes) -> tuple[str, str]:
    _validate_upload(filename, data)
    ext = Path(filename).suffix.lower()
    upload_id = uuid.uuid4().hex[:12]
    safe_name = slugify(Path(filename).stem)
    rel_path = f"references/uploads/{upload_id}-{safe_name}{ext}"
    out = project.path(*rel_path.split("/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return upload_id, rel_path


def _looks_like_image(data: bytes, ext: str) -> bool:
    if ext == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8")
    if ext == ".webp":
        return data.startswith(b"RIFF") and data[8:12] == b"WEBP"
    if ext == ".gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    return False


def _normalize_target_type(value: str) -> str:
    target_type = str(value or "").strip().lower()
    if target_type not in TARGET_TYPES:
        raise ValueError("reference target must be character, location, or style")
    return target_type


def _target_slug(value: str) -> str:
    """Distinct, stable match-slug for a target name.

    ``slugify`` strips all non-ASCII and falls back to the literal ``"project"``, so every
    all-CJK name (美女, 男生, 数字美女, …) collapsed onto one slug and a reference resolved to
    one Chinese character leaked onto every other (invariant #10). ASCII names keep their
    familiar slug; names with no ASCII get a stable hash suffix so distinct names stay
    distinct. Mirrors the bible's ``char_slug`` hashing approach.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    ascii_slug = slugify(text)
    if ascii_slug != "project" or text.casefold() == "project":
        return ascii_slug
    return f"ref-{format(zlib.crc32(text.encode('utf-8')) & 0xFFFFFFFF, '08x')}"


def _record_is_resolved(record: dict[str, Any]) -> bool:
    return str(record.get("status") or "resolved") == "resolved"


def _record_file_exists(project: Project, record: dict[str, Any]) -> bool:
    rel = str(record.get("path") or "")
    if not rel:
        return False
    path = project.path(*rel.split("/"))
    try:
        path.resolve().relative_to(project.dir.resolve())
    except ValueError:
        return False
    return path.is_file()


def _require_record_file(project: Project, record: dict[str, Any]) -> Path:
    rel = str(record.get("path") or "")
    if not rel or not _record_file_exists(project, record):
        display = rel or f"references/references.json#{record.get('id') or 'unknown'}"
        raise FileNotFoundError(f"resolved reference image is missing: {display}")
    return project.path(*rel.split("/"))


def _project_relative_path(project: Project, value: str) -> str:
    path = Path(str(value or ""))
    if not path.is_absolute():
        return path.as_posix().strip("/")
    try:
        return path.resolve().relative_to(project.dir.resolve()).as_posix()
    except ValueError:
        return str(value)


def _read_manifest_doc(project: Project) -> dict[str, Any]:
    path = project.path(*MANIFEST_REL)
    if not path.is_file():
        return {"references": []}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {"references": data}
    if isinstance(data, dict):
        data.setdefault("references", [])
        return data
    return {"references": []}


def _upload_attr(upload: Any, name: str, default: Any) -> Any:
    if isinstance(upload, dict):
        return upload.get(name, default)
    return getattr(upload, name, default)


def _next_reference_number(records: list[dict[str, Any]]) -> int:
    highest = 0
    for record in records:
        alias = str(record.get("alias") or "")
        match = re.fullmatch(r"@(image|photo)(\d+)", alias)
        if match:
            highest = max(highest, int(match.group(2)))
    return highest + 1


def _apply_intake_note(records: list[dict[str, Any]], note: str) -> tuple[list[dict[str, Any]], list[str]]:
    constraints: list[dict[str, Any]] = []
    warnings = _missing_alias_warnings(note, records)
    for clause in _note_clauses(note):
        mentioned = _mentioned_records(clause, records)
        if len(mentioned) >= 2:
            constraints.append({
                "text": clause.strip(),
                "aliases": [str(record.get("alias")) for record in mentioned if record.get("alias")],
            })
            continue
        if len(mentioned) != 1:
            continue
        target = _target_from_clause(clause)
        if not target:
            continue
        _apply_target(mentioned[0], **target)
    return constraints, warnings


def _note_clauses(note: str) -> list[str]:
    return [
        piece.strip()
        for piece in re.split(r"[;\n]+", str(note or ""))
        if piece.strip()
    ]


def _mentioned_records(clause: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for record in records:
        aliases = [record.get("alias"), *(record.get("aliases") or [])]
        if any(_phrase_in_text(alias, clause) for alias in aliases if alias):
            found.append(record)
    return found


def _phrase_in_text(phrase: str, text: str) -> bool:
    phrase = _norm(phrase)
    text = _norm(text)
    if not phrase:
        return False
    pattern = r"(?<![a-z0-9@])" + re.escape(phrase) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


def _target_from_clause(clause: str) -> dict[str, Any] | None:
    low = _norm(clause)
    value = _target_value_after_mapping_verb(clause)
    if any(word in low for word in _STYLE_WORDS):
        return {
            "target_type": "style",
            "target_id": "global",
            "status": "resolved",
            "pending_target": None,
        }
    for word, role in _ROLE_WORDS.items():
        if word in low:
            return {
                "target_type": "character",
                "target_id": role,
                "status": "unresolved",
                "pending_target": {"kind": "character_role", "value": role},
            }
    if any(word in low for word in _LOCATION_WORDS):
        # The cleaned value (the place's name, e.g. "clock shop"/"公寓") is best; fall back to
        # a concrete place word, then a generic marker.
        target_id = (
            value
            or _first_present_word(low, _LOCATION_PLACES)
            or _first_present_word(low, _LOCATION_MARKERS)
            or "location"
        )
        return {
            "target_type": "location",
            "target_id": target_id,
            "status": "unresolved",
            "pending_target": {"kind": "location", "value": target_id},
        }
    if any(word in low for word in _CHARACTER_WORDS):
        # Generic character reference with no specific name -> the lead character.
        return {
            "target_type": "character",
            "target_id": "lead",
            "status": "unresolved",
            "pending_target": {"kind": "character_role", "value": "lead"},
        }
    if value:
        # A named subject ("...是美女", "= Mara"). Don't freeze it as a literal target: leave
        # it pending so resolve_pending_reference_targets binds it to a real cast member — an
        # exact name match wins (value IS a character), otherwise it falls to the lead. This is
        # what makes a descriptive note ("美女") attach instead of resolving to a phrase that
        # only matches a generated character name by coincidence.
        return {
            "target_type": "character",
            "target_id": value,
            "status": "unresolved",
            "pending_target": {"kind": "character_role", "value": value},
        }
    return None


def _target_value_after_mapping_verb(clause: str) -> str:
    # Mapping verbs: English is/as/for/= and 中文 是/为/＝ (invariant #10). CJK terms use no
    # word boundary because \b does not apply between CJK characters.
    match = re.search(
        r"(?:\b(?:is|as|for)\b|=|＝|是|为)\s*(.*)$", str(clause or ""), re.IGNORECASE
    )
    if not match:
        return ""
    value = _norm(match.group(1))
    value = re.sub(r"^(the|a|an)\s+", "", value)
    value = re.sub(r"\b(reference|photo|image|picture|for|location|background|place)\b", "", value)
    # 中文 demonstratives and trailing markers (这个公寓的场景 -> 公寓) so the name can match.
    value = re.sub(r"^(这个|那个|这|那|该)", "", value)
    value = re.sub(r"(的)?(场景|背景|地点|场所|环境|参考|画面|角色|人物)$", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _lead_character_name(characters: list[dict[str, Any]]) -> str:
    if not characters:
        return ""
    for character in characters:
        role = _norm(character.get("role") or "")
        if "lead" in role or "protagonist" in role or "main" in role:
            return str(character.get("name") or "")
    return str(characters[0].get("name") or "")


def _first_present_word(text: str, words: set[str]) -> str:
    for word in sorted(words, key=len, reverse=True):
        if word in text:
            return word
    return ""


def _apply_target(
    record: dict[str, Any],
    *,
    target_type: str,
    target_id: str,
    status: str,
    pending_target: dict[str, Any] | None,
) -> None:
    record["status"] = status
    record["target_type"] = target_type
    record["target_id"] = target_id
    record["target_slug"] = _target_slug(target_id)
    if pending_target:
        record["pending_target"] = pending_target
    else:
        record.pop("pending_target", None)


def _has_explicit_target(record: dict[str, Any]) -> bool:
    return bool(record.get("target_type") or record.get("pending_target"))


def _apply_inference(record: dict[str, Any], inference: Any) -> None:
    if not isinstance(inference, dict):
        return
    record["inference"] = dict(inference)
    confidence = float(inference.get("confidence") or 0.0)
    if confidence < INFERENCE_CONFIDENCE_THRESHOLD:
        record["status"] = "unresolved"
        return
    target_type = str(inference.get("target_type") or "").strip().lower()
    if target_type not in TARGET_TYPES:
        return
    target_id = str(inference.get("target_id") or "").strip()
    if target_type == "style":
        _apply_target(
            record,
            target_type="style",
            target_id="global",
            status="resolved",
            pending_target=None,
        )
        return
    if not target_id:
        record["target_type"] = target_type
        return
    role = _ROLE_WORDS.get(_norm(target_id))
    if target_type == "character" and role:
        _apply_target(
            record,
            target_type="character",
            target_id=role,
            status="unresolved",
            pending_target={"kind": "character_role", "value": role},
        )
        return
    _apply_target(
        record,
        target_type=target_type,
        target_id=target_id,
        status="unresolved",
        pending_target={"kind": target_type, "value": target_id},
    )


def _missing_alias_warnings(note: str, records: list[dict[str, Any]]) -> list[str]:
    warnings = []
    text = _norm(note)
    if not text:
        return warnings
    known_aliases = {
        _norm(alias)
        for record in records
        for alias in [record.get("alias"), *(record.get("aliases") or [])]
        if alias
    }
    for ordinal in _ORDINALS.values():
        for noun in ("image", "photo", "one"):
            phrase = f"{ordinal} {noun}"
            if phrase in text and phrase not in known_aliases:
                warnings.append(f"Reference note mentions '{phrase}', but no uploaded image matched it.")
    return _dedupe(warnings)


def _norm(text: Any) -> str:
    value = str(text or "").lower()
    value = value.replace("_", " ")
    value = re.sub(r"[^\w@]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _mime_for_ext(ext: str) -> str:
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
