"""Deterministic Markdown/plain-text knowledge ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .schema import KnowledgeEntry, KnowledgeSource, load_knowledge_entries


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_CHARS = 1_200


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.lower()).strip("-") or "source"


def _bounded(text: str) -> list[str]:
    text = text.strip()
    chunks: list[str] = []
    while len(text) > _MAX_CHARS:
        cut = text.rfind(" ", 0, _MAX_CHARS + 1)
        if cut < _MAX_CHARS // 2:
            cut = _MAX_CHARS
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def _markdown_chunks(text: str, fallback_title: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    title = fallback_title
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        for bounded in _bounded(content):
            chunks.append((title, bounded))

    for line in text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            title = heading.group(2).strip()
            body = []
        else:
            body.append(line)
    flush()
    return chunks


def _plain_chunks(text: str, fallback_title: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        chunks.extend((fallback_title, item) for item in _bounded(paragraph))
    return chunks


def _sequence(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def import_knowledge_file(
    source: Path,
    *,
    root: Path,
    metadata: dict[str, Any],
) -> list[KnowledgeEntry]:
    source = Path(source)
    root = Path(root)
    if source.suffix.lower() not in {".md", ".txt"}:
        raise ValueError("knowledge imports must be Markdown or plain text")

    text = source.read_text()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source_id = f"{_slug(source.stem)}-{digest[:8]}"
    import_dir = root / "imports" / source_id
    import_dir.mkdir(parents=True, exist_ok=True)
    normalized_source = import_dir / "source.md"
    normalized_source.write_text(text.rstrip() + "\n")

    raw_chunks = (
        _markdown_chunks(text, source.stem)
        if source.suffix.lower() == ".md"
        else _plain_chunks(text, source.stem)
    )
    entries: list[KnowledgeEntry] = []
    errors: list[dict[str, Any]] = []
    for index, (title, content) in enumerate(raw_chunks, start=1):
        raw = {
            "id": f"import.{digest[:12]}.{index:03d}",
            "domain": str(metadata.get("domain") or "general"),
            "stages": _sequence(metadata, "stages"),
            "intents": _sequence(metadata, "intents"),
            "styles": _sequence(metadata, "styles"),
            "formats": _sequence(metadata, "formats"),
            "language": str(metadata.get("language") or "und"),
            "title": title,
            "principle": content,
            "use_when": "",
            "recipe": content,
            "avoid": [],
            "requires_capabilities": _sequence(metadata, "requires_capabilities"),
            "conflicts_with": _sequence(metadata, "conflicts_with"),
            "source": {
                "kind": "import",
                "path": normalized_source.relative_to(root).as_posix(),
                "sha256": digest,
            },
        }
        try:
            entries.append(KnowledgeEntry.from_dict(raw))
        except ValueError as exc:
            quarantine = root / "imports" / "quarantine" / f"{source_id}-{index:03d}.md"
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            quarantine.write_text(f"# {title}\n\n{content}\n")
            errors.append({"source_id": source_id, "chunk": index, "error": str(exc)})

    _atomic_json(import_dir / "entries.json", {
        "version": 1,
        "entries": [entry.to_dict() for entry in entries],
    })

    manifest_path = root / "imports" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {
        "version": 1,
        "sources": [],
    }
    record = {
        "id": source_id,
        "original_path": str(source),
        "normalized_path": normalized_source.relative_to(root).as_posix(),
        "sha256": digest,
        "chunks": len(entries),
    }
    manifest["sources"] = [
        item for item in manifest.get("sources", []) if item.get("id") != source_id
    ] + [record]
    manifest["sources"].sort(key=lambda item: item["id"])
    _atomic_json(manifest_path, manifest)

    if errors:
        report_path = root / "imports" / "import-report.json"
        report = json.loads(report_path.read_text()) if report_path.is_file() else {
            "version": 1,
            "errors": [],
        }
        report["errors"].extend(errors)
        _atomic_json(report_path, report)
    return entries


def rebuild_index(root: Path) -> Path:
    root = Path(root)
    entries = load_knowledge_entries(root)
    path = root / "index.json"
    _atomic_json(path, {
        "version": 1,
        "entries": [entry.to_dict() for entry in entries],
    })
    return path
