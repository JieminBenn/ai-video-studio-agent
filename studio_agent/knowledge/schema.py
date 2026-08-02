"""Validated file contracts for the local filmmaking knowledge corpus."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGED_ROOT = REPO_ROOT / "knowledge"

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_DOMAIN_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class KnowledgeSource:
    kind: str
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, raw: Any) -> "KnowledgeSource":
        data = dict(raw) if isinstance(raw, dict) else {}
        return cls(
            kind=str(data.get("kind") or "core"),
            path=str(data.get("path") or ""),
            sha256=str(data.get("sha256") or ""),
        )


@dataclass(frozen=True)
class KnowledgeEntry:
    id: str
    domain: str
    stages: tuple[str, ...]
    intents: tuple[str, ...]
    styles: tuple[str, ...]
    formats: tuple[str, ...]
    language: str
    title: str
    principle: str
    use_when: str
    recipe: str
    avoid: tuple[str, ...]
    requires_capabilities: tuple[str, ...]
    conflicts_with: tuple[str, ...]
    source: KnowledgeSource

    @classmethod
    def from_dict(
        cls,
        raw: Any,
        *,
        source: KnowledgeSource | None = None,
    ) -> "KnowledgeEntry":
        data = dict(raw) if isinstance(raw, dict) else {}
        entry_id = str(data.get("id") or "").strip()
        domain = str(data.get("domain") or "").strip()
        title = str(data.get("title") or "").strip()
        if not _ID_RE.fullmatch(entry_id):
            raise ValueError(f"invalid knowledge entry id: {entry_id!r}")
        if not _DOMAIN_RE.fullmatch(domain):
            raise ValueError(f"invalid knowledge domain: {domain!r}")
        if not title:
            raise ValueError(f"knowledge entry {entry_id!r} has no title")
        return cls(
            id=entry_id,
            domain=domain,
            stages=_strings(data.get("stages")),
            intents=_strings(data.get("intents")),
            styles=_strings(data.get("styles")),
            formats=_strings(data.get("formats")),
            language=str(data.get("language") or "und").strip(),
            title=title,
            principle=str(data.get("principle") or "").strip(),
            use_when=str(data.get("use_when") or "").strip(),
            recipe=str(data.get("recipe") or "").strip(),
            avoid=_strings(data.get("avoid")),
            requires_capabilities=_strings(data.get("requires_capabilities")),
            conflicts_with=_strings(data.get("conflicts_with")),
            source=source or KnowledgeSource.from_dict(data.get("source")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "stages",
            "intents",
            "styles",
            "formats",
            "avoid",
            "requires_capabilities",
            "conflicts_with",
        ):
            data[key] = list(data[key])
        return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_core(root: Path) -> list[KnowledgeEntry]:
    core = root / "core"
    if not core.is_dir():
        return []
    entries: list[KnowledgeEntry] = []
    for path in sorted(core.rglob("*.yaml")):
        if path.name == "manifest.yaml":
            continue
        rel = path.relative_to(root).as_posix()
        try:
            document = yaml.safe_load(path.read_text()) or {}
            if isinstance(document, dict) and isinstance(
                document.get("bilingual_entries"), list
            ):
                source = KnowledgeSource(kind="core", path=rel, sha256=_sha256(path))
                for item in document["bilingual_entries"]:
                    entries.extend(_expand_bilingual_entry(item, source=source))
                continue
            raw_entries = document.get("entries") if isinstance(document, dict) else document
            if not isinstance(raw_entries, list):
                raise ValueError("top-level entries must be a list")
            source = KnowledgeSource(kind="core", path=rel, sha256=_sha256(path))
            entries.extend(
                KnowledgeEntry.from_dict(item, source=source) for item in raw_entries
            )
        except Exception as exc:
            raise ValueError(f"invalid knowledge file {rel}: {exc}") from exc
    return entries


def _expand_bilingual_entry(
    raw: dict[str, Any],
    *,
    source: KnowledgeSource,
) -> list[KnowledgeEntry]:
    entries = []
    for language in ("en", "zh"):
        entries.append(KnowledgeEntry.from_dict({
            "id": f"{raw['id']}.{language}",
            "domain": raw["domain"],
            "stages": raw.get("stages", []),
            "intents": raw.get("intents", []),
            "styles": raw.get("styles", []),
            "formats": raw.get("formats", []),
            "language": language,
            "title": raw.get(f"title_{language}"),
            "principle": raw.get(f"principle_{language}"),
            "use_when": raw.get(f"use_when_{language}"),
            "recipe": raw.get(f"recipe_{language}"),
            "avoid": raw.get(f"avoid_{language}", []),
            "requires_capabilities": raw.get("requires_capabilities", []),
            "conflicts_with": raw.get("conflicts_with", []),
        }, source=source))
    return entries


def _load_skill_entries(manifest: dict[str, Any]) -> list[KnowledgeEntry]:
    entries: list[KnowledgeEntry] = []
    for mapping in manifest.get("skills", []):
        skill_path = REPO_ROOT / "skills" / f"{mapping['skill']}.md"
        source = KnowledgeSource(
            kind="core",
            path=skill_path.relative_to(REPO_ROOT).as_posix(),
            sha256=_sha256(skill_path),
        )
        skill_text = skill_path.read_text().strip()
        for language in ("en", "zh"):
            summary = str(mapping.get(f"summary_{language}") or "")
            entries.append(KnowledgeEntry.from_dict({
                "id": f"skill.{mapping['id']}.{language}",
                "domain": mapping["domain"],
                "stages": mapping.get("stages", []),
                "intents": mapping.get("intents", []),
                "styles": [],
                "formats": [],
                "language": language,
                "title": mapping.get(f"title_{language}"),
                "principle": summary,
                "use_when": summary,
                "recipe": skill_text if language == "en" else summary,
                "avoid": mapping.get(f"avoid_{language}", []),
                "requires_capabilities": [],
                "conflicts_with": [],
            }, source=source))
    return entries


def _load_camera_entries(manifest: dict[str, Any]) -> list[KnowledgeEntry]:
    mapping = manifest.get("camera_recipes", {})
    camera_path = REPO_ROOT / str(mapping.get("path"))
    recipes = yaml.safe_load(camera_path.read_text()) or []
    source = KnowledgeSource(
        kind="core",
        path=camera_path.relative_to(REPO_ROOT).as_posix(),
        sha256=_sha256(camera_path),
    )
    entries: list[KnowledgeEntry] = []
    for recipe in recipes:
        if not isinstance(recipe, dict) or not recipe.get("id"):
            continue
        for language in ("en", "zh"):
            entries.append(KnowledgeEntry.from_dict({
                "id": f"camera.{recipe['id']}.{language}",
                "domain": mapping.get("domain", "camera_movement"),
                "stages": mapping.get("stages", []),
                "intents": recipe.get(f"intents_{language}", []),
                "styles": [],
                "formats": [],
                "language": language,
                "title": recipe.get(f"name_{language}") or recipe["id"],
                "principle": f"Use the {recipe.get('family', 'camera')} move only when its dramatic intent matches the beat.",
                "use_when": ", ".join(recipe.get(f"intents_{language}", [])),
                "recipe": recipe.get(f"prompt_{language}", ""),
                "avoid": ["camera movement without a visible dramatic beat"] if language == "en" else ["没有可见戏剧节点的无动机运镜"],
                "requires_capabilities": mapping.get("requires_capabilities", []),
                "conflicts_with": [],
            }, source=source))
    return entries


def load_packaged_core() -> list[KnowledgeEntry]:
    manifest_path = PACKAGED_ROOT / "core" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    entries = _load_core(PACKAGED_ROOT)
    entries.extend(_load_skill_entries(manifest))
    entries.extend(_load_camera_entries(manifest))
    return sorted(entries, key=lambda entry: entry.id)


def _load_imports(root: Path) -> list[KnowledgeEntry]:
    entries: list[KnowledgeEntry] = []
    imports = root / "imports"
    if not imports.is_dir():
        return entries
    for path in sorted(imports.glob("*/entries.json")):
        document = json.loads(path.read_text())
        raw_entries = document.get("entries", []) if isinstance(document, dict) else []
        entries.extend(KnowledgeEntry.from_dict(item) for item in raw_entries)
    return entries


def load_knowledge_entries(root: Path) -> list[KnowledgeEntry]:
    """Load authoritative core and imported entry files in stable order."""
    root = Path(root)
    core_entries = (
        load_packaged_core()
        if root.resolve() == PACKAGED_ROOT.resolve()
        else _load_core(root)
    )
    entries = core_entries + _load_imports(root)
    return sorted(entries, key=lambda entry: entry.id)
