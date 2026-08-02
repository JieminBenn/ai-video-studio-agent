"""Deterministic stage-aware retrieval and frozen project packets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schema import KnowledgeEntry


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SAFE_TARGET_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _values(value: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()) if str(item))


@dataclass(frozen=True)
class KnowledgeQuery:
    stage: str
    domains: tuple[str, ...]
    intent_text: str
    intents: tuple[str, ...]
    style: str
    format_name: str
    language: str
    capabilities: frozenset[str]
    hard_avoidances: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "domains": list(self.domains),
            "intent_text": self.intent_text,
            "intents": list(self.intents),
            "style": self.style,
            "format_name": self.format_name,
            "language": self.language,
            "capabilities": sorted(self.capabilities),
            "hard_avoidances": list(self.hard_avoidances),
        }


@dataclass(frozen=True)
class RankedEntry:
    entry: KnowledgeEntry
    score: int
    reasons: tuple[str, ...]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def lexical_overlap(query_text: str, entry_text: str) -> int:
    query_tokens = _tokens(query_text)
    if not query_tokens:
        return 0
    return round(
        100 * len(query_tokens & _tokens(entry_text)) / max(1, len(query_tokens))
    )


def _eligible(entry: KnowledgeEntry, query: KnowledgeQuery) -> bool:
    if entry.stages and query.stage not in entry.stages:
        return False
    if query.domains and entry.domain not in query.domains and entry.domain != "general":
        return False
    if entry.formats and query.format_name not in entry.formats:
        return False
    if not set(entry.requires_capabilities) <= set(query.capabilities):
        return False
    if set(entry.conflicts_with) & set(query.hard_avoidances):
        return False
    return True


def _rank(entry: KnowledgeEntry, query: KnowledgeQuery) -> RankedEntry:
    reasons: list[str] = []
    intent_matches = len(set(query.intents) & set(entry.intents))
    if intent_matches:
        reasons.append("intent")
    style_match = int(bool(query.style) and query.style in entry.styles)
    if style_match:
        reasons.append("style")
    format_match = int(bool(query.format_name) and query.format_name in entry.formats)
    if format_match:
        reasons.append("format")
    language_match = int(query.language == entry.language)
    if language_match:
        reasons.append("language")
    lexical = lexical_overlap(
        query.intent_text,
        " ".join((entry.title, entry.principle, entry.use_when, entry.recipe)),
    )
    if lexical:
        reasons.append(f"lexical:{lexical}")
    score = (
        10 * intent_matches
        + 4 * style_match
        + 3 * format_match
        + 2 * language_match
        + lexical
    )
    return RankedEntry(entry=entry, score=score, reasons=tuple(reasons))


def retrieve(
    entries: Iterable[KnowledgeEntry],
    query: KnowledgeQuery,
    *,
    limit: int = 6,
) -> list[RankedEntry]:
    ranked = sorted(
        (_rank(entry, query) for entry in entries if _eligible(entry, query)),
        key=lambda item: (-item.score, item.entry.id),
    )
    selected: list[RankedEntry] = []
    domains: defaultdict[str, int] = defaultdict(int)
    for item in ranked:
        if len(selected) >= max(0, limit):
            break
        if domains[item.entry.domain] >= 2:
            continue
        selected.append(item)
        domains[item.entry.domain] += 1
    return selected


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _packet_path(project, purpose: str, target: str) -> Path:
    safe_purpose = _SAFE_TARGET_RE.sub("-", purpose).strip("-") or "knowledge"
    safe_target = _SAFE_TARGET_RE.sub("-", target).strip("-") or "target"
    return project.path("knowledge", "packets", f"{safe_purpose}-{safe_target}.json")


def _entry_excerpt(item: RankedEntry) -> dict[str, Any]:
    entry = item.entry
    return {
        "id": entry.id,
        "domain": entry.domain,
        "title": entry.title,
        "score": item.score,
        "reasons": list(item.reasons),
        "principle": entry.principle,
        "use_when": entry.use_when,
        "recipe": entry.recipe,
        "avoid": list(entry.avoid),
        "source": {
            "kind": entry.source.kind,
            "path": entry.source.path,
            "sha256": entry.source.sha256,
        },
    }


def get_or_create_packet(
    project,
    *,
    purpose: str,
    target: str,
    query: KnowledgeQuery,
    entries: Iterable[KnowledgeEntry],
    limit: int = 6,
) -> dict[str, Any]:
    path = _packet_path(project, purpose, target)
    input_fingerprint = _canonical_hash(query.to_dict())
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("input_fingerprint") == input_fingerprint:
            return existing

    all_entries = sorted(list(entries), key=lambda entry: entry.id)
    ranked = retrieve(all_entries, query, limit=limit)
    packet = {
        "version": 1,
        "purpose": purpose,
        "target": target,
        "language": query.language,
        "query": query.to_dict(),
        "input_fingerprint": input_fingerprint,
        "corpus_hash": _canonical_hash([entry.to_dict() for entry in all_entries]),
        "selected_entries": [_entry_excerpt(item) for item in ranked],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    return packet


def refresh_packet(project, *, purpose: str, target: str) -> list[str]:
    path = _packet_path(project, purpose, target)
    if not path.is_file():
        return []
    relative = path.relative_to(project.dir).as_posix()
    path.unlink()
    return [relative]


def packet_guidance(packet: dict[str, Any], *, max_chars: int = 6_000) -> str:
    lines: list[str] = []
    for entry in packet.get("selected_entries") or []:
        guidance = " ".join(filter(None, [
            str(entry.get("principle") or ""),
            str(entry.get("use_when") or ""),
            str(entry.get("recipe") or ""),
        ])).strip()
        avoid = ", ".join(str(item) for item in entry.get("avoid") or [])
        line = f"- [{entry.get('id')}] {entry.get('title')}: {guidance}"
        if avoid:
            line += f" Avoid: {avoid}."
        lines.append(line)
    return "\n".join(lines)[:max_chars]
