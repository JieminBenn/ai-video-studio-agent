import json

from studio_agent.knowledge.retrieval import (
    KnowledgeQuery,
    get_or_create_packet,
    refresh_packet,
    retrieve,
)
from studio_agent.knowledge.schema import KnowledgeEntry, KnowledgeSource
from studio_agent.orchestrator.project import Project


def _entry(
    entry_id,
    *,
    domain="camera_movement",
    intents=(),
    capabilities=(),
    stages=("storyboard",),
    language="en",
):
    return KnowledgeEntry(
        id=entry_id,
        domain=domain,
        stages=tuple(stages),
        intents=tuple(intents),
        styles=("cinematic",),
        formats=("short_film",),
        language=language,
        title=entry_id,
        principle="Close distance with meaning",
        use_when="A realization lands",
        recipe="Push from medium to close on the emotional beat",
        avoid=("movement without a beat",),
        requires_capabilities=tuple(capabilities),
        conflicts_with=(),
        source=KnowledgeSource(
            kind="core", path=f"{entry_id}.yaml", sha256=entry_id
        ),
    )


ENTRIES = [
    _entry(
        "camera.intimate_reveal.push_in",
        intents=("realization",),
        capabilities=("camera_motion",),
    ),
    _entry("camera.locked_observation", intents=("observation",)),
    _entry(
        "lighting.soft_motivation",
        domain="lighting",
        intents=("realization",),
    ),
]

QUERY = KnowledgeQuery(
    stage="storyboard",
    domains=("camera_movement", "lighting"),
    intent_text="quiet realization",
    intents=("realization",),
    style="cinematic",
    format_name="short_film",
    language="en",
    capabilities=frozenset({"camera_motion"}),
    hard_avoidances=(),
)


def _project(tmp_path):
    return Project.create("idea", root=tmp_path, stages=["storyboard"])


def test_retrieval_filters_capabilities_and_ranks_intent():
    ranked = retrieve(ENTRIES, QUERY, limit=3)

    assert ranked[0].entry.id == "camera.intimate_reveal.push_in"
    assert "intent" in ranked[0].reasons
    assert all(
        set(item.entry.requires_capabilities) <= {"camera_motion"}
        for item in ranked
    )


def test_retrieval_excludes_missing_capability():
    query = KnowledgeQuery(
        **{**QUERY.to_dict(), "capabilities": frozenset()}
    )

    ranked = retrieve(ENTRIES, query, limit=6)

    assert "camera.intimate_reveal.push_in" not in {
        item.entry.id for item in ranked
    }


def test_retrieval_diversifies_one_domain_to_two_entries():
    entries = ENTRIES + [
        _entry("camera.third", intents=("realization",)),
        _entry("camera.fourth", intents=("realization",)),
    ]

    ranked = retrieve(entries, QUERY, limit=6)

    camera_entries = [
        item for item in ranked if item.entry.domain == "camera_movement"
    ]
    assert len(camera_entries) == 2
    assert any(item.entry.domain == "lighting" for item in ranked)


def test_existing_packet_is_reused_when_only_corpus_changes(tmp_path):
    project = _project(tmp_path)
    first = get_or_create_packet(
        project,
        purpose="shot",
        target="sh-001",
        query=QUERY,
        entries=ENTRIES,
    )
    changed_entries = ENTRIES + [
        _entry("camera.newer", intents=("realization",))
    ]

    second = get_or_create_packet(
        project,
        purpose="shot",
        target="sh-001",
        query=QUERY,
        entries=changed_entries,
    )

    assert second == first
    assert project.path(
        "knowledge", "packets", "shot-sh-001.json"
    ).is_file()


def test_changed_direct_query_replaces_packet(tmp_path):
    project = _project(tmp_path)
    first = get_or_create_packet(
        project,
        purpose="shot",
        target="sh-001",
        query=QUERY,
        entries=ENTRIES,
    )
    changed = KnowledgeQuery(
        **{**QUERY.to_dict(), "intent_text": "detached observation", "intents": ("observation",)}
    )

    second = get_or_create_packet(
        project,
        purpose="shot",
        target="sh-001",
        query=changed,
        entries=ENTRIES,
    )

    assert second["input_fingerprint"] != first["input_fingerprint"]
    stored = json.loads(
        project.path("knowledge", "packets", "shot-sh-001.json").read_text()
    )
    assert stored == second


def test_refresh_removes_only_selected_packet(tmp_path):
    project = _project(tmp_path)
    get_or_create_packet(
        project, purpose="shot", target="sh-001", query=QUERY, entries=ENTRIES
    )
    get_or_create_packet(
        project, purpose="shot", target="sh-002", query=QUERY, entries=ENTRIES
    )

    removed = refresh_packet(project, purpose="shot", target="sh-001")

    assert removed == ["knowledge/packets/shot-sh-001.json"]
    assert not project.path("knowledge", "packets", "shot-sh-001.json").exists()
    assert project.path("knowledge", "packets", "shot-sh-002.json").is_file()
