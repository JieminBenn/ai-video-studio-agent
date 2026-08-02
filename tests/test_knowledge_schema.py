import json

import pytest

from studio_agent.knowledge.schema import (
    KnowledgeEntry,
    load_knowledge_entries,
    load_packaged_core,
)


def test_loads_valid_core_yaml_entry(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    (core / "lighting.yaml").write_text("""
entries:
  - id: lighting.motivated.single_source
    domain: lighting
    stages: [bible, storyboard]
    intents: [intimacy]
    styles: [cinematic]
    formats: [short_film]
    language: en
    title: Motivated single-source light
    principle: Let one believable source establish direction and contrast.
    use_when: The scene should feel grounded and intimate.
    recipe: Name the source, direction, softness, color, and falloff.
    avoid: [flat omnidirectional light]
    requires_capabilities: []
    conflicts_with: []
""")

    entries = load_knowledge_entries(tmp_path)

    assert len(entries) == 1
    assert isinstance(entries[0], KnowledgeEntry)
    assert entries[0].domain == "lighting"
    assert entries[0].source.path == "core/lighting.yaml"
    assert entries[0].source.sha256


def test_malformed_core_entry_fails_with_source_path(tmp_path):
    core = tmp_path / "core"
    core.mkdir()
    (core / "bad.yaml").write_text("entries:\n  - id: bad entry id\n    domain: lighting\n")

    with pytest.raises(ValueError, match="core/bad.yaml"):
        load_knowledge_entries(tmp_path)


def test_import_entries_json_round_trips(tmp_path):
    entry = KnowledgeEntry.from_dict({
        "id": "import.abc.001",
        "domain": "general",
        "stages": [],
        "intents": [],
        "styles": [],
        "formats": [],
        "language": "zh",
        "title": "运镜笔记",
        "principle": "镜头运动必须服务情绪节点。",
        "use_when": "",
        "recipe": "先确定起幅和落幅。",
        "avoid": [],
        "requires_capabilities": [],
        "conflicts_with": [],
        "source": {"kind": "import", "path": "imports/abc/source.md", "sha256": "abc"},
    })

    assert KnowledgeEntry.from_dict(json.loads(json.dumps(entry.to_dict()))) == entry


def test_packaged_core_covers_all_domains_and_is_bilingual():
    entries = load_packaged_core()

    assert len(entries) >= 24
    assert {entry.domain for entry in entries} >= {
        "camera_movement",
        "lens_angle",
        "lighting",
        "composition",
        "identity",
        "performance",
        "prompting",
        "editing",
    }
    foundations = [
        entry for entry in entries if entry.source.path.endswith("foundations.yaml")
    ]
    assert foundations
    for entry_id in {
        entry.id.rsplit(".", 1)[0] for entry in foundations
    }:
        languages = {
            entry.language
            for entry in foundations
            if entry.id.rsplit(".", 1)[0] == entry_id
        }
        assert languages == {"en", "zh"}
