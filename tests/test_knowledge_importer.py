import json

import pytest

from studio_agent.knowledge.importer import import_knowledge_file, rebuild_index


def test_markdown_headings_become_hashed_entries(tmp_path):
    source = tmp_path / "lighting.md"
    source.write_text(
        "# Motivated light\nUse a visible source.\n\n"
        "## Avoid\nDo not flatten faces."
    )

    entries = import_knowledge_file(
        source,
        root=tmp_path / "knowledge",
        metadata={
            "domain": "lighting",
            "stages": ["bible", "storyboard"],
            "language": "en",
        },
    )

    assert [entry.title for entry in entries] == ["Motivated light", "Avoid"]
    assert entries[0].domain == "lighting"
    assert entries[0].source.sha256
    assert (tmp_path / "knowledge" / "imports" / "manifest.json").is_file()


def test_untagged_plain_text_stays_general(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Foreground occlusion can create unease without camera motion.")

    entry = import_knowledge_file(
        source, root=tmp_path / "knowledge", metadata={}
    )[0]

    assert entry.domain == "general"
    assert entry.stages == ()
    assert entry.language == "und"


def test_unsupported_import_type_is_rejected(tmp_path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"not a supported import")

    with pytest.raises(ValueError, match="Markdown or plain text"):
        import_knowledge_file(source, root=tmp_path / "knowledge", metadata={})


def test_invalid_chunk_is_quarantined_and_reported(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("# Note\nA useful note.")

    entries = import_knowledge_file(
        source,
        root=tmp_path / "knowledge",
        metadata={"domain": "not valid!"},
    )

    assert entries == []
    report = json.loads(
        (tmp_path / "knowledge" / "imports" / "import-report.json").read_text()
    )
    assert report["errors"][0]["chunk"] == 1
    assert list((tmp_path / "knowledge" / "imports" / "quarantine").glob("*.md"))


def test_rebuild_index_contains_imported_entries(tmp_path):
    root = tmp_path / "knowledge"
    source = tmp_path / "notes.txt"
    source.write_text("Foreground occlusion creates depth.")
    import_knowledge_file(source, root=root, metadata={"language": "en"})

    index_path = rebuild_index(root)
    index = json.loads(index_path.read_text())

    assert index["version"] == 1
    assert len(index["entries"]) == 1
