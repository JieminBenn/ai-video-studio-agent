import json

import pytest

import studio_agent.reference_assets as reference_assets
from studio_agent.orchestrator.project import Project
from studio_agent.reference_assets import (
    cap_reference_paths,
    cap_references,
    natural_aliases_for_index,
    next_reference_alias,
    load_reference_manifest,
    mark_reference_upload_revised,
    reference_intent_context,
    reference_paths_for_target,
    reference_paths_in_text,
    regeneration_reference_images,
    resolve_pending_reference_targets,
    retarget_reference,
    save_reference_intake_uploads,
    save_reference_upload,
    style_anchor_paths,
    style_reference_paths,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"


def test_save_reference_upload_writes_manifest_and_file(tmp_path):
    p = Project.create("reference upload", root=tmp_path, stages=["storyboard"])

    record = save_reference_upload(
        p,
        filename="hero-face.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="Mara",
        label="hero face",
        note="match this protagonist",
    )

    assert record["target_type"] == "character"
    assert record["target_id"] == "Mara"
    assert record["alias"] == "@image1"
    assert {"photo1", "image 1", "the image"} <= set(record["aliases"])
    assert record["path"].startswith("references/uploads/")
    assert p.path(*record["path"].split("/")).read_bytes() == PNG_BYTES
    manifest = load_reference_manifest(p)
    assert manifest == [record]
    raw = json.loads(p.path("references", "references.json").read_text())
    assert raw["references"][0]["label"] == "hero face"
    assert raw["references"][0]["status"] == "resolved"


def test_reference_aliases_continue_across_targeted_and_intake_batches(tmp_path):
    p = Project.create("reference alias sequence", root=tmp_path, stages=["storyboard"])
    first = save_reference_upload(
        p,
        filename="mara.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="Mara",
    )

    later = save_reference_intake_uploads(
        p,
        [
            {"filename": "lee.png", "data": PNG_BYTES, "content_type": "image/png"},
            {"filename": "street.png", "data": PNG_BYTES, "content_type": "image/png"},
        ],
    )

    assert first["alias"] == "@image1"
    assert [record["alias"] for record in later] == ["@image2", "@image3"]
    assert [record["alias"] for record in load_reference_manifest(p)] == [
        "@image1", "@image2", "@image3",
    ]


def test_reference_alias_sequence_continues_after_legacy_photo_alias(tmp_path):
    p = Project.create("legacy reference alias sequence", root=tmp_path, stages=["storyboard"])
    p.path("references", "references.json").write_text(json.dumps({
        "references": [{"alias": "@photo7", "aliases": ["photo7", "image7"]}],
    }))

    record = save_reference_upload(
        p,
        filename="new.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="Mara",
    )

    assert record["alias"] == "@image8"


def test_style_reference_is_not_image_numbered_and_subjects_start_at_one(tmp_path):
    # Bug: the style upload is saved first and used to claim @image1, shifting the user's
    # character/location references by one (displayed @image1 at upload, persisted @image2).
    # The style/global image must not consume a subject @imageN slot.
    p = Project.create("style alias consistency", root=tmp_path, stages=["storyboard"])
    style = save_reference_upload(
        p, filename="mood.png", data=PNG_BYTES, target_type="style", target_id="global"
    )
    char = save_reference_upload(
        p, filename="hero.png", data=PNG_BYTES, target_type="character", target_id="Mara"
    )

    assert style["alias"] == "@style"
    assert "@image1" not in style["aliases"]
    assert char["alias"] == "@image1"            # subject starts at 1 despite style saved first
    assert next_reference_alias(p) == "@image2"


def test_reference_paths_match_target_slug_and_style(tmp_path):
    p = Project.create("reference matching", root=tmp_path, stages=["storyboard"])
    char = save_reference_upload(
        p,
        filename="mara.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="Mara Lee",
    )
    style = save_reference_upload(
        p,
        filename="mood.png",
        data=PNG_BYTES,
        target_type="style",
        target_id="global",
    )

    assert reference_paths_for_target(p, "character", "mara-lee") == [char["path"]]
    assert style_reference_paths(p) == [style["path"]]


def test_style_anchors_exclude_raw_uploaded_style_photo_content(tmp_path):
    p = Project.create("style anchor safety", root=tmp_path, stages=["style", "bible"])
    style = save_reference_upload(
        p,
        filename="style-with-people.png",
        data=PNG_BYTES,
        target_type="style",
        target_id="global",
    )

    assert style_reference_paths(p) == [style["path"]]
    assert style_anchor_paths(p) == []

    p.path("bible", "style_sample.png").write_bytes(PNG_BYTES)

    assert style_anchor_paths(p) == ["bible/style_sample.png"]


def test_unresolved_references_do_not_match_targets(tmp_path):
    p = Project.create("unresolved reference matching", root=tmp_path, stages=["storyboard"])
    records = save_reference_intake_uploads(
        p,
        [{"filename": "unknown.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="",
    )

    assert records[0]["status"] == "unresolved"
    assert reference_paths_for_target(p, "character", "unknown") == []


def test_chinese_character_note_attaches_reference_to_lead(tmp_path):
    # Invariant #10: a Chinese intent note must attach the upload to the character. The
    # user does not know the generated name at intake, so a generic "这个女子角色" maps to
    # the lead character.
    p = Project.create("中文角色参考", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_intake_uploads(
        p,
        [{"filename": "lady.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="@image1 是这个女子角色",
    )

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "林月", "role": "主角"}],
        locations=[],
    )

    stored = load_reference_manifest(p)[0]
    assert stored["status"] == "resolved"
    assert stored["target_type"] == "character"
    assert stored["target_id"] == "林月"
    assert reference_paths_for_target(p, "character", "林月") == [stored["path"]]


def test_generic_english_character_note_attaches_to_lead(tmp_path):
    p = Project.create("generic character ref", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_intake_uploads(
        p,
        [{"filename": "her.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="@image1 is the character",
    )

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "Mara", "role": "lead"}, {"name": "Otto", "role": "side"}],
        locations=[],
    )

    stored = load_reference_manifest(p)[0]
    assert stored["status"] == "resolved"
    assert stored["target_id"] == "Mara"


def test_chinese_location_note_attaches_reference_to_location(tmp_path):
    p = Project.create("中文场景参考", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_intake_uploads(
        p,
        [{"filename": "room.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="@image1 是公寓的场景",
    )

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "林月", "role": "主角"}],
        locations=[{"name": "公寓"}],
    )

    stored = load_reference_manifest(p)[0]
    assert stored["status"] == "resolved"
    assert stored["target_type"] == "location"
    assert stored["target_id"] == "公寓"


def test_chinese_style_note_resolves_style_reference(tmp_path):
    p = Project.create("中文风格参考", root=tmp_path, stages=["bible", "storyboard"])
    records = save_reference_intake_uploads(
        p,
        [{"filename": "look.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="@image1 是画风参考",
    )

    assert records[0]["target_type"] == "style"
    assert records[0]["target_id"] == "global"
    assert records[0]["status"] == "resolved"


class _MappingLLM:
    """Stub LLM that returns a fixed reference→target mapping and records its calls."""

    def __init__(self, mappings):
        self._mappings = mappings
        self.calls = []

    def complete_json(self, prompt, *, system=None):
        from studio_agent.providers.base import Generation

        self.calls.append(prompt)
        return Generation(
            content={"mappings": self._mappings},
            provider="stub", model="stub", cost_usd=0.0, seconds=0.0,
        )


def test_llm_resolver_maps_note_the_keyword_parser_misses(tmp_path):
    # A free-form note with no mapping verb / known vocab leaves the deterministic parser
    # stuck; the LLM resolver maps it to the real character name using the cast list.
    p = Project.create("llm ref map", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_intake_uploads(
        p,
        [{"filename": "girl.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="把这张参考用在小美身上",
    )
    assert load_reference_manifest(p)[0]["status"] == "unresolved"

    llm = _MappingLLM([{
        "alias": "@image1",
        "target_type": "character",
        "target_id": "小美",
        "confidence": 0.95,
    }])
    resolve_pending_reference_targets(
        p, characters=[{"name": "小美", "role": "主角"}], locations=[], llm=llm,
    )

    stored = load_reference_manifest(p)[0]
    assert stored["status"] == "resolved"
    assert stored["target_type"] == "character"
    assert stored["target_id"] == "小美"
    assert llm.calls, "the LLM resolver was not consulted"
    assert reference_paths_for_target(p, "character", "小美") == [stored["path"]]


def test_llm_resolver_does_not_bind_empty_note_without_high_confidence_vision(tmp_path):
    p = Project.create("ungrounded auto mapping", root=tmp_path, stages=["bible"])
    save_reference_intake_uploads(
        p,
        [{"filename": "unknown.png", "data": PNG_BYTES, "content_type": "image/png"}],
    )
    llm = _MappingLLM([{
        "alias": "@image1",
        "target_type": "character",
        "target_id": "Mara",
        "confidence": 0.99,
    }])

    resolve_pending_reference_targets(
        p, characters=[{"name": "Mara"}, {"name": "Theo"}], locations=[], llm=llm,
    )

    assert load_reference_manifest(p)[0]["status"] == "unresolved"


def test_llm_resolver_rejects_target_name_not_in_cast(tmp_path):
    # Safety: the LLM cannot invent a target. An unknown name leaves the record unresolved.
    p = Project.create("llm ref reject", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_intake_uploads(
        p,
        [{"filename": "girl.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="把这张参考用在小美身上",
    )
    llm = _MappingLLM([{
        "alias": "@image1",
        "target_type": "character",
        "target_id": "Nobody",
        "confidence": 0.95,
    }])
    resolve_pending_reference_targets(
        p, characters=[{"name": "小美"}], locations=[], llm=llm,
    )

    assert load_reference_manifest(p)[0]["status"] == "unresolved"


def test_llm_resolver_not_called_when_everything_already_resolved(tmp_path):
    # Cost control: when the deterministic pass resolves all references, skip the LLM.
    p = Project.create("llm ref skip", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="@image1 是主角",
    )
    llm = _MappingLLM([])
    resolve_pending_reference_targets(
        p, characters=[{"name": "小美", "role": "主角"}], locations=[], llm=llm,
    )

    assert load_reference_manifest(p)[0]["status"] == "resolved"
    assert llm.calls == []


def test_reference_upload_archives_stale_outputs_and_marks_downstream_pending(tmp_path):
    p = Project.create(
        "reference invalidation",
        root=tmp_path,
        stages=["plot", "storyboard", "video", "review", "audio", "assemble"],
    )
    output = p.path("output", f"{p.project_id}.mp4")
    output.write_bytes(b"old-paid-output")
    for stage in p.stages:
        p.set_stage_status(stage, "approved")

    invalidated = mark_reference_upload_revised(p)

    assert invalidated == ["storyboard", "video", "review", "audio", "assemble"]
    assert p.current_stage == "storyboard"
    assert p.stage_status("plot") == "approved"
    assert p.stage_status("storyboard") == "pending"
    assert p.stage_status("assemble") == "pending"
    assert output.exists() is False
    archived = list(p.path("history").rglob(f"output/{p.project_id}.mp4"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"old-paid-output"


def test_late_unresolved_upload_restarts_at_bible_for_target_resolution(tmp_path):
    p = Project.create(
        "late unresolved reference",
        root=tmp_path,
        stages=["plot", "bible", "storyboard", "video"],
    )
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.current_stage = None
    p.status = "done"
    p.save()

    record = save_reference_intake_uploads(
        p,
        [{"filename": "unknown.png", "data": PNG_BYTES, "content_type": "image/png"}],
    )[0]

    assert record["status"] == "unresolved"
    assert p.current_stage == "bible"
    assert p.stage_status("plot") == "approved"
    assert p.stage_status("bible") == "pending"
    assert p.stage_status("storyboard") == "pending"


def test_reference_upload_rejects_bad_image(tmp_path):
    p = Project.create("bad reference", root=tmp_path, stages=["storyboard"])

    with pytest.raises(ValueError, match="valid image"):
        save_reference_upload(
            p,
            filename="not-really.png",
            data=b"not an image",
            target_type="style",
            target_id="global",
        )


def test_natural_aliases_cover_order_and_single_image_language():
    first = natural_aliases_for_index(1, 2)
    second = natural_aliases_for_index(2, 2)
    only = natural_aliases_for_index(1, 1)

    assert {"photo1", "photo 1", "image1", "image 1", "first image", "first photo"} <= set(first)
    assert {"photo2", "photo 2", "image2", "image 2", "second image", "second photo"} <= set(second)
    assert {"the image", "this image", "the photo", "this photo"} <= set(only)


def test_intake_uploads_assign_aliases_and_parse_natural_targets(tmp_path):
    p = Project.create("reference intake", root=tmp_path, stages=["storyboard"])

    records = save_reference_intake_uploads(
        p,
        [
            {"filename": "hero.png", "data": PNG_BYTES, "content_type": "image/png"},
            {"filename": "apartment.png", "data": PNG_BYTES, "content_type": "image/png"},
        ],
        note="first image is protagonist; second image is the apartment",
    )

    assert [r["alias"] for r in records] == ["@image1", "@image2"]
    assert "first image" in records[0]["aliases"]
    assert records[0]["status"] == "unresolved"
    assert records[0]["target_type"] == "character"
    assert records[0]["pending_target"] == {"kind": "character_role", "value": "protagonist"}
    assert records[1]["target_type"] == "location"
    assert records[1]["pending_target"] == {"kind": "location", "value": "apartment"}


def test_single_image_singular_language_targets_upload(tmp_path):
    p = Project.create("single image reference", root=tmp_path, stages=["storyboard"])

    records = save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="the image is protagonist",
    )

    assert records[0]["alias"] == "@image1"
    assert "the image" in records[0]["aliases"]
    assert records[0]["target_type"] == "character"
    assert records[0]["pending_target"]["value"] == "protagonist"


def test_intake_uploads_store_relationship_constraints(tmp_path):
    p = Project.create("relationship reference", root=tmp_path, stages=["plot"])

    save_reference_intake_uploads(
        p,
        [
            {"filename": "a.png", "data": PNG_BYTES, "content_type": "image/png"},
            {"filename": "b.png", "data": PNG_BYTES, "content_type": "image/png"},
        ],
        note="photo1 kisses photo2",
    )

    context = reference_intent_context(p)
    assert context["story_constraints"] == [
        {"text": "photo1 kisses photo2", "aliases": ["@image1", "@image2"]}
    ]


class _StaticAnalyzer:
    def __init__(self, *, target_type="style", confidence=0.75, target_id=None):
        self.target_type = target_type
        self.confidence = confidence
        self.target_id = target_id
        self.calls = []

    def analyze(self, image_path, *, aliases, user_note):
        from studio_agent.providers.base import Generation

        self.calls.append({"image_path": image_path, "aliases": aliases, "user_note": user_note})
        return Generation(
            content={
                "target_type": self.target_type,
                "target_id": self.target_id,
                "confidence": self.confidence,
                "reason": "test inference",
                "visual_summary": "test visual summary",
            },
            provider="test",
            model="static-analyzer",
        )


class _SequenceAnalyzer:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def analyze(self, image_path, *, aliases, user_note):
        from studio_agent.providers.base import Generation

        self.calls.append({"image_path": image_path, "aliases": aliases, "user_note": user_note})
        return Generation(content=next(self.results), provider="test", model="sequence-analyzer")


def test_intake_batch_analyzes_each_image_independently(tmp_path):
    p = Project.create("mixed reference inference", root=tmp_path, stages=["storyboard"])
    analyzer = _SequenceAnalyzer([
        {
            "target_type": "character",
            "target_id": "Mara",
            "confidence": 0.91,
            "reason": "person",
        },
        {
            "target_type": "location",
            "target_id": "Clock Shop",
            "confidence": 0.89,
            "reason": "interior",
        },
    ])

    records = save_reference_intake_uploads(
        p,
        [
            {"filename": "one.png", "data": PNG_BYTES, "content_type": "image/png"},
            {"filename": "two.png", "data": PNG_BYTES, "content_type": "image/png"},
        ],
        analyzer=analyzer,
    )

    assert len(analyzer.calls) == 2
    assert [(record["target_type"], record["target_id"]) for record in records] == [
        ("character", "Mara"),
        ("location", "Clock Shop"),
    ]
    assert [record["alias"] for record in records] == ["@image1", "@image2"]
    assert [entry["stage"] for entry in p.cost_log] == ["references", "references"]


def test_manual_batch_target_skips_analyzer(tmp_path):
    p = Project.create("manual reference target", root=tmp_path, stages=["storyboard"])
    analyzer = _StaticAnalyzer(target_type="style", confidence=0.99)

    records = save_reference_intake_uploads(
        p,
        [
            {"filename": "front.png", "data": PNG_BYTES, "content_type": "image/png"},
            {"filename": "side.png", "data": PNG_BYTES, "content_type": "image/png"},
        ],
        analyzer=analyzer,
        target_type="character",
        target_id="Mara",
        label="Mara angles",
    )

    assert analyzer.calls == []
    assert all(record["status"] == "resolved" for record in records)
    assert all(record["target_type"] == "character" for record in records)
    assert all(record["target_id"] == "Mara" for record in records)
    assert all(record["label"] == "Mara angles" for record in records)


def test_invalid_intake_batch_does_not_write_partial_manifest(tmp_path):
    p = Project.create("invalid reference batch", root=tmp_path, stages=["storyboard"])

    with pytest.raises(ValueError, match="valid image"):
        save_reference_intake_uploads(
            p,
            [
                {"filename": "valid.png", "data": PNG_BYTES, "content_type": "image/png"},
                {"filename": "invalid.png", "data": b"not-an-image", "content_type": "image/png"},
            ],
        )

    assert load_reference_manifest(p) == []
    assert list(p.path("references", "uploads").glob("*")) == []


def test_high_confidence_inference_resolves_style_reference(tmp_path):
    p = Project.create("style inference", root=tmp_path, stages=["storyboard"])

    records = save_reference_intake_uploads(
        p,
        [{"filename": "mood.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="",
        analyzer=_StaticAnalyzer(confidence=0.75),
    )

    assert records[0]["status"] == "resolved"
    assert records[0]["target_type"] == "style"
    assert records[0]["target_id"] == "global"
    assert records[0]["inference"]["confidence"] == 0.75
    assert style_reference_paths(p) == [records[0]["path"]]


def test_inferred_visual_label_auto_binds_to_only_compatible_character(tmp_path):
    p = Project.create("pending visual label", root=tmp_path, stages=["bible", "storyboard"])
    records = save_reference_intake_uploads(
        p,
        [{"filename": "person.png", "data": PNG_BYTES, "content_type": "image/png"}],
        analyzer=_StaticAnalyzer(
            target_type="character", target_id="young woman", confidence=0.92
        ),
    )

    assert records[0]["status"] == "unresolved"
    assert records[0]["pending_target"] == {
        "kind": "character",
        "value": "young woman",
    }

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "Mara", "role": "lead"}],
        locations=[],
    )

    stored = load_reference_manifest(p)[0]
    assert stored["status"] == "resolved"
    assert stored["target_id"] == "Mara"


def test_inferred_visual_label_auto_binds_to_only_compatible_location(tmp_path):
    p = Project.create("pending location label", root=tmp_path, stages=["bible"])
    save_reference_intake_uploads(
        p,
        [{"filename": "interior.png", "data": PNG_BYTES, "content_type": "image/png"}],
        analyzer=_StaticAnalyzer(
            target_type="location", target_id="warm interior", confidence=0.90
        ),
    )

    resolve_pending_reference_targets(
        p,
        characters=[],
        locations=[{"name": "Clock Shop"}],
    )

    stored = load_reference_manifest(p)[0]
    assert stored["status"] == "resolved"
    assert stored["target_id"] == "Clock Shop"


def test_llm_resolver_applies_high_confidence_semantic_match(tmp_path):
    p = Project.create("semantic match", root=tmp_path, stages=["bible"])
    save_reference_intake_uploads(
        p,
        [{"filename": "person.png", "data": PNG_BYTES, "content_type": "image/png"}],
        analyzer=_StaticAnalyzer(
            target_type="character", target_id="woman in red", confidence=0.93
        ),
    )
    llm = _MappingLLM([{
        "alias": "@image1",
        "target_type": "character",
        "target_id": "Mara",
        "confidence": 0.91,
    }])

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "Mara"}, {"name": "Theo"}],
        locations=[],
        llm=llm,
    )

    assert load_reference_manifest(p)[0]["target_id"] == "Mara"
    assert 'vision={"confidence": 0.93' in llm.calls[0]
    assert "note or high-confidence vision evidence" in llm.calls[0]


def test_llm_resolver_rejects_low_confidence_semantic_match(tmp_path):
    p = Project.create("ambiguous match", root=tmp_path, stages=["bible"])
    save_reference_intake_uploads(
        p,
        [{"filename": "person.png", "data": PNG_BYTES, "content_type": "image/png"}],
        analyzer=_StaticAnalyzer(
            target_type="character", target_id="young person", confidence=0.90
        ),
    )
    llm = _MappingLLM([{
        "alias": "@image1",
        "target_type": "character",
        "target_id": "Mara",
        "confidence": 0.74,
    }])

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "Mara"}, {"name": "Theo"}],
        locations=[],
        llm=llm,
    )

    assert load_reference_manifest(p)[0]["status"] == "unresolved"


def test_llm_resolver_cannot_override_low_confidence_analyzer_inference(tmp_path):
    p = Project.create("weak visual evidence", root=tmp_path, stages=["bible"])
    save_reference_intake_uploads(
        p,
        [{"filename": "unknown.png", "data": PNG_BYTES, "content_type": "image/png"}],
        analyzer=_StaticAnalyzer(
            target_type="character", target_id="possibly a person", confidence=0.20
        ),
    )
    llm = _MappingLLM([{
        "alias": "@image1",
        "target_type": "character",
        "target_id": "Mara",
        "confidence": 0.99,
    }])

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "Mara"}, {"name": "Theo"}],
        locations=[],
        llm=llm,
    )

    stored = load_reference_manifest(p)[0]
    assert stored["inference"]["confidence"] == 0.20
    assert stored["status"] == "unresolved"
    assert llm.calls, "the mapper should inspect but not override weak visual evidence"


def test_inferred_exact_name_resolves_against_canonical_bible_target(tmp_path):
    p = Project.create("pending exact name", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_intake_uploads(
        p,
        [{"filename": "mara.png", "data": PNG_BYTES, "content_type": "image/png"}],
        analyzer=_StaticAnalyzer(target_type="character", target_id="Mara", confidence=0.92),
    )

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "Mara", "role": "lead"}],
        locations=[],
    )

    stored = load_reference_manifest(p)[0]
    assert stored["status"] == "resolved"
    assert stored["target_id"] == "Mara"
    assert reference_paths_for_target(p, "character", "Mara") == [stored["path"]]


def test_low_confidence_inference_stays_unresolved(tmp_path):
    p = Project.create("low confidence inference", root=tmp_path, stages=["storyboard"])

    records = save_reference_intake_uploads(
        p,
        [{"filename": "unknown.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="",
        analyzer=_StaticAnalyzer(confidence=0.74),
    )

    assert records[0]["status"] == "unresolved"
    assert records[0]["inference"]["confidence"] == 0.74
    assert style_reference_paths(p) == []


def test_explicit_mapping_overrides_analyzer(tmp_path):
    p = Project.create("explicit beats inference", root=tmp_path, stages=["storyboard"])
    analyzer = _StaticAnalyzer(target_type="style", confidence=0.95)

    records = save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="image 1 is protagonist",
        analyzer=analyzer,
    )

    assert analyzer.calls == []
    assert records[0]["target_type"] == "character"
    assert records[0]["pending_target"]["kind"] == "character_role"


def test_missing_natural_alias_records_warning(tmp_path):
    p = Project.create("missing alias warning", root=tmp_path, stages=["storyboard"])

    save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="the second one is the apartment",
    )

    context = reference_intent_context(p)
    assert any("second" in warning.lower() for warning in context["warnings"])


def test_retarget_reference_updates_manifest_without_rewriting_file(tmp_path):
    p = Project.create("retarget reference", root=tmp_path, stages=["storyboard"])
    records = save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="",
    )
    path = p.path(*records[0]["path"].split("/"))
    before = path.read_bytes()

    updated = retarget_reference(p, records[0]["id"], "character", "Mara")

    assert updated["status"] == "resolved"
    assert updated["target_type"] == "character"
    assert updated["target_id"] == "Mara"
    assert path.read_bytes() == before
    assert reference_paths_for_target(p, "character", "Mara") == [updated["path"]]


def test_style_anchor_includes_generated_sample_when_present(tmp_path):
    p = Project.create("anchor", root=tmp_path, stages=["bible"])
    # No uploaded style ref, no sample yet -> empty.
    assert style_anchor_paths(p) == []
    # Generated sample exists -> it is the anchor.
    sample = p.path("bible", "style_sample.png")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"\x89PNG\r\n\x1a\nsample")
    assert style_anchor_paths(p) == ["bible/style_sample.png"]


def test_style_anchor_keeps_uploaded_refs_out_of_downstream_anchors(tmp_path):
    p = Project.create("anchor2", root=tmp_path, stages=["bible"])
    style = save_reference_upload(
        p, target_type="style", target_id="global",
        filename="ref.png", data=b"\x89PNG\r\n\x1a\nstyle", content_type="image/png",
    )
    sample = p.path("bible", "style_sample.png")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"\x89PNG\r\n\x1a\nsample")
    paths = style_anchor_paths(p)
    assert style_reference_paths(p) == [style["path"]]
    assert paths == ["bible/style_sample.png"]


def test_cap_references_drops_style_first_and_dedupes():
    subjects = ["a.png", "b.png", "c.png"]
    styles = ["bible/style_sample.png"]
    # 4 refs capped to 3 -> style anchor dropped, subjects kept in order
    assert cap_references(subjects, styles, 3) == ["a.png", "b.png", "c.png"]
    # under budget -> everything kept, style last
    assert cap_references(["a.png"], styles, 3) == ["a.png", "bible/style_sample.png"]
    # dedupe (style also present among subjects)
    assert cap_references(["a.png", "a.png"], [], 3) == ["a.png"]
    # no cap
    assert cap_references(subjects, styles, 0) == ["a.png", "b.png", "c.png", "bible/style_sample.png"]


def test_cap_reference_paths_dual_form_anchor_classification(tmp_path):
    """cap_reference_paths must classify the style anchor in BOTH relative and absolute form.

    The dual-form line ``anchors |= {str(project.dir / rel) for rel in anchors}``
    ensures callers that pass absolute paths (e.g. bible stage) also have the anchor
    shed first when over budget. The discriminating assertion places the absolute anchor
    FIRST in the input list with max_n=1 and one subject: without the dual-form line the
    anchor would be misclassified as a subject and selected (wrong); with the line it is
    recognised as style and shed, keeping the subject (correct).
    """
    p = Project.create("cap-ref-paths", root=tmp_path, stages=["bible"])

    # Create bible/style_sample.png so style_anchor_paths returns it.
    sample = p.path("bible", "style_sample.png")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"\x89PNG\r\n\x1a\nsample")

    rel_anchor = "bible/style_sample.png"
    abs_anchor = str(p.dir / rel_anchor)
    subject_a = "references/uploads/a.png"
    subjects = ["references/uploads/a.png", "references/uploads/b.png", "references/uploads/c.png"]

    # --- over-budget: 3 subjects + relative anchor, max_n=3 → anchor shed, subjects kept ---
    result_rel = cap_reference_paths(p, subjects + [rel_anchor], max_n=3)
    assert result_rel == subjects, (
        "relative-form anchor should be dropped when over budget, keeping all 3 subjects"
    )

    # --- over-budget: absolute anchor FIRST, 1 subject, max_n=1 — THE DUAL-FORM DISCRIMINATOR ---
    # Without the dual-form line: abs_anchor classified as subject → selected (wrong).
    # With the dual-form line: abs_anchor classified as style → shed, subject kept (correct).
    result_abs_first = cap_reference_paths(p, [abs_anchor, subject_a], max_n=1)
    assert result_abs_first == [subject_a], (
        "absolute-form anchor placed first must be classified as style and shed, "
        "keeping the subject — this fails if the dual-form line is removed"
    )

    # --- over-budget: 3 subjects + absolute anchor, max_n=3 → absolute anchor shed ---
    result_abs = cap_reference_paths(p, subjects + [abs_anchor], max_n=3)
    assert result_abs == subjects, (
        "absolute-form anchor must also be classified as a style anchor and dropped first"
    )

    # --- under-budget: 2 subjects + anchor, max_n=4 → anchor kept and ordered last ---
    two_subjects = subjects[:2]

    result_under_rel = cap_reference_paths(p, two_subjects + [rel_anchor], max_n=4)
    assert result_under_rel == two_subjects + [rel_anchor], (
        "under budget with relative anchor: anchor should be kept and placed last"
    )

    result_under_abs = cap_reference_paths(p, two_subjects + [abs_anchor], max_n=4)
    assert result_under_abs == two_subjects + [abs_anchor], (
        "under budget with absolute anchor: anchor should be kept and placed last"
    )


def test_cap_reference_paths_orders_bible_sheet_ahead_of_raw_upload(tmp_path):
    """The bible-generated character sheet is the downstream identity anchor (invariant #4).

    Regression: raw uploads were ordered ahead of the derived ``bible/characters/.../reference.png``
    sheet, so reference-conditioned image models keyed identity on the ORIGINAL upload and ignored
    a regenerated bible character — "did not use my generated character, stuck to the initial bible."
    The derived bible sheet must lead so a regenerated sheet actually drives downstream keyframes.
    """
    p = Project.create("cap-order", root=tmp_path, stages=["bible"])

    upload = "references/uploads/original-photo.jpeg"
    char_sheet = "bible/characters/character-56fd8bed/reference.png"
    loc_sheet = "bible/locations/location-704416dc/reference.png"

    # Input mirrors live_reference_paths_for_shot: raw upload first, bible sheets after.
    result = cap_reference_paths(p, [upload, char_sheet, loc_sheet], max_n=9)
    assert result.index(char_sheet) < result.index(upload), (
        "derived bible character sheet must lead the raw upload downstream"
    )
    assert result.index(loc_sheet) < result.index(upload), (
        "derived bible location sheet must also lead the raw upload downstream"
    )

    # Explicitly named uploads (user said "match @image1" in feedback) still lead everything.
    named = cap_reference_paths(p, [char_sheet, upload], max_n=9, named_refs=[upload])
    assert named[0] == upload, "an explicitly named upload keeps top priority"


def test_reference_does_not_leak_onto_other_chinese_character(tmp_path):
    # Defect A: distinct CJK names must produce distinct match slugs. `slugify` collapsed
    # every non-ASCII name to the same fallback ("project"), so a reference resolved to one
    # Chinese character silently attached to every other Chinese character/location too.
    p = Project.create("中文双角色", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_upload(
        p,
        filename="lady.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="美女",
    )

    assert reference_paths_for_target(p, "character", "美女") != []
    assert reference_paths_for_target(p, "character", "男生") == []


def test_descriptive_chinese_character_note_binds_to_lead(tmp_path):
    # Defect B: a note that *describes* the subject ("美女") rather than naming a cast member
    # must be rebound to the lead character. It used to resolve to the literal phrase "美女",
    # which only matched a real character by coincidence.
    p = Project.create("中文描述参考", root=tmp_path, stages=["bible", "storyboard"])
    save_reference_intake_uploads(
        p,
        [{"filename": "her.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="@image1是美女",
    )

    resolve_pending_reference_targets(
        p,
        characters=[{"name": "晚晚", "role": "主角"}, {"name": "阿明", "role": "配角"}],
        locations=[],
    )

    stored = load_reference_manifest(p)[0]
    assert stored["status"] == "resolved"
    assert stored["target_type"] == "character"
    assert stored["target_id"] == "晚晚"
    assert reference_paths_for_target(p, "character", "晚晚") == [stored["path"]]
    assert reference_paths_for_target(p, "character", "阿明") == []


def test_regeneration_reference_images_rejects_named_raw_style_alias(tmp_path):
    p = Project.create("regen refs", root=tmp_path, stages=["bible", "storyboard"])
    style = save_reference_upload(
        p, filename="look.png", data=PNG_BYTES, target_type="style", label="style",
    )
    subject = save_reference_upload(
        p, filename="mara.png", data=PNG_BYTES, target_type="character", target_id="Mara",
    )

    subject_abs = str(p.dir / subject["path"])
    images = regeneration_reference_images(
        p, comment="use the @style image again", subject_refs=[subject_abs],
    )

    assert images[0] == subject_abs  # subjects first
    assert str(p.dir / style["path"]) not in images


def test_style_upload_is_named_at_style_and_resolves_from_a_comment(tmp_path):
    p = Project.create("style alias", root=tmp_path, stages=["bible"])
    style = save_reference_upload(
        p, filename="look.png", data=PNG_BYTES, target_type="style", label="style",
    )

    assert style["alias"] == "@style"
    assert reference_paths_in_text(p, "use the @style image again") == [style["path"]]


def test_reference_paths_in_text_filters_aliases_to_subject_types(tmp_path):
    p = Project.create("filtered aliases", root=tmp_path, stages=["storyboard"])
    character = save_reference_upload(
        p,
        filename="mara.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="Mara",
    )
    save_reference_upload(
        p,
        filename="look.png",
        data=PNG_BYTES,
        target_type="style",
        target_id="global",
    )

    assert reference_paths_in_text(
        p,
        "keep @image1 and @style",
        target_types={"character", "location"},
    ) == [character["path"]]


def test_filtered_alias_lookup_rejects_unresolved_upload(tmp_path):
    p = Project.create("unresolved explicit alias", root=tmp_path, stages=["storyboard"])
    unresolved = save_reference_intake_uploads(
        p,
        [{"filename": "unknown.png", "data": PNG_BYTES, "content_type": "image/png"}],
    )[0]

    assert unresolved["status"] == "unresolved"
    assert reference_paths_in_text(
        p,
        f"use {unresolved['alias']} exactly",
        target_types={"character", "location"},
    ) == []


def test_resolved_reference_missing_on_disk_raises_with_manifest_path(tmp_path):
    p = Project.create("missing resolved reference", root=tmp_path, stages=["bible"])
    record = save_reference_upload(
        p,
        filename="mara.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="Mara",
    )
    p.path(*record["path"].split("/")).unlink()

    with pytest.raises(FileNotFoundError, match=record["path"]):
        reference_paths_for_target(p, "character", "Mara")


def test_live_shot_refs_drop_manifest_upload_retargeted_to_another_character(tmp_path):
    p = Project.create("retargeted live refs", root=tmp_path, stages=["storyboard"])
    record = save_reference_upload(
        p,
        filename="mara.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="Mara",
    )
    shot = {
        "characters": ["Mara"],
        "reference_characters": ["Mara"],
        "reference_images": [record["path"]],
        "named_reference_images": [record["path"]],
    }

    retarget_reference(p, record["id"], "character", "Theo")

    assert reference_assets.live_reference_paths_for_shot(
        p, shot, ("reference_images",)
    ) == []
    stale = str(p.dir / record["path"])
    assert cap_reference_paths(p, [], 1, named_refs=[stale]) == []


def test_live_shot_refs_include_raw_subject_and_block_raw_style(tmp_path):
    p = Project.create("live refs", root=tmp_path, stages=["storyboard"])
    character = save_reference_upload(
        p,
        filename="mara.png",
        data=PNG_BYTES,
        target_type="character",
        target_id="Mara",
    )
    style = save_reference_upload(
        p,
        filename="look.png",
        data=PNG_BYTES,
        target_type="style",
        target_id="global",
    )
    bible = p.path("bible", "characters", "mara", "reference.png")
    bible.parent.mkdir(parents=True, exist_ok=True)
    bible.write_bytes(PNG_BYTES)
    sample = p.path("bible", "style_sample.png")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(PNG_BYTES)
    shot = {
        "characters": ["Mara"],
        "reference_locations": [],
        "reference_images": [
            "bible/characters/mara/reference.png",
            style["path"],
            str(p.dir / style["path"]),
            "bible/style_sample.png",
        ],
    }

    refs = reference_assets.live_reference_paths_for_shot(
        p, shot, ("reference_images",)
    )

    # The raw subject upload is still included (and leads at the assembly layer), while raw
    # style uploads are blocked from conditioning.
    assert refs[0] == character["path"]
    assert character["path"] in refs
    assert style["path"] not in refs
    assert str(p.dir / style["path"]) not in refs
    # But cap_reference_paths is the final ordering authority: the derived bible character
    # sheet is the downstream identity anchor (invariant #4), so under a tight cap it wins
    # over the raw upload that merely fed it — this is what lets a regenerated bible
    # character actually drive the keyframe instead of the original photo.
    absolute = [str(p.dir / ref) for ref in refs]
    assert cap_reference_paths(p, absolute, 1) == [
        str(p.dir / "bible/characters/mara/reference.png")
    ]
