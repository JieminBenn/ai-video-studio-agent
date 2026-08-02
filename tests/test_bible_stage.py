"""Tests for the bible stage — the consistency anchor (invariant #4).

Generates canonical character descriptions + reference images + locked seeds from the
plot. This is what makes everything downstream consistent.
"""

import json
from pathlib import Path

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.keyframe_prompt import STYLE_REFERENCE_GUARD
from studio_agent.providers.base import Generation
from studio_agent.providers.fake import FakeImageGen, FakeLLM, FakeReferenceAnalyzer
from studio_agent.providers.xai_files import StoredXAIFile
from studio_agent.providers.xai_image import XAIImageGen
from studio_agent.reference_assets import (
    load_reference_manifest,
    reference_paths_for_target,
    save_reference_intake_uploads,
    save_reference_upload,
    style_reference_paths,
)
from studio_agent.stages.base import Providers
from studio_agent.stages.bible import BibleStage, char_slug, location_slug, seed_for
from studio_agent.stages.concept import ConceptStage

PIPELINE = ["plot", "script", "bible"]

PLOT = {
    "logline": "A short film about a clockmaker who repairs memories.",
    "synopsis": "A clockmaker discovers a watch that mends the past.",
    "themes": ["memory", "regret"],
    "characters": [
        {"name": "Mara", "role": "lead", "description": "the clockmaker"},
        {"name": "The Customer", "role": "supporting", "description": "a grieving man"},
    ],
    "arc": [{"episode": 1, "scenes": [{"scene": 1, "summary": "Mara opens the shop."}]}],
}

LOCATION_PLOT = {
    **PLOT,
    "arc": [{"episode": 1, "scenes": [
        {
            "scene": 1,
            "heading": "INT. CLOCK SHOP - DAWN",
            "summary": "Mara opens the shop while dust spins through brass light.",
        },
        {
            "scene": 2,
            "heading": "EXT. ROOFTOP - NIGHT",
            "summary": "Mara tests the repaired watch under city rain.",
        },
    ]}],
}


def _project(tmp_path):
    p = Project.create("a clockmaker who repairs memories", root=tmp_path, stages=PIPELINE)
    p.story_dir.joinpath("idea.md").write_text("a clockmaker who repairs memories")
    p.path("story", "plot.json").write_text(json.dumps(PLOT))
    return p


def _location_project(tmp_path):
    p = Project.create("a clockmaker who repairs memories", root=tmp_path, stages=PIPELINE)
    p.story_dir.joinpath("idea.md").write_text("a clockmaker who repairs memories")
    p.path("story", "plot.json").write_text(json.dumps(LOCATION_PLOT))
    return p


def _xai_limit_location_prompt() -> str:
    prefix = (
        "Location scene sheet for 电脑桌前的房间一角. "
        "现代风格的电竞房，碳纤维纹理电竞桌，霓虹紫与青色灯光，屏幕正对镜头，"
        "冷色黑蓝调，固定相机。 # Location Identity Skill\n"
    )
    remaining = 7896 - len(prefix.encode("utf-8"))
    assert remaining > 0
    prompt = prefix + ("x" * remaining)
    assert len(prompt.encode("utf-8")) == 7896
    return prompt


def _styled_project(tmp_path):
    p = Project.create(
        "an anime clockmaker who repairs memories",
        root=tmp_path,
        stages=PIPELINE,
        model_config={"style": {
            "look": "anime",
            "palette": "moonlit teal and warm brass",
            "aspect_ratio": "16:9",
            "rendering": "clean cel shading",
            "line_style": "crisp expressive ink lines",
        }},
    )
    p.story_dir.joinpath("idea.md").write_text("an anime clockmaker who repairs memories")
    p.path("story", "plot.json").write_text(json.dumps(PLOT))
    return p


def _styled_location_project(tmp_path):
    p = Project.create(
        "an anime clockmaker who repairs memories",
        root=tmp_path,
        stages=PIPELINE,
        model_config={"style": {
            "look": "anime",
            "palette": "moonlit teal and warm brass",
            "aspect_ratio": "16:9",
            "rendering": "clean cel shading",
            "line_style": "crisp expressive ink lines",
        }},
    )
    p.story_dir.joinpath("idea.md").write_text("an anime clockmaker who repairs memories")
    p.path("story", "plot.json").write_text(json.dumps(LOCATION_PLOT))
    return p


def _providers():
    return Providers(llm=FakeLLM(), image=FakeImageGen())


class StructuredLocationLLM(FakeLLM):
    def complete_json(self, prompt, *, system=None):
        result = super().complete_json(prompt, system=system)
        if "[task:location]" not in prompt:
            return result
        return Generation(
            content={
                "description": "A precise recurring set.",
                "palette": ["cold blue", {"accent": "amber"}],
                "materials": ["glass", "concrete"],
                "lighting": "blue dusk",
                "hero_props": [
                    {"name": "chrome table", "continuity": "centered"},
                    "city skyline",
                ],
                "continuity_rules": [
                    {"rule": "table remains centered"},
                    "windows stay unobstructed",
                ],
                "prompt_aliases": ["penthouse"],
            },
            provider="structured-test",
            model="structured-test-1",
            cost_usd=0.0,
            seconds=0.01,
        )


class RecordingImage(FakeImageGen):
    def __init__(self):
        self.calls = []

    def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "out_path": out_path,
            "reference_images": list(reference_images or []),
            "kwargs": dict(kwargs),
        })
        return super().generate(prompt, out_path=out_path,
                                reference_images=reference_images, **kwargs)


class _RecordingXAIFileStore:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def resolve_input(self, path):
        local_path = str(Path(path).resolve())
        if local_path not in self.inputs:
            index = len(self.inputs) + 1
            self.inputs[local_path] = StoredXAIFile(
                file_id=f"file-reference-{index}",
                filename=f"reference-{index}.png",
                content_sha256=f"reference-hash-{index}",
                bytes=Path(path).stat().st_size,
            )
        return self.inputs[local_path]

    def file_id_for(self, path):
        return self.inputs[str(Path(path).resolve())].file_id

    def find_by_filename(self, filename):
        return None

    def output_record(self, key):
        return self.outputs.get(key)

    def mark_output_submitted(self, key, **row):
        self.outputs[key] = {**row, "request_status": "submitted"}

    def mark_output_completed(self, key, *, file_id):
        self.outputs.setdefault(key, {}).update(
            {"file_id": file_id, "request_status": "completed"}
        )

    def poll_for_filename(self, filename, timeout_s):
        return None

    def download(self, file_id, out_path):
        _write_fake_png(f"https://x.ai/files/{file_id}", out_path)

    def submitted_local_path_for(self, filename):
        return next(
            row["local_path"]
            for row in self.outputs.values()
            if row.get("filename") == filename
        )


def _stored_xai_response_transport(calls):
    def request(url, *, api_key, json_body, timeout_s):
        calls.append((url, json.loads(json.dumps(json_body))))
        return {
            "data": [{
                "url": f"https://x.ai/generated/{len(calls)}.png",
                "file_output": {"file_id": f"file-output-{len(calls)}"},
            }],
        }

    return request


def _write_fake_png(_url, out_path):
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"\x89PNG\r\n\x1a\nxai-bible")


def _single_character_project(tmp_path):
    p = _project(tmp_path)
    plot = json.loads(p.path("story", "plot.json").read_text())
    plot["characters"] = plot["characters"][:1]
    p.path("story", "plot.json").write_text(json.dumps(plot))
    return p


def _transformation_project(tmp_path):
    idea = "一个男人在马路上，身上冒出红色气体，变成一个狼人"
    p = Project.create(
        idea,
        root=tmp_path,
        stages=["concept", "bible", "clip", "video", "audio", "assemble"],
        model_config={"language": "zh", "product_format": {"name": "short_video", "mode": "clip"}},
    )
    p.story_dir.joinpath("idea.md").write_text(idea)
    ConceptStage().run(p, Providers(llm=FakeLLM()))
    return p


def test_bible_formats_structured_location_prompt_values(tmp_path):
    p = _location_project(tmp_path)

    result = BibleStage().run(
        p, Providers(llm=StructuredLocationLLM(), image=FakeImageGen())
    )

    assert result.status == "complete"
    location_files = sorted(p.path("bible", "locations").glob("*/location.json"))
    data = json.loads(location_files[0].read_text())
    assert data["hero_props"][0]["name"] == "chrome table"
    assert '\"name\": \"chrome table\"' in data["image_prompt"]


def test_bible_model_sheet_uses_character_seed_and_grid_layout_prompt(tmp_path):
    # The combined model sheet (角色三视图) uses the character seed and contains grid
    # layout keywords — the cinematic director is bypassed for grids.
    p = _project(tmp_path)
    image = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=image))

    char_calls = [c for c in image.calls if "/characters/" in c["out_path"]
                  and c["out_path"].endswith("reference.png")]
    assert len(char_calls) >= 1
    sheet = char_calls[0]
    assert "three-quarter" in sheet["prompt"]
    assert "expression strip" in sheet["prompt"]
    # The model sheet must use the deterministic character seed.
    assert sheet["kwargs"].get("seed") == seed_for("Mara")


def test_model_sheet_prompt_fits_tight_provider_budget():
    # Reproduces project-61171d: the model sheet bypasses the director and interpolates
    # the raw character `description` verbatim. A verbose description (4KB+) blew the
    # prompt past xAI Grok's 8000 utf8-byte limit, and fitting could only shed the single
    # skill section — not enough. The model sheet must stay within a tight image-provider
    # budget on its own (board fields already carry the canonical identity).
    from studio_agent.image_generation import prepare_image_prompt
    from studio_agent.providers.base import ImageCapabilities
    from studio_agent.runtime_skills import load_prompt_skill
    from studio_agent.stages.bible import _model_sheet_prompt

    long_desc = "小雨" * 2500  # ~15000 utf8 bytes of narrative prose
    board = {
        "canonical_face": "a" * 400,
        "canonical_body": "b" * 400,
        "hair": "c" * 400,
        "wardrobe": "d" * 400,
        "palette": "e" * 200,
    }
    character = {"description": long_desc}
    skill = load_prompt_skill("character_identity")
    prompt = _model_sheet_prompt("小雨", board, character, {}, skill)
    prompt = f"{prompt}\n\n{STYLE_REFERENCE_GUARD}"

    caps = ImageCapabilities(max_prompt_length=8000, prompt_length_unit="utf8_bytes")
    prepared = prepare_image_prompt(prompt, caps)  # must not raise ImagePromptLimitError

    assert prepared.measured_submitted <= 8000
    # The full narrative description must not be embedded verbatim.
    assert long_desc not in prompt


class CountingLLM(FakeLLM):
    def __init__(self):
        self.image_prompt_calls = 0

    def complete_json(self, prompt, *, system=None):
        if "[task:image_prompt]" in prompt:
            self.image_prompt_calls += 1
        return super().complete_json(prompt, system=system)


def test_bible_model_sheet_uses_grid_prompt_not_director(tmp_path):
    # The combined model sheet bypasses the cinematic prompt director (which is tuned
    # for single frames) and uses the structural grid prompt directly.
    p = _project(tmp_path)
    image = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=image))

    cdir = p.path("bible", "characters", char_slug("Mara"))
    # The model sheet does not write a reference.prompt.md (no director involved).
    assert not (cdir / "reference.prompt.md").exists()
    ref_call = next(c for c in image.calls if c["out_path"].endswith(
        str(cdir / "reference.png")
    ) or (
        "/characters/" in c["out_path"] and c["out_path"].endswith("reference.png")
    ))
    # Grid layout keywords must be in the prompt
    assert "orthographic turnaround" in ref_call["prompt"]
    assert "expression strip" in ref_call["prompt"]
    # Identity rationale is still written (the packet still runs for state/knowledge use)
    assert (cdir / "identity.rationale.md").is_file()


def test_bible_does_not_regenerate_existing_character_model_sheet(tmp_path):
    # Idempotency: re-running the bible stage when reference.png already exists must NOT
    # generate the character model sheet again — no extra paid image call (invariant #3).
    p = _project(tmp_path)
    image = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=image))
    char_calls_after_first = sum(
        1 for c in image.calls
        if "/characters/" in c["out_path"] and c["out_path"].endswith("reference.png")
    )
    assert char_calls_after_first >= 1  # sanity: at least one sheet was generated

    # Re-run with stage reset to pending — reference.png still on disk.
    p.set_stage_status("bible", "pending")
    BibleStage().run(p, Providers(llm=FakeLLM(), image=image))
    char_calls_after_second = sum(
        1 for c in image.calls
        if "/characters/" in c["out_path"] and c["out_path"].endswith("reference.png")
    )
    assert char_calls_after_second == char_calls_after_first, (
        "character model sheet was re-generated even though reference.png already existed"
    )


def test_bible_writes_style_and_per_character_assets(tmp_path):
    p = _project(tmp_path)

    result = BibleStage().run(p, _providers())

    assert result.status == "complete"
    assert p.path("bible", "style.md").is_file()

    for character in PLOT["characters"]:
        cdir = p.path("bible", "characters", char_slug(character["name"]))
        assert (cdir / "character.json").is_file()
        ref = cdir / "reference.png"
        assert ref.is_file()
        assert ref.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

        cdata = json.loads((cdir / "character.json").read_text())
        for key in ["name", "description", "seed", "reference_image", "image_prompt"]:
            assert key in cdata


def test_bible_style_artifacts_include_extra_preset_guidance(tmp_path):
    p = _styled_project(tmp_path)

    BibleStage().run(p, _providers())

    style_md = p.path("bible", "style.md").read_text()
    assert "Look: anime" in style_md
    assert "Rendering: clean cel shading" in style_md
    assert "Line style: crisp expressive ink lines" in style_md

    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdata = json.loads((cdir / "character.json").read_text())
    assert "clean cel shading" in cdata["image_prompt"]
    assert "crisp expressive ink lines" in cdata["image_prompt"]


def test_bible_locks_a_stable_seed_per_character(tmp_path):
    p = _project(tmp_path)
    BibleStage().run(p, _providers())

    cdir = p.path("bible", "characters", char_slug("Mara"))
    seed = json.loads((cdir / "character.json").read_text())["seed"]
    # Seed is derived deterministically from the character name.
    from studio_agent.stages.bible import seed_for
    assert seed == seed_for("Mara")


def test_bible_logs_image_generation_cost(tmp_path):
    p = _project(tmp_path)
    BibleStage().run(p, _providers())
    assert any(entry["stage"] == "bible" for entry in p.cost_log)


def test_bible_is_idempotent_no_regeneration(tmp_path):
    p = _project(tmp_path)
    providers = _providers()

    BibleStage().run(p, providers)
    p.set_stage_status("bible", "complete")
    cost_after_first = len(p.cost_log)
    ref = p.path("bible", "characters", char_slug("Mara"), "reference.png")
    bytes_before = ref.read_bytes()

    result = BibleStage().run(p, providers)
    assert result.status == "skipped"
    assert len(p.cost_log) == cost_after_first
    assert ref.read_bytes() == bytes_before  # not regenerated


def test_bible_regenerates_missing_reference_without_overwriting_character_json(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    BibleStage().run(p, providers)

    cdir = p.path("bible", "characters", char_slug("Mara"))
    character_json = cdir / "character.json"
    data = json.loads(character_json.read_text())
    data["description"] = "human-edited canonical description"
    data["image_prompt"] = "human-edited reference prompt"
    character_json.write_text(json.dumps(data, indent=2))
    (cdir / "reference.png").unlink()
    cost_after_first = len(p.cost_log)

    result = BibleStage().run(p, providers)

    assert result.status == "complete"
    assert (cdir / "reference.png").is_file()
    assert len(p.cost_log) == cost_after_first + 1
    assert json.loads(character_json.read_text())["description"] == "human-edited canonical description"


def test_bible_writes_identity_board(tmp_path):
    # Identity board is written with canonical look keys; no turnaround/expressions
    # sheet filenames since they are collapsed into the single model sheet at reference.png.
    p = _project(tmp_path)
    BibleStage().run(p, _providers())

    cdir = p.path("bible", "characters", char_slug("Mara"))
    board_path = cdir / "identity_board.json"
    assert board_path.is_file()
    assert not (cdir / "turnaround.png").exists()
    assert not (cdir / "expressions.png").exists()

    board = json.loads(board_path.read_text())
    for key in ["canonical_face", "canonical_body", "hair", "wardrobe",
                "palette", "do", "dont", "prompt_aliases"]:
        assert key in board, f"missing {key}"
    assert "turnaround" not in board
    assert "expressions" not in board
    assert isinstance(board["do"], list) and isinstance(board["prompt_aliases"], list)


def test_bible_keeps_baseline_identity_and_generates_only_endpoint_state_reference(tmp_path):
    p = _transformation_project(tmp_path)
    rec = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=rec))

    cdir = p.path("bible", "characters", char_slug("男人"))
    character = json.loads((cdir / "character.json").read_text())
    board = json.loads((cdir / "identity_board.json").read_text())
    states = json.loads((cdir / "states.json").read_text())

    assert "正在变" not in character["description"]
    assert "mid-transformation" not in json.dumps(board).lower()
    assert states["initial_state"] == "human"
    assert (cdir / "states" / "werewolf" / "reference.png").is_file()
    assert not (cdir / "states" / "gas-onset" / "reference.png").exists()
    assert not (cdir / "states" / "partial-werewolf" / "reference.png").exists()

    endpoint_call = next(
        call for call in rec.calls
        if call["out_path"].endswith("states/werewolf/reference.png")
    )
    assert str(cdir / "reference.png") in endpoint_call["reference_images"]
    assert "完整狼人" in endpoint_call["prompt"] or "werewolf" in endpoint_call["prompt"].lower()


def test_bible_context_only_endpoint_creates_no_second_reference(tmp_path):
    p = _transformation_project(tmp_path)
    states_path = p.path("story", "visual_state_changes.json")
    document = json.loads(states_path.read_text())
    plan = document["characters"][0]
    plan["states"] = [
        {
            "id": "human",
            "label": "正常人类",
            "kind": "base",
            "description": "同一个人",
            "appearance_changes": [],
            "reference_required": True,
        },
        {
            "id": "outside-monitor",
            "label": "显示器外",
            "kind": "endpoint",
            "description": "同一个人现在站在显示器外",
            "appearance_changes": [],
            "reference_required": False,
        },
    ]
    plan["initial_state"] = "human"
    plan["transitions"] = []
    states_path.write_text(json.dumps(document, ensure_ascii=False))
    image = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=image))

    cdir = p.path("bible", "characters", char_slug("男人"))
    assert (cdir / "reference.png").is_file()
    assert not (cdir / "states" / "outside-monitor" / "reference.png").exists()
    assert not any("outside-monitor" in call["out_path"] for call in image.calls)


def test_bible_endpoint_keeps_character_upload_and_style_as_three_xai_file_ids(tmp_path):
    p = _transformation_project(tmp_path)
    uploaded = save_reference_upload(
        p,
        filename="hero.png",
        data=b"\x89PNG\r\n\x1a\nhero",
        target_type="character",
        target_id="男人",
    )
    p.path("bible", "style_sample.png").write_bytes(b"\x89PNG\r\n\x1a\nstyle")
    calls = []
    store = _RecordingXAIFileStore()
    image = XAIImageGen(
        requester=_stored_xai_response_transport(calls),
        downloader=_write_fake_png,
        file_store_factory=lambda _out_path, _api_key: store,
    )

    BibleStage().run(
        p,
        Providers(
            llm=FakeLLM(),
            image=image,
            reference_analyzer=FakeReferenceAnalyzer(),
        ),
    )

    endpoint = next(
        body for url, body in calls
        if url.endswith("/images/edits")
        and "states" in store.submitted_local_path_for(
            body["storage_options"]["filename"]
        )
    )
    assert endpoint["images"] == [
        {
            "file_id": store.file_id_for(
                p.path("bible", "characters", char_slug("男人"), "reference.png")
            )
        },
        {"file_id": store.file_id_for(p.dir / uploaded["path"])},
        {"file_id": store.file_id_for(p.path("bible", "style_sample.png"))},
    ]
    assert "data:image" not in json.dumps(endpoint)


def test_bible_endpoint_state_reference_is_idempotent(tmp_path):
    p = _transformation_project(tmp_path)
    providers = _providers()
    BibleStage().run(p, providers)
    cost_before = len(p.cost_log)

    BibleStage().run(p, providers)

    assert len(p.cost_log) == cost_before


def test_bible_writes_identity_packet_and_rationale(tmp_path):
    p = _project(tmp_path)

    BibleStage().run(p, _providers())

    packet = p.path("knowledge", "packets", "identity-mara.json")
    rationale = p.path(
        "bible", "characters", "mara", "identity.rationale.md"
    )
    board = json.loads(
        p.path("bible", "characters", "mara", "identity_board.json").read_text()
    )
    assert packet.is_file()
    assert rationale.is_file()
    assert "Selected knowledge" in rationale.read_text()
    assert board["identity_signature"]
    assert board["wardrobe_details"]["immutable"]


def test_bible_preflight_asks_only_when_character_direction_is_weak(tmp_path):
    p = _single_character_project(tmp_path)

    result = BibleStage().preflight(
        p, Providers(llm=FakeLLM(), image=FakeImageGen()), auto=False
    )

    assert result.decision_required is True
    assert result.request_path.endswith("story/decisions/bible.json")


def test_bible_writes_location_bibles_and_reference_boards(tmp_path):
    p = _location_project(tmp_path)

    BibleStage().run(p, _providers())

    clock_shop = p.path("bible", "locations", "clock-shop")
    rooftop = p.path("bible", "locations", "rooftop")
    for ldir in (clock_shop, rooftop):
        data = json.loads((ldir / "location.json").read_text())
        assert data["name"]
        assert "description" in data
        assert "palette" in data
        assert "continuity_rules" in data
        assert "reference.png" == data["reference_image"]
        assert "environment_board" not in data
        assert (ldir / "reference.png").is_file()
        assert not (ldir / "environment_board.png").exists()

    assert json.loads((clock_shop / "location.json").read_text())["scene_numbers"] == [1]
    assert json.loads((rooftop / "location.json").read_text())["scene_numbers"] == [2]


def test_bible_location_board_is_idempotent_and_preserves_edits(tmp_path):
    p = _location_project(tmp_path)
    providers = _providers()
    BibleStage().run(p, providers)
    lfile = p.path("bible", "locations", "clock-shop", "location.json")
    data = json.loads(lfile.read_text())
    data["lighting"] = "human-edited dawn light"
    lfile.write_text(json.dumps(data, indent=2))
    cost_before = len(p.cost_log)

    BibleStage().run(p, providers)

    assert json.loads(lfile.read_text())["lighting"] == "human-edited dawn light"
    assert len(p.cost_log) == cost_before


def test_bible_identity_board_is_idempotent_and_preserves_edits(tmp_path):
    p = _project(tmp_path)
    providers = _providers()
    BibleStage().run(p, providers)

    cdir = p.path("bible", "characters", char_slug("Mara"))
    board_path = cdir / "identity_board.json"
    board = json.loads(board_path.read_text())
    board["wardrobe"] = "human-edited wardrobe"
    board_path.write_text(json.dumps(board, indent=2))
    cost_before = len(p.cost_log)

    # Re-run the (not-yet-complete) stage: existing artifacts are not regenerated.
    BibleStage().run(p, providers)
    assert len(p.cost_log) == cost_before  # nothing re-generated
    assert json.loads(board_path.read_text())["wardrobe"] == "human-edited wardrobe"


def test_bible_grounds_identity_in_reference_image_when_vision_available(tmp_path):
    # The bug: the canonical description + identity board were written by a text-only LLM
    # that never saw the uploaded reference, so they contradicted it. When a reference and a
    # vision analyzer are present, the identity must be derived from the IMAGE.
    from studio_agent.providers.base import Generation

    class _VisionAnalyzer(FakeReferenceAnalyzer):
        def __init__(self):
            self.described = []

        def describe(self, image_paths, *, prompt, language="en"):
            self.described.append(list(image_paths))
            return Generation(
                content={
                    "visual_description": "ornate fantasy goddess with a gold headdress",
                    "wardrobe": "gold headdress, red-and-gold armored dress, gold choker, "
                                "earrings and bracelets",
                    "palette": "gold, crimson, ivory",
                    "canonical_face": "pale refined face, red lips, pointed ears",
                    "canonical_body": "tall and slender",
                    "hair": "long straight black hair",
                    "do": ["keep the gold headdress and all gold jewelry"],
                    "dont": ["do not remove the jewelry or armor"],
                    "prompt_aliases": ["goddess"],
                },
                provider="stub", model="stub", cost_usd=0.0, seconds=0.0,
            )

    p = _project(tmp_path)
    save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": b"\x89PNG\r\n\x1a\nhero", "content_type": "image/png"}],
        note="the image is protagonist",
    )
    analyzer = _VisionAnalyzer()

    BibleStage().run(
        p, Providers(llm=FakeLLM(), image=RecordingImage(), reference_analyzer=analyzer)
    )

    cdir = p.path("bible", "characters", char_slug("Mara"))
    char = json.loads((cdir / "character.json").read_text())
    board = json.loads((cdir / "identity_board.json").read_text())
    assert "headdress" in char["wardrobe"]
    assert "headdress" in json.dumps(board, ensure_ascii=False)
    assert analyzer.described, "the vision describe path was not used for the bible identity"


def test_vision_identity_prompt_forbids_deferring_to_image_alias(tmp_path):
    # Defect C: the VLM echoed "参考image2" / "as in image1" into the bible instead of
    # transcribing what it saw, leaving the identity hollow. The identity prompt must tell
    # the model not to refer to the image by its alias and to write the concrete details out.
    from studio_agent.providers.base import Generation

    captured = {}

    class _CapturingAnalyzer(FakeReferenceAnalyzer):
        def describe(self, image_paths, *, prompt, language="en"):
            captured.setdefault("prompt", prompt)
            return Generation(
                content={"visual_description": "x", "wardrobe": "y"},
                provider="stub", model="stub", cost_usd=0.0, seconds=0.0,
            )

    p = _project(tmp_path)
    save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": b"\x89PNG\r\n\x1a\nhero", "content_type": "image/png"}],
        note="the image is protagonist",
    )
    BibleStage().run(
        p, Providers(llm=FakeLLM(), image=RecordingImage(), reference_analyzer=_CapturingAnalyzer())
    )

    assert "Do not refer to the image by its alias" in captured["prompt"]


def test_bible_rejects_qc_shaped_identity_before_paid_image_generation(tmp_path):
    class _QCShapedAnalyzer(FakeReferenceAnalyzer):
        name = "broken-vlm"
        model = "broken-vlm-1"

        def describe(self, image_paths, *, prompt, language="en"):
            return Generation(
                content={"summary": "looks correct", "checks": []},
                provider=self.name,
                model=self.model,
                cost_usd=0.37,
                seconds=1.25,
            )

    p = _single_character_project(tmp_path)
    save_reference_upload(
        p,
        filename="hero.png",
        data=b"\x89PNG\r\n\x1a\nhero",
        target_type="character",
        target_id="Mara",
    )
    images = RecordingImage()

    with pytest.raises(ValueError, match="broken-vlm.*visual_description"):
        BibleStage().run(
            p,
            Providers(
                llm=FakeLLM(),
                image=images,
                reference_analyzer=_QCShapedAnalyzer(),
            ),
        )

    assert images.calls == []
    assert len(p.cost_log) == 1
    assert p.cost_log[0]["provider"] == "broken-vlm"
    assert p.cost_log[0]["cost_usd"] == 0.37
    assert p.cost_log[0]["seconds"] == 1.25


def test_bible_requires_vision_analyzer_for_authoritative_upload(tmp_path):
    p = _single_character_project(tmp_path)
    save_reference_upload(
        p,
        filename="hero.png",
        data=b"\x89PNG\r\n\x1a\nhero",
        target_type="character",
        target_id="Mara",
    )

    with pytest.raises(ValueError, match="vision analyzer.*authoritative"):
        BibleStage().run(p, Providers(llm=FakeLLM(), image=RecordingImage()))


def test_bible_reuses_current_reference_grounded_character_without_analyzer(tmp_path):
    p = _single_character_project(tmp_path)
    record = save_reference_upload(
        p,
        filename="hero.png",
        data=b"\x89PNG\r\n\x1a\nhero",
        target_type="character",
        target_id="Mara",
    )
    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdir.mkdir(parents=True, exist_ok=True)
    character_path = cdir / "character.json"
    character_path.write_text(json.dumps({
        "name": "Mara",
        "description": "reference-grounded heroine",
        "seed": 123,
        "image_prompt": "current grounded prompt",
        "reference_image": "reference.png",
        "source_references": [record["path"]],
    }))
    (cdir / "identity_board.json").write_text(json.dumps({
        "canonical_face": "reference-grounded face",
        "wardrobe": "reference-grounded wardrobe",
    }))
    (cdir / "reference.png").write_bytes(b"\x89PNG\r\n\x1a\ncurrent")
    images = RecordingImage()

    result = BibleStage().run(p, Providers(llm=FakeLLM(), image=images))

    assert result.status == "complete"
    assert json.loads(character_path.read_text())["description"] == (
        "reference-grounded heroine"
    )
    assert not any("/characters/" in call["out_path"] for call in images.calls)


def test_bible_regrounds_legacy_character_when_source_references_are_missing(tmp_path):
    p = _single_character_project(tmp_path)
    record = save_reference_upload(
        p,
        filename="goddess.png",
        data=b"\x89PNG\r\n\x1a\ngoddess",
        target_type="character",
        target_id="Mara",
    )
    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({
        "name": "Mara",
        "description": "invented cyberpunk woman",
        "seed": 123,
        "regen_count": 2,
        "reference_image": "reference.png",
    }))

    BibleStage().run(
        p,
        Providers(
            llm=FakeLLM(),
            image=RecordingImage(),
            reference_analyzer=FakeReferenceAnalyzer(),
        ),
    )

    data = json.loads((cdir / "character.json").read_text())
    assert "goddess" in data["description"]
    assert data["source_references"] == [record["path"]]
    assert data["seed"] == 123
    assert data["regen_count"] == 2


def test_clip_bible_uses_explicit_chinese_upload_instead_of_concept_description(tmp_path):
    idea = "背景是一张POV视角电脑照片，要真实，赛博朋克。把美女从电脑里拉出来。"
    p = Project.create(
        idea,
        root=tmp_path,
        stages=["concept", "bible", "clip"],
        model_config={"language": "zh", "product_format": {"mode": "clip"}},
    )
    p.story_dir.joinpath("idea.md").write_text(idea)
    p.path("story", "plot.json").write_text(json.dumps({
        "characters": [{
            "name": "赛博少女",
            "role": "被从屏幕中拉出的数字生命体",
            "description": "银蓝短发和发光连体衣",
        }],
        "locations": [{"name": "电脑桌", "description": "真实电竞房", "scene": 1}],
    }, ensure_ascii=False))
    record = save_reference_intake_uploads(
        p,
        [{
            "filename": "fantasy-woman.png",
            "data": b"\x89PNG\r\n\x1a\nfantasy",
            "content_type": "image/png",
        }],
        note="@image1 是美女",
    )[0]
    images = RecordingImage()

    BibleStage().run(
        p,
        Providers(
            llm=FakeLLM(),
            image=images,
            reference_analyzer=FakeReferenceAnalyzer(),
        ),
    )

    manifest = load_reference_manifest(p)
    assert manifest[0]["target_id"] == "赛博少女"
    cdir = p.path("bible", "characters", char_slug("赛博少女"))
    character = json.loads((cdir / "character.json").read_text())
    assert "fantasy-woman" in character["description"]
    assert "银蓝短发" not in character["description"]
    sheet_call = next(
        call for call in images.calls
        if call["out_path"].endswith(f"{char_slug('赛博少女')}/reference.png")
    )
    assert str(p.dir / record["path"]) in sheet_call["reference_images"]


def test_bible_regrounding_replaces_stale_character_sheet_once(tmp_path):
    p = _single_character_project(tmp_path)
    record = save_reference_upload(
        p,
        filename="goddess.png",
        data=b"\x89PNG\r\n\x1a\ngoddess",
        target_type="character",
        target_id="Mara",
    )
    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdir.mkdir(parents=True, exist_ok=True)
    character_path = cdir / "character.json"
    character_path.write_text(json.dumps({
        "name": "Mara",
        "description": "invented cyberpunk woman",
        "seed": 123,
        "regen_count": 2,
        "reference_image": "reference.png",
    }))
    (cdir / "identity_board.json").write_text(json.dumps({
        "canonical_face": "stale invented face",
        "wardrobe": "stale cyberpunk wardrobe",
    }))
    ref_path = cdir / "reference.png"
    stale_bytes = b"\x89PNG\r\n\x1a\nstale-character-sheet"
    ref_path.write_bytes(stale_bytes)
    images = RecordingImage()
    providers = Providers(
        llm=FakeLLM(),
        image=images,
        reference_analyzer=FakeReferenceAnalyzer(),
    )

    BibleStage().run(p, providers)

    character_calls = [
        call for call in images.calls
        if call["out_path"].endswith("bible/characters/mara/reference.png")
    ]
    assert len(character_calls) == 1
    assert ref_path.read_bytes() != stale_bytes
    data = json.loads(character_path.read_text())
    assert data["source_references"] == [record["path"]]
    assert data["seed"] == 123
    assert data["regen_count"] == 2

    BibleStage().run(p, providers)
    assert len([
        call for call in images.calls
        if call["out_path"].endswith("bible/characters/mara/reference.png")
    ]) == 1


def test_bible_regrounding_replaces_stale_location_sheet_once(tmp_path):
    p = _location_project(tmp_path)
    record = save_reference_upload(
        p,
        filename="clock-shop.png",
        data=b"\x89PNG\r\n\x1a\nclock-shop",
        target_type="location",
        target_id="Clock Shop",
    )
    ldir = p.path("bible", "locations", location_slug("Clock Shop"))
    ldir.mkdir(parents=True, exist_ok=True)
    location_path = ldir / "location.json"
    location_path.write_text(json.dumps({
        "name": "Clock Shop",
        "description": "invented glass penthouse",
        "seed": 456,
        "regen_count": 3,
        "image_prompt": "stale scene prompt",
        "reference_image": "reference.png",
    }))
    ref_path = ldir / "reference.png"
    stale_bytes = b"\x89PNG\r\n\x1a\nstale-location-sheet"
    ref_path.write_bytes(stale_bytes)
    images = RecordingImage()
    providers = Providers(
        llm=FakeLLM(),
        image=images,
        reference_analyzer=FakeReferenceAnalyzer(),
    )

    BibleStage().run(p, providers)

    location_calls = [
        call for call in images.calls
        if call["out_path"].endswith("bible/locations/clock-shop/reference.png")
    ]
    assert len(location_calls) == 1
    assert ref_path.read_bytes() != stale_bytes
    data = json.loads(location_path.read_text())
    assert data["source_references"] == [record["path"]]
    assert data["seed"] == 456
    assert data["regen_count"] == 3

    BibleStage().run(p, providers)
    assert len([
        call for call in images.calls
        if call["out_path"].endswith("bible/locations/clock-shop/reference.png")
    ]) == 1


def test_bible_retries_stale_character_sheet_without_repeating_vision(tmp_path):
    class _CountingAnalyzer(FakeReferenceAnalyzer):
        name = "counting-character-analyzer"
        model = "counting-character-analyzer-1"

        def __init__(self):
            self.describe_calls = 0

        def describe(self, image_paths, *, prompt, language="en"):
            self.describe_calls += 1
            return super().describe(image_paths, prompt=prompt, language=language)

    class _FlakyCharacterImage(RecordingImage):
        def __init__(self):
            super().__init__()
            self.sheet_attempts = 0

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            if out_path.endswith("bible/characters/mara/reference.png"):
                self.sheet_attempts += 1
                if self.sheet_attempts == 1:
                    raise RuntimeError("temporary character image failure")
            return super().generate(
                prompt,
                out_path=out_path,
                reference_images=reference_images,
                **kwargs,
            )

    p = _single_character_project(tmp_path)
    record = save_reference_upload(
        p,
        filename="goddess.png",
        data=b"\x89PNG\r\n\x1a\ngoddess",
        target_type="character",
        target_id="Mara",
    )
    cdir = p.path("bible", "characters", char_slug("Mara"))
    cdir.mkdir(parents=True, exist_ok=True)
    character_path = cdir / "character.json"
    character_path.write_text(json.dumps({
        "name": "Mara",
        "description": "invented cyberpunk woman",
        "seed": 123,
        "regen_count": 2,
        "reference_image": "reference.png",
    }))
    ref_path = cdir / "reference.png"
    stale_bytes = b"\x89PNG\r\n\x1a\nstale-character-sheet"
    ref_path.write_bytes(stale_bytes)
    analyzer = _CountingAnalyzer()
    images = _FlakyCharacterImage()
    providers = Providers(
        llm=FakeLLM(),
        image=images,
        reference_analyzer=analyzer,
    )

    with pytest.raises(RuntimeError, match="temporary character image failure"):
        BibleStage().run(p, providers)

    failed_data = json.loads(character_path.read_text())
    assert failed_data["source_references"] == [record["path"]]
    assert failed_data["reference_image_stale"] is True
    assert ref_path.read_bytes() == stale_bytes
    assert analyzer.describe_calls == 2
    vision_costs = [
        entry for entry in p.cost_log if entry["provider"] == analyzer.name
    ]
    assert len(vision_costs) == 2

    BibleStage().run(p, providers)

    completed_data = json.loads(character_path.read_text())
    assert completed_data["reference_image_stale"] is False
    assert completed_data["seed"] == 123
    assert completed_data["regen_count"] == 2
    assert ref_path.read_bytes() != stale_bytes
    assert images.sheet_attempts == 2
    assert analyzer.describe_calls == 2
    assert len([
        entry for entry in p.cost_log if entry["provider"] == analyzer.name
    ]) == 2


def test_bible_retries_stale_location_sheet_without_repeating_vision(tmp_path):
    class _CountingAnalyzer(FakeReferenceAnalyzer):
        name = "counting-location-analyzer"
        model = "counting-location-analyzer-1"

        def __init__(self):
            self.describe_calls = 0

        def describe(self, image_paths, *, prompt, language="en"):
            self.describe_calls += 1
            return super().describe(image_paths, prompt=prompt, language=language)

    class _FlakyLocationImage(RecordingImage):
        def __init__(self):
            super().__init__()
            self.sheet_attempts = 0

        def generate(self, prompt, *, out_path, reference_images=None, **kwargs):
            if out_path.endswith("bible/locations/clock-shop/reference.png"):
                self.sheet_attempts += 1
                if self.sheet_attempts == 1:
                    raise RuntimeError("temporary location image failure")
            return super().generate(
                prompt,
                out_path=out_path,
                reference_images=reference_images,
                **kwargs,
            )

    p = _location_project(tmp_path)
    record = save_reference_upload(
        p,
        filename="clock-shop.png",
        data=b"\x89PNG\r\n\x1a\nclock-shop",
        target_type="location",
        target_id="Clock Shop",
    )
    ldir = p.path("bible", "locations", location_slug("Clock Shop"))
    ldir.mkdir(parents=True, exist_ok=True)
    location_path = ldir / "location.json"
    location_path.write_text(json.dumps({
        "name": "Clock Shop",
        "description": "invented glass penthouse",
        "seed": 456,
        "regen_count": 3,
        "image_prompt": "stale scene prompt",
        "reference_image": "reference.png",
    }))
    ref_path = ldir / "reference.png"
    stale_bytes = b"\x89PNG\r\n\x1a\nstale-location-sheet"
    ref_path.write_bytes(stale_bytes)
    analyzer = _CountingAnalyzer()
    images = _FlakyLocationImage()
    providers = Providers(
        llm=FakeLLM(),
        image=images,
        reference_analyzer=analyzer,
    )

    with pytest.raises(RuntimeError, match="temporary location image failure"):
        BibleStage().run(p, providers)

    failed_data = json.loads(location_path.read_text())
    assert failed_data["source_references"] == [record["path"]]
    assert failed_data["reference_image_stale"] is True
    assert ref_path.read_bytes() == stale_bytes
    assert analyzer.describe_calls == 1
    vision_costs = [
        entry for entry in p.cost_log if entry["provider"] == analyzer.name
    ]
    assert len(vision_costs) == 1

    BibleStage().run(p, providers)

    completed_data = json.loads(location_path.read_text())
    assert completed_data["reference_image_stale"] is False
    assert completed_data["seed"] == 456
    assert completed_data["regen_count"] == 3
    assert ref_path.read_bytes() != stale_bytes
    assert images.sheet_attempts == 2
    assert analyzer.describe_calls == 1
    assert len([
        entry for entry in p.cost_log if entry["provider"] == analyzer.name
    ]) == 1


def test_bible_character_sheet_guards_subject_reference_style(tmp_path):
    # When a character has an uploaded subject reference, the model sheet must be told to use
    # it for identity only and re-render in the project style (so a realistic photo converts
    # to the project's look) — without a subject reference the guard must not appear.
    from studio_agent.keyframe_prompt import SUBJECT_REFERENCE_GUARD

    p = _project(tmp_path)
    save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": b"\x89PNG\r\n\x1a\nhero", "content_type": "image/png"}],
        note="the image is protagonist",
    )
    rec = RecordingImage()

    BibleStage().run(
        p,
        Providers(
            llm=FakeLLM(),
            image=rec,
            reference_analyzer=FakeReferenceAnalyzer(),
        ),
    )

    mara_ref_call = next(
        call for call in rec.calls
        if call["out_path"].endswith("bible/characters/mara/reference.png")
    )
    assert mara_ref_call["reference_images"]
    assert SUBJECT_REFERENCE_GUARD in mara_ref_call["prompt"]
    # A different character with no uploaded reference does not get the subject guard.
    other = next(
        call for call in rec.calls
        if "/characters/" in call["out_path"]
        and call["out_path"].endswith("reference.png")
        and not call["out_path"].endswith("bible/characters/mara/reference.png")
    )
    assert SUBJECT_REFERENCE_GUARD not in other["prompt"]


def test_bible_resolves_initial_protagonist_reference_to_lead_character(tmp_path):
    p = _project(tmp_path)
    records = save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": b"\x89PNG\r\n\x1a\nhero", "content_type": "image/png"}],
        note="the image is protagonist",
    )
    rec = RecordingImage()

    BibleStage().run(
        p,
        Providers(
            llm=FakeLLM(),
            image=rec,
            reference_analyzer=FakeReferenceAnalyzer(),
        ),
    )

    manifest = load_reference_manifest(p)
    updated = next(record for record in manifest if record["id"] == records[0]["id"])
    assert updated["status"] == "resolved"
    assert updated["target_type"] == "character"
    assert updated["target_id"] == "Mara"
    assert reference_paths_for_target(p, "character", "Mara") == [updated["path"]]

    mara_ref_call = next(
        call for call in rec.calls
        if call["out_path"].endswith("bible/characters/mara/reference.png")
    )
    assert str(p.dir / updated["path"]) in mara_ref_call["reference_images"]


def test_bible_resolves_initial_location_reference_by_slug(tmp_path):
    p = _location_project(tmp_path)
    records = save_reference_intake_uploads(
        p,
        [{"filename": "clock-shop.png", "data": b"\x89PNG\r\n\x1a\nshop", "content_type": "image/png"}],
        note="first image is the clock shop location",
    )
    rec = RecordingImage()

    BibleStage().run(
        p,
        Providers(
            llm=FakeLLM(),
            image=rec,
            reference_analyzer=FakeReferenceAnalyzer(),
        ),
    )

    updated = next(record for record in load_reference_manifest(p) if record["id"] == records[0]["id"])
    assert updated["status"] == "resolved"
    assert updated["target_type"] == "location"
    assert updated["target_id"] == "Clock Shop"

    loc_call = next(
        call for call in rec.calls
        if call["out_path"].endswith("bible/locations/clock-shop/reference.png")
    )
    assert str(p.dir / updated["path"]) in loc_call["reference_images"]


def test_bible_keeps_ambiguous_location_reference_unresolved(tmp_path):
    p = _location_project(tmp_path)
    records = save_reference_intake_uploads(
        p,
        [{"filename": "apartment.png", "data": b"\x89PNG\r\n\x1a\napt", "content_type": "image/png"}],
        note="the image is the apartment background",
    )
    rec = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=rec))

    updated = next(record for record in load_reference_manifest(p) if record["id"] == records[0]["id"])
    assert updated["status"] == "unresolved"
    assert reference_paths_for_target(p, "location", "Apartment") == []
    assert all(
        str(p.dir / updated["path"]) not in call["reference_images"]
        for call in rec.calls
    )


def test_bible_does_not_pass_raw_style_reference_to_character_sheets(tmp_path):
    # Raw style/global uploads are for style extraction and style-sample generation only.
    # Passing them into character sheets lets their people/subjects bleed into the cast.
    p = _styled_project(tmp_path)
    save_reference_upload(
        p,
        target_type="style",
        target_id="global",
        filename="painted-figure.png",
        data=b"\x89PNG\r\n\x1a\nstyle",
        content_type="image/png",
    )
    rec = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=rec))

    style_path = str(p.dir / style_reference_paths(p)[0])
    sheet_calls = [
        call for call in rec.calls
        if call["out_path"].endswith("reference.png")
    ]
    assert sheet_calls
    assert all(style_path not in call["reference_images"] for call in sheet_calls)


def test_bible_does_not_pass_raw_style_reference_to_location_sheets(tmp_path):
    # Location sheets should get location content refs plus the extracted style text/sample,
    # not the raw style photo's subjects, vehicles, people, or composition.
    p = _styled_location_project(tmp_path)
    save_reference_upload(
        p,
        target_type="style",
        target_id="global",
        filename="painted-figure.png",
        data=b"\x89PNG\r\n\x1a\nstyle",
        content_type="image/png",
    )
    rec = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=rec))

    style_path = str(p.dir / style_reference_paths(p)[0])
    scene_sheets = [
        call for call in rec.calls
        if "/locations/" in call["out_path"]
        and call["out_path"].endswith("reference.png")
    ]
    assert scene_sheets
    assert all(style_path not in call["reference_images"] for call in scene_sheets)


def test_character_bible_emits_one_combined_model_sheet(tmp_path):
    # A character is a single 角色三视图 model sheet (turnaround + expressions in one
    # image), not three separate files. This collapses downstream references to one.
    p = _single_character_project(tmp_path)
    rec = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=rec))

    cdir = next((p.path("bible", "characters")).iterdir())
    assert (cdir / "reference.png").is_file()
    assert not (cdir / "turnaround.png").exists()
    assert not (cdir / "expressions.png").exists()
    # exactly one character image generation (the model sheet); no turnaround/expressions calls
    char_calls = [c for c in rec.calls if "/characters/" in c["out_path"]
                  and c["out_path"].endswith("reference.png")]
    assert len(char_calls) == 1
    assert not any(c["out_path"].endswith(("turnaround.png", "expressions.png"))
                   for c in rec.calls)


def test_location_bible_emits_one_combined_scene_sheet(tmp_path):
    # A location is a single 场景四视图 scene sheet (establishing + alternate angles +
    # material/palette/lighting callouts), not a separate establishing image and board.
    p = _styled_location_project(tmp_path)
    save_reference_upload(
        p, target_type="style", target_id="global",
        filename="painted.png", data=b"\x89PNG\r\n\x1a\nstyle", content_type="image/png",
    )
    rec = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=rec))

    ldir = next((p.path("bible", "locations")).iterdir())
    assert (ldir / "reference.png").is_file()
    assert not (ldir / "environment_board.png").exists()
    ldir_str = str(ldir)
    loc_calls = [c for c in rec.calls if ldir_str in c["out_path"]
                 and c["out_path"].endswith("reference.png")]
    assert len(loc_calls) == 1
    # Raw style upload does not ride along into the scene sheet where its subject can leak.
    style_path = str(p.dir / style_reference_paths(p)[0])
    assert all(style_path not in c["reference_images"] for c in loc_calls)


def test_bible_character_sheet_anchors_on_generated_style_sample(tmp_path):
    p = _single_character_project(tmp_path)
    sample = p.path("bible", "style_sample.png")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"\x89PNG\r\n\x1a\nsample")
    rec = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=rec))

    sample_abs = str(p.dir / "bible/style_sample.png")
    sheet_calls = [
        c for c in rec.calls
        if c["out_path"].endswith("reference.png") and "/characters/" in c["out_path"]
    ]
    assert sheet_calls, "expected a character sheet generation"
    for c in sheet_calls:
        assert sample_abs in c["reference_images"]
        assert "Do not copy its subject, characters, or composition." in c["prompt"]


def test_bible_location_sheet_anchors_on_generated_style_sample(tmp_path):
    p = _location_project(tmp_path)
    style_src = tmp_path / "style-with-people.png"
    FakeImageGen().generate("style people", out_path=str(style_src), seed=4)
    style_record = save_reference_upload(
        p,
        target_type="style",
        target_id="global",
        filename="style-with-people.png",
        data=style_src.read_bytes(),
        content_type="image/png",
    )
    sample = p.path("bible", "style_sample.png")
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_bytes(b"\x89PNG\r\n\x1a\nsample")
    rec = RecordingImage()

    BibleStage().run(p, Providers(llm=FakeLLM(), image=rec))

    sample_abs = str(p.dir / "bible/style_sample.png")
    loc_calls = [
        c for c in rec.calls
        if c["out_path"].endswith("reference.png") and "/locations/" in c["out_path"]
    ]
    assert loc_calls, "expected a location sheet generation"
    for c in loc_calls:
        assert sample_abs in c["reference_images"]
        assert str(p.dir / style_record["path"]) not in c["reference_images"]


def test_style_medium_lead_front_loads_medium_and_guards_substitution():
    # A 2D project's medium must surface as an explicit leading art-style instruction
    # plus a do-not-substitute guard, so a 3D-leaning provider does not silently render
    # the 2D illustration as a 3D model (the reported 2D->3D regression).
    from studio_agent.style import style_medium_lead

    style = {
        "look": "梦幻国漫插画",
        "medium": "2D手绘",
        "idiom": "国漫",
        "rendering": "数字绘画，柔和纹理",
    }
    lead = style_medium_lead(style)
    assert "2D手绘" in lead
    assert "国漫" in lead
    # explicit do-not-substitute guard naming the 3D failure mode
    assert "do not" in lead.lower()
    assert "3d" in lead.lower()


def test_style_medium_lead_is_empty_when_no_medium_descriptors():
    from studio_agent.style import style_medium_lead

    assert style_medium_lead({}) == ""
    assert style_medium_lead({"palette": "stormy blues"}) == ""


def test_model_sheet_prompt_leads_with_medium_before_sheet_framing():
    from studio_agent.stages.bible import _model_sheet_prompt

    style = {"look": "梦幻国漫插画", "palette": "青绿", "medium": "2D手绘", "idiom": "国漫"}
    board = {
        "canonical_face": "face", "canonical_body": "body",
        "hair": "hair", "wardrobe": "wardrobe", "palette": "palette",
    }
    prompt = _model_sheet_prompt("林", board, {"description": "desc"}, style)
    # The medium leads the prompt — it appears before the "model sheet" turnaround framing
    # that otherwise biases the model toward a glossy 3D design sheet.
    assert "2D手绘" in prompt
    assert prompt.index("2D手绘") < prompt.index("model sheet")
    assert "3D" in prompt or "3d" in prompt.lower()


def test_character_image_prompt_leads_with_medium():
    from studio_agent.stages.bible import _image_prompt

    style = {"look": "梦幻国漫插画", "palette": "青绿", "medium": "2D手绘"}
    prompt = _image_prompt("林", {"visual_description": "a young man"}, style)
    assert "2D手绘" in prompt
    assert prompt.index("2D手绘") < prompt.index("a young man")


class PromptRecordingLLM(FakeLLM):
    """Records every prompt so language directives can be asserted."""
    def __init__(self):
        self.prompts = []

    def complete_json(self, prompt, *, system=None):
        self.prompts.append(prompt)
        return super().complete_json(prompt, system=system)


def _prompts_for_task(llm, task):
    return [p for p in llm.prompts if f"[task:{task}]" in p]


def test_bible_prompts_carry_chinese_language_directive(tmp_path):
    p = _project(tmp_path)
    p.model_config["language"] = "zh"  # _project defaults to en; force zh here
    llm = PromptRecordingLLM()
    BibleStage().run(p, Providers(llm=llm, image=RecordingImage()))
    for task in ("character", "identity_board", "location"):
        prompts = _prompts_for_task(llm, task)
        assert prompts, f"no {task} prompt was issued"
        assert all("LANGUAGE: Chinese" in pr for pr in prompts), (
            f"{task} prompt missing Chinese language directive"
        )


def test_bible_prompts_carry_english_language_directive(tmp_path):
    p = _project(tmp_path)
    p.model_config["language"] = "en"
    llm = PromptRecordingLLM()
    BibleStage().run(p, Providers(llm=llm, image=RecordingImage()))
    char_prompts = _prompts_for_task(llm, "character")
    assert char_prompts and all("LANGUAGE: English" in pr for pr in char_prompts)


class ChineseCharacterLLM(FakeLLM):
    """Returns Chinese canonical text to prove the JSON is written unescaped."""
    def complete_json(self, prompt, *, system=None):
        gen = super().complete_json(prompt, system=system)
        if "[task:character]" in prompt:
            gen.content = {
                "visual_description": "一位沉稳的女性，黑发",
                "wardrobe": "深色风衣",
                "palette": "冷调",
            }
        return gen


def test_bible_writes_chinese_character_json_unescaped(tmp_path):
    p = _project(tmp_path)
    BibleStage().run(p, Providers(llm=ChineseCharacterLLM(), image=RecordingImage()))
    cdir = p.path("bible", "characters", char_slug("Mara"))
    raw = (cdir / "character.json").read_text()
    assert "\\u" not in raw, "Chinese was unicode-escaped in character.json"
    assert "深色风衣" in raw


def test_char_slug_distinguishes_chinese_names(tmp_path):
    # Two distinct Chinese names must produce two distinct, non-empty slugs.
    # Previously both stripped to "" -> fallback "character" -> collision.
    assert char_slug("美女") != char_slug("男生")
    assert char_slug("美女").strip("-") not in ("", "character")
    assert location_slug("现代客厅") != location_slug("卧室")
    # ASCII names are unchanged so existing projects keep their paths.
    assert char_slug("Mara") == "mara"


def test_bible_builds_distinct_dirs_for_two_chinese_characters(tmp_path):
    # The real bug: a 中文 plot with two characters produced ONE bible directory,
    # silently dropping the second character (no reference image downstream).
    p = _project(tmp_path)
    p.model_config["language"] = "zh"
    plot = json.loads(p.path("story", "plot.json").read_text())
    plot["characters"] = [
        {"name": "美女", "role": "lead", "description": "屏幕里的年轻女子"},
        {"name": "男生", "role": "lead", "description": "屏幕外的男生，只见双手"},
    ]
    p.path("story", "plot.json").write_text(json.dumps(plot, ensure_ascii=False))

    BibleStage().run(p, Providers(llm=FakeLLM(), image=FakeImageGen()))

    names = {
        json.loads(f.read_text())["name"]
        for f in p.path("bible", "characters").glob("*/character.json")
    }
    assert names == {"美女", "男生"}, f"expected both characters, got {names}"


def test_xai_limit_fits_chinese_location_prompt_without_changing_canonical_artifact(
    tmp_path,
):
    project = _location_project(tmp_path)
    canonical_prompt = _xai_limit_location_prompt()
    location_dir = project.path("bible", "locations", "clock-shop")
    location_dir.mkdir(parents=True, exist_ok=True)
    location_path = location_dir / "location.json"
    location_path.write_text(json.dumps({
        "name": "Clock Shop",
        "description": "现代风格的电竞房",
        "palette": "冷色黑蓝调，霓虹紫与青色",
        "materials": "碳纤维纹理电竞桌",
        "lighting": "屏幕光与RGB灯光",
        "hero_props": ["打开的游戏笔记本电脑"],
        "continuity_rules": ["屏幕正对镜头"],
        "prompt_aliases": ["Clock Shop"],
        "scene_numbers": [1],
        "reference_image": "reference.png",
        "reference_image_stale": True,
        "source_references": [],
        "seed": 42,
        "image_prompt": canonical_prompt,
    }, ensure_ascii=False))
    project.path("bible", "style_sample.png").write_bytes(b"style")
    images = RecordingImage()
    images.max_prompt_length = 8000
    images.prompt_length_unit = "utf8_bytes"
    providers = Providers(llm=FakeLLM(), image=images)

    BibleStage().run(project, providers)

    location_calls = [
        call for call in images.calls
        if call["out_path"].endswith("bible/locations/clock-shop/reference.png")
    ]
    assert len(location_calls) == 1
    submitted = location_calls[0]["prompt"]
    assert len((canonical_prompt + "\n\n" + STYLE_REFERENCE_GUARD).encode("utf-8")) == 8117
    assert len(submitted.encode("utf-8")) <= 8000
    for required in (
        "现代风格的电竞房",
        "碳纤维纹理电竞桌",
        "霓虹紫与青色",
        "屏幕正对镜头",
    ):
        assert required in submitted
    assert STYLE_REFERENCE_GUARD in submitted
    assert "Location Identity Skill" not in submitted
    assert json.loads(location_path.read_text())["image_prompt"] == canonical_prompt
    assert location_calls[0]["reference_images"] == [
        str(project.path("bible", "style_sample.png"))
    ]
    provider_prompt = location_dir / "reference.provider-prompt.md"
    provider_metadata = location_dir / "reference.provider-prompt.json"
    assert provider_prompt.read_text() == submitted
    metadata = json.loads(provider_metadata.read_text())
    assert metadata["original_length"] == 8117
    assert metadata["submitted_length"] == len(submitted.encode("utf-8"))

    BibleStage().run(project, providers)
    assert len([
        call for call in images.calls
        if call["out_path"].endswith("bible/locations/clock-shop/reference.png")
    ]) == 1
