"""Bible stage: the consistency anchor (invariant #4).

For each character in the plot it generates a canonical *visual* description (LLM),
locks a deterministic seed, builds an identity board of canonical look rules (LLM), and
renders ONE combined character model sheet (角色三视图: turnaround + expression strip +
palette on a single page) written to ``reference.png``. That approved sheet + identity
board + locked seed is the single source of truth every downstream image/video generation
is conditioned on, so characters stay consistent across shots.

Writes::

    bible/style.md                              global visual style guide
    bible/characters/<slug>/character.json      canonical desc + seed + image prompt + ref path
    bible/characters/<slug>/identity_board.json canonical look rules + aliases
    bible/characters/<slug>/reference.png       combined model sheet (角色三视图), the identity anchor

Idempotent (invariant #3): skips the whole stage if complete, and guards each per-character
artifact independently so a re-run backfills only what is missing and never duplicates a
paid generation or clobbers a hand-edited file.
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

from .base import Providers, Stage, StagePreflight, StageResult
from ..creative_brief import load_creative_brief, resolved_value
from ..creative_decisions import decision_preflight
from ..identity_board import coerce_aliases, normalize_identity_board
from ..image_generation import generate_project_image
from ..image_prompt_director import direct_image_prompt, directed_prompt_text, director_enabled
from ..keyframe_prompt import STYLE_REFERENCE_GUARD, SUBJECT_REFERENCE_GUARD
from ..knowledge import KnowledgeQuery, get_or_create_packet, load_packaged_core
from ..reference_assets import (
    cap_reference_paths,
    reference_paths_for_target,
    resolve_pending_reference_targets,
    style_anchor_paths,
)
from ..language import language_label
from ..runtime_skills import load_prompt_skill
from ..style import format_style_markdown, format_style_prompt, style_medium_lead
from ..visual_states import state_plan_for_character

_CHARACTER_PROMPT = """[task:character]
You are a character designer. Write a canonical visual description for this character,
detailed and consistent enough to anchor every future image/video generation. Return
JSON with keys: visual_description, wardrobe, palette.

This is the BASELINE identity only. Never bake transformations, temporary visual effects,
damage, disguises, aging, or later forms from STATE PLAN into the canonical description.

Use this character identity structure when useful:
{skill}

NAME: {name}
ROLE: {role}
DESC: {desc}
STATE PLAN: {state_plan}
LANGUAGE: {language}
"""

_IDENTITY_BOARD_PROMPT = """[task:identity_board]
You are a character designer. Produce a canonical identity board to lock this character's
look across every scene. Return JSON with keys: identity_signature, canonical_face,
canonical_body, face, body, hair, wardrobe, wardrobe_details, palette, hero_props,
continuity_priority, allowed_variation, do (list), dont (list), prompt_aliases (list),
appearance_lock.

"appearance_lock" is ONE dense paragraph (about 60-120 words) that locks this exact person's
look for every future image: face geometry, every distinctive mark and its placement, skin,
hair, eyes, build and age read, the immutable wardrobe, and the palette / rendering treatment.
Ground it in the reference image where one is shown. Write it so an image model cannot draw a
different-looking person or drift the colors. No references to image filenames or aliases.

Lock only the BASELINE person. STATE PLAN is temporal direction, not permanent identity.
Do not require temporary effects or transformed anatomy in every image.

Use this character identity structure when useful:
{skill}

Use only relevant guidance from this retrieved knowledge packet:
{knowledge}

NAME: {name}
DESC: {desc}
STATE PLAN: {state_plan}
LANGUAGE: {language}
"""

_LOCATION_PROMPT = """[task:location]
You are a production designer. Produce a reusable location bible for AI video continuity.
Return JSON with keys: description, palette, materials, lighting, hero_props (list),
continuity_rules (list), prompt_aliases (list).

Use this location identity structure when useful:
{skill}

NAME: {name}
SCENES: {scenes}
DESC: {desc}
LANGUAGE: {language}
"""


_VISION_IDENTITY_PREAMBLE = (
    "You are shown the uploaded REFERENCE IMAGE(S) of this subject. The image is the single "
    "source of truth for appearance: describe the canonical identity EXACTLY as shown — face, "
    "hair, wardrobe, accessories/jewelry, props, and colors. Where any text below conflicts "
    "with the image, the IMAGE wins. Do not invent details absent from the image, and do not "
    "omit distinctive elements that are present (e.g. headwear, jewelry, armor, ornaments). "
    "Keep wardrobe and do/don't rules faithful to what is visible. "
    "Do not refer to the image by its alias or filename (e.g. 'see image1', '参考image2', "
    "'as shown in the reference'): transcribe the concrete visible details into every field "
    "instead. Never leave a field as null or 'None' — describe what you actually see."
)


def _require_reference_schema(
    generation,
    *,
    target: str,
    required_any: tuple[str, ...],
) -> dict:
    content = generation.content if isinstance(generation.content, dict) else {}
    present = [key for key in required_any if content.get(key)]
    if not present:
        expected = ", ".join(required_any)
        provider = getattr(generation, "provider", "unknown")
        model = getattr(generation, "model", "unknown")
        raise ValueError(
            f"{provider}/{model} returned invalid reference identity for {target}: "
            f"expected at least one non-empty field: {expected}"
        )
    return dict(content)


# The model sheet interpolates the character `description` verbatim (it bypasses the
# prompt director, which would otherwise condense it). The board fields below already
# carry the canonical face/body/hair/wardrobe/palette — the narrative description is only
# supplementary context, so it is capped to a bounded UTF-8 budget. Without this an
# unusually verbose LLM description (4KB+) pushes the sheet past a tight image-provider
# limit (xAI Grok = 8000 utf8 bytes) and the fitter, which can only shed skill sections,
# cannot recover it.
_MODEL_SHEET_DESCRIPTION_BUDGET = 700


def _clip_utf8(text: str, max_bytes: int) -> str:
    """Trim ``text`` to at most ``max_bytes`` UTF-8 bytes on a codepoint boundary."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip() + "…"


def _model_sheet_prompt(
    name: str, board: dict, character: dict, style: dict, skill: str = ""
) -> str:
    # One coherent character model sheet (角色三视图/设定): turnaround + expression strip +
    # palette on a single page. Described as an explicit panel grid (grids are far more
    # consistent than "same character from N angles") and forced orthographic/flat so it
    # reads as a design sheet, not a scene. Bypasses the cinematic prompt director.
    medium_lead = style_medium_lead(style)
    medium_prefix = f"{medium_lead} " if medium_lead else ""
    description = _clip_utf8(
        str(character.get("description", "")), _MODEL_SHEET_DESCRIPTION_BUDGET
    )
    return (
        f"{medium_prefix}"
        f"Character model sheet (角色设定三视图) for {name}, one coherent page, clean grid layout. "
        f"Top row: full-body orthographic turnaround — front, three-quarter, side, and back "
        f"views of the SAME character, identical face/hair/wardrobe and consistent proportions "
        f"across every view. Bottom row: a six-panel expression strip (neutral, happy, sad, "
        f"angry, surprised, thoughtful) head-and-shoulders, same identity throughout. Include a "
        f"small color-palette swatch bar and the character name label. "
        f"Face: {board.get('canonical_face', '')}. Body: {board.get('canonical_body', '')}. "
        f"Hair: {board.get('hair', '')}. Wardrobe: {board.get('wardrobe', '')}. "
        f"Palette: {board.get('palette', '')}. "
        f"Description: {description}. "
        f"{format_style_prompt(style)} Flat neutral studio background, consistent lighting across "
        f"all panels, orthographic, no perspective distortion. {skill}"
    )


def _slugify(name: str, fallback: str) -> str:
    """Stable, filesystem-safe, ASCII slug that stays UNIQUE per distinct name.

    ASCII names keep their familiar slug ("Mara" -> "mara"). Names that carry no
    ASCII (e.g. Chinese "美女"/"男生") used to all strip to "" and collapse onto the
    bare fallback, so two such characters collided on one directory and the second
    was silently dropped (invariant #10: Chinese is first-class). When no ASCII
    survives we suffix a stable hash of the original name so distinct names get
    distinct directories.
    """
    text = (name or "").strip()
    ascii_slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if ascii_slug:
        return ascii_slug
    if not text:
        return fallback
    digest = format(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF, "08x")
    return f"{fallback}-{digest}"


def char_slug(name: str) -> str:
    return _slugify(name, "character")


def location_slug(name: str) -> str:
    return _slugify(name, "location")


def seed_for(name: str) -> int:
    """A stable, locked seed derived from the character name (invariant #4)."""
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def _prompt_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _join_prompt_values(value: object) -> str:
    if not isinstance(value, list):
        return ""
    return ", ".join(_prompt_value(item) for item in value)


def _packet_guidance(packet: dict) -> str:
    sections = []
    for entry in packet.get("selected_entries") or []:
        guidance = " ".join(filter(None, [
            str(entry.get("principle") or ""),
            str(entry.get("use_when") or ""),
            str(entry.get("recipe") or ""),
        ])).strip()
        if guidance:
            sections.append(f"- {entry.get('id')}: {guidance}")
    return "\n".join(sections) or "(no relevant retrieved guidance; use the packaged identity skill)"


def _identity_rationale(
    name: str,
    packet: dict,
    references: list[str],
) -> str:
    lines = [f"# Identity rationale — {name}", "", "## Selected knowledge", ""]
    entries = packet.get("selected_entries") or []
    if entries:
        for entry in entries:
            source = (entry.get("source") or {}).get("path") or "unknown source"
            lines.append(f"- `{entry.get('id')}` — {entry.get('title')} ({source})")
    else:
        lines.append("- Packaged character-identity skill fallback")
    lines.extend(["", "## Reference bindings", ""])
    lines.extend(f"- `{reference}`" for reference in references)
    return "\n".join(lines).rstrip() + "\n"


def _image_prompt(name: str, canonical: dict, style: dict) -> str:
    medium_lead = style_medium_lead(style)
    medium_prefix = f"{medium_lead} " if medium_lead else ""
    return (
        f"{medium_prefix}"
        f"Reference portrait of {name}. {canonical.get('visual_description', '')} "
        f"Wardrobe: {canonical.get('wardrobe', '')}. "
        f"Palette: {canonical.get('palette', '')}. "
        f"{format_style_prompt(style)} "
        f"Neutral studio background, full character visible. Motivated soft studio lighting, "
        f"85mm portrait lens at eye-level."
    )


def _state_reference_prompt(
    name: str,
    state: dict,
    board: dict,
    transitions: list[dict],
    style: dict,
) -> str:
    preserved: list[str] = []
    for transition in transitions:
        if state.get("id") in {
            transition.get("to"),
            *(transition.get("ordered_state_ids") or []),
        }:
            preserved.extend(str(value) for value in transition.get("preserve") or [])
    preserve_text = ", ".join(dict.fromkeys(preserved)) or (
        "eyes, facial geometry, age, body proportions, and wardrobe continuity"
    )
    appearance_changes = ", ".join(
        str(value) for value in state.get("appearance_changes") or [] if str(value).strip()
    )
    return (
        f"Derived visual-state reference for {name}: {state.get('label') or state.get('id')}. "
        f"State description: {state.get('description', '')}. Preserve recognizable identity "
        f"from the supplied approved baseline references, especially {preserve_text}. "
        f"Material appearance changes to depict: {appearance_changes}. "
        f"Canonical face: {board.get('canonical_face', '')}. "
        f"Canonical body: {board.get('canonical_body', '')}. "
        f"Wardrobe continuity: {board.get('wardrobe', '')}. {format_style_prompt(style)} "
        "Show the clean completed state as one coherent full-body character reference on a "
        "neutral studio background, photographed with an 85mm portrait lens at eye-level under "
        "motivated soft studio lighting. Do not make a transformation sequence, split panel, "
        "collage, or storyboard grid."
    )


def _scene_sheet_prompt(name: str, data: dict, style: dict, skill: str = "") -> str:
    # One coherent location scene sheet (场景四视图): a large establishing view plus
    # alternate angles, with the design-board callouts (material/palette swatches, lighting
    # notes, continuity rules) on the same page. Single downstream location anchor.
    props = _join_prompt_values(data.get("hero_props"))
    rules = _join_prompt_values(data.get("continuity_rules"))
    medium_lead = style_medium_lead(style)
    medium_prefix = f"{medium_lead} " if medium_lead else ""
    return (
        f"{medium_prefix}"
        f"Location scene sheet (场景四视图) for {name}, one coherent page. Large establishing "
        f"wide view plus smaller alternate-angle panels of the SAME location, with material "
        f"swatches, color-palette swatches, and lighting notes alongside. "
        f"Description: {data.get('description', '')}. Palette: {data.get('palette', '')}. "
        f"Materials: {data.get('materials', '')}. Lighting: {data.get('lighting', '')}. "
        f"Hero props: {props}. Continuity rules: {rules}. {format_style_prompt(style)} "
        f"Designed, inspectable environment; clean production-design reference board; no generic "
        f"backdrop. {_LOCATION_UNPOPULATED_GUARD} {skill}"
    )


# Location boards are environment plates: keyframes place characters into them later, so the
# reference itself must stay empty. Without this the image model tends to populate the scene
# with incidental people. Language-agnostic fixed instruction (no word pattern-matching).
_LOCATION_UNPOPULATED_GUARD = (
    "This is an unpopulated environment plate: show the location completely empty — absolutely "
    "no people, characters, human figures, silhouettes, crowds, or reflections of people "
    "anywhere in frame, including the background. Depict only the space, architecture, set "
    "dressing, and props."
)


class BibleStage(Stage):
    name = "bible"

    def preflight(
        self,
        project,
        providers: Providers,
        *,
        auto: bool = False,
    ) -> StagePreflight:
        plot_path = project.path("story", "plot.json")
        if not plot_path.is_file():
            return StagePreflight()
        brief = load_creative_brief(project)
        if resolved_value(brief, "character_treatment"):
            return StagePreflight()
        plot = json.loads(plot_path.read_text())
        weak = []
        for character in plot.get("characters") or []:
            name = str(character.get("name") or "Character")
            description = "".join(str(character.get("description") or "").split())
            refs = reference_paths_for_target(
                project, "character", name, char_slug(name)
            )
            if len(description) < 80 and not refs:
                weak.append(name)
        if not weak:
            return StagePreflight()
        return decision_preflight(
            project,
            providers,
            stage=self.name,
            gap={
                "dimension": "character_treatment",
                "default": "restrained and lived-in",
                "why": "Character treatment shapes every identity and downstream image.",
                "evidence": weak,
            },
            auto=auto,
        )

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="bible already complete")

        plot = json.loads(project.path("story", "plot.json").read_text())
        style = dict(project.model_config.get("style") or {})
        character_skill = load_prompt_skill("character_identity")
        location_skill = load_prompt_skill("location_identity")

        self._write_style(project, plot, style)

        characters = plot.get("characters") or []
        states_path = project.path("story", "visual_state_changes.json")
        state_document = (
            json.loads(states_path.read_text()) if states_path.is_file()
            else {"version": 1, "characters": []}
        )
        locations = self._location_candidates(project, plot)
        resolve_pending_reference_targets(
            project, characters=characters, locations=locations, llm=providers.llm
        )
        for character in characters:
            self._build_character(
                project,
                character,
                style,
                providers,
                character_skill,
                state_plan=state_plan_for_character(
                    state_document, str(character.get("name") or "")
                ),
            )

        for location in locations:
            self._build_location(project, location, style, providers, location_skill)

        return StageResult(
            status="complete",
            message=f"bible: {len(characters)} character(s), {len(locations)} location(s)",
        )

    def _write_style(self, project, plot: dict, style: dict) -> None:
        themes = _join_prompt_values(plot.get("themes")) or "n/a"
        style_doc = {"look": "cinematic", "palette": "natural", "aspect_ratio": "16:9", **style}
        style_lines = format_style_markdown(style_doc)
        text = (
            "# Show Bible — Visual Style\n\n"
            f"{style_lines}\n"
            f"- Themes: {themes}\n\n"
            "Keep character appearance, wardrobe, and palette identical across every scene.\n"
        )
        project.path("bible", "style.md").write_text(text)

    def _directed_hero_prompt(
        self,
        project,
        providers: Providers,
        *,
        brief: str,
        prompt_path: Path,
        style: dict,
        purpose: str,
        aliases: list[str],
        knowledge_packet: dict | None = None,
        has_style_reference: bool = False,
    ) -> str:
        """Return the director's dense prompt for a hero reference image.

        The compiled template stays the brief (persisted in ``*.json`` ``image_prompt``);
        the directed prompt is persisted alongside the image as a hand-editable artifact
        (invariant #8) and reused on re-runs so a missing image never re-calls the LLM
        (invariant #3). The character model sheet (reference.png) is generated directly
        via ``_generate_sheet`` and does not use this director — the director is tuned
        for single cinematic frames, not the structured 角色三视图 grid layout.

        When a project style-reference image conditions the generation, the directed
        prompt is post-processed to carry a style-only guard so the style image's own
        *subject* (e.g. a painted person) never bleeds into a character/location — only
        its look does. The guard is appended after the director runs so it survives the
        LLM rewrite, and is persisted once so re-runs neither drop nor duplicate it.
        """
        directed = self._compose_hero_prompt(
            project,
            providers,
            brief=brief,
            prompt_path=prompt_path,
            style=style,
            purpose=purpose,
            aliases=aliases,
            knowledge_packet=knowledge_packet,
        )
        if has_style_reference and STYLE_REFERENCE_GUARD not in directed:
            directed = f"{directed}\n\n{STYLE_REFERENCE_GUARD}"
            prompt_path.write_text(directed)
        return directed

    def _compose_hero_prompt(
        self,
        project,
        providers: Providers,
        *,
        brief: str,
        prompt_path: Path,
        style: dict,
        purpose: str,
        aliases: list[str],
        knowledge_packet: dict | None = None,
    ) -> str:
        """Compile/direct the dense hero prompt (no style-reference guard)."""
        config = project.model_config or {}
        hard_avoidances = list(load_creative_brief(project).get("hard_avoidances") or [])
        enriched_brief = brief
        if knowledge_packet:
            knowledge_path = prompt_path.with_name("reference.knowledge.json")
            if not knowledge_path.is_file():
                knowledge_path.write_text(
                    json.dumps(knowledge_packet, ensure_ascii=False, indent=2) + "\n"
                )
            enriched_brief += "\nRetrieved identity guidance:\n" + _packet_guidance(
                knowledge_packet
            )
        if hard_avoidances:
            enriched_brief += "\nHard avoid: " + "; ".join(hard_avoidances)

        if prompt_path.is_file():
            return prompt_path.read_text()
        if not director_enabled(config) or providers.llm is None:
            prompt_path.write_text(enriched_brief)
            return enriched_brief
        gen = direct_image_prompt(
            providers,
            project,
            stage=self.name,
            brief=enriched_brief,
            style=style,
            style_name=str(config.get("style_name") or style.get("look") or ""),
            intent=str(config.get("creative_intent") or project.idea or ""),
            purpose=purpose,
            reference_aliases=aliases,
            hard_avoidances=hard_avoidances,
            language=str(config.get("language") or "en"),
        )
        directed = directed_prompt_text(
            gen.content if isinstance(gen.content, dict) else {}
        ) or enriched_brief
        prompt_path.write_text(directed)
        return directed

    def _generate_sheet(
        self,
        project,
        providers: Providers,
        *,
        prompt: str,
        out_path: Path,
        refs: list[str],
        seed: int,
    ) -> None:
        refs = cap_reference_paths(project, refs, providers.image.capabilities.max_reference_images)
        generation = generate_project_image(
            project,
            providers.image,
            prompt,
            out_path=str(out_path),
            reference_images=refs,
            seed=seed,
        )
        project.add_generation_cost(stage=self.name, generation=generation)

    def _identity_describer(self, project, providers: Providers, subject_refs: list[str]):
        """Return a callable ``prompt -> Generation`` for identity generation.

        When generation is needed for an uploaded subject reference, a vision-capable
        analyzer is mandatory because silently falling back to text would replace the
        authoritative image with invented details. Current artifacts can still be reused
        without configuring an analyzer. Projects without references keep the text path."""
        analyzer = getattr(providers, "reference_analyzer", None)
        language = str(project.model_config.get("language") or "en")

        def generate(prompt: str):
            if subject_refs:
                if analyzer is None:
                    raise ValueError(
                        "A vision analyzer is required for authoritative reference images: "
                        + ", ".join(subject_refs)
                    )
                try:
                    return analyzer.describe(
                        subject_refs,
                        prompt=f"{_VISION_IDENTITY_PREAMBLE}\n\n{prompt}",
                        language=language,
                    )
                except NotImplementedError as exc:
                    raise ValueError(
                        "The configured vision analyzer cannot describe authoritative "
                        "references: " + ", ".join(subject_refs)
                    ) from exc
            return providers.llm.complete_json(prompt)

        return generate

    def _build_character(
        self,
        project,
        character: dict,
        style: dict,
        providers: Providers,
        character_skill: str = "",
        state_plan: dict | None = None,
    ) -> None:
        name = character.get("name", "Character")
        slug = char_slug(name)
        cdir = project.path("bible", "characters", slug)
        cdir.mkdir(parents=True, exist_ok=True)
        character_path = cdir / "character.json"
        ref_path = cdir / "reference.png"
        source_references = reference_paths_for_target(
            project, "character", name, slug
        )
        subject_refs = [
            str(project.dir / rel)
            for rel in source_references
        ]
        uploaded_refs = list(subject_refs)
        # Carry the project-wide style reference so identity boards inherit the locked
        # look from frame one (style-only: the prompt forbids copying its content).
        uploaded_refs += [str(project.dir / rel) for rel in style_anchor_paths(project)]
        relative_refs = [
            f"bible/characters/{slug}/reference.png",
            *source_references,
        ]
        packet = self._identity_packet(
            project,
            character,
            slug=slug,
            style=style,
            reference_count=len(relative_refs),
        )
        knowledge = _packet_guidance(packet)
        rationale_path = cdir / "identity.rationale.md"
        if not rationale_path.is_file():
            rationale_path.write_text(_identity_rationale(name, packet, relative_refs))

        # No blanket early-return: each artifact below guards itself with is_file(), so
        # identity boards also backfill on pre-existing projects while a re-run never
        # duplicates a paid generation (invariant #3).

        # A reference image is the source of truth for identity: when a subject reference and
        # a vision analyzer are both available, describe the canonical look + board FROM the
        # image so the written identity matches it instead of an invented, conflicting one.
        describe_identity = self._identity_describer(project, providers, subject_refs)

        # 1. Canonical description (LLM/VLM) -> character.json. A reference-path change
        # invalidates only the written identity; unchanged bindings remain idempotent.
        existing_character = (
            json.loads(character_path.read_text()) if character_path.is_file() else {}
        )
        needs_canonical = (
            not character_path.is_file()
            or list(existing_character.get("source_references") or [])
            != source_references
        )
        if needs_canonical:
            prompt = _CHARACTER_PROMPT.format(
                name=name,
                role=character.get("role", ""),
                desc=character.get("description", ""),
                skill=character_skill,
                state_plan=json.dumps(state_plan or {}, ensure_ascii=False),
                language=language_label(project.model_config.get("language") or "en"),
            )
            project.assert_budget_available()
            canonical_gen = describe_identity(prompt)
            project.add_generation_cost(
                stage=self.name, generation=canonical_gen
            )
            canonical = (
                _require_reference_schema(
                    canonical_gen,
                    target=name,
                    required_any=("visual_description",),
                )
                if subject_refs
                else dict(canonical_gen.content)
            )
            character_data = {
                "name": name,
                "role": character.get("role", ""),
                "description": canonical.get("visual_description", character.get("description", "")),
                "wardrobe": canonical.get("wardrobe", ""),
                "palette": canonical.get("palette", ""),
                "seed": int(existing_character.get("seed", seed_for(name))),
                "image_prompt": _image_prompt(name, canonical, style),
                "reference_image": ref_path.name,
                "reference_image_stale": True,
                "source_references": source_references,
            }
            if "regen_count" in existing_character:
                character_data["regen_count"] = existing_character["regen_count"]
            character_path.write_text(json.dumps(character_data, ensure_ascii=False, indent=2))
        else:
            character_data = existing_character

        seed = int(character_data.get("seed", seed_for(name)))

        # 2. Identity board (invariant #4): canonical look rules drive the model-sheet prompt.
        board_path = cdir / "identity_board.json"
        if not board_path.is_file() or needs_canonical:
            project.assert_budget_available()
            board_gen = describe_identity(
                _IDENTITY_BOARD_PROMPT.format(
                    name=name,
                    desc=character_data.get("description", ""),
                    skill=character_skill,
                    knowledge=knowledge,
                    state_plan=json.dumps(state_plan or {}, ensure_ascii=False),
                    language=language_label(project.model_config.get("language") or "en"),
                )
            )
            project.add_generation_cost(stage=self.name, generation=board_gen)
            board_content = (
                _require_reference_schema(
                    board_gen,
                    target=name,
                    required_any=(
                        "identity_signature",
                        "canonical_face",
                        "face",
                        "hair",
                        "wardrobe",
                    ),
                )
                if subject_refs
                else board_gen.content
            )
            board = normalize_identity_board(
                board_content,
                name=name,
                canonical=character_data,
                references=relative_refs,
            )
            board_path.write_text(json.dumps(board, ensure_ascii=False, indent=2))
        else:
            board = json.loads(board_path.read_text())

        # 3. One combined model sheet (角色三视图) written to reference.png — the single
        #    downstream identity anchor. Direct grid prompt (the cinematic director is tuned
        #    for single frames, not grids), with the style-only guard appended when a project
        #    style upload conditions the generation so its subject never bleeds in.
        if character_data.get("reference_image_stale") or not ref_path.is_file():
            sheet_prompt = _model_sheet_prompt(
                name, board, character_data, style, character_skill
            )
            if subject_refs:
                # Use the uploaded photo for identity/wardrobe only; render in project style.
                sheet_prompt = f"{sheet_prompt}\n\n{SUBJECT_REFERENCE_GUARD}"
            if style_anchor_paths(project):
                sheet_prompt = f"{sheet_prompt}\n\n{STYLE_REFERENCE_GUARD}"
            self._generate_sheet(
                project,
                providers,
                prompt=sheet_prompt,
                out_path=ref_path,
                refs=uploaded_refs,
                seed=seed,
            )
            if (
                character_data.get("reference_image") != ref_path.name
                or character_data.get("reference_image_stale") is not False
            ):
                character_data["reference_image"] = ref_path.name
                character_data["reference_image_stale"] = False
                character_path.write_text(json.dumps(character_data, ensure_ascii=False, indent=2))

        # 4. Materially distinct visual states. Transient states stay as editable text
        #    rules; only explicit non-base reference states spend on a derived anchor.
        if state_plan:
            self._build_character_states(
                project,
                providers,
                name=name,
                cdir=cdir,
                state_plan=state_plan,
                board=board,
                style=style,
                baseline_refs=[str(ref_path)] + uploaded_refs if ref_path.is_file()
                else list(uploaded_refs),
            )

    def _build_character_states(
        self,
        project,
        providers: Providers,
        *,
        name: str,
        cdir: Path,
        state_plan: dict,
        board: dict,
        style: dict,
        baseline_refs: list[str],
    ) -> None:
        states_path = cdir / "states.json"
        if states_path.is_file():
            document = json.loads(states_path.read_text())
        else:
            document = json.loads(json.dumps(state_plan, ensure_ascii=False))

        changed = not states_path.is_file()
        initial = str(document.get("initial_state") or "")
        transitions = list(document.get("transitions") or [])
        for state in document.get("states") or []:
            if not isinstance(state, dict):
                continue
            sid = str(state.get("id") or "").strip()
            if not sid:
                continue
            if sid == initial or state.get("kind") == "base":
                if state.get("reference_image") != "reference.png":
                    state["reference_image"] = "reference.png"
                    changed = True
                continue
            # New normalized plans explicitly carry material deltas.  Do not create a
            # paid identity anchor for pose/location/camera-only endpoints.  Legacy
            # states without this field retain their existing file-backed behavior.
            if "appearance_changes" in state and not state.get("appearance_changes"):
                state["reference_required"] = False
            if not bool(state.get("reference_required")):
                continue

            relative = f"states/{sid}/reference.png"
            if state.get("reference_image") != relative:
                state["reference_image"] = relative
                changed = True
            state_dir = cdir / "states" / sid
            state_dir.mkdir(parents=True, exist_ok=True)
            out_path = state_dir / "reference.png"
            if out_path.is_file():
                continue
            brief = _state_reference_prompt(name, state, board, transitions, style)
            prompt = self._directed_hero_prompt(
                project,
                providers,
                brief=brief,
                prompt_path=state_dir / "reference.prompt.md",
                style=style,
                purpose="character_state_reference",
                aliases=[name, str(state.get("label") or sid)],
                has_style_reference=bool(style_anchor_paths(project)),
            )
            baseline_refs = cap_reference_paths(
                project,
                baseline_refs,
                providers.image.capabilities.max_reference_images,
                named_refs=[str(cdir / "reference.png")],
            )
            generation = generate_project_image(
                project,
                providers.image,
                prompt,
                out_path=str(out_path),
                reference_images=baseline_refs,
                seed=seed_for(f"{name}:state:{sid}"),
            )
            project.add_generation_cost(stage=self.name, generation=generation)

        if changed:
            states_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n"
            )

    def _identity_packet(
        self,
        project,
        character: dict,
        *,
        slug: str,
        style: dict,
        reference_count: int,
    ) -> dict:
        brief = load_creative_brief(project)
        decision_path = project.path("story", "decisions", "bible.json")
        decision = json.loads(decision_path.read_text()) if decision_path.is_file() else {}
        resolution = decision.get("resolution") or {}
        description = str(character.get("description") or "")
        intent_text = " ".join(filter(None, [
            project.idea,
            str(character.get("role") or ""),
            description,
            resolved_value(brief, "tone"),
            resolved_value(brief, "character_treatment"),
            str(resolution.get("value") or ""),
            f"reference_count={reference_count}",
        ]))
        query = KnowledgeQuery(
            stage="bible",
            domains=("identity", "performance", "composition", "lighting"),
            intent_text=intent_text,
            intents=("consistency", "character_design"),
            style=str(project.model_config.get("style_name") or style.get("look") or ""),
            format_name=str(
                project.model_config.get("format_name")
                or (project.model_config.get("product_format") or {}).get("name")
                or ""
            ),
            language=str(project.model_config.get("language") or "en"),
            capabilities=frozenset(),
            hard_avoidances=tuple(brief.get("hard_avoidances") or []),
        )
        retrieval = project.model_config.get("knowledge_retrieval") or {}
        return get_or_create_packet(
            project,
            purpose="identity",
            target=slug,
            query=query,
            entries=load_packaged_core(),
            limit=int(retrieval.get("max_entries", 6)),
        )

    def _build_location(
        self,
        project,
        location: dict,
        style: dict,
        providers: Providers,
        location_skill: str = "",
    ) -> None:
        name = location.get("name", "Location")
        slug = location_slug(name)
        ldir = project.path("bible", "locations", slug)
        ldir.mkdir(parents=True, exist_ok=True)
        location_path = ldir / "location.json"
        ref_path = ldir / "reference.png"
        source_references = reference_paths_for_target(
            project, "location", name, slug
        )
        subject_refs = [
            str(project.dir / rel)
            for rel in source_references
        ]
        uploaded_refs = list(subject_refs)
        # Inherit the project-wide style reference (style-only) for location boards too.
        uploaded_refs += [str(project.dir / rel) for rel in style_anchor_paths(project)]

        describe_identity = self._identity_describer(project, providers, subject_refs)
        existing_location = (
            json.loads(location_path.read_text()) if location_path.is_file() else {}
        )
        needs_canonical = (
            not location_path.is_file()
            or list(existing_location.get("source_references") or [])
            != source_references
        )
        if needs_canonical:
            prompt = _LOCATION_PROMPT.format(
                name=name,
                scenes=", ".join(str(s) for s in location.get("scene_numbers", [])),
                desc=location.get("description", ""),
                skill=location_skill,
                language=language_label(project.model_config.get("language") or "en"),
            )
            project.assert_budget_available()
            gen = describe_identity(prompt)
            project.add_generation_cost(stage=self.name, generation=gen)
            generated = (
                _require_reference_schema(
                    gen,
                    target=name,
                    required_any=("description",),
                )
                if subject_refs
                else dict(gen.content)
            )
            location_data = {
                "name": name,
                "description": generated.get("description", location.get("description", "")),
                "palette": generated.get("palette", ""),
                "materials": generated.get("materials", ""),
                "lighting": generated.get("lighting", ""),
                "hero_props": generated.get("hero_props", []),
                "continuity_rules": generated.get("continuity_rules", []),
                "prompt_aliases": coerce_aliases(generated.get("prompt_aliases")) or [name],
                "scene_numbers": list(location.get("scene_numbers", [])),
                "reference_image": ref_path.name,
                "reference_image_stale": True,
                "source_references": source_references,
                "seed": int(
                    existing_location.get("seed", seed_for(f"location:{name}"))
                ),
            }
            if "regen_count" in existing_location:
                location_data["regen_count"] = existing_location["regen_count"]
            location_path.write_text(
                json.dumps(location_data, ensure_ascii=False, indent=2)
            )
        else:
            location_data = existing_location

        changed = False
        if not location_data.get("image_prompt"):
            location_data["image_prompt"] = _scene_sheet_prompt(
                name, location_data, style, location_skill
            )
            changed = True
        if location_data.get("reference_image") != ref_path.name:
            location_data["reference_image"] = ref_path.name
            changed = True
        if "scene_numbers" not in location_data:
            location_data["scene_numbers"] = list(location.get("scene_numbers", []))
            changed = True
        if changed:
            location_path.write_text(
                json.dumps(location_data, ensure_ascii=False, indent=2)
            )

        seed = int(location_data.get("seed", seed_for(f"location:{name}")))
        if location_data.get("reference_image_stale") or not ref_path.is_file():
            prompt = location_data.get("image_prompt") or _scene_sheet_prompt(
                name, location_data, style, location_skill
            )
            if subject_refs:
                prompt = f"{prompt}\n\n{SUBJECT_REFERENCE_GUARD}"
            if style_anchor_paths(project):
                prompt = f"{prompt}\n\n{STYLE_REFERENCE_GUARD}"
            self._generate_sheet(
                project,
                providers,
                prompt=prompt,
                out_path=ref_path,
                refs=uploaded_refs,
                seed=seed,
            )
            location_data["reference_image_stale"] = False
            location_path.write_text(
                json.dumps(location_data, ensure_ascii=False, indent=2)
            )

    def _location_candidates(self, project, plot: dict) -> list[dict]:
        candidates: dict[str, dict] = {}

        def add(name: str, description: str = "", scene_number=None) -> None:
            clean_name = _clean_location_name(name)
            if not clean_name:
                return
            slug = location_slug(clean_name)
            item = candidates.setdefault(
                slug,
                {"name": clean_name, "description": "", "scene_numbers": []},
            )
            if description and description not in item["description"]:
                item["description"] = (item["description"] + " " + description).strip()
            if scene_number is not None and scene_number not in item["scene_numbers"]:
                item["scene_numbers"].append(scene_number)

        for raw in plot.get("locations") or []:
            if isinstance(raw, dict):
                add(
                    raw.get("name", ""),
                    raw.get("description", ""),
                    raw.get("scene"),
                )
            else:
                add(str(raw), "")

        script_path = project.path("story", "script.json")
        if script_path.is_file():
            try:
                script = json.loads(script_path.read_text())
            except json.JSONDecodeError:
                script = {}
            for scene in _script_scenes(script):
                add(
                    _location_from_heading(scene.get("heading", "")),
                    " ".join(str(b) for b in scene.get("beats", [])[:3]),
                    scene.get("scene"),
                )

        for arc in plot.get("arc") or []:
            for scene in arc.get("scenes", []):
                name = scene.get("location") or _location_from_heading(scene.get("heading", ""))
                add(name, scene.get("summary", ""), scene.get("scene"))

        if not candidates:
            add("Primary Setting", plot.get("synopsis") or plot.get("logline", ""), None)

        for item in candidates.values():
            item["scene_numbers"] = sorted(item["scene_numbers"], key=lambda value: str(value))
        return list(candidates.values())


def _script_scenes(script: dict) -> list[dict]:
    scenes = []
    for episode in script.get("episodes", []):
        scenes.extend(episode.get("scenes", []))
    return scenes


def _location_from_heading(heading: str) -> str:
    heading = str(heading or "").strip()
    if not heading or heading.upper().startswith("SCENE "):
        return ""
    text = re.sub(r"^(INT|EXT|INT/EXT|I/E)\.?\s+", "", heading, flags=re.IGNORECASE)
    if " - " in text:
        text = text.rsplit(" - ", 1)[0]
    return text


def _clean_location_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    if text.upper().startswith("SCENE "):
        return ""
    return text.title()
