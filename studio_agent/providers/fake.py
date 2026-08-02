"""Fake providers — deterministic, offline, zero-cost.

Every provider needs a ``Fake*`` so the whole pipeline runs offline, in CI, and
without spending money (AGENTS.md §6). The foundational spine only needs an LLM, so
that is all that is implemented here; later M0 steps add ``FakeImageGen`` etc.

``FakeLLM.complete_json`` dispatches on a ``[task:<name>]`` marker that stages embed
at the start of their prompt, so each LLM-backed stage gets a deterministic,
idea-aware structured response. As new LLM stages land, add a branch here.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import wave
import zlib
from pathlib import Path

from ..cinematography_recipes import load_camera_recipes
from .base import (
    QC_DIMENSIONS as _QC_DIMENSIONS,
    Generation,
    ImageGen,
    LLM,
    Music,
    ReferenceAnalyzer,
    StyleProfiler,
    TTS,
    VideoCapabilities,
    VideoGen,
    VLMCheck,
    qc_report_status,
)

_TASK_RE = re.compile(r"\[task:([a-z_]+)\]")
_IDEA_RE = re.compile(r"IDEA:\s*(.+?)\s*$", re.MULTILINE)
_PLOT_RE = re.compile(r"PLOT:\s*(\{.*\})\s*$", re.MULTILINE)
_SCRIPT_RE = re.compile(r"SCRIPT:\s*(\{.*\})\s*$", re.MULTILINE)
_FORMAT_RE = re.compile(r"FORMAT:\s*(\{.*\})\s*$", re.MULTILINE)
_LANGUAGE_RE = re.compile(r"LANGUAGE:\s*(.+?)\s*$", re.MULTILINE)
_NAME_RE = re.compile(r"NAME:\s*(.+?)\s*$", re.MULTILINE)
_DESC_RE = re.compile(r"DESC:\s*(.+?)\s*$", re.MULTILINE)
_INSTRUCTION_RE = re.compile(r"INSTRUCTION:\s*(.+?)\s*$", re.MULTILINE)
_CURRENT_JSON_RE = re.compile(r"CURRENT_JSON:\s*(\{.*\}|\[.*\])\s*$", re.DOTALL)
_BRIEF_RE = re.compile(r"BRIEF:\s*(.*)\s*$", re.DOTALL)
_GAP_RE = re.compile(r"GAP:\s*(\{.*\})\s*$", re.MULTILINE)


def _prompt_conversation_revision(prompt: str, *, grounded_in: list[str] | None = None) -> str:
    marker = "CURRENT PROMPT:\n"
    current = prompt.split(marker, 1)[1] if marker in prompt else ""
    current = current.split("\n\nRETURN ONLY", 1)[0].strip()
    prompt_type = ""
    match = re.search(r"^PROMPT TYPE:\s*(.+)$", prompt, re.MULTILINE)
    if match:
        prompt_type = match.group(1).strip().casefold()
    if prompt_type == "grid":
        note = "Grid adjustment applied: every labeled panel remains a separate frozen still."
    elif prompt_type == "static keyframe":
        note = "Static adjustment applied: the subject remains in one clear frozen held pose."
    else:
        note = "Motion adjustment applied while preserving approved timing and continuity."
    if grounded_in:
        note += " Identity grounded in the supplied project references."
    return "\n".join(part for part in (current, note) if part).strip()

_CAMERAS = ["wide establishing shot", "medium shot", "close-up"]
_ALTERNATIVE_DECISIONS = {
    "camera_language": "energetic and immediate",
    "character_treatment": "iconic and carefully composed",
}

# Deterministic per-format scene counts so the fake plot exercises multi-scene
# orchestration offline. Unknown/absent formats stay single-scene (the M0 slice).
_FORMAT_SCENE_COUNTS = {
    "short_drama_episode": 6,
    "short_film": 4,
    "series_episode": 8,
}

# A fake "scene" yields roughly this many seconds of footage (two ~8s shots), so
# the assembled film lands inside the format's duration band when scene count is
# derived from the requested target duration.
_SECONDS_PER_SCENE = 16.0

# Distinct-but-reused locations so multi-location continuity is exercisable
# offline: scenes cycle through this pool, visiting several locations while
# reusing each across scenes.
_LOCATION_POOL_EN = [
    "Main Interior",
    "Exterior Approach",
    "Secondary Interior",
    "Distant Vista",
]
_LOCATION_POOL_ZH = ["主场景内景", "外景入口", "次要内景", "远景全貌"]


def _scene_count_for_format(fmt: dict) -> int:
    target = fmt.get("target_duration_s")
    if target:
        return max(1, round(float(target) / _SECONDS_PER_SCENE))
    return _FORMAT_SCENE_COUNTS.get(fmt.get("name"), 1)


def _scene_locations(scene_count: int, language: str = "en") -> list[str]:
    """One location name per scene (1-indexed), cycling a small pool with reuse.

    A multi-scene film visits at most three distinct locations so locations are
    reused across scenes, exercising cross-scene location continuity offline.
    """
    pool = _LOCATION_POOL_ZH if language == "zh" else _LOCATION_POOL_EN
    distinct = 1 if scene_count <= 1 else min(scene_count, 3, len(pool))
    return [pool[(i - 1) % distinct] for i in range(1, scene_count + 1)]


def _location_entries(scenes: list[dict]) -> list[dict]:
    """Unique locations with the scene numbers that visit them, in first-seen order."""
    entries: dict[str, dict] = {}
    for scene in scenes:
        name = scene.get("location")
        if not name:
            continue
        entry = entries.setdefault(name, {"name": name, "scene_numbers": []})
        entry["scene_numbers"].append(scene.get("scene"))
    return list(entries.values())


class FakeLLM(LLM):
    """Deterministic stand-in for a real LLM. No network, no cost."""

    name = "fake"
    model = "fake-llm-1"

    def complete(self, prompt: str, *, system: str | None = None) -> Generation:
        if self._task(prompt) == "prompt_conversation":
            return Generation(
                content=_prompt_conversation_revision(prompt),
                provider=self.name,
                model=self.model,
                cost_usd=0.0,
                seconds=0.01,
                meta={"task": "prompt_conversation"},
            )
        if self._task(prompt) == "artifact_revision":
            instruction = self._instruction(prompt)
            content = f"{self._current_text(prompt).strip()}\n\nRevision note: {instruction}".strip()
            return Generation(
                content=content,
                provider=self.name,
                model=self.model,
                cost_usd=0.0,
                seconds=0.01,
                meta={"task": "artifact_revision"},
            )
        return Generation(
            content=f"[fake completion]\n{prompt.strip()}",
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=0.01,
        )

    def complete_json(self, prompt: str, *, system: str | None = None) -> Generation:
        task = self._task(prompt)
        idea = self._idea(prompt)
        fmt = self._format(prompt)
        language = self._language(prompt)
        if task == "plot":
            plot = self._plot(idea, fmt, language)
            content = {
                "creative_brief": self._creative_brief(idea, language),
                "plot": plot,
                "visual_state_changes": self._visual_state_changes(
                    idea, plot.get("characters") or [], language
                ),
            }
        elif task == "concept":
            concept = self._concept(idea, language)
            content = {
                "creative_brief": self._creative_brief(idea, language),
                "concept": concept,
                "visual_state_changes": self._visual_state_changes(
                    idea, concept.get("characters") or [], language
                ),
            }
        elif task == "script":
            content = self._script(self._plot_context(prompt), fmt, language)
        elif task == "character":
            content = self._character(prompt)
        elif task == "identity_board":
            content = self._identity_board(prompt)
        elif task == "location":
            content = self._location(prompt)
        elif task == "storyboard":
            content = self._storyboard(self._script_context(prompt))
        elif task == "clip":
            content = self._clip(idea)
        elif task == "shot_insert":
            content = self._shot_insert(prompt)
        elif task == "scene_sequence":
            content = self._scene_sequence(self._language(prompt))
        elif task == "flow_review":
            content = {
                "redundancies": [],
                "progression": [],
                "summary": "fake flow review: no redundancies detected",
            }
        elif task in ("image_prompt", "video_prompt", "grid_prompt"):
            media = {"video_prompt": "video", "grid_prompt": "grid"}.get(task, "image")
            content = self._directed_prompt(prompt, media=media)
        elif task == "creative_decision":
            content = self._creative_decision(prompt)
        elif task == "artifact_revision":
            content = self._artifact_revision(prompt)
        elif task == "reference_mapping":
            # Offline default: defer to the deterministic keyword/role resolver (no guesses).
            content = {"mappings": []}
        else:
            # Unknown task: echo a minimal, still-valid object so callers don't crash.
            content = {"task": task or "unknown", "idea": idea}
        return Generation(
            content=content,
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=0.01,
            meta={"task": task or "unknown"},
        )

    @staticmethod
    def _task(prompt: str) -> str | None:
        m = _TASK_RE.search(prompt)
        return m.group(1) if m else None

    @staticmethod
    def _idea(prompt: str) -> str:
        m = _IDEA_RE.search(prompt)
        return m.group(1).strip() if m else "an untitled idea"

    @staticmethod
    def _plot_context(prompt: str) -> dict:
        m = _PLOT_RE.search(prompt)
        return json.loads(m.group(1)) if m else {}

    @staticmethod
    def _script_context(prompt: str) -> dict:
        m = _SCRIPT_RE.search(prompt)
        return json.loads(m.group(1)) if m else {}

    @staticmethod
    def _format(prompt: str) -> dict:
        m = _FORMAT_RE.search(prompt)
        return json.loads(m.group(1)) if m else {}

    @staticmethod
    def _language(prompt: str) -> str:
        m = _LANGUAGE_RE.search(prompt)
        if not m:
            return "en"
        value = m.group(1).strip().lower()
        return "zh" if "chinese" in value or value == "zh" else "en"

    @staticmethod
    def _instruction(prompt: str) -> str:
        m = _INSTRUCTION_RE.search(prompt)
        return m.group(1).strip() if m else "fake revision"

    @staticmethod
    def _current_text(prompt: str) -> str:
        marker = "CURRENT_TEXT:"
        if marker not in prompt:
            return ""
        return prompt.split(marker, 1)[1].strip()

    @classmethod
    def _artifact_revision(cls, prompt: str) -> dict | list:
        m = _CURRENT_JSON_RE.search(prompt)
        current = json.loads(m.group(1)) if m else {}
        instruction = cls._instruction(prompt)
        if isinstance(current, dict):
            revised = dict(current)
            revised["revision_note"] = instruction
            return revised
        if isinstance(current, list):
            return current
        return {"revision_note": instruction}

    @classmethod
    def _directed_prompt(cls, prompt: str, media: str = "image") -> dict:
        """Deterministically distill the BRIEF into a dense {prompt, negative}.

        Mirrors what a real prompt director does without a model: drop the markdown
        scaffolding and skill dumps, keep the salient sections as one paragraph, and
        surface the avoid terms as a negative. Stable input -> stable output. Video briefs
        keep motion + camera so the directed prompt still drives motion. Grid briefs
        keep panel sequence so composition is preserved.
        """
        m = _BRIEF_RE.search(prompt)
        brief = m.group(1) if m else prompt
        if media == "video":
            headers = [
                "Performance objective",
                "Subject & action",
                "State trajectory",
                "Target state lock",
                "Storyboard grid",
                "Motion beats",
                "Camera",
                "Movement motivation",
                "Lighting & lens",
                "Blocking & visual intent",
                "Sound",
                "Continuity",
                "Identity & references",
                "Retrieved filmmaking guidance",
                "Style",
            ]
        elif media == "grid":
            headers = [
                "Grid layout",
                "Consistency",
                "Subject & action",
                "Panel sequence",
                "State trajectory",
                "Style",
            ]
        else:
            headers = [
                "Shot & composition",
                "Subject & moment",
                "Opening visual state",
                "Lighting & lens",
                "Identity & references",
                "Location & environment references",
                "Retrieved filmmaking guidance",
                "Style",
            ]
        parts = [cls._brief_section(brief, h) for h in headers]
        parts = [p for p in parts if p]
        directed = " ".join(parts) if parts else cls._collapse(brief)
        avoid = cls._brief_section(brief, "Avoid")
        return {
            "prompt": directed.strip(),
            "negative": avoid or "low detail, blurry, deformed hands, extra fingers",
        }

    @staticmethod
    def _brief_section(brief: str, header: str) -> str:
        """Return the collapsed text under a ``## <header>`` section (header may carry a
        trailing suffix such as ``Motion beats (~3s)``)."""
        pattern = re.compile(
            rf"^##\s+{re.escape(header)}[^\n]*$\n(.*?)(?=^##\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        m = pattern.search(brief)
        return FakeLLM._collapse(m.group(1)) if m else ""

    @staticmethod
    def _collapse(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _storyboard(script: dict) -> dict:
        """Break each scene into one shot per beat; dialogue drives duration.

        Each shot lists the characters speaking in it so the stage can look up their
        locked bible seeds and reference-condition the keyframe.
        """
        # Cycle through real recipe ids so the offline path exercises the
        # cinematography-recipe selection deterministically.
        recipe_ids = sorted(load_camera_recipes())
        shots = []
        for ep in script.get("episodes", []):
            for sc in ep.get("scenes", []):
                scene_no = sc.get("scene")
                dialogue = sc.get("dialogue", [])
                beats = sc.get("beats") or ["establishing beat"]
                for i, beat in enumerate(beats):
                    line = dialogue[i] if i < len(dialogue) else None
                    shot_dialogue = [line] if line else []
                    speakers = [line["character"]] if line else []
                    words = _speech_units(line["line"]) if line else 0
                    shots.append(
                        {
                            "scene": scene_no,
                            "camera": _CAMERAS[i % len(_CAMERAS)],
                            "shot_size": _CAMERAS[i % len(_CAMERAS)],
                            "camera_angle": "eye-level",
                            "lens_intent": "natural perspective with readable depth",
                            "camera_movement": "slow push-in on the decisive action",
                            "camera_recipe": recipe_ids[i % len(recipe_ids)] if recipe_ids else "",
                            "action": beat,
                            "visual_beats": [
                                {"action": f"the subject begins {beat}",
                                 "secondary_motion": "clothing and hair shift with the movement",
                                 "weight": 1},
                                {"action": f"the subject completes {beat} with clear follow-through",
                                 "secondary_motion": "cloth settles a beat later",
                                 "weight": 2},
                            ],
                            "description": f"{_CAMERAS[i % len(_CAMERAS)]}: {beat}",
                            "dramatic_purpose": f"make the audience register: {beat}",
                            "start_frame": f"establish {_CAMERAS[i % len(_CAMERAS)]} and spatial context",
                            "end_frame": f"resolve on the visible consequence of {beat}",
                            "subject_blocking": f"the subject physically performs: {beat}",
                            "movement_motivation": f"push closer only as {beat} changes the moment",
                            "lighting_state": "motivated directional light held continuous through the beat",
                            "edit_relationship": "cut on the completed action into the next consequence",
                            "craft_recipe_ids": [],
                            "dialogue": shot_dialogue,
                            # Dialogue length drives duration (PLAN.md); 2s base + read time.
                            "duration_s": round(2.0 + 0.4 * words, 1),
                            "characters": speakers,
                        }
                    )
        return {"shots": shots}

    @staticmethod
    def _shot_insert(prompt: str) -> dict:
        """A deterministic inserted-shot draft that echoes the user's request verbatim.

        Language-agnostic (invariant #11): the request text is copied into the shot's
        description/action without any keyword interpretation.
        """
        match = re.search(
            r"USER REQUEST[^\n]*:\s*\n(.*?)\nLANGUAGE:", prompt, re.DOTALL
        )
        request = (match.group(1).strip() if match else "").strip() or "a new connecting shot"
        return {
            "description": request,
            "action": request,
            "camera": "medium shot",
            "camera_movement": "static locked-off camera",
            "camera_recipe": "",
            "composition": "subject centered with readable depth",
            "emotion": "consistent with the surrounding shots",
            "dramatic_purpose": "bridge the neighboring shots",
            "start_frame": "continue exactly from the previous shot's end state",
            "end_frame": "hand off cleanly into the next shot's start state",
            "subject_blocking": "subject holds a readable screen position",
            "shot_size": "medium shot",
            "camera_angle": "eye-level",
            "lens_intent": "natural perspective preserving subject and environment",
            "movement_motivation": "stillness lets the inserted beat read clearly",
            "lighting_state": "preserve the neighboring shots' light and exposure",
            "edit_relationship": "continue screen direction across the insertion",
            "duration_s": 4,
            "characters": [],
            "dialogue": [],
            "narration": "",
        }

    @staticmethod
    def _clip(idea: str) -> dict:
        """A deterministic global clip sequence with no invented dialogue."""
        common = {
            "camera": "slow tracking shot",
            "shot_size": "medium-wide tracking shot",
            "camera_angle": "eye-level with a slight heroic lift",
            "lens_intent": "natural wide perspective with strong foreground depth",
            "camera_movement": "slow dolly following the subject",
            "camera_recipe": "",  # the stage resolves/sets a real recipe id
            "description": f"A cinematic moment: {idea}.",
            "action": f"The subject performs the central motion of: {idea}.",
            "dramatic_purpose": "deliver one immediately readable visual payoff",
            "start_frame": "establish the subject and a clear path through the environment",
            "end_frame": "resolve on a strong final pose that can cut cleanly",
            "subject_blocking": "the subject travels through foreground and settles on the focal mark",
            "movement_motivation": "the dolly follows the subject's central physical motion",
            "lighting_state": "motivated practical light with stable exposure and color",
            "edit_relationship": "begin clearly and finish on a self-contained cut point",
            "craft_recipe_ids": [],
            "composition": "subject centered, strong depth, atmospheric lighting",
            "emotion": "cool, confident, striking",
            "dialogue": [],
        }
        if ("狼人" in idea and ("红色气体" in idea or "红色" in idea)) or (
            "werewolf" in idea.lower() and ("gas" in idea.lower() or "mist" in idea.lower())
        ):
            common.update({
                "intent_class": "transformation",
                "recommended_duration_s": 30,
                "dramatic_purpose": "make the transformation readable before the final reveal",
                "start_frame": "the ordinary man walks naturally on the road with no gas or transformed anatomy",
                "end_frame": "the completed werewolf holds frame, recognizably the same person",
                "beats": [
                    {
                        "id": "human",
                        "action": "男人以普通人形态走在马路上",
                        "secondary_motion": "衣物随步伐自然摆动，呼吸平稳",
                        "description": "普通男人走在马路上，身体正常，周围没有红色气体",
                        "camera": "medium-wide tracking shot",
                        "camera_movement": "parallel tracking movement",
                        "start_frame": "普通男人走在干净的马路上",
                        "end_frame": "他短暂停步，仍是完整普通人",
                        "duration_s": 6,
                        "start_states": {"男人": "human"},
                        "end_states": {"男人": "human"},
                    },
                    {
                        "id": "gas-onset",
                        "action": "红色气体从男人身体周围缓慢冒出",
                        "secondary_motion": "气体边缘随体温轻轻卷动，衣物微微翻动",
                        "description": "红色气体刚刚出现，但男人外形仍然清楚可辨",
                        "camera": "slow medium push-in",
                        "camera_movement": "push closer as the gas appears",
                        "start_frame": "普通男人身边没有气体",
                        "end_frame": "红色气体包围男人但没有遮住身份",
                        "duration_s": 8,
                        "start_states": {"男人": "human"},
                        "end_states": {"男人": "gas-onset"},
                    },
                    {
                        "id": "partial-werewolf",
                        "action": "男人在红色气体中逐步变成狼人",
                        "secondary_motion": "皮肤下的肌肉逐渐隆起，毛发一缕缕生长，呼吸变粗重",
                        "description": "面部与四肢按连续步骤产生狼人特征",
                        "camera": "controlled close tracking shot",
                        "camera_movement": "orbit subtly around the transforming body",
                        "start_frame": "红色气体中的普通男人",
                        "end_frame": "半狼人形态保留原本眼睛与面部结构",
                        "duration_s": 10,
                        "start_states": {"男人": "gas-onset"},
                        "end_states": {"男人": "partial-werewolf"},
                    },
                    {
                        "id": "werewolf-reveal",
                        "action": "完整狼人从消散的红色气体中显露",
                        "secondary_motion": "毛发随最后一缕气体散去而轻轻抖动落定",
                        "description": "完整狼人形态定格，仍能认出是同一个男人",
                        "camera": "low-angle hero reveal",
                        "camera_movement": "settle into a slow final push-in",
                        "start_frame": "半狼人形态被稀薄红气包围",
                        "end_frame": "完整狼人清晰站在马路上，红气逐渐消散",
                        "duration_s": 6,
                        "start_states": {"男人": "partial-werewolf"},
                        "end_states": {"男人": "werewolf"},
                    },
                ],
            })
            return common
        common.update({
            "intent_class": "showcase",
            "recommended_duration_s": 15,
            "beats": [
                {
                    "id": "hook",
                    "action": f"Establish the visual hook of: {idea}",
                    "secondary_motion": "clothing and hair settle into the opening pose",
                    "description": f"A clear opening image of: {idea}",
                    "camera": "medium-wide tracking shot",
                    "camera_movement": "begin with a measured tracking move",
                    "start_frame": "subject and environment are immediately readable",
                    "end_frame": "the central motion begins",
                    "duration_s": 5,
                    "start_states": {}, "end_states": {},
                },
                {
                    "id": "development",
                    "action": f"Develop the central physical motion of: {idea}",
                    "secondary_motion": "cloth and hair trail the motion and settle a beat later",
                    "description": f"The visual motion gains depth and energy: {idea}",
                    "camera": "moving medium shot",
                    "camera_movement": "continue smoothly with the subject",
                    "start_frame": "the central motion is underway",
                    "end_frame": "the motion reaches its strongest point",
                    "duration_s": 6,
                    "start_states": {}, "end_states": {},
                },
                {
                    "id": "payoff",
                    "action": "Resolve the movement on one striking final image",
                    "secondary_motion": "breath and residual sway ease to stillness on the final pose",
                    "description": f"A cinematic payoff for: {idea}",
                    "camera": "hero framing",
                    "camera_movement": "settle gently into the final composition",
                    "start_frame": "the movement approaches its resolution",
                    "end_frame": "a strong final pose holds cleanly",
                    "duration_s": 4,
                    "start_states": {}, "end_states": {},
                },
            ],
        })
        return common

    @staticmethod
    def _scene_sequence(language: str = "en") -> dict:
        """A deterministic per-scene clip plan: two action beats (first narrated),
        one atomic dialogue beat, one action beat — enough to exercise merge + atomic."""
        if language == "zh":
            narration, line, speaker = "旁白：清晨的灯塔。", "暴风雨要来了。", "守塔人"
        else:
            narration, line, speaker = "Narrator: dawn at the lighthouse.", "A storm is coming.", "Keeper"
        return {
            "intent_class": "micro-arc",
            "recommended_duration_s": 16,
            "camera": "locked then slow push-in",
            "camera_movement": "slow push-in",
            "camera_recipe": "",
            "dramatic_purpose": "establish then turn",
            "start_frame": "wide establishing", "end_frame": "close reaction",
            "subject_blocking": "subject crosses to window",
            "shot_size": "wide to medium", "camera_angle": "eye level",
            "lens_intent": "deep focus then shallow", "movement_motivation": "follow the turn",
            "lighting_state": "cold dawn light", "edit_relationship": "cut on the line",
            "description": "a sustained scene beat", "composition": "subject framed by window",
            "emotion": "tense resolve",
            "beats": [
                {"id": "beat-01", "action": "wide establishing of the room",
                 "description": "establish the space", "narration": narration,
                 "start_states": {}, "end_states": {}, "dialogue": []},
                {"id": "beat-02", "action": "subject crosses to the window",
                 "description": "subject moves through frame",
                 "start_states": {}, "end_states": {}, "dialogue": []},
                {"id": "beat-03", "action": "subject turns and speaks",
                 "description": "subject delivers the line",
                 "start_states": {}, "end_states": {},
                 "dialogue": [{"character": speaker, "line": line}]},
                {"id": "beat-04", "action": "the window rattles in the wind",
                 "description": "environment reacts",
                 "start_states": {}, "end_states": {}, "dialogue": []},
            ],
        }

    @staticmethod
    def _creative_brief(idea: str, language: str = "en") -> dict:
        if language == "zh":
            return {
                "intent_summary": f"以清晰、可拍摄的视觉方向呈现：{idea}",
                "visual_direction": {
                    "tone": "克制而富有情绪",
                    "visual_world": "具有触感与空间层次的电影化世界",
                    "camera_language": "先观察环境，再在关键情绪节点靠近人物",
                    "light_texture": "有动机的方向性光线与清晰材质",
                    "character_treatment": "轮廓明确、表演克制、细节连续",
                },
                "hard_avoidances": ["无动机运镜", "平淡无方向光线"],
            }
        return {
            "intent_summary": f"A specific, filmable visual treatment of: {idea}",
            "visual_direction": {
                "tone": "restrained and emotionally legible",
                "visual_world": "tactile cinematic realism with spatial depth",
                "camera_language": "observe the world, then move closer on emotional turns",
                "light_texture": "motivated directional light and readable materials",
                "character_treatment": "distinct silhouette, restrained acting, stable details",
            },
            "hard_avoidances": ["unmotivated camera movement", "flat directionless lighting"],
        }

    @staticmethod
    def _creative_decision(prompt: str) -> dict:
        match = _GAP_RE.search(prompt)
        gap = json.loads(match.group(1)) if match else {}
        dimension = str(gap.get("dimension") or "creative_direction")
        default = str(gap.get("default") or "restrained and intentional")
        alternative = _ALTERNATIVE_DECISIONS.get(dimension, "bold and immediate")
        return {
            "question": f"Which feeling should guide the {dimension.replace('_', ' ')}?",
            "why_it_matters": str(gap.get("why") or "It shapes downstream visual choices."),
            "evidence": ["The project has not locked this creative dimension."],
            "choices": [
                {"value": default, "label": default, "description": "Director recommendation"},
                {
                    "value": alternative,
                    "label": alternative,
                    "description": "Contrasting interpretation",
                },
            ],
            "default": default,
        }

    @staticmethod
    def _character(prompt: str) -> dict:
        nm = _NAME_RE.search(prompt)
        ds = _DESC_RE.search(prompt)
        name = nm.group(1).strip() if nm else "Character"
        desc = ds.group(1).strip() if ds else ""
        return {
            "visual_description": (
                f"{name}: {desc}. Consistent face and build across shots; "
                "naturalistic features, distinctive silhouette."
            ),
            "wardrobe": "signature outfit kept identical across every scene",
            "palette": "muted, cinematic",
        }

    @staticmethod
    def _identity_board(prompt: str) -> dict:
        nm = _NAME_RE.search(prompt)
        name = nm.group(1).strip() if nm else "Character"
        return {
            "canonical_face": f"{name}: consistent facial structure, same features every shot",
            "canonical_body": f"{name}: consistent build, height, and posture",
            "hair": "same hairstyle and colour across all scenes",
            "wardrobe": "signature outfit kept identical across every scene",
            "palette": "muted, cinematic",
            "do": [f"keep {name}'s face and wardrobe identical", "match the reference image"],
            "dont": ["change hairstyle", "alter wardrobe", "drift facial features"],
            "prompt_aliases": [name, f"{name} (lead)"],
        }

    @staticmethod
    def _location(prompt: str) -> dict:
        nm = _NAME_RE.search(prompt)
        ds = _DESC_RE.search(prompt)
        name = nm.group(1).strip() if nm else "Location"
        desc = ds.group(1).strip() if ds else ""
        return {
            "description": f"{name}: {desc}. A stable, reusable story environment.",
            "palette": "story-specific practical colors",
            "materials": "consistent walls, floor, surfaces, and weathering",
            "lighting": "motivated cinematic light kept consistent across shots",
            "hero_props": ["recognizable anchor prop"],
            "continuity_rules": [
                "keep entrances, windows, and hero props in the same place",
                "preserve palette and lighting direction",
            ],
            "prompt_aliases": [name, f"{name} location bible"],
        }

    @staticmethod
    def _plot(idea: str, fmt: dict | None = None, language: str = "en") -> dict:
        """A deterministic, idea-aware plot skeleton matching the plot.json schema."""
        fmt = dict(fmt or {})
        scene_count = _scene_count_for_format(fmt)
        multi_scene = scene_count > 1
        short_drama = fmt.get("name") == "short_drama_episode"
        locations = _scene_locations(scene_count, language)
        if language == "zh":
            scenes = [
                {"scene": i, "summary": f"第{i}场推进这个故事：{idea}。", "location": locations[i - 1]}
                for i in range(1, scene_count + 1)
            ]
            return {
                "logline": f"一部关于{idea}的短片。",
                "synopsis": f"故事围绕{idea}展开，主角在连续的选择中完成一次清晰的转变。",
                "themes": ["连接", "改变", "孤独"],
                "characters": [
                    {
                        "name": "主角",
                        "role": "lead",
                        "description": f"这个故事的核心人物：{idea}。",
                    },
                ],
                "locations": _location_entries(scenes),
                "arc": [{"episode": 1, "scenes": scenes}],
                "format": fmt,
            }

        scenes = [
            {
                "scene": i,
                "summary": (
                    f"Short drama beat {i} turning the premise of {idea}."
                    if short_drama
                    else f"Scene {i} of {scene_count} advancing {idea}."
                    if multi_scene
                    else f"Opening beat establishing {idea}."
                ),
                "location": locations[i - 1],
            }
            for i in range(1, scene_count + 1)
        ]
        return {
            "logline": f"A short film about {idea}.",
            "synopsis": (
                (
                    f"Across {scene_count} connected scenes, we explore {idea}. "
                    if multi_scene
                    else f"In a single self-contained scene, we explore {idea}. "
                )
                + "The protagonist faces a small but meaningful turn that pays off the premise."
            ),
            "themes": ["connection", "change", "solitude"],
            "characters": [
                {
                    "name": "Protagonist",
                    "role": "lead",
                    "description": f"The central figure of: {idea}.",
                },
            ],
            "locations": _location_entries(scenes),
            "arc": [{"episode": 1, "scenes": scenes}],
            "format": fmt,
        }

    @staticmethod
    def _concept(idea: str, language: str = "en") -> dict:
        """A deterministic non-narrative concept: one subject + one location, no story."""
        if language == "zh":
            if "男人" in idea and "狼人" in idea:
                subject = {
                    "name": "男人",
                    "role": "主体",
                    "description": "穿着完整深色西装的中年男人，正常人类形态。",
                }
                location = {
                    "name": "城市马路",
                    "description": "夜晚空旷的城市柏油马路，昏黄路灯与远处建筑轮廓。",
                    "scene": 1,
                }
            else:
                subject = {"name": "主角", "role": "主体", "description": f"画面主体：{idea}。"}
                location = {"name": "场景", "description": f"{idea} 的拍摄环境。", "scene": 1}
            logline = f"关于{idea}的短视频镜头。"
        else:
            subject = {"name": "Subject", "role": "subject", "description": f"The on-screen subject of: {idea}."}
            location = {"name": "Setting", "description": f"The environment for: {idea}.", "scene": 1}
            logline = f"A short-video shot of {idea}."
        return {
            "logline": logline,
            "synopsis": "",
            "themes": [],
            "characters": [subject],
            "location": location,
            "arc": [],
        }

    @staticmethod
    def _visual_state_changes(
        idea: str,
        characters: list[dict],
        language: str = "en",
    ) -> dict:
        """Exercise transformation identity offline without contaminating the baseline."""
        transforming = (
            ("狼人" in idea and ("变成" in idea or "变为" in idea))
            or ("werewolf" in idea.lower() and "transform" in idea.lower())
        )
        if not transforming or not characters:
            return {"characters": []}
        name = str(characters[0].get("name") or ("男人" if language == "zh" else "Man"))
        if language == "zh":
            labels = ("正常人类", "红气出现", "局部变形", "完整狼人")
            descriptions = (
                "穿完整深色西装的中年男人，尚无红气或狼化特征",
                "红色气体从身体表面出现，人体尚未变形",
                "骨骼、毛发和手部逐步狼化，仍可认出本人",
                "完整狼人形态，保留眼睛、面部结构、年龄感与服装残片",
            )
            trigger = "红色气体从身体出现"
        else:
            labels = ("Human", "Gas onset", "Partial werewolf", "Full werewolf")
            descriptions = (
                "middle-aged man in an intact dark suit, with no gas or wolf traits",
                "red gas emerges while anatomy remains fully human",
                "bones, fur, and hands progressively change while identity remains readable",
                "full werewolf retaining the man's eyes, facial geometry, age, and suit remnants",
            )
            trigger = "red gas emerges from the body"
        ids = ("human", "gas-onset", "partial-werewolf", "werewolf")
        return {
            "characters": [{
                "character": name,
                "initial_state": "human",
                "states": [
                    {
                        "id": sid,
                        "label": label,
                        "kind": "base" if index == 0 else "endpoint" if index == 3 else "transient",
                        "description": description,
                        "appearance_changes": (
                            ["species", "anatomy", "face", "body hair", "wardrobe damage"]
                            if index == 3 else []
                        ),
                        "reference_required": index in {0, 3},
                    }
                    for index, (sid, label, description) in enumerate(
                        zip(ids, labels, descriptions)
                    )
                ],
                "transitions": [{
                    "from": "human",
                    "to": "werewolf",
                    "trigger": trigger,
                    "ordered_state_ids": list(ids),
                    "preserve": ["eyes", "facial geometry", "age", "wardrobe continuity"],
                }],
            }]
        }

    @staticmethod
    def _script(plot: dict, fmt: dict | None = None, language: str = "en") -> dict:
        """A deterministic screenplay derived from the plot's arc and characters.

        One script scene per plot scene; dialogue is voiced by the plot's lead so the
        script is grounded in the plot rather than invented from nothing.
        """
        fmt = dict(fmt or plot.get("format") or {})
        short_drama = fmt.get("name") == "short_drama_episode"
        characters = plot.get("characters") or [{"name": "Protagonist"}]
        lead = characters[0]["name"]
        other = characters[1]["name"] if len(characters) > 1 else lead

        episodes = []
        for ep in plot.get("arc") or [{"episode": 1, "scenes": []}]:
            scenes = []
            for sc in ep.get("scenes", []):
                summary = sc.get("summary", "")
                if language == "zh":
                    beats = [
                        f"建立：{summary}",
                        "转折：新的压力迫使主角立刻做选择。",
                    ]
                    narration = f"旁白：{summary}" if summary else ""
                    dialogue = [
                        {"character": lead, "line": f"我们必须抓住这一刻，不能再犹豫了。"},
                        {"character": other, "line": "如果这是真的，下一步就会改变所有人。"},
                    ]
                elif short_drama:
                    beats = [
                        f"Hook: {summary}",
                        "Button: a sharp reversal forces an immediate choice.",
                    ]
                    narration = f"Narrator: {summary}" if summary else ""
                    dialogue = [
                        {
                            "character": lead,
                            "line": (
                                "This changes everything for us tonight and we will "
                                "only get a single chance to act."
                            ),
                        },
                        {
                            "character": other,
                            "line": (
                                "Then choose right now before the whole secret breaks "
                                "open and drags everyone down with it."
                            ),
                        },
                    ]
                else:
                    beats = [
                        f"Establish: {summary}",
                        "Turn: a small complication raises the stakes.",
                    ]
                    narration = f"Narrator: {summary}" if summary else ""
                    # Fixed-length lines keep the fake's per-scene runtime
                    # deterministic and independent of the idea text, so the
                    # assembled film lands inside the format's duration band.
                    dialogue = [
                        {
                            "character": lead,
                            "line": (
                                "We have waited a long time for this turning point "
                                "and now it is finally here."
                            ),
                        },
                        {
                            "character": other,
                            "line": (
                                "Then we move ahead together because everything we "
                                "have built depends on this next decisive step."
                            ),
                        },
                    ]
                scenes.append(
                    {
                        "scene": sc.get("scene"),
                        "heading": f"SCENE {sc.get('scene')}",
                        "beats": beats,
                        "dialogue": dialogue,
                        "narration": narration,
                    }
                )
            episodes.append({"episode": ep.get("episode"), "scenes": scenes})

        scenes_flat = [sc for ep in episodes for sc in ep["scenes"]]
        return {
            "title": plot.get("logline", "Untitled"),
            "episodes": episodes,
            "format": fmt,
            "structure_review": _qcz_structure_review(scenes_flat, language),
        }


_QCZ_STAGES = [
    ("起", "setup / 引入"),
    ("承", "development / 推进"),
    ("转", "turn / 转折"),
    ("合", "resolution / 升华"),
]


def _qcz_structure_review(scenes: list[dict], language: str = "en") -> dict:
    """A deterministic 起承转合 + 可执行性 review derived from the scene list.

    Mirrors the human review the tutorial calls for: which dramatic stages the script
    covers (结构完整性) and whether each scene can be rendered as an image (可执行性).
    """
    n = len(scenes)
    assigned: dict[int, list] = {}
    for i, scene in enumerate(scenes):
        idx = min(3, (i * 4) // n) if n else 0
        assigned.setdefault(idx, []).append(scene.get("scene"))

    coverage = [
        {
            "stage": stage,
            "label": label,
            "covered": bool(assigned.get(idx)),
            "scenes": assigned.get(idx, []),
        }
        for idx, (stage, label) in enumerate(_QCZ_STAGES)
    ]
    note = "画面可由分镜直接生成" if language == "zh" else "renderable from the scene description"
    executability = [
        {"scene": scene.get("scene"), "executable": True, "note": note}
        for scene in scenes
    ]
    complete = all(entry["covered"] for entry in coverage)
    if language == "zh":
        summary = "结构完整,起承转合齐备。" if complete else "结构待完善,部分阶段尚未覆盖。"
    else:
        summary = (
            "Structure complete: setup, development, turn, and resolution are all present."
            if complete
            else "Structure incomplete: some 起承转合 stages are not yet covered."
        )
    return {
        "coverage": coverage,
        "executability": executability,
        "complete": complete,
        "summary": summary,
    }


def _speech_units(text: str) -> int:
    words = text.split()
    if len(words) > 1:
        return len(words)
    cjk = re.findall(r"[\u3400-\u9fff]", text)
    if cjk:
        return max(1, round(len(cjk) / 2))
    return len(words)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A minimal, valid solid-color PNG built with the standard library only."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + bytes(rgb) * width  # filter byte 0 + pixels
    raw = row * height
    idat = zlib.compress(raw, 9)
    return sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


class FakeImageGen(ImageGen):
    """Deterministic stand-in for an image model. No network, no cost.

    Writes a solid-color PNG whose colour is derived from the locked ``seed`` so the
    same seed always yields the identical reference image (the consistency anchor,
    invariant #4) and different seeds differ.
    """

    name = "fake"
    model = "fake-image-1"

    def __init__(self, *, supports_storyboard_grid: bool = True):
        self._supports_storyboard_grid = supports_storyboard_grid

    def generate(
        self,
        prompt: str,
        *,
        out_path: str,
        reference_images: list[str] | None = None,
        **kwargs,
    ) -> Generation:
        seed = int(kwargs.get("seed", 0))
        refs = list(reference_images or [])
        # Fold the references' content into the deterministic colour so the same seed +
        # same refs always match, while different refs change the output (proves wiring).
        ref_mix = 0
        for ref in refs:
            ref_mix = zlib.crc32(_sha(ref).encode("utf-8"), ref_mix)
        mix = (seed + ref_mix) & 0x7FFFFFFF
        rgb = (mix * 53 % 256, mix * 97 % 256, mix * 193 % 256)
        png = _solid_png(64, 64, rgb)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        return Generation(
            content=str(path),
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=0.02,
            meta={"seed": seed, "prompt": prompt, "reference_images": refs},
        )


class FakeReferenceAnalyzer(ReferenceAnalyzer):
    """Deterministic uploaded-reference classifier. No network, no cost."""

    name = "fake"
    model = "fake-reference-analyzer-1"

    def analyze(
        self,
        image_path: str,
        *,
        aliases: list[str] | None = None,
        user_note: str = "",
    ) -> Generation:
        text = " ".join([Path(image_path).stem, " ".join(aliases or []), user_note]).lower()
        target_type = ""
        target_id = ""
        confidence = 0.25
        reason = "no strong filename, alias, or note cue"

        if any(word in text for word in ("style", "mood", "palette", "look", "vibe")):
            target_type = "style"
            target_id = "global"
            confidence = 0.82
            reason = "style cue in filename, alias, or note"
        elif any(word in text for word in (
            "location", "background", "room", "apartment", "house", "home",
            "school", "shop", "street", "forest", "city", "beach",
        )):
            target_type = "location"
            target_id = _title_from_filename(image_path, drop={
                "location", "background", "room", "reference", "ref", "image", "photo",
            }) or "Location"
            confidence = 0.82
            reason = "location cue in filename, alias, or note"
        elif any(word in text for word in (
            "character", "person", "face", "hero", "heroine", "protagonist", "lead",
        )):
            target_type = "character"
            target_id = _title_from_filename(image_path, drop={
                "character", "person", "face", "reference", "ref", "image", "photo",
            }) or "protagonist"
            confidence = 0.82
            reason = "character cue in filename, alias, or note"

        return Generation(
            content={
                "target_type": target_type,
                "target_id": target_id,
                "confidence": confidence,
                "reason": reason,
                "visual_summary": f"Fake analysis of {Path(image_path).name}",
            },
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=0.01,
        )

    def describe(
        self,
        image_paths: list[str],
        *,
        prompt: str,
        language: str = "en",
    ) -> Generation:
        # Deterministic offline stand-in: echo a stable identity tagged to the reference file
        # so the vision path is exercised without a real model.
        stem = Path(image_paths[0]).stem if image_paths else "reference"
        return Generation(
            content={
                "visual_description": f"subject as shown in reference {stem}",
                "wardrobe": f"wardrobe as shown in reference {stem}",
                "palette": "as shown in the reference",
                "canonical_face": f"face as shown in reference {stem}",
                "canonical_body": "body as shown in the reference",
                "hair": "hair as shown in the reference",
                "description": f"location as shown in reference {stem}",
                "materials": "materials as shown in the reference",
                "lighting": "lighting as shown in the reference",
                "do": ["match the reference exactly"],
                "dont": ["do not deviate from the reference"],
                "prompt_aliases": [stem],
            },
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=0.01,
        )

    def revise(
        self,
        image_paths: list[str],
        *,
        prompt: str,
        language: str = "en",
    ) -> Generation:
        # Deterministic offline stand-in: echo the current artifact with a grounded note that
        # records which reference images were seen, so the vision-revision path is exercised
        # and assertable without a real model.
        stems = [Path(p).stem for p in image_paths]
        if FakeLLM._task(prompt) == "prompt_conversation":
            return Generation(
                content=_prompt_conversation_revision(prompt, grounded_in=stems),
                provider=self.name,
                model=self.model,
                cost_usd=0.0,
                seconds=0.01,
                meta={"task": "prompt_conversation"},
            )
        json_match = _CURRENT_JSON_RE.search(prompt)
        if json_match:
            current = json.loads(json_match.group(1))
            instr = _INSTRUCTION_RE.search(prompt)
            instruction = instr.group(1).strip() if instr else "fake revision"
            if isinstance(current, dict):
                revised = dict(current)
                revised["revision_note"] = instruction
                revised["grounded_in"] = stems
                text = json.dumps(revised, ensure_ascii=False)
            else:
                text = json_match.group(1)
        else:
            marker = "CURRENT_TEXT:"
            current_text = prompt.split(marker, 1)[1].strip() if marker in prompt else ""
            text = (current_text + f"\n\nGrounded in: {', '.join(stems)}").strip()
        return Generation(
            content=text,
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=0.01,
        )


class FakeStyleProfiler(StyleProfiler):
    """Deterministic style profiler. No network, no cost.

    Builds a complete style dict from the user's text and/or the uploaded reference's
    filename so the offline pipeline exercises the same threading a real VLM profiler
    would feed. A real profiler would *look* at the image; the fake derives a stable
    descriptor from its name so tests stay deterministic.
    """

    name = "fake"
    model = "fake-style-profiler-1"

    def profile(
        self,
        *,
        description: str = "",
        image_path: str | None = None,
        language: str = "en",
        feedback: str = "",
    ) -> Generation:
        text = (description or "").strip()
        image_label = ""
        if image_path:
            image_label = _title_from_filename(image_path, drop={
                "style", "mood", "look", "palette", "reference", "ref", "image", "photo",
            })

        # Text refines an uploaded reference; otherwise whichever was provided wins.
        look = text or (f"{image_label} reference style".strip() if image_label else "cinematic")
        label = (text or image_label or "Custom style").strip()
        # Keep the label short and human-facing for the gate / bible.
        label = " ".join(label.split()[:6]) or "Custom style"

        source = "uploaded reference image" if image_path else "the described style"
        playbook_bits = [
            f"Render every shot in this locked project style: {look}.",
            f"Match the medium, color palette, lighting, lens/grain, and texture of {source}.",
        ]
        if image_path:
            playbook_bits.append(
                "Borrow style only from the reference — never copy its subject or composition."
            )
        prompt_playbook = " ".join(playbook_bits)

        fb = (feedback or "").strip()
        if fb:
            look = f"{look} (adjusted: {fb})"
            label = " ".join(f"{label} {fb}".split()[:6]) or "Custom style"
            prompt_playbook = f"{prompt_playbook} Adjust per the user's note: {fb}."

        style = {
            "look": look,
            "label": label,
            "palette": "as described" if not image_path else "sampled from the reference",
            "aspect_ratio": "16:9",
            "medium": "as described" if not image_path else "read from the reference",
            "idiom": look,
            "rendering": "consistent project look",
            "lighting": "consistent key/soft fill",
            "color_grade": "consistent project grade",
            "line_style": "consistent with the chosen style",
            "lens": "consistent depth of field",
            "motion": "measured pacing",
            "atmosphere": "consistent ambient texture",
            "notes": f"Custom user-defined style ({language}).",
            "prompt_playbook": prompt_playbook,
        }
        return Generation(
            content=style,
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=0.01,
            meta={"input": "style", "saw_image": bool(image_path)},
        )


def _title_from_filename(path: str, *, drop: set[str]) -> str:
    stem = re.sub(r"^[0-9a-f]{12}-", "", Path(path).stem, flags=re.IGNORECASE)
    words = [
        word for word in re.split(r"[^a-zA-Z0-9]+", stem)
        if word and word.lower() not in drop
    ]
    return " ".join(words).strip().title()


def _sha(path) -> str:
    p = Path(path) if path else None
    if p and p.is_file():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return ""


class FakeVideoGen(VideoGen):
    """Deterministic stand-in for a video model. No network, no cost, no ffmpeg.

    Writes a deterministic placeholder clip from the shot's keyframe, locked seed,
    duration, bible reference images, and the previous shot's last frame (carry-forward
    continuity). The bytes are a stable function of those inputs, so re-running yields
    the identical clip while a different duration, reference, or carried frame changes
    it. The assemble stage is where ffmpeg turns keyframes + durations into the actual
    playable mp4.
    """

    name = "fake"
    model = "fake-video-1"

    def __init__(self, *, supports_storyboard_grid: bool = True):
        self._supports_storyboard_grid = supports_storyboard_grid

    @property
    def capabilities(self) -> VideoCapabilities:
        # getattr fallback so FakeVideoGen subclasses that override __init__ without calling
        # super() (e.g. test recorders) don't raise here — a raised property would be
        # swallowed by video.py's getattr(..., default) and silently drop native-audio.
        return VideoCapabilities(
            supports_native_audio=True,
            supports_storyboard_grid=getattr(self, "_supports_storyboard_grid", True),
            min_duration_s=1,
            max_duration_s=15,
        )

    def generate(self, prompt: str, *, out_path: str, **kwargs) -> Generation:
        seed = int(kwargs.get("seed", 0))
        # ``duration_s=None`` means "provider default" — the fake resolves it to a
        # deterministic length and records it in meta so tests can assert the path.
        raw_duration = kwargs.get("duration_s", 1.0)
        duration = 4.0 if raw_duration is None else float(raw_duration)
        keyframe_path = kwargs.get("keyframe_path")
        last_frame_ref = kwargs.get("last_frame_ref")
        reference_images = list(kwargs.get("reference_images") or [])

        header = json.dumps(
            {
                "seed": seed,
                "duration_s": duration,
                "keyframe": _sha(keyframe_path),
                "last_frame": _sha(last_frame_ref),
                "reference_images": [_sha(ref) for ref in reference_images],
                "prompt": prompt,
            },
            sort_keys=True,
        )
        kf_bytes = Path(keyframe_path).read_bytes() if keyframe_path and Path(keyframe_path).is_file() else b""
        content = b"FAKECLIP\n" + header.encode("utf-8") + b"\n" + kf_bytes

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        generate_audio = bool(kwargs.get("generate_audio", False))
        native_audio_path = None
        if generate_audio:
            audio = path.with_name(f"{path.stem}.native.source.wav")
            audio.write_bytes(_silent_wav(duration, sample_rate=48000, channels=2))
            native_audio_path = str(audio)
        return Generation(
            content=str(path),
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=duration,
            meta={
                "seed": seed,
                "duration_s": duration,
                "last_frame_ref": last_frame_ref,
                "reference_images": reference_images,
                "generate_audio": generate_audio,
                "native_audio_path": native_audio_path,
            },
        )


class FakeVLMCheck(VLMCheck):
    """Deterministic stand-in for a video-understanding QC model. No network, no cost.

    A structurally valid fake clip or MP4 container passes every dimension. A clip that
    is missing, empty, not a recognizable clip, or carries a ``QC_FAIL`` marker fails
    with a timestamped high-severity issue — giving an offline, deterministic way to
    exercise the review gate (a real VLM-backed check replaces this in M1).
    """

    name = "fake"
    model = "fake-vlm-1"

    @property
    def supports_audio_review(self) -> bool:
        return True

    def review(self, clip_path: str, *, prompt: str, **kwargs) -> Generation:
        data = Path(clip_path).read_bytes() if Path(clip_path).is_file() else b""
        playable = _looks_like_fake_clip(data) or _looks_like_mp4(data)
        flawed = (b"QC_FAIL" in data) or (not playable)

        checks = []
        for dim in _QC_DIMENSIONS:
            # The injected/structural flaw surfaces on motion + artifacts.
            fails = flawed and dim in ("motion_anatomy", "artifacts")
            if not playable and dim == "shot_instruction_adherence":
                fails = True
            checks.append(
                {
                    "dimension": dim,
                    "passed": not fails,
                    "severity": "high" if fails else "none",
                    "detail": ("unplayable or corrupted clip" if not playable
                               else "injected QC_FAIL marker" if fails
                               else "ok"),
                    "timestamp": "00:00" if fails else None,
                }
            )

        status = qc_report_status(checks)
        return Generation(
            content={
                "clip": Path(clip_path).name,
                "prompt": prompt,
                "checks": checks,
                **status,
            },
            provider=self.name,
            model=self.model,
            cost_usd=0.0,
            seconds=0.03,
            meta={"overall_pass": status["overall_pass"]},
        )


def _looks_like_fake_clip(data: bytes) -> bool:
    return data.startswith(b"FAKECLIP\n") and len(data) > len(b"FAKECLIP\n")


def _looks_like_mp4(data: bytes) -> bool:
    # MP4 files normally carry the ftyp box near the start; this is not a full media
    # validator, just a deterministic fake-QC stand-in that accepts real provider MP4s.
    return len(data) > 12 and b"ftyp" in data[:64]


def estimate_speech_seconds(text: str) -> float:
    """Rough read time used when no explicit duration is given (~150 wpm)."""
    words = len(text.split())
    return max(1.0, round(words * 0.4, 1))


def _silent_wav(
    duration_s: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """A valid, deterministic silent PCM WAV — stdlib only, ffmpeg-readable."""
    frames = max(1, int(round(duration_s * sample_rate)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * frames * channels)
    return buf.getvalue()


class FakeTTS(TTS):
    """Deterministic stand-in for text-to-dialogue. Writes silent WAV, no cost."""

    name = "fake"
    model = "fake-tts-1"

    def speak(self, text: str, *, out_path: str, **kwargs) -> Generation:
        duration = kwargs.get("duration_s")
        duration = float(duration) if duration is not None else estimate_speech_seconds(text)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_silent_wav(duration))
        return Generation(
            content=str(path), provider=self.name, model=self.model,
            cost_usd=0.0, seconds=duration, meta={"duration_s": duration},
        )


class FakeMusic(Music):
    """Deterministic stand-in for music generation. Writes silent WAV, no cost."""

    name = "fake"
    model = "fake-music-1"

    def compose(self, prompt: str, *, out_path: str, **kwargs) -> Generation:
        duration = float(kwargs.get("duration_s", 10.0))
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_silent_wav(duration))
        return Generation(
            content=str(path), provider=self.name, model=self.model,
            cost_usd=0.0, seconds=duration, meta={"duration_s": duration, "prompt": prompt},
        )
