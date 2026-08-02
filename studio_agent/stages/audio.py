"""Audio stage: editable native tracks for new projects, legacy synthesis on resume.

For each shot that has dialogue it synthesizes a line of speech sized to the shot's
duration (dialogue already drove shot durations upstream, so audio and video stay in
sync), and — when ``music_enabled`` is set — it composes a single music bed spanning the
whole piece. Spoken text and the
music brief are saved as editable artifacts under ``storyboard/prompts/`` (invariant #8),
and ``assets/audio/audio.json`` lists the tracks for the assemble stage.

Idempotent (invariants #3/#5): skips the whole stage if complete; reuses hand-edited
prompts and re-renders only audio files that are missing.
"""

from __future__ import annotations

import json

from .base import Providers, Stage, StageResult
from ..audio_native import NativeAudioError, extract_native_audio
from ..style import format_style_prompt


def _music_enabled(project) -> bool:
    return bool(project.model_config.get("music_enabled"))


class AudioStage(Stage):
    name = "audio"

    def __init__(self, *, extractor=extract_native_audio):
        self._extractor = extractor

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="audio already complete")
        if project.model_config.get("audio_mode") == "native_video":
            return self._run_native(project, providers)
        return self._run_legacy(project, providers)

    def _run_native(self, project, providers: Providers) -> StageResult:
        clips_dir = project.path("assets", "clips")
        clips = json.loads(clips_dir.joinpath("clips.json").read_text())["clips"]
        # 旁白: narration is non-diegetic — the native clip never carries it (the video
        # prompt forbids lip-syncing it), so synthesize it as a separate VO stem that the
        # assemble stage mixes over the native production track.
        narration_by_id = self._narration_by_shot(project)
        tracks = []
        narration = []
        for entry in clips:
            shot_id = str(entry["id"])
            source = clips_dir / str(entry.get("native_audio") or entry["clip"])
            destination = project.path("assets", "audio", f"{shot_id}.native.wav")
            if not destination.is_file():
                try:
                    self._extractor(source, destination)
                except NativeAudioError as exc:
                    return StageResult(status="failed", message=f"{shot_id}: {exc}")
            tracks.append({
                "id": shot_id,
                "file": destination.name,
                "source_clip": str(entry["clip"]),
                "duration_s": float(entry.get("duration_s") or 0.0),
            })
            shot = narration_by_id.get(shot_id)
            if shot is not None and str(shot.get("narration") or "").strip():
                narration.append(self._narration_for(project, shot, providers))

        music = None
        if _music_enabled(project):
            total_s = round(sum(float(t.get("duration_s") or 0.0) for t in tracks), 1)
            music = self._music_for(project, total_s, providers)

        manifest = {"mode": "native_video", "tracks": tracks}
        if music:
            manifest["music"] = music
        if narration:
            manifest["narration"] = narration
        project.path("assets", "audio", "audio.json").write_text(
            json.dumps(manifest, indent=2)
        )
        message = f"native audio: {len(tracks)} track(s)"
        if music:
            message += " + music"
        if narration:
            message += f" + {len(narration)} narration"
        return StageResult(status="complete", message=message)

    def _narration_by_shot(self, project) -> dict:
        shots_path = project.path("storyboard", "shots.json")
        if not shots_path.is_file():
            return {}
        shots = json.loads(shots_path.read_text()).get("shots", [])
        return {str(s.get("id")): s for s in shots}

    def _run_legacy(self, project, providers: Providers) -> StageResult:
        shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
        rendered = self._rendered_durations(project)
        total_s = round(
            sum(self._shot_duration(s, rendered, 0.0) for s in shots), 1
        )

        dialogue = []
        narration = []
        for shot in shots:
            if shot.get("dialogue"):
                dialogue.append(self._dialogue_for(project, shot, providers))
            # 旁白: off-screen narrator VO, a separate stem from lip-synced dialogue.
            if str(shot.get("narration") or "").strip():
                narration.append(self._narration_for(project, shot, providers))

        music = self._music_for(project, total_s, providers) if _music_enabled(project) else None

        manifest = {"dialogue": dialogue, "narration": narration}
        if music:
            manifest["music"] = music
        project.path("assets", "audio", "audio.json").write_text(json.dumps(manifest, indent=2))
        return StageResult(
            status="complete",
            message=(
                f"audio: {len(dialogue)} line(s) + {len(narration)} narration"
                + (" + music" if music else "")
            ),
        )

    @staticmethod
    def _rendered_durations(project) -> dict[str, float]:
        """Real seconds recorded per clip by the video stage (clips.json)."""
        from .video import VideoStage

        return VideoStage._manifest_durations(project)

    @staticmethod
    def _shot_duration(shot, rendered: dict[str, float], default: float) -> float:
        """A concrete duration for audio work.

        ``duration_s: null`` deferred the clip's length to the video model; by the audio
        stage the rendered length is in clips.json, so read it from there.
        """
        value = shot.get("duration_s", default)
        if value is None:
            value = rendered.get(str(shot.get("id")), default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _dialogue_for(self, project, shot, providers: Providers) -> dict:
        shot_id = shot["id"]
        duration = self._shot_duration(shot, self._rendered_durations(project), 2.0)
        text = self._ensure_dialogue_prompt(project, shot)
        wav = project.path("assets", "audio", f"{shot_id}.dialogue.wav")
        if not wav.is_file():
            project.assert_budget_available()
            gen = providers.tts.speak(text, out_path=str(wav), duration_s=duration)
            project.add_generation_cost(stage=self.name, generation=gen)
        return {"id": shot_id, "file": wav.name, "duration_s": duration}

    def _narration_for(self, project, shot, providers: Providers) -> dict:
        shot_id = shot["id"]
        duration = self._shot_duration(shot, self._rendered_durations(project), 2.0)
        text = self._ensure_narration_prompt(project, shot)
        wav = project.path("assets", "audio", f"{shot_id}.narration.wav")
        if not wav.is_file():
            project.assert_budget_available()
            gen = providers.tts.speak(text, out_path=str(wav), duration_s=duration)
            project.add_generation_cost(stage=self.name, generation=gen)
        return {"id": shot_id, "file": wav.name, "duration_s": duration}

    def _music_for(self, project, total_s: float, providers: Providers) -> dict:
        prompt = self._ensure_music_prompt(project)
        wav = project.path("assets", "audio", "music.wav")
        if not wav.is_file():
            project.assert_budget_available()
            gen = providers.music.compose(prompt, out_path=str(wav), duration_s=total_s)
            project.add_generation_cost(stage=self.name, generation=gen)
        return {"file": wav.name, "duration_s": total_s}

    def _ensure_dialogue_prompt(self, project, shot) -> str:
        path = project.path("storyboard", "prompts", f"{shot['id']}.audio.md")
        if path.is_file():
            return path.read_text()
        text = "\n".join(f"{d.get('character', '')}: {d.get('line', '')}"
                         for d in shot.get("dialogue", []))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return text

    def _ensure_narration_prompt(self, project, shot) -> str:
        path = project.path("storyboard", "prompts", f"{shot['id']}.narration.md")
        if path.is_file():
            return path.read_text()
        text = str(shot.get("narration") or "").strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return text

    def _ensure_music_prompt(self, project) -> str:
        path = project.path("storyboard", "prompts", "music.audio.md")
        if path.is_file():
            return path.read_text()
        plot_path = project.path("story", "plot.json")
        plot = json.loads(plot_path.read_text()) if plot_path.is_file() else {}
        style = dict(project.model_config.get("style") or {})
        themes = ", ".join(plot.get("themes", [])) or "the story's mood"
        mood = str(project.model_config.get("music_mood") or "").strip()
        mood_clause = f"Mood: {mood}. " if mood else ""
        prompt = (
            f"Instrumental score for: {plot.get('logline', 'the film')}. "
            f"Evoke {themes}. {mood_clause}{format_style_prompt(style)} "
            "Underscore, no vocals; dynamics follow the cut."
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt)
        return prompt
