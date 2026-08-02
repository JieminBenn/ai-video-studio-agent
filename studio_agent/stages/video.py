"""Video stage: image-to-video clip per shot, with honest last-frame carry-forward.

For each shot in ``storyboard/shots.json`` it animates the shot's keyframe into a clip,
conditioned on the shot's locked reference seed, bible reference images where present,
and on the **previous shot's actual final frame** for continuity across cuts. After each
clip is produced, the shot's real last frame is captured as an artifact
(``assets/clips/<id>.last_frame.png``) — downloaded from the provider's returned last
frame when available, else extracted from the generated mp4, else (offline/fake) the
shot's keyframe — and that artifact is fed into the next shot's generation. The carry is
only sent to providers whose capabilities report last-frame support.

The preceding ``video_prompts`` stage saves and approves each editable motion prompt.
This stage verifies those exact bytes, calls no LLM or prompt compiler, and writes an
ordered ``assets/clips/clips.json`` manifest (clip path + duration + produced/carried
last frame) for the later assemble stage.

Idempotent (invariants #3/#5): skips the whole stage if complete; otherwise reuses an
existing last-frame artifact and re-renders only clips that are missing, so a re-run
never duplicates a paid generation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .base import Providers, Stage, StageResult
from .video_prompts import (
    capability_signature,
    prepare_video_binding,
    resolve_video_binding,
)
from ..asset_regeneration import (
    pending_regeneration,
    promote_candidate,
    record_regeneration_error,
)
from ..assembly.ffmpeg_edit import _probe_duration, extract_last_frame
from ..invalidation import invalidate_shots
from ..prompt_approvals import (
    PromptApprovalError,
    prompt_path,
    require_prompt_batch_approval,
)
from ..providers.base import VideoCapabilities


def _download_last_frame(url: str, out_path: str) -> bool:
    """Download a provider-returned last-frame image to ``out_path``."""
    from urllib.request import urlopen

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response, out.open("wb") as fh:  # noqa: S310 - provider URL
            shutil.copyfileobj(response, fh)
    except Exception:  # pragma: no cover - exercised only by live network failures
        return False
    return out.is_file() and out.stat().st_size > 0


class VideoStage(Stage):
    name = "video"

    def __init__(self, *, downloader=None, frame_extractor=None):
        # Injectable so the provider-URL and mp4-extraction paths are testable offline.
        self._download = downloader or _download_last_frame
        self._extract_last_frame = frame_extractor or extract_last_frame

    def run(self, project, providers: Providers) -> StageResult:
        if project.stage_status(self.name) == "complete":
            return StageResult(status="skipped", message="video already complete")

        require_prompt_batch_approval(project, "videos")
        shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
        capabilities = getattr(providers.video, "capabilities", VideoCapabilities())
        native_audio = str(project.model_config.get("audio_mode") or "") == "native_video"
        if native_audio and not capabilities.supports_native_audio:
            raise ValueError(
                f"video provider '{getattr(providers.video, 'name', 'unknown')}' does not support "
                "the project's native audio mode"
            )

        clips = []
        previous_durations = self._manifest_durations(project)
        prev_last_frame = None  # the previous shot's real final frame, carried forward
        for index, shot in enumerate(shots):
            shot_id = str(shot["id"])
            keyframe = project.path("storyboard", "keyframes", shot["keyframe"])
            clip_path = project.path("assets", "clips", f"{shot_id}.mp4")
            last_frame_path = project.path("assets", "clips", f"{shot_id}.last_frame.png")
            request = pending_regeneration(project, shot_id, "video")

            binding = shot.get("video_prompt_binding")
            if isinstance(binding, dict):
                if binding.get("capabilities") != capability_signature(capabilities):
                    raise PromptApprovalError(
                        f"video provider capabilities changed after prompt preparation for "
                        f"{shot['id']}"
                    )
            else:
                _effective, _effective_caps, binding = prepare_video_binding(
                    project,
                    shot,
                    capabilities,
                    has_expected_carry=(
                        index > 0
                        and capabilities.supports_last_frame
                        and int(capabilities.max_image_inputs) > 1
                    ),
                )

            reference_images, target_state_reference_images = resolve_video_binding(
                project, binding
            )
            if len(reference_images) != len(binding.get("reference_images") or []):
                raise PromptApprovalError(
                    f"video reference binding changed after prompt preparation for {shot['id']}"
                )
            carry = (
                str(prev_last_frame)
                if prev_last_frame and binding.get("expects_previous_last_frame")
                else None
            )
            prompt_text = prompt_path(project, "videos", shot).read_text()

            gen = None
            if request is not None:
                candidate_root = project.path(
                    "assets", ".candidates", str(request["request_id"])
                )
                candidate_clip = candidate_root / f"{shot_id}.mp4"
                candidate_last = candidate_root / f"{shot_id}.last_frame.png"
                try:
                    project.assert_budget_available()
                    gen = providers.video.generate(
                        prompt_text,
                        out_path=str(candidate_clip),
                        keyframe_path=str(keyframe),
                        last_frame_ref=carry,
                        reference_images=reference_images,
                        target_state_reference_images=target_state_reference_images,
                        seed=shot.get("reference_seed", 0),
                        duration_s=shot.get("duration_s", 2.0),
                        generate_audio=native_audio,
                    )
                    project.add_generation_cost(stage=self.name, generation=gen)
                    self._ensure_last_frame(candidate_clip, keyframe, candidate_last, gen)
                    live_family = [
                        f"assets/clips/{shot_id}.mp4",
                        f"assets/clips/{shot_id}.last_frame.png",
                    ]
                    moves = {
                        candidate_clip: live_family[0],
                        candidate_last: live_family[1],
                    }
                    if native_audio:
                        candidate_audio = candidate_root / f"{shot_id}.native.source.wav"
                        live_audio_rel = f"assets/clips/{shot_id}.native.source.wav"
                        live_family.append(live_audio_rel)
                        if candidate_audio.is_file():
                            moves[candidate_audio] = live_audio_rel
                    invalidate_shots(
                        project,
                        [shot_id],
                        from_stage="video",
                        reason="video-candidate-ready",
                        source_path=live_family[0],
                        preserve_paths=live_family,
                    )
                    promote_candidate(
                        project,
                        request,
                        candidate_to_live=moves,
                        live_family=live_family,
                    )
                    if native_audio and gen is not None:
                        live_audio = project.path(
                            "assets", "clips", f"{shot_id}.native.source.wav"
                        )
                        gen.meta = dict(gen.meta or {})
                        gen.meta["native_audio_path"] = str(live_audio)
                except Exception as exc:
                    record_regeneration_error(project, shot_id, "video", exc)
                    raise
            elif not clip_path.is_file():
                project.assert_budget_available()  # honor the cap before paid work
                gen = providers.video.generate(
                    prompt_text,
                    out_path=str(clip_path),
                    keyframe_path=str(keyframe),
                    last_frame_ref=carry,
                    reference_images=reference_images,
                    target_state_reference_images=target_state_reference_images,
                    seed=shot.get("reference_seed", 0),
                    duration_s=shot.get("duration_s", 2.0),
                    generate_audio=native_audio,
                )
                project.add_generation_cost(stage=self.name, generation=gen)

            if request is None:
                self._ensure_last_frame(clip_path, keyframe, last_frame_path, gen)

            clip = {
                "id": shot["id"],
                "clip": clip_path.name,
                "duration_s": self._resolved_duration(
                    shot, gen, clip_path, previous_durations
                ),
                "last_frame": last_frame_path.name,
                "last_frame_ref": carry,
            }
            if native_audio:
                native_audio_source = self._native_audio_source(project, shot["id"], gen)
                if native_audio_source:
                    clip["native_audio"] = native_audio_source
            clips.append(clip)
            prev_last_frame = last_frame_path

        project.path("assets", "clips", "clips.json").write_text(
            json.dumps({"clips": clips}, indent=2)
        )
        return StageResult(status="complete", message=f"video: {len(clips)} clip(s)")

    @staticmethod
    def _manifest_durations(project) -> dict[str, float]:
        """Durations already recorded in clips.json, for idempotent re-runs."""
        manifest_path = project.path("assets", "clips", "clips.json")
        if not manifest_path.is_file():
            return {}
        try:
            entries = json.loads(manifest_path.read_text()).get("clips", [])
        except ValueError:
            return {}
        return {
            str(entry.get("id")): float(entry["duration_s"])
            for entry in entries
            if isinstance(entry, dict) and entry.get("duration_s") is not None
        }

    @staticmethod
    def _resolved_duration(shot, gen, clip_path, previous_durations) -> float:
        """A concrete duration for the clips.json manifest.

        A shot with ``duration_s: null`` deferred the length to the video model, but the
        assemble/audio stages need real seconds — so record what actually rendered: the
        provider-reported duration, the previously recorded manifest value on a re-run,
        or the file itself via ffprobe.
        """
        duration = shot.get("duration_s", 2.0)
        if duration is not None:
            return duration
        meta_duration = (gen.meta or {}).get("duration_s") if gen is not None else None
        try:
            if meta_duration is not None:
                return float(meta_duration)
        except (TypeError, ValueError):
            pass
        prior = previous_durations.get(str(shot["id"]))
        if prior is not None:
            return prior
        probed = _probe_duration(clip_path)
        return probed if probed else 2.0

    @staticmethod
    def _native_audio_source(project, shot_id: str, gen) -> str | None:
        raw_path = gen.meta.get("native_audio_path") if gen is not None else None
        candidate = (
            Path(raw_path)
            if raw_path
            else project.path("assets", "clips", f"{shot_id}.native.source.wav")
        )
        project_root = project.dir.resolve()
        if not candidate.is_absolute():
            cwd_relative = candidate.resolve()
            try:
                cwd_relative.relative_to(project_root)
                candidate = cwd_relative
            except ValueError:
                candidate = (project.dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        try:
            candidate.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(f"native audio sidecar is outside project: {candidate}") from exc
        clips_root = project.path("assets", "clips").resolve()
        if candidate.parent != clips_root:
            raise ValueError(
                f"native audio sidecar must be stored directly under assets/clips: {candidate}"
            )
        if not candidate.is_file():
            return None
        return candidate.name

    def _ensure_last_frame(self, clip_path: Path, keyframe: Path, out_path: Path, gen) -> Path:
        """Capture the clip's real final frame, honestly and idempotently.

        Precedence: a fresh provider's returned ``last_frame_url`` -> extract from the
        generated mp4 -> fall back to the shot's keyframe (the fake clip's own frame
        offline). An existing artifact is reused so re-runs do no extra work.
        """
        if out_path.is_file():
            return out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        url = gen.meta.get("last_frame_url") if gen is not None else None
        if url and self._download(url, str(out_path)):
            return out_path
        if self._extract_last_frame(str(clip_path), str(out_path)):
            return out_path
        shutil.copy(keyframe, out_path)
        return out_path
