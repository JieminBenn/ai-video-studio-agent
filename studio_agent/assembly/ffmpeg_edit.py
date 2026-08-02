"""EDL builder + FFmpeg renderer (timeline.json -> output/<id>.mp4).

``build_timeline`` is pure and deterministic — it turns the clips/audio/shots manifests
into an ordered edit decision list and writes ``edit/timeline.json``. The render step is
isolated behind a renderer object so the deterministic core can be tested without ffmpeg.

For the fake path the generated "clips" are placeholders, so the playable mp4 falls
back to real keyframe PNGs (one segment per shot, held for the shot's duration) with
dialogue + music muxed in. When a real provider produced a playable MP4 clip, the
renderer uses that clip directly in the final cut.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

FPS = 25
WIDTH = 1280
HEIGHT = 720


def build_timeline(project) -> dict:
    """Build the EDL from the clips/audio/shots manifests and persist timeline.json."""
    shots = json.loads(project.path("storyboard", "shots.json").read_text())["shots"]
    clips = json.loads(project.path("assets", "clips", "clips.json").read_text())["clips"]
    audio, mode = _load_audio_manifest(project)

    keyframe_by_id = {s["id"]: s.get("keyframe") for s in shots}
    dialogue_by_id = {d["id"]: d["file"] for d in audio.get("dialogue", [])}
    narration_by_id = {n["id"]: n["file"] for n in audio.get("narration", [])}
    native_by_id = _native_tracks_by_id(audio) if mode == "native_video" else {}

    video_track = []
    for clip in clips:  # clips.json defines the ordered cut
        sid = clip["id"]
        segment = {
            "id": sid,
            "clip": clip.get("clip"),
            "keyframe": keyframe_by_id.get(sid),
            "duration_s": clip.get("duration_s", 0.0),
        }
        if mode == "native_video":
            if sid not in native_by_id:
                raise ValueError(f"native audio manifest missing track for shot {sid}")
            segment["audio"] = native_by_id[sid]
            segment["narration"] = narration_by_id.get(sid)
        else:
            segment["dialogue"] = dialogue_by_id.get(sid)
            segment["narration"] = narration_by_id.get(sid)
        video_track.append(segment)

    total = round(sum(seg["duration_s"] for seg in video_track), 1)
    timeline = {
        "fps": FPS,
        "total_duration_s": total,
        "audio_mode": mode,
        "video_track": video_track,
    }
    timeline["music"] = (audio.get("music") or {}).get("file")
    project.path("edit", "timeline.json").write_text(json.dumps(timeline, indent=2))
    return timeline


def _load_audio_manifest(project) -> tuple[dict, str]:
    configured_mode = project.model_config.get("audio_mode")
    if configured_mode not in (None, "legacy", "native_video"):
        raise ValueError(f"unsupported project audio_mode: {configured_mode!r}")

    audio_path = project.path("assets", "audio", "audio.json")
    if not audio_path.is_file():
        if configured_mode == "native_video":
            raise RuntimeError("native audio manifest missing: assets/audio/audio.json")
        return {}, "legacy"

    try:
        audio = json.loads(audio_path.read_text())
    except json.JSONDecodeError as exc:
        label = "native audio" if configured_mode == "native_video" else "audio"
        raise ValueError(f"invalid {label} manifest JSON: {exc.msg}") from exc
    if not isinstance(audio, dict):
        raise ValueError("audio manifest must be a JSON object")

    manifest_mode = audio.get("mode")
    if configured_mode == "native_video":
        if manifest_mode != "native_video":
            raise ValueError(
                "native audio manifest must declare mode 'native_video' "
                f"(got {manifest_mode!r})"
            )
        return audio, "native_video"

    if configured_mode == "legacy":
        if manifest_mode not in (None, "legacy"):
            raise ValueError(
                "legacy audio manifest must omit mode or declare mode 'legacy' "
                f"(got {manifest_mode!r})"
            )
        return audio, "legacy"

    # Backward-compatible inference for projects created before audio_mode was stored.
    if manifest_mode in (None, "legacy"):
        return audio, "legacy"
    if manifest_mode == "native_video":
        return audio, "native_video"
    raise ValueError(f"unsupported audio manifest mode: {manifest_mode!r}")


def _native_tracks_by_id(audio: dict) -> dict[str, str]:
    tracks = audio.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("native audio manifest tracks must be a list")

    by_id = {}
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError("native audio manifest track must be a JSON object")
        shot_id = track.get("id")
        filename = track.get("file")
        if not isinstance(shot_id, str) or not shot_id.strip():
            raise ValueError("native audio manifest track id must be a non-empty string")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                f"native audio manifest track file must be a non-empty string for shot {shot_id}"
            )
        if shot_id in by_id:
            raise ValueError(f"duplicate native audio track for shot {shot_id}")
        by_id[shot_id] = filename
    return by_id


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_last_frame(clip_path, out_path) -> bool:
    """Write the final frame of a real mp4 to ``out_path`` as a PNG.

    Used by the video stage to make last-frame carry-forward honest: the next shot is
    conditioned on the *actual* final frame of the previous clip, not its first frame.
    Returns ``False`` (without raising) when the clip is not a real mp4 or ffmpeg is
    unavailable, so the caller can fall back to another source.
    """
    clip = Path(clip_path)
    if not _looks_like_mp4(clip) or not ffmpeg_available():
        return False
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Seek near the end and grab the last decodable frame.
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-1", "-i", str(clip),
             "-update", "1", "-frames:v", "1", "-q:v", "2", str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        return False
    return out.is_file() and out.stat().st_size > 0


def extract_frames(clip_path, out_dir, count: int = 4) -> list:
    """Sample ``count`` frames evenly across a real mp4 for VLM-based QC.

    Returns the written PNG paths in temporal order. Falls back to an empty list when the
    clip is not a real mp4 or ffmpeg is unavailable, so the caller can decide what to do.
    """
    clip = Path(clip_path)
    out = Path(out_dir)
    if count < 1 or not _looks_like_mp4(clip) or not ffmpeg_available():
        return []
    out.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(clip)
    # Evenly spaced timestamps inside the clip (avoid the exact first/last frame).
    if duration and duration > 0:
        timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
    else:
        timestamps = [0.0]  # unknown duration: at least grab the opening frame

    frames = []
    for i, ts in enumerate(timestamps):
        frame = out / f"frame_{i:03d}.png"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ts}", "-i", str(clip),
                 "-frames:v", "1", "-q:v", "2", str(frame)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError:
            continue
        if frame.is_file() and frame.stat().st_size > 0:
            frames.append(frame)
    return frames


def _probe_duration(clip: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


class FFmpegRenderer:
    """Renders timeline.json to a playable mp4 from real clips or keyframe fallbacks."""

    def render(self, project, timeline: dict, out_path: str) -> str:
        if not ffmpeg_available():
            raise RuntimeError(
                "ffmpeg not found on PATH — install it to render the final mp4."
            )

        keyframes = project.path("storyboard", "keyframes")
        clips_dir = project.path("assets", "clips")
        audio_dir = project.path("assets", "audio")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            segments = []
            audio_concat_inputs = []
            narration_inputs = []
            native_mode = timeline.get("audio_mode") == "native_video"
            # 旁白: an off-screen narrator stem laid over the cut, in both legacy and native
            # modes (the native clip never carries the non-diegetic narration).
            has_narration = any(
                seg.get("narration") for seg in timeline["video_track"]
            )
            for i, seg in enumerate(timeline["video_track"]):
                kf = keyframes / seg["keyframe"]
                clip = clips_dir / seg["clip"] if seg.get("clip") else None
                seg_mp4 = tmpdir / f"seg_{i:03d}.mp4"
                dur = max(0.1, float(seg["duration_s"]))
                if clip and _looks_like_mp4(clip):
                    self._clip_to_segment(clip, dur, seg_mp4)
                else:
                    self._image_to_segment(kf, dur, seg_mp4)
                segments.append(seg_mp4)
                # Native mode requires a complete production track for every shot.
                # Legacy mode retains dialogue-or-silence behavior.
                wav = tmpdir / f"aud_{i:03d}.wav"
                if native_mode:
                    native_audio = _native_audio_artifact(audio_dir, seg.get("audio"))
                    if native_audio is None or not native_audio.is_file():
                        raise RuntimeError(
                            f"native audio artifact missing for shot {seg['id']}"
                        )
                    self._fit_audio_segment(native_audio, dur, wav)
                else:
                    dialogue_name = seg.get("dialogue")
                    dialogue_path = audio_dir / dialogue_name if dialogue_name else None
                    if dialogue_path is not None and dialogue_path.is_file():
                        self._fit_audio_segment(dialogue_path, dur, wav)
                    else:
                        self._silence(dur, wav)
                audio_concat_inputs.append(wav)

                # Build a parallel narration stem (VO on the narrated shot, silence else)
                # so it stays aligned to the cut and mixes over dialogue + music.
                if has_narration:
                    nar_wav = tmpdir / f"nar_{i:03d}.wav"
                    narration_name = seg.get("narration")
                    narration_path = (
                        audio_dir / narration_name if narration_name else None
                    )
                    if narration_path is not None and narration_path.is_file():
                        self._fit_audio_segment(narration_path, dur, nar_wav)
                    else:
                        self._silence(dur, nar_wav)
                    narration_inputs.append(nar_wav)

            video = tmpdir / "video.mp4"
            self._concat(segments, video, kind="v")
            production_audio = tmpdir / "production.wav"
            self._concat(audio_concat_inputs, production_audio, kind="a")

            narration_audio = None
            if has_narration:
                narration_audio = tmpdir / "narration.wav"
                self._concat(narration_inputs, narration_audio, kind="a")

            music = timeline.get("music")
            music_path = audio_dir / music if music and (audio_dir / music).is_file() else None
            self._mux(video, production_audio, narration_audio, music_path, out)
        return str(out)

    # -- ffmpeg primitives ---------------------------------------------------
    def _run(self, args: list[str]) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *args],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )

    def _image_to_segment(self, image: Path, duration: float, out: Path) -> None:
        # Even dimensions + yuv420p so the mp4 is widely playable.
        self._run([
            "-loop", "1", "-t", f"{duration}", "-i", str(image),
            "-vf", _video_filter(),
            "-an",
            "-r", str(FPS), "-c:v", "libx264", str(out),
        ])

    def _clip_to_segment(self, clip: Path, duration: float, out: Path) -> None:
        self._run([
            "-i", str(clip),
            "-vf", f"{_video_filter()},tpad=stop_mode=clone:stop_duration={duration}",
            "-t", f"{duration}",
            "-an",
            "-r", str(FPS), "-c:v", "libx264", str(out),
        ])

    def _silence(self, duration: float, out: Path) -> None:
        self._run([
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t", f"{duration}", "-c:a", "pcm_s16le", str(out),
        ])

    def _fit_audio_segment(self, source: Path, duration: float, out: Path) -> None:
        """Normalize one shot track and trim/pad it to the exact edit duration."""
        exact_duration = f"{duration:.6f}"
        self._run([
            "-i", str(source),
            "-map", "0:a:0",
            "-af", (
                f"atrim=start=0:end={exact_duration},"
                f"asetpts=PTS-STARTPTS,apad=pad_dur={exact_duration}"
            ),
            "-t", exact_duration,
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
            str(out),
        ])

    def _concat(self, parts: list[Path], out: Path, *, kind: str) -> None:
        listfile = out.with_suffix(".txt")
        listfile.write_text("".join(f"file '{p}'\n" for p in parts))
        codec = ["-c:v", "libx264"] if kind == "v" else ["-c:a", "pcm_s16le"]
        self._run(["-f", "concat", "-safe", "0", "-i", str(listfile), *codec, str(out)])

    def _mux(
        self,
        video: Path,
        production_audio: Path,
        narration: Path | None,
        music: Path | None,
        out: Path,
    ) -> None:
        # Always carry the production (dialogue) track; mix in the narration VO and the
        # music bed when present. With only the production track, pass it through directly.
        extra = [stem for stem in (narration, music) if stem is not None]
        if not extra:
            self._run([
                "-i", str(video), "-i", str(production_audio),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-shortest", str(out),
            ])
            return

        inputs = ["-i", str(video), "-i", str(production_audio)]
        for stem in extra:
            inputs += ["-i", str(stem)]
        # Audio inputs are ffmpeg inputs 1..N (input 0 is the video).
        mix_labels = "".join(f"[{idx}:a]" for idx in range(1, len(extra) + 2))
        self._run([
            *inputs,
            "-filter_complex",
            f"{mix_labels}amix=inputs={len(extra) + 1}:duration=longest[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(out),
        ])


def _native_audio_artifact(audio_dir: Path, filename: object) -> Path | None:
    """Resolve a native WAV only when it is a direct child of the audio folder."""
    if not isinstance(filename, str) or not filename.strip():
        return None
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.suffix.lower() != ".wav":
        return None
    root = audio_dir.resolve()
    candidate = (audio_dir / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _video_filter() -> str:
    return (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    )


def _looks_like_mp4(path: Path) -> bool:
    if not path.is_file():
        return False
    return b"ftyp" in path.read_bytes()[:64]
