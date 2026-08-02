"""Provider-neutral probing and extraction for native video soundtracks."""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path


class NativeAudioError(RuntimeError):
    """A provider-native audio source is missing or cannot be decoded."""


def _probe_decodable_wav(
    source: Path, *, require_normalized: bool = False
) -> bool | None:
    """Return validity, or ``None`` when Python cannot parse the WAV encoding."""
    try:
        with wave.open(str(source), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            frame_rate = reader.getframerate()
            frame_count = reader.getnframes()
            if frame_count <= 0 or channels <= 0 or sample_width <= 0:
                return False
            expected_bytes = frame_count * channels * sample_width
            decoded_bytes = 0
            while decoded_bytes < expected_bytes:
                chunk = reader.readframes(min(8192, frame_count))
                if not chunk:
                    break
                decoded_bytes += len(chunk)
            if decoded_bytes != expected_bytes:
                return False
            if require_normalized:
                return (
                    channels == 2
                    and sample_width == 2
                    and frame_rate == 48000
                    and reader.getcomptype() == "NONE"
                )
            return True
    except wave.Error as exc:
        if str(exc).lower().startswith("unknown format"):
            return None
        return False
    except (OSError, EOFError):
        return False


def _is_decodable_wav(source: Path, *, require_normalized: bool = False) -> bool:
    return _probe_decodable_wav(source, require_normalized=require_normalized) is True


def has_audio_stream(source: Path) -> bool:
    """Return whether ``source`` contains at least one decodable audio stream."""
    if source.suffix.lower() == ".wav":
        wav_probe = _probe_decodable_wav(source)
        if wav_probe is not None:
            return wav_probe
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return bool(result.stdout.strip())


def extract_native_audio(source: Path, destination: Path) -> Path:
    """Atomically decode ``source`` into an editable 48 kHz stereo PCM WAV."""
    if not source.is_file() or not has_audio_stream(source):
        raise NativeAudioError(f"native audio stream missing or undecodable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp.wav",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "48000",
                "-c:a",
                "pcm_s16le",
                temporary_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _is_decodable_wav(temporary_path, require_normalized=True):
            raise NativeAudioError(
                f"failed to extract valid native audio from {source.name}"
            )
        temporary_path.replace(destination)
    except NativeAudioError:
        raise
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeAudioError(f"failed to extract native audio from {source.name}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
