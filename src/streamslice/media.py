from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
from pathlib import Path
from typing import Any

from .models import Candidate
from .process import ProcessError, require_binary, run

LOGGER = logging.getLogger(__name__)


def probe(path: str | Path) -> dict[str, Any]:
    media = Path(path).expanduser().resolve()
    result = run(
        [
            require_binary("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration,size,start_time:"
                "stream=index,codec_type,codec_name,width,height,r_frame_rate,"
                "sample_rate,channels,start_time,duration,time_base"
            ),
            "-of",
            "json",
            media,
        ]
    )
    data = json.loads(result.stdout)
    data["path"] = str(media)
    data["duration"] = float(data["format"]["duration"])
    data["start_time"] = _optional_float(data["format"].get("start_time"))
    for stream in data.get("streams", []):
        stream["start_time"] = _optional_float(stream.get("start_time"))
    return data


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fingerprint(path: str | Path) -> str:
    media = Path(path).expanduser().resolve()
    stat = media.stat()
    digest = hashlib.sha256()
    digest.update(str(media).encode())
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    with media.open("rb") as handle:
        digest.update(handle.read(4 * 1024 * 1024))
    return digest.hexdigest()[:16]


def stream_id(path: str | Path) -> str:
    media = Path(path).expanduser().resolve()
    parent = media.parent.name.replace(" ", "_")
    return f"{media.parent.parent.name}_{parent}_{media.stem}_{fingerprint(media)}"


def extract_audio_segments(
    source: Path,
    output_dir: Path,
    *,
    duration: float,
    segment_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
    bitrate: str,
) -> list[tuple[float, float, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[tuple[float, float, Path]] = []
    cursor = 0.0
    index = 0
    while cursor < duration - 0.05:
        length = min(segment_seconds, duration - cursor)
        start = max(0.0, cursor - (overlap_seconds if index else 0.0))
        requested = length + (cursor - start)
        target = output_dir / f"audio-{index:03d}-{start:.3f}.mp3"
        if not target.is_file() or target.stat().st_size < 1024:
            run(
                [
                    require_binary("ffmpeg"),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start:.3f}",
                    "-i",
                    source,
                    "-t",
                    f"{requested:.3f}",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-c:a",
                    "libmp3lame",
                    "-b:a",
                    bitrate,
                    "-y",
                    target,
                ],
                timeout=max(120, requested * 2),
            )
        segments.append((start, min(duration, start + requested), target))
        cursor += length
        index += 1
    return segments


def cut_source_clip(
    source: Path,
    candidate: Candidate,
    target: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    runtime = config["runtime"]
    run(
        [
            require_binary("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{candidate.start_time:.3f}",
            "-i",
            source,
            "-t",
            f"{candidate.duration:.3f}",
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-vf",
            "setpts=PTS-STARTPTS,fps=60",
            "-af",
            "aresample=async=1:first_pts=0",
            "-c:v",
            "libx264",
            "-preset",
            str(runtime["ffmpeg_preset"]),
            "-crf",
            str(runtime["ffmpeg_crf"]),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            target,
        ],
        timeout=max(
            float(runtime.get("source_clip_timeout_seconds", 1800)),
            candidate.duration * 20,
        ),
    )
    info = probe(target)
    if not any(item.get("codec_type") == "video" for item in info["streams"]):
        raise ProcessError(f"No video stream in {target}")
    if not any(item.get("codec_type") == "audio" for item in info["streams"]):
        raise ProcessError(f"No audio stream in {target}")
    _validate_zero_based_timestamps(info, target)
    tolerance = float(runtime["duration_tolerance_seconds"])
    if math.fabs(info["duration"] - candidate.duration) > tolerance:
        raise ProcessError(
            f"Clip duration drift for {target}: {info['duration']:.3f} vs {candidate.duration:.3f}"
        )
    return info


def _validate_zero_based_timestamps(
    info: dict[str, Any],
    target: Path,
    *,
    tolerance: float = 0.05,
) -> None:
    starts = [("format", info.get("start_time"))]
    starts.extend(
        (str(stream.get("codec_type", "stream")), stream.get("start_time"))
        for stream in info.get("streams", [])
        if stream.get("codec_type") in {"audio", "video"}
    )
    invalid = [
        f"{kind}={value:.6f}"
        for kind, value in starts
        if value is not None and abs(value) > tolerance
    ]
    if invalid:
        raise ProcessError(
            f"Non-zero timestamps in exact source clip {target}: {', '.join(invalid)}"
        )


def detect_audio_activity(
    source: Path,
    *,
    duration: float,
    noise_db: float = -32.0,
    minimum_silence_seconds: float = 0.12,
) -> list[tuple[float, float]]:
    """Return non-silent intervals on the exact audio timeline.

    This is deliberately text-independent.  The intervals are used as a guard
    against model timestamps which place captions several seconds inside a
    confirmed silent region; they are not treated as speech recognition.
    """
    result = run(
        [
            require_binary("ffmpeg"),
            "-hide_banner",
            "-nostats",
            "-i",
            source,
            "-map",
            "0:a:0",
            "-af",
            f"silencedetect=noise={noise_db:g}dB:d={minimum_silence_seconds:g}",
            "-f",
            "null",
            "-",
        ],
        timeout=max(120.0, duration * 2),
    )
    silence: list[tuple[float, float]] = []
    current_start: float | None = None
    for raw_line in (result.stderr or "").splitlines():
        start_match = re.search(r"silence_start:\s*([-+0-9.eE]+)", raw_line)
        if start_match:
            current_start = max(0.0, min(duration, float(start_match.group(1))))
            continue
        end_match = re.search(r"silence_end:\s*([-+0-9.eE]+)", raw_line)
        if end_match and current_start is not None:
            end = max(current_start, min(duration, float(end_match.group(1))))
            silence.append((current_start, end))
            current_start = None
    if current_start is not None:
        silence.append((current_start, duration))

    activity: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in silence:
        if start > cursor:
            activity.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        activity.append((cursor, duration))
    return activity


def copy_for_remotion(source: Path, public_target: Path) -> None:
    public_target.parent.mkdir(parents=True, exist_ok=True)
    if public_target.exists() and public_target.stat().st_size == source.stat().st_size:
        return
    shutil.copy2(source, public_target)
