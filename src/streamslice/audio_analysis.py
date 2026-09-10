from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .models import AudioPeak
from .process import require_binary

LOGGER = logging.getLogger(__name__)


def analyze_audio(
    source: Path,
    *,
    duration: float,
    work_dir: Path,
    config: dict[str, Any],
) -> list[AudioPeak]:
    output = work_dir / "audio-peaks.json"
    if output.is_file():
        data = json.loads(output.read_text(encoding="utf-8"))
        return [AudioPeak(**item) for item in data["peaks"]]

    settings = config["audio_analysis"]
    rate = int(settings["sample_rate"])
    window_seconds = float(settings["window_seconds"])
    window_samples = max(1, round(rate * window_seconds))
    byte_count = window_samples * 4
    command = [
        require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    rows: list[dict[str, float]] = []
    previous_rms = 0.0
    index = 0
    while True:
        block = process.stdout.read(byte_count)
        if not block:
            break
        samples = np.frombuffer(block, dtype="<f4")
        if not len(samples):
            break
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        absolute_peak = float(np.max(np.abs(samples)))
        crest = absolute_peak / max(rms, 1e-7)
        onset = max(0.0, rms - previous_rms)
        rows.append(
            {
                "time": min(duration, (index + 0.5) * window_seconds),
                "rms": rms,
                "peak": absolute_peak,
                "crest": crest,
                "onset": onset,
            }
        )
        previous_rms = rms
        index += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg audio decode failed: {stderr[-2000:]}")
    if not rows:
        raise RuntimeError("Audio analysis produced no windows")

    rms_values = np.array([item["rms"] for item in rows], dtype=np.float64)
    onset_values = np.array([item["onset"] for item in rows], dtype=np.float64)
    crest_values = np.array([min(12.0, item["crest"]) for item in rows], dtype=np.float64)
    combined = (
        _robust_z(rms_values) * 0.55
        + _robust_z(onset_values) * 0.30
        + _robust_z(crest_values) * 0.15
    )
    threshold = float(settings["peak_z_score"])
    candidates: list[AudioPeak] = []
    for idx in range(1, len(rows) - 1):
        score = float(combined[idx])
        if score < threshold or score < combined[idx - 1] or score < combined[idx + 1]:
            continue
        item = rows[idx]
        candidates.append(
            AudioPeak(
                time=item["time"],
                score=round(score, 4),
                rms=round(item["rms"], 7),
                peak=round(item["peak"], 7),
                crest=round(item["crest"], 4),
                onset=round(item["onset"], 7),
            )
        )
    peaks = _merge_peaks(
        candidates,
        distance=float(settings["merge_distance_seconds"]),
        limit=int(settings["max_peaks"]),
    )
    output.write_text(
        json.dumps(
            {
                "source": str(source),
                "duration": duration,
                "window_seconds": window_seconds,
                "librosa_used": _librosa_available() and bool(settings["use_librosa_if_available"]),
                "peaks": [item.to_dict() for item in peaks],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Audio analysis found %s peaks", len(peaks))
    return peaks


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(mad * 1.4826, float(np.std(values)) * 0.25, 1e-9)
    return (values - median) / scale


def _merge_peaks(peaks: list[AudioPeak], *, distance: float, limit: int) -> list[AudioPeak]:
    accepted: list[AudioPeak] = []
    for peak in sorted(peaks, key=lambda item: item.score, reverse=True):
        if any(math.fabs(peak.time - current.time) < distance for current in accepted):
            continue
        accepted.append(peak)
        if len(accepted) >= limit:
            break
    return sorted(accepted, key=lambda item: item.time)


def _librosa_available() -> bool:
    try:
        import librosa  # noqa: F401
    except ImportError:
        return False
    return True
