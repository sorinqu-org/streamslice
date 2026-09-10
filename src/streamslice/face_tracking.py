"""Detects and smooths a dominant face trajectory across a source video via YuNet."""
from __future__ import annotations

import contextlib
import json
import logging
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .media import probe
from .process import ProcessError, require_binary

try:
    import cv2
except ImportError:  # pragma: no cover - exercised via the degradation test
    cv2 = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

FACE_TRACK_VERSION = 1

DEFAULT_MODEL_PATH = (
    Path(__file__).parent / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
)

# YuNet's internal top-k candidate cap before NMS; a modest value is plenty for a
# single webcam-sized region that may also contain a couple of background faces.
_TOP_K = 50

# Discard tracks that only flash for one or two sampled frames (noise, not a face
# worth following with the crop).
_MIN_TRACK_SAMPLES = 3

# Two detections are considered the same identity if they are close in space
# (normalized center distance) or overlap substantially (IoU).
_MATCH_MAX_CENTER_DIST = 0.15
_MATCH_MIN_IOU = 0.1

# bbox_at() refuses to interpolate across a gap larger than this multiple of the
# nominal sample spacing (1/sample_fps). Any gap longer than max_gap_seconds is
# deliberately left unfilled during track construction, so it always exceeds this
# factor while normal consecutive samples never do.
_INTERP_GAP_FACTOR = 1.5


class FaceTrackingError(RuntimeError):
    """Raised when ffmpeg itself fails to read the source video."""


@dataclass(frozen=True, slots=True)
class FaceSample:
    time: float
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0


def _sample_bbox(sample: FaceSample) -> tuple[float, float, float, float]:
    return (sample.x, sample.y, sample.width, sample.height)


@dataclass(frozen=True, slots=True)
class FaceTrack:
    samples: tuple[FaceSample, ...]
    detector: str
    coverage: float
    sample_fps: float
    duration: float

    def is_usable(self, min_coverage: float = 0.4) -> bool:
        return bool(self.samples) and self.detector != "none" and self.coverage >= min_coverage

    def points(self) -> list[dict[str, float]]:
        return [
            {"time": sample.time, "focus_x": sample.center_x, "focus_y": sample.center_y}
            for sample in self.samples
        ]

    def bbox_at(self, time: float) -> tuple[float, float, float, float] | None:
        samples = self.samples
        if not samples:
            return None
        if time < samples[0].time or time > samples[-1].time:
            return None
        if len(samples) == 1:
            return _sample_bbox(samples[0])

        gap_limit = _INTERP_GAP_FACTOR / self.sample_fps if self.sample_fps > 0 else math.inf

        left, right = samples[0], samples[-1]
        for i in range(len(samples) - 1):
            if samples[i].time <= time <= samples[i + 1].time:
                left, right = samples[i], samples[i + 1]
                break

        if right.time - left.time > gap_limit:
            if math.isclose(time, left.time, abs_tol=1e-6):
                return _sample_bbox(left)
            if math.isclose(time, right.time, abs_tol=1e-6):
                return _sample_bbox(right)
            return None

        span = right.time - left.time
        frac = (time - left.time) / span if span > 1e-9 else 0.0
        return (
            left.x + (right.x - left.x) * frac,
            left.y + (right.y - left.y) * frac,
            left.width + (right.width - left.width) * frac,
            left.height + (right.height - left.height) * frac,
        )

    def max_bbox(self) -> tuple[float, float, float, float] | None:
        if not self.samples:
            return None
        x0 = min(sample.x for sample in self.samples)
        y0 = min(sample.y for sample in self.samples)
        x1 = max(sample.x + sample.width for sample in self.samples)
        y1 = max(sample.y + sample.height for sample in self.samples)
        return (x0, y0, x1 - x0, y1 - y0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_version": FACE_TRACK_VERSION,
            "detector": self.detector,
            "coverage": self.coverage,
            "sample_fps": self.sample_fps,
            "duration": self.duration,
            "samples": [
                {
                    "time": s.time,
                    "x": s.x,
                    "y": s.y,
                    "width": s.width,
                    "height": s.height,
                    "confidence": s.confidence,
                }
                for s in self.samples
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaceTrack:
        samples = tuple(
            FaceSample(
                time=float(item["time"]),
                x=float(item["x"]),
                y=float(item["y"]),
                width=float(item["width"]),
                height=float(item["height"]),
                confidence=float(item["confidence"]),
            )
            for item in data.get("samples", [])
        )
        return cls(
            samples=samples,
            detector=str(data.get("detector", "none")),
            coverage=float(data.get("coverage", 0.0)),
            sample_fps=float(data.get("sample_fps", 0.0)),
            duration=float(data.get("duration", 0.0)),
        )


def _positive_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _positive_int(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    half = window // 2
    out = np.empty_like(values)
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(values[lo:hi])
    return out


def _deadband_filter(values: np.ndarray, threshold: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        if abs(values[i] - out[i - 1]) < threshold:
            out[i] = out[i - 1]
        else:
            out[i] = values[i]
    return out


def _ema_filter(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _velocity_limit(values: np.ndarray, times: np.ndarray, max_velocity: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        dt = max(1e-6, float(times[i] - times[i - 1]))
        max_delta = max_velocity * dt
        delta = float(values[i] - out[i - 1])
        delta = max(-max_delta, min(max_delta, delta))
        out[i] = out[i - 1] + delta
    return out


def smooth_samples(
    samples: Sequence[FaceSample],
    *,
    median_window: int = 5,
    deadband: float = 0.015,
    ema_alpha: float = 0.25,
    max_velocity: float = 0.35,
) -> list[FaceSample]:
    """Smooth a face sample sequence: temporal median -> deadband -> EMA -> velocity limit."""
    items = list(samples)
    if len(items) <= 1:
        return items

    times = np.array([s.time for s in items], dtype=np.float64)
    channels = {
        "cx": np.array([s.center_x for s in items], dtype=np.float64),
        "cy": np.array([s.center_y for s in items], dtype=np.float64),
        "w": np.array([s.width for s in items], dtype=np.float64),
        "h": np.array([s.height for s in items], dtype=np.float64),
    }

    smoothed: dict[str, np.ndarray] = {}
    for key, values in channels.items():
        stage = _median_filter(values, median_window)
        stage = _deadband_filter(stage, deadband)
        stage = _ema_filter(stage, ema_alpha)
        stage = _velocity_limit(stage, times, max_velocity)
        smoothed[key] = stage

    result: list[FaceSample] = []
    for i, original in enumerate(items):
        cx, cy = float(smoothed["cx"][i]), float(smoothed["cy"][i])
        w, h = max(0.0, float(smoothed["w"][i])), max(0.0, float(smoothed["h"][i]))
        result.append(
            FaceSample(
                time=original.time,
                x=cx - w / 2.0,
                y=cy - h / 2.0,
                width=w,
                height=h,
                confidence=original.confidence,
            )
        )
    return result


def _bbox_region_to_full(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    region_x: float,
    region_y: float,
    region_width: float,
    region_height: float,
) -> tuple[float, float, float, float]:
    """Map a bbox normalized within a search region to full-frame normalized coordinates."""
    return (
        region_x + x * region_width,
        region_y + y * region_height,
        width * region_width,
        height * region_height,
    )


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _plan_sampling(
    width: int, height: int, region: dict[str, float] | None, detect_width: int
) -> tuple[str, int, int, float, float, float, float]:
    """Build the ffmpeg crop/scale filter and coordinate mapping for the search area."""
    if region is not None:
        rx = round(float(region["x"]) * width)
        ry = round(float(region["y"]) * height)
        rw = round(float(region["width"]) * width)
        rh = round(float(region["height"]) * height)
        rx = max(0, min(width - 1, rx))
        ry = max(0, min(height - 1, ry))
        rw = max(2, min(width - rx, rw))
        rh = max(2, min(height - ry, rh))
        # ffmpeg's rawvideo pipe pads odd crop dimensions to even ones for
        # chroma-subsampled sources, which desyncs any fixed frame_size read
        # downstream unless the crop itself is forced to even width/height.
        rw -= rw % 2
        rh -= rh % 2
        region_x, region_y = rx / width, ry / height
        region_w, region_h = rw / width, rh / height
        crop_w, crop_h = rw, rh
        vf = f"crop={rw}:{rh}:{rx}:{ry}"
    else:
        region_x, region_y, region_w, region_h = 0.0, 0.0, 1.0, 1.0
        crop_w, crop_h = width, height
        vf = ""

    if crop_w > detect_width:
        decode_w = detect_width
        decode_h = max(2, round(crop_h * detect_width / crop_w / 2) * 2)
    else:
        decode_w, decode_h = crop_w, crop_h

    if (decode_w, decode_h) != (crop_w, crop_h):
        scale = f"scale={decode_w}:{decode_h}:flags=lanczos"
        vf = f"{vf},{scale}" if vf else scale

    return vf, decode_w, decode_h, region_x, region_y, region_w, region_h


def _read_exact(stream: Any, size: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _load_cache(cache_path: Path) -> FaceTrack | None:
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("track_version") != FACE_TRACK_VERSION:
        return None
    try:
        return FaceTrack.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("Failed to parse cached face track %s: %s", cache_path, exc)
        return None


def _save_cache(cache_path: Path, track: FaceTrack) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(track.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _empty_track(detector: str, sample_fps: float, duration: float | None) -> FaceTrack:
    return FaceTrack(
        samples=(),
        detector=detector,
        coverage=0.0,
        sample_fps=sample_fps,
        duration=duration if duration is not None else 0.0,
    )


def _select_dominant_track(
    frames: list[tuple[int, float, list[tuple[float, float, float, float, float]]]],
    *,
    sample_fps: float,
    max_gap_seconds: float,
) -> dict[str, Any] | None:
    active: list[dict[str, Any]] = []
    for idx, _time, detections in frames:
        order = sorted(range(len(detections)), key=lambda i: -detections[i][4])
        used: set[int] = set()
        for i in order:
            fx, fy, fw, fh, conf = detections[i]
            best_j: int | None = None
            best_score: float | None = None
            for j, tr in enumerate(active):
                if j in used:
                    continue
                if (idx - tr["last_idx"]) / sample_fps > max_gap_seconds:
                    continue
                lx, ly, lw, lh = tr["last_bbox"]
                dist = math.hypot((fx + fw / 2) - (lx + lw / 2), (fy + fh / 2) - (ly + lh / 2))
                iou = _iou((fx, fy, fw, fh), (lx, ly, lw, lh))
                if dist >= _MATCH_MAX_CENTER_DIST and iou <= _MATCH_MIN_IOU:
                    continue
                score = iou - dist
                if best_score is None or score > best_score:
                    best_score, best_j = score, j
            if best_j is not None:
                used.add(best_j)
                tr = active[best_j]
                tr["items"].append((idx, fx, fy, fw, fh, conf))
                tr["last_idx"] = idx
                tr["last_bbox"] = (fx, fy, fw, fh)
            else:
                active.append(
                    {
                        "items": [(idx, fx, fy, fw, fh, conf)],
                        "last_idx": idx,
                        "last_bbox": (fx, fy, fw, fh),
                    }
                )

    candidates = [tr for tr in active if len(tr["items"]) >= _MIN_TRACK_SAMPLES]
    if not candidates:
        return None

    def _score(tr: dict[str, Any]) -> float:
        items = tr["items"]
        total_area = sum(w * h for _, _, _, w, h, _ in items)
        span = (items[-1][0] - items[0][0]) / sample_fps if len(items) > 1 else 1.0 / sample_fps
        avg_conf = sum(item[5] for item in items) / len(items)
        return total_area * span * avg_conf

    return max(candidates, key=_score)


def _build_samples(
    idx_to_bbox: dict[int, tuple[float, float, float, float, float]],
    *,
    sample_fps: float,
    max_gap_seconds: float,
    median_window: int,
    deadband: float,
    ema_alpha: float,
    max_velocity: float,
) -> list[FaceSample]:
    present = sorted(idx_to_bbox.items())
    max_gap_frames = max(1, round(max_gap_seconds * sample_fps))

    segments: list[list[FaceSample]] = []
    current: list[FaceSample] = []
    for k, (idx, bbox) in enumerate(present):
        if k > 0:
            prev_idx, prev_bbox = present[k - 1]
            gap = idx - prev_idx
            if gap > max_gap_frames:
                if current:
                    segments.append(current)
                current = []
            elif gap > 1:
                for fill_idx in range(prev_idx + 1, idx):
                    frac = (fill_idx - prev_idx) / gap
                    current.append(
                        FaceSample(
                            time=fill_idx / sample_fps,
                            x=prev_bbox[0] + (bbox[0] - prev_bbox[0]) * frac,
                            y=prev_bbox[1] + (bbox[1] - prev_bbox[1]) * frac,
                            width=prev_bbox[2] + (bbox[2] - prev_bbox[2]) * frac,
                            height=prev_bbox[3] + (bbox[3] - prev_bbox[3]) * frac,
                            confidence=min(prev_bbox[4], bbox[4]),
                        )
                    )
        fx, fy, fw, fh, conf = bbox
        current.append(
            FaceSample(
                time=idx / sample_fps,
                x=fx,
                y=fy,
                width=fw,
                height=fh,
                confidence=conf,
            )
        )
    if current:
        segments.append(current)

    smoothed: list[FaceSample] = []
    for segment in segments:
        smoothed.extend(
            smooth_samples(
                segment,
                median_window=median_window,
                deadband=deadband,
                ema_alpha=ema_alpha,
                max_velocity=max_velocity,
            )
        )
    return smoothed


def track_face(
    video: Path,
    *,
    config: dict[str, Any],
    region: dict[str, float] | None = None,
    cache_path: Path | None = None,
    duration: float | None = None,
) -> FaceTrack:
    """Detect the dominant face and return a smoothed trajectory in full-frame coordinates."""
    video = Path(video)

    if cache_path is not None:
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    settings = config.get("face_tracking", {}) if isinstance(config, dict) else {}
    sample_fps = _positive_float(settings.get("sample_fps"), 4.0)

    if not settings.get("enabled", True):
        LOGGER.info("Face tracking disabled by config for %s", video)
        return _empty_track("none", sample_fps, duration)

    if cv2 is None:
        LOGGER.warning("OpenCV is not available, skipping face tracking for %s", video)
        return _empty_track("none", sample_fps, duration)

    model_path = Path(settings.get("model_path") or DEFAULT_MODEL_PATH).expanduser()
    if not model_path.is_file():
        LOGGER.warning("Face detector model not found at %s, skipping face tracking", model_path)
        return _empty_track("none", sample_fps, duration)

    detect_width = _positive_int(settings.get("detect_width"), 640)
    min_detect_width = _positive_int(settings.get("min_detect_width"), 320)
    score_threshold = _positive_float(settings.get("score_threshold"), 0.6)
    nms_threshold = _positive_float(settings.get("nms_threshold"), 0.3)
    max_gap_seconds = _positive_float(settings.get("max_gap_seconds"), 1.0)
    median_window = _positive_int(settings.get("median_window"), 5)
    deadband = _positive_float(settings.get("deadband"), 0.015)
    ema_alpha = _positive_float(settings.get("ema_alpha"), 0.25)
    max_velocity = _positive_float(settings.get("max_velocity"), 0.35)
    upscale_cfg = _positive_float(settings.get("upscale"), 2.0)

    try:
        info = probe(video)
    except ProcessError as exc:
        raise FaceTrackingError(f"Failed to probe video {video}: {exc}") from exc

    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise FaceTrackingError(f"No video stream found in {video}")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    total_duration = float(duration) if duration is not None else float(info["duration"])

    vf, decode_w, decode_h, region_x, region_y, region_w, region_h = _plan_sampling(
        width, height, region, detect_width
    )
    full_vf = f"{vf},fps={sample_fps}" if vf else f"fps={sample_fps}"

    upscale_factor = 1.0
    if decode_w < min_detect_width:
        upscale_factor = max(upscale_cfg, min_detect_width / decode_w)
    final_w = max(1, round(decode_w * upscale_factor))
    final_h = max(1, round(decode_h * upscale_factor))

    # cv2.utils.logging is absent from some OpenCV builds; silencing the model
    # loader's stderr chatter is a nicety, not a requirement.
    with contextlib.suppress(AttributeError):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    detector = cv2.FaceDetectorYN.create(
        str(model_path), "", (final_w, final_h), score_threshold, nms_threshold, _TOP_K
    )
    detector.setInputSize((final_w, final_h))

    ffmpeg_bin = require_binary("ffmpeg")
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        full_vf,
        "-an",
        "-sn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    frame_size = decode_w * decode_h * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None

    frames: list[tuple[int, float, list[tuple[float, float, float, float, float]]]] = []
    frame_idx = 0
    interp = cv2.INTER_CUBIC if upscale_factor > 1.0 else cv2.INTER_AREA
    while True:
        block = _read_exact(process.stdout, frame_size)
        if block is None:
            break
        frame = np.frombuffer(block, dtype=np.uint8).reshape(decode_h, decode_w, 3)
        detect_frame = frame
        if upscale_factor != 1.0:
            detect_frame = cv2.resize(frame, (final_w, final_h), interpolation=interp)
        _, faces = detector.detect(detect_frame)
        detections: list[tuple[float, float, float, float, float]] = []
        if faces is not None:
            for row in faces:
                bx, by, bw, bh, conf = (
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[-1]),
                )
                bx = max(0.0, min(float(final_w), bx))
                by = max(0.0, min(float(final_h), by))
                bw = max(0.0, min(final_w - bx, bw))
                bh = max(0.0, min(final_h - by, bh))
                if bw <= 0 or bh <= 0:
                    continue
                nx = (bx / upscale_factor) / decode_w
                ny = (by / upscale_factor) / decode_h
                nw = (bw / upscale_factor) / decode_w
                nh = (bh / upscale_factor) / decode_h
                full_frame_bbox = _bbox_region_to_full(
                    nx,
                    ny,
                    nw,
                    nh,
                    region_x=region_x,
                    region_y=region_y,
                    region_width=region_w,
                    region_height=region_h,
                )
                detections.append((*full_frame_bbox, conf))
        frames.append((frame_idx, frame_idx / sample_fps, detections))
        frame_idx += 1

    stderr_data = process.stderr.read() if process.stderr else b""
    process.stdout.close()
    if process.stderr:
        process.stderr.close()
    returncode = process.wait()
    if returncode != 0:
        detail = stderr_data.decode("utf-8", errors="replace").strip()
        raise FaceTrackingError(f"ffmpeg failed to read video {video}: {detail[-2000:]}")

    n_frames = len(frames)
    empty = FaceTrack(
        samples=(),
        detector="yunet",
        coverage=0.0,
        sample_fps=sample_fps,
        duration=total_duration,
    )
    if n_frames == 0:
        return empty

    dominant = _select_dominant_track(
        frames, sample_fps=sample_fps, max_gap_seconds=max_gap_seconds
    )
    if dominant is None:
        LOGGER.info("No stable dominant face found in %s", video)
        return empty

    idx_to_bbox = {item[0]: item[1:] for item in dominant["items"]}
    coverage = len(idx_to_bbox) / n_frames

    samples = _build_samples(
        idx_to_bbox,
        sample_fps=sample_fps,
        max_gap_seconds=max_gap_seconds,
        median_window=median_window,
        deadband=deadband,
        ema_alpha=ema_alpha,
        max_velocity=max_velocity,
    )

    track = FaceTrack(
        samples=tuple(samples),
        detector="yunet",
        coverage=coverage,
        sample_fps=sample_fps,
        duration=total_duration,
    )
    LOGGER.info(
        "Face tracking for %s: coverage=%.2f samples=%d duration=%.1fs",
        video,
        coverage,
        len(samples),
        total_duration,
    )
    if cache_path is not None:
        _save_cache(cache_path, track)
    return track
