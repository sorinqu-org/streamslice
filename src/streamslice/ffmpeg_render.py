"""Native FFmpeg renderer: template layouts, face-aware framing and montage cuts.

The renderer turns one cut source clip into the final vertical video in a single
FFmpeg invocation. Three inputs shape the filtergraph:

* a :class:`~streamslice.templates.Template` describes the bands of the frame,
* a :class:`~streamslice.face_tracking.FaceTrack` says where the face actually
  is, so a webcam band frames the head instead of the middle of its bounding box,
* an :class:`~streamslice.montage.EditPlan` says which stretches of the clip
  survive, with which layout and zoom.

Requires FFmpeg 7.0 or newer: the moving crop relies on ``x``/``y`` expressions
being re-evaluated per frame, which older builds only did with ``eval=frame``.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .face_tracking import FaceTrack, track_face
from .media import detect_audio_activity, probe
from .montage import Cut, EditPlan, passthrough_plan, plan_edit
from .process import ProcessError, require_binary, run
from .templates import Band, Template, TemplateError, default_template_name, load_template
from .tracking import PointTracker

LOGGER = logging.getLogger(__name__)

FALLBACK_WEBCAM = {"x": 0.738, "y": 0.739, "width": 0.262, "height": 0.261}
FALLBACK_GAMEPLAY = {"x": 0.0, "y": 0.0, "width": 0.738, "height": 1.0}
FULL_FRAME = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}

# A crop must keep the whole head plus breathing room. Without this margin the
# window can sit flush against the chin or the hairline and the framing reads as
# an accident rather than a choice.
FACE_MARGIN = 0.55

# Blurred backdrop strength, in output pixels. Strong enough that the backdrop
# never competes with the sharp foreground layer for attention.
BLUR_SIGMA = 24

_ATEMPO_MIN = 0.5
_ATEMPO_MAX = 2.0

# PointTracker renders one nested if() per keyframe, and FFmpeg's expression
# parser gives up somewhere past a few hundred of them. A 4 fps face track over a
# 30 s clip is 126 samples, which produced an 18 KB expression that failed to
# configure the crop filter at all. A smooth camera move needs far fewer
# keyframes than the tracker samples, so the trajectory is simplified first.
MAX_FOCUS_KEYFRAMES = 24
FOCUS_TOLERANCE = 0.004


def _escape_filter_path(path: Path | str) -> str:
    """Escape a path so it survives being embedded in a filtergraph argument."""
    text = str(Path(path).resolve())
    return (
        text.replace("\\", "/")
        .replace(":", "\\:")
        .replace("'", "'\\''")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _even(value: float, minimum: int = 2) -> int:
    """Round to an even integer: chroma-subsampled encoders reject odd sizes."""
    return max(minimum, round(value) // 2 * 2)


def _finite(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


@dataclass(frozen=True, slots=True)
class _Rect:
    """A pixel rectangle in source-frame coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0


@dataclass(frozen=True, slots=True)
class _BandPlan:
    """Everything the filtergraph needs to draw one band of one cut."""

    band: Band
    out_w: int
    out_h: int
    crop_w: int
    crop_h: int
    blur_fill: bool
    upscale: float
    focus: tuple[dict[str, float], ...]
    region: _Rect
    masked: bool


def _normalized_rect(value: Any, fallback: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict):
        return dict(fallback)
    try:
        rect = {key: float(value[key]) for key in ("x", "y", "width", "height")}
    except (KeyError, TypeError, ValueError):
        return dict(fallback)
    if not all(math.isfinite(item) for item in rect.values()):
        return dict(fallback)
    if (
        rect["width"] <= 0
        or rect["height"] <= 0
        or rect["x"] < 0
        or rect["y"] < 0
        or rect["x"] + rect["width"] > 1.001
        or rect["y"] + rect["height"] > 1.001
    ):
        return dict(fallback)
    return rect


def webcam_region(recommendation: dict[str, Any], layout_cfg: dict[str, Any]) -> dict[str, float]:
    """Resolve where the webcam lives, preferring visually validated coordinates."""
    for candidate in (
        recommendation.get("webcam_crop"),
        recommendation.get("webcam_box"),
        layout_cfg.get("webcam"),
    ):
        rect = _normalized_rect(candidate, FALLBACK_WEBCAM)
        if candidate is not None:
            return rect
    return dict(FALLBACK_WEBCAM)


def _source_region(
    source: str,
    *,
    recommendation: dict[str, Any],
    layout_cfg: dict[str, Any],
    source_w: int,
    source_h: int,
) -> _Rect:
    if source == "webcam":
        rect = webcam_region(recommendation, layout_cfg)
    elif source == "gameplay":
        rect = _normalized_rect(layout_cfg.get("gameplay"), FALLBACK_GAMEPLAY)
    else:
        rect = dict(FULL_FRAME)
    return _Rect(
        x=rect["x"] * source_w,
        y=rect["y"] * source_h,
        width=rect["width"] * source_w,
        height=rect["height"] * source_h,
    )


def _largest_inner_rect(region: _Rect, aspect: float) -> tuple[float, float]:
    """Largest ``aspect``-ratio rectangle that still fits inside ``region``."""
    if region.aspect >= aspect:
        return region.height * aspect, region.height
    return region.width, region.width / aspect


def _plan_band(
    band: Band,
    *,
    out_w: int,
    out_h: int,
    region: _Rect,
    face: tuple[float, float, float, float] | None,
    focus_points: list[dict[str, float]],
    source_w: int,
    source_h: int,
) -> _BandPlan:
    """Size and place the crop window for one band.

    The window is grown, never shrunk, by three independent constraints, in this
    order: it must contain the tracked face with margin, it must not demand more
    upscaling than the template allows, and it must stay inside the source region.
    Growing keeps the band aspect; only the final clamp to the region can break
    it, and that is exactly the case that needs a blurred backdrop instead of a
    crop, because no window inside this region can fill the band sharply.
    """
    aspect = out_w / out_h
    crop_w, crop_h = _largest_inner_rect(region, aspect)

    zoom = max(1.0, _finite(band.zoom, 1.0))
    crop_w /= zoom
    crop_h /= zoom

    if face is not None:
        need_w = face[2] * source_w * (1.0 + 2.0 * FACE_MARGIN)
        need_h = face[3] * source_h * (1.0 + 2.0 * FACE_MARGIN)
        grow = max(1.0, need_w / crop_w if crop_w else 1.0, need_h / crop_h if crop_h else 1.0)
        crop_w *= grow
        crop_h *= grow

    max_upscale = max(1.0, _finite(band.max_upscale, 2.0))
    upscale = max(out_w / crop_w if crop_w else 1.0, out_h / crop_h if crop_h else 1.0)
    if upscale > max_upscale:
        grow = upscale / max_upscale
        crop_w *= grow
        crop_h *= grow

    blur_fill = band.fit == "blur_fill"
    if crop_w > region.width + 0.5 or crop_h > region.height + 0.5:
        # The region is simply too small to fill this band at the requested
        # quality. Take all of it and letterbox over a blurred copy: a soft
        # backdrop reads far better than a 6x upscale of a webcam thumbnail.
        crop_w = min(crop_w, region.width)
        crop_h = min(crop_h, region.height)
        blur_fill = True
    if band.fit == "contain":
        blur_fill = False

    crop_w_px = _even(min(crop_w, region.width))
    crop_h_px = _even(min(crop_h, region.height))
    effective_upscale = max(out_w / crop_w_px, out_h / crop_h_px)

    focus = _crop_origin_points(
        focus_points,
        band=band,
        crop_w=crop_w_px,
        crop_h=crop_h_px,
        region=region,
        source_w=source_w,
        source_h=source_h,
    )
    return _BandPlan(
        band=band,
        out_w=out_w,
        out_h=out_h,
        crop_w=crop_w_px,
        crop_h=crop_h_px,
        blur_fill=blur_fill,
        upscale=effective_upscale,
        focus=tuple(focus),
        region=region,
        masked=band.source in ("gameplay", "full"),
    )


def _crop_origin_points(
    focus_points: list[dict[str, float]],
    *,
    band: Band,
    crop_w: int,
    crop_h: int,
    region: _Rect,
    source_w: int,
    source_h: int,
) -> list[dict[str, float]]:
    """Convert focus points into clamped crop centres PointTracker can consume.

    ``PointTracker`` emits ``clip(focus*in - crop/2, 0, in - crop)``, i.e. it
    always centres the window on the focus point. Alignment and the clamp to the
    source region are therefore folded in here: each point is turned into the
    centre of the window we actually want, expressed back as a fraction of the
    full frame so the tracker's own arithmetic reproduces it exactly.
    """
    align_x = min(1.0, max(0.0, _finite(band.align_x, 0.5)))
    align_y = min(1.0, max(0.0, _finite(band.align_y, 0.5)))
    min_x = region.x
    max_x = max(region.x, region.right - crop_w)
    min_y = region.y
    max_y = max(region.y, region.bottom - crop_h)

    converted: list[dict[str, float]] = []
    for point in focus_points or [{"time": 0.0, "focus_x": 0.5, "focus_y": 0.5}]:
        focus_x = min(1.0, max(0.0, _finite(point.get("focus_x"), 0.5)))
        focus_y = min(1.0, max(0.0, _finite(point.get("focus_y"), 0.5)))
        origin_x = min(max_x, max(min_x, focus_x * source_w - crop_w * align_x))
        origin_y = min(max_y, max(min_y, focus_y * source_h - crop_h * align_y))
        converted.append(
            {
                "time": _finite(point.get("time"), 0.0),
                "focus_x": (origin_x + crop_w / 2.0) / source_w,
                "focus_y": (origin_y + crop_h / 2.0) / source_h,
            }
        )
    return converted


def _simplify_trajectory(
    points: list[dict[str, float]],
    *,
    max_points: int = MAX_FOCUS_KEYFRAMES,
    tolerance: float = FOCUS_TOLERANCE,
) -> list[dict[str, float]]:
    """Drop keyframes that linear interpolation already reproduces.

    Greedily removes the interior point whose removal changes the interpolated
    path least, until every remaining point matters by more than ``tolerance`` or
    the budget is met. Endpoints are always kept so the move still starts and
    ends where the tracker said it should.
    """
    if len(points) <= 2:
        return points

    kept = list(points)

    def deviation(index: int) -> float:
        before, current, after = kept[index - 1], kept[index], kept[index + 1]
        span = after["time"] - before["time"]
        if span <= 0:
            return 0.0
        ratio = (current["time"] - before["time"]) / span
        return max(
            abs(before["focus_x"] + (after["focus_x"] - before["focus_x"]) * ratio
                - current["focus_x"]),
            abs(before["focus_y"] + (after["focus_y"] - before["focus_y"]) * ratio
                - current["focus_y"]),
        )

    while len(kept) > 2:
        errors = [(deviation(index), index) for index in range(1, len(kept) - 1)]
        smallest, index = min(errors)
        if len(kept) <= max_points and smallest > tolerance:
            break
        del kept[index]
    return kept


def _rebase_points(
    points: tuple[dict[str, float], ...] | list[dict[str, float]],
    start: float,
    end: float,
) -> list[dict[str, float]]:
    """Clip a focus trajectory to ``[start, end]`` and rebase it to zero.

    Every cut is trimmed and ``setpts``-reset, so inside a cut FFmpeg's ``t``
    restarts at zero. A trajectory expressed in clip time would otherwise point
    the camera at whatever was happening at the start of the clip.
    """
    ordered = sorted(points, key=lambda item: _finite(item.get("time"), 0.0))
    if not ordered:
        return [{"time": 0.0, "focus_x": 0.5, "focus_y": 0.5}]
    tracker = PointTracker(list(ordered), duration=max(0.0, end - start), method="cubic")
    times = {start, end}
    times.update(
        _finite(item.get("time"), 0.0)
        for item in ordered
        if start < _finite(item.get("time"), 0.0) < end
    )
    rebased: list[dict[str, float]] = []
    for absolute in sorted(times):
        focus_x, focus_y = tracker.evaluate(absolute)
        rebased.append(
            {"time": max(0.0, absolute - start), "focus_x": focus_x, "focus_y": focus_y}
        )
    return _simplify_trajectory(rebased)


def _mask_boxes(
    masks: list[dict[str, Any]], *, region: _Rect, source_w: int, source_h: int
) -> list[tuple[int, int, int, int]]:
    """Convert overlay masks into source-pixel boxes clipped to the band region.

    ``overlay_analysis`` normalises masks against the whole source frame, so they
    are applied before the crop rather than after the scale.
    """
    boxes: list[tuple[int, int, int, int]] = []
    for mask in masks:
        try:
            mx = float(mask["x"]) * source_w
            my = float(mask["y"]) * source_h
            mw = float(mask["width"]) * source_w
            mh = float(mask["height"]) * source_h
        except (KeyError, TypeError, ValueError):
            continue
        left = max(region.x, mx)
        top = max(region.y, my)
        right = min(region.right, mx + mw)
        bottom = min(region.bottom, my + mh)
        # delogo needs at least a pixel of surrounding source to interpolate from.
        if right - left < 8 or bottom - top < 8:
            continue
        left = max(1.0, left)
        top = max(1.0, top)
        width = min(right - left, source_w - left - 1)
        height = min(bottom - top, source_h - top - 1)
        if width < 8 or height < 8:
            continue
        boxes.append((int(left), int(top), int(width), int(height)))
    return boxes


def _band_chain(
    *,
    plan: _BandPlan,
    cut: Cut,
    out_label: str,
    masks: list[tuple[int, int, int, int]],
    source_w: int,
) -> list[str]:
    """Emit the filter chain that renders one band of one cut."""
    prefix = out_label
    steps = [
        f"trim=start={cut.source_start:.4f}:end={cut.source_end:.4f}",
        "setpts=PTS-STARTPTS",
    ]
    if plan.masked:
        steps.extend(f"delogo=x={x}:y={y}:w={w}:h={h}" for x, y, w, h in masks)

    points = _rebase_points(plan.focus, cut.source_start, cut.source_end)
    tracker = PointTracker(
        points, duration=max(0.0, cut.source_end - cut.source_start), method="cubic"
    )
    if not plan.band.track:
        points = points[:1]
        tracker = PointTracker(points, duration=0.0, method="cubic")
    x_expr = tracker.ffmpeg_x_expr(crop_w_expr=str(plan.crop_w), in_w_expr=str(source_w))
    y_expr = tracker.ffmpeg_y_expr(crop_h_expr=str(plan.crop_h), in_h_expr="in_h")
    steps.append(
        f"crop=w={plan.crop_w}:h={plan.crop_h}:x='{x_expr}':y='{y_expr}'"
    )

    filters: list[str] = []
    if plan.blur_fill:
        head = f"[0:v]{','.join(steps)}[{prefix}_src]"
        filters.append(head)
        filters.append(f"[{prefix}_src]split=2[{prefix}_fg][{prefix}_bg]")
        filters.append(
            f"[{prefix}_bg]scale={plan.out_w}:{plan.out_h}"
            ":force_original_aspect_ratio=increase:flags=bilinear,"
            f"crop=w={plan.out_w}:h={plan.out_h},gblur=sigma={BLUR_SIGMA}[{prefix}_bgb]"
        )
        filters.append(
            f"[{prefix}_fg]scale={plan.out_w}:{plan.out_h}"
            f":force_original_aspect_ratio=decrease:flags=lanczos[{prefix}_fgs]"
        )
        filters.append(
            f"[{prefix}_bgb][{prefix}_fgs]overlay=(W-w)/2:(H-h)/2"
            f":format=auto[{prefix}_composed]"
        )
        composed = f"{prefix}_composed"
    else:
        steps.append(f"scale={plan.out_w}:{plan.out_h}:flags=lanczos")
        filters.append(f"[0:v]{','.join(steps)}[{prefix}_composed]")
        composed = f"{prefix}_composed"

    divider = max(0, int(plan.band.divider_px))
    if divider > 0:
        colour = plan.band.divider_color or "#000000"
        filters.append(
            f"[{composed}]drawbox=x=0:y={max(0, plan.out_h - divider)}"
            f":w={plan.out_w}:h={divider}:color={colour}:t=fill[{out_label}]"
        )
    else:
        filters.append(f"[{composed}]null[{out_label}]")
    return filters


def _band_heights(template: Template, bands: tuple[Band, ...]) -> list[int]:
    """Split the canvas height across bands so the rows always add back up."""
    heights: list[int] = []
    remaining = template.height
    for index, band in enumerate(bands):
        if index == len(bands) - 1:
            heights.append(max(2, remaining))
            break
        height = _even(template.height * max(0.0, _finite(band.height, 0.0)))
        height = min(height, remaining - 2 * (len(bands) - index - 1))
        heights.append(max(2, height))
        remaining -= heights[-1]
    return heights


def _atempo_chain(speed: float) -> str:
    """Chain atempo stages, because a single one only spans 0.5x..2x."""
    stages: list[float] = []
    remaining = speed
    while remaining > _ATEMPO_MAX:
        stages.append(_ATEMPO_MAX)
        remaining /= _ATEMPO_MAX
    while remaining < _ATEMPO_MIN:
        stages.append(_ATEMPO_MIN)
        remaining /= _ATEMPO_MIN
    stages.append(remaining)
    return ",".join(f"atempo={stage:.6f}" for stage in stages)


def build_ffmpeg_filtergraph(
    *,
    props: dict[str, Any],
    ass_path: Path,
    source_duration: float,
    template: Template,
    plan: EditPlan,
    face_track: FaceTrack | None = None,
    source_size: tuple[int, int] = (1920, 1080),
    has_audio: bool = True,
) -> str:
    """Build the complete filtergraph for a template-driven, montaged render."""
    source_w, source_h = source_size
    recommendation = dict(props.get("layoutRecommendation") or {})
    layout_cfg = dict(props.get("layout") or {})
    masks = list(props.get("overlayMasks") or [])

    gameplay_points = _extract_focal_trajectory(recommendation, source_duration)
    if face_track is not None and face_track.samples:
        face_points = face_track.points()
        face_box = face_track.max_bbox()
    else:
        webcam = webcam_region(recommendation, layout_cfg)
        face_points = [
            {
                "time": 0.0,
                "focus_x": webcam["x"] + webcam["width"] / 2.0,
                "focus_y": webcam["y"] + webcam["height"] / 2.0,
            }
        ]
        face_box = None

    filters: list[str] = []
    concat_labels: list[str] = []

    for index, cut in enumerate(plan.cuts):
        spec = template.layout(cut.layout)
        heights = _band_heights(template, spec.bands)
        band_labels: list[str] = []
        for band_index, band in enumerate(spec.bands):
            zoomed = _zoomed_band(band, cut.zoom)
            region = _source_region(
                band.source,
                recommendation=recommendation,
                layout_cfg=layout_cfg,
                source_w=source_w,
                source_h=source_h,
            )
            tracked_face = face_box if band.source == "webcam" else None
            points = face_points if band.source == "webcam" else gameplay_points
            band_plan = _plan_band(
                zoomed,
                out_w=template.width,
                out_h=heights[band_index],
                region=region,
                face=tracked_face,
                focus_points=points,
                source_w=source_w,
                source_h=source_h,
            )
            label = f"c{index}b{band_index}"
            filters.extend(
                _band_chain(
                    plan=band_plan,
                    cut=cut,
                    out_label=label,
                    masks=_mask_boxes(
                        masks, region=region, source_w=source_w, source_h=source_h
                    ),
                    source_w=source_w,
                )
            )
            band_labels.append(label)
            LOGGER.debug(
                "cut %s band %s (%s): crop=%sx%s upscale=%.2f blur_fill=%s",
                index,
                band_index,
                band.source,
                band_plan.crop_w,
                band_plan.crop_h,
                band_plan.upscale,
                band_plan.blur_fill,
            )

        stacked = f"c{index}v"
        if len(band_labels) == 1:
            filters.append(f"[{band_labels[0]}]null[{stacked}]")
        else:
            joined = "".join(f"[{label}]" for label in band_labels)
            filters.append(f"{joined}vstack=inputs={len(band_labels)}[{stacked}]")

        if not math.isclose(cut.speed, 1.0, abs_tol=1e-6):
            sped = f"{stacked}s"
            filters.append(f"[{stacked}]setpts=PTS/{cut.speed:.6f}[{sped}]")
            stacked = sped
        concat_labels.append(stacked)

        if has_audio:
            audio = f"c{index}a"
            steps = [
                f"atrim=start={cut.source_start:.4f}:end={cut.source_end:.4f}",
                "asetpts=PTS-STARTPTS",
            ]
            if not math.isclose(cut.speed, 1.0, abs_tol=1e-6):
                steps.append(_atempo_chain(cut.speed))
            filters.append(f"[0:a]{','.join(steps)}[{audio}]")

    if len(concat_labels) == 1 and not has_audio:
        filters.append(f"[{concat_labels[0]}]null[vcat]")
    elif len(concat_labels) == 1:
        filters.append(f"[{concat_labels[0]}]null[vcat]")
        filters.append("[c0a]anull[acat]")
    else:
        if has_audio:
            joined = "".join(f"[{label}][c{i}a]" for i, label in enumerate(concat_labels))
            filters.append(f"{joined}concat=n={len(concat_labels)}:v=1:a=1[vcat][acat]")
        else:
            joined = "".join(f"[{label}]" for label in concat_labels)
            filters.append(f"{joined}concat=n={len(concat_labels)}:v=1:a=0[vcat]")

    escaped_ass = _escape_filter_path(ass_path)
    filters.append(f"[vcat]subtitles=filename='{escaped_ass}'[outv]")
    return ";".join(filters)


def _zoomed_band(band: Band, zoom: float) -> Band:
    """Fold a cut's punch-in zoom into the band's own zoom."""
    factor = max(1.0, _finite(zoom, 1.0))
    if math.isclose(factor, 1.0, abs_tol=1e-6):
        return band
    return Band(
        source=band.source,
        height=band.height,
        fit=band.fit,
        align_x=band.align_x,
        align_y=band.align_y,
        zoom=max(1.0, _finite(band.zoom, 1.0)) * factor,
        track=band.track,
        max_upscale=band.max_upscale,
        divider_px=band.divider_px,
        divider_color=band.divider_color,
    )


def _extract_focal_trajectory(
    recommendation: dict[str, Any], duration: float
) -> list[dict[str, float]]:
    """Extract gameplay trajectory keypoints, or fall back to a single focus point."""
    trajectory = recommendation.get("focal_trajectory") or recommendation.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        points: list[dict[str, float]] = []
        for item in trajectory:
            if not isinstance(item, dict) or "time" not in item:
                continue
            points.append(
                {
                    "time": _finite(item.get("time"), 0.0),
                    "focus_x": _finite(item.get("focus_x"), 0.5),
                    "focus_y": _finite(item.get("focus_y"), 0.5),
                }
            )
        if points:
            return points

    default_x = _finite(recommendation.get("focus_x"), 0.5)
    default_y = _finite(recommendation.get("focus_y"), 0.5)
    event_time = recommendation.get("event_time")
    event_end = recommendation.get("event_end")
    if (
        recommendation.get("gameplay_event_validated") is True
        and isinstance(event_time, (int, float))
        and isinstance(event_end, (int, float))
        and 0 <= float(event_time) < float(event_end) <= duration
    ):
        event_x = _finite(recommendation.get("event_focus_x"), default_x)
        event_y = _finite(recommendation.get("event_focus_y"), default_y)
        return [
            {"time": 0.0, "focus_x": default_x, "focus_y": default_y},
            {"time": max(0.0, float(event_time) - 0.5), "focus_x": default_x, "focus_y": default_y},
            {"time": float(event_time), "focus_x": event_x, "focus_y": event_y},
            {"time": float(event_end), "focus_x": event_x, "focus_y": event_y},
            {
                "time": min(duration, float(event_end) + 0.5),
                "focus_x": default_x,
                "focus_y": default_y,
            },
            {"time": duration, "focus_x": default_x, "focus_y": default_y},
        ]
    return [{"time": 0.0, "focus_x": default_x, "focus_y": default_y}]


def resolve_template(props: dict[str, Any], config: dict[str, Any]) -> Template:
    """Load the template named by the props, then the config, then the built-in default."""
    for name in (props.get("template"), default_template_name(config)):
        if not name:
            continue
        try:
            return load_template(str(name), config)
        except TemplateError as exc:
            LOGGER.warning("Template %r unavailable (%s); falling back", name, exc)
    return load_template("classic-split", config)


def resolve_face_track(
    source_clip: Path,
    *,
    props: dict[str, Any],
    config: dict[str, Any],
    duration: float,
    cache_dir: Path | None = None,
) -> FaceTrack | None:
    """Track the streamer's face inside the webcam region, or return ``None``.

    Face tracking is an enrichment pass: a missing model, an unreadable clip or a
    scene with no visible face must fall back to the configured geometry rather
    than fail the render.
    """
    settings = config.get("face_tracking") or {}
    if not settings.get("enabled", True):
        return None
    recommendation = dict(props.get("layoutRecommendation") or {})
    layout_cfg = dict(props.get("layout") or {})
    region = webcam_region(recommendation, layout_cfg)
    cache_path = (cache_dir / "face-track.json") if cache_dir else None
    try:
        track = track_face(
            source_clip,
            config=config,
            region=region,
            cache_path=cache_path,
            duration=duration,
        )
    except Exception:
        LOGGER.warning("Face tracking failed; keeping configured framing", exc_info=True)
        return None
    minimum = _finite(settings.get("min_coverage"), 0.4)
    if not track.is_usable(minimum):
        LOGGER.info(
            "Face track unusable (detector=%s coverage=%.2f < %.2f); keeping configured framing",
            track.detector,
            track.coverage,
            minimum,
        )
        return None
    LOGGER.info(
        "Face track: detector=%s coverage=%.2f samples=%s",
        track.detector,
        track.coverage,
        len(track.samples),
    )
    return track


def stored_edit_plan(props: dict[str, Any], duration: float) -> EditPlan:
    """Return the plan carried by the props, or one uncut take.

    Cutting footage shifts every caption after each removed gap, so the plan has
    to be built by whoever also writes ``subtitles.ass`` (see
    :func:`streamslice.render.render_clip`). If no plan reached the renderer, the
    safe reading is that nothing was remapped, so nothing may be cut either.
    """
    stored = props.get("editPlan")
    if isinstance(stored, dict):
        try:
            return EditPlan.from_dict(stored)
        except Exception:
            LOGGER.warning("Stored edit plan is unusable; rendering one take", exc_info=True)
    layout = str((props.get("layoutRecommendation") or {}).get("layout") or "split")
    return passthrough_plan(duration, layout)


def resolve_edit_plan(
    source_clip: Path,
    *,
    props: dict[str, Any],
    config: dict[str, Any],
    template: Template,
    duration: float,
) -> EditPlan:
    """Plan the montage for one clip."""
    recommendation = dict(props.get("layoutRecommendation") or {})
    settings = config.get("montage") or {}
    if not settings.get("enabled", True):
        return passthrough_plan(duration, str(recommendation.get("layout") or "split"))

    activity: list[tuple[float, float]] | None = None
    if settings.get("trim_silence", True) or settings.get("speed_up_silence", False):
        try:
            activity = detect_audio_activity(source_clip, duration=duration)
        except Exception:
            LOGGER.warning("Silence detection failed; montage keeps every frame", exc_info=True)
            activity = None
    try:
        return plan_edit(
            duration=duration,
            words=list(props.get("words") or []),
            recommendation=recommendation,
            config=config,
            peaks=list(props.get("peaks") or []),
            activity=activity,
            template_motion=template.motion,
        )
    except Exception:
        LOGGER.warning("Edit planning failed; rendering one continuous take", exc_info=True)
        return passthrough_plan(duration, str(recommendation.get("layout") or "split"))


def render_with_ffmpeg(
    source_clip: Path,
    props: dict[str, Any],
    ass_path: Path,
    output_path: Path,
    config: dict[str, Any],
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Render the final vertical clip in one hardware-accelerated FFmpeg pass."""
    source_clip = Path(source_clip).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe(source_clip)
    streams = info.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), {})
    has_audio = any(item.get("codec_type") == "audio" for item in streams)
    source_w = int(video_stream.get("width") or 1920)
    source_h = int(video_stream.get("height") or 1080)
    duration = _finite(info.get("duration"), 0.0)
    if duration <= 0:
        duration = _finite(props.get("durationInSeconds"), 30.0)

    template = resolve_template(props, config)
    face_track = resolve_face_track(
        source_clip,
        props=props,
        config=config,
        duration=duration,
        cache_dir=output_path.parent,
    )
    plan = stored_edit_plan(props, duration)
    LOGGER.info(
        "Render plan: template=%s cuts=%s output=%.2fs (trimmed %.2fs of %.2fs)",
        template.name,
        plan.cut_count,
        plan.output_duration,
        plan.removed_seconds,
        plan.source_duration,
    )
    (output_path.parent / "edit-plan.json").write_text(
        json.dumps(
            {"template": template.name, "plan": plan.to_dict()}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    filtergraph = build_ffmpeg_filtergraph(
        props=props,
        ass_path=ass_path,
        source_duration=duration,
        template=template,
        plan=plan,
        face_track=face_track,
        source_size=(source_w, source_h),
        has_audio=has_audio,
    )

    render_cfg = config.get("render", {})
    encoder = str(render_cfg.get("upscale_encoder") or render_cfg.get("codec") or "h264_nvenc")
    if encoder in ("h264", "libx264"):
        encoder = "h264_nvenc"
    fps = int(render_cfg.get("fps", template.fps) or template.fps)
    ffmpeg_bin = require_binary("ffmpeg")

    def command(video_args: list[str]) -> list[str]:
        args = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_clip),
            "-filter_complex",
            filtergraph,
            "-map",
            "[outv]",
        ]
        if has_audio:
            args += ["-map", "[acat]", "-c:a", "aac", "-b:a", "192k"]
        args += video_args
        args += ["-r", str(fps), "-movflags", "+faststart", str(output_path)]
        return args

    gpu_args = [
        "-c:v",
        encoder,
        "-preset",
        "p5",
        "-cq",
        str(render_cfg.get("cq", 19)),
        "-b:v",
        str(render_cfg.get("video_bitrate", "8M")),
        "-maxrate",
        "12M",
        "-bufsize",
        "16M",
    ]
    cpu_args = [
        "-c:v",
        "libx264",
        "-preset",
        str(render_cfg.get("ffmpeg_preset", "veryfast")),
        "-crf",
        str(render_cfg.get("crf", 18)),
    ]

    LOGGER.info("Starting FFmpeg render -> %s", output_path.name)
    try:
        run(command(gpu_args), timeout=timeout_seconds, capture=False)
    except Exception as exc:
        LOGGER.warning("GPU render failed (%s); retrying on CPU libx264", exc)
        run(command(cpu_args), timeout=timeout_seconds, capture=False)

    out_info = probe(output_path)
    out_streams = out_info.get("streams", [])
    video = next((item for item in out_streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in out_streams if item.get("codec_type") == "audio"), None)
    if video is None:
        raise ProcessError(f"Rendered clip has no video stream: {output_path}")
    if (video.get("width"), video.get("height")) != (template.width, template.height):
        raise ProcessError(
            f"Unexpected output resolution {video.get('width')}x{video.get('height')}, "
            f"expected {template.width}x{template.height}"
        )
    if has_audio and not audio:
        raise ProcessError(f"Rendered clip lost its audio: {output_path}")
    return out_info
