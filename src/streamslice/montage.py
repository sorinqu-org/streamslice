"""Deterministic edit-plan generation: jump cuts, layout switches, punches and hooks."""
from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

_LAYOUTS = frozenset({"split", "webcam_full", "gameplay_full"})

_BOOL_KEYS = frozenset({"enabled", "trim_silence", "punch_in", "hook_punch", "speed_up_silence"})
_STR_KEYS = frozenset({"event_layout"})
_INT_KEYS = frozenset({"max_punches"})

_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "trim_silence": True,
    "max_silence_seconds": 1.2,
    "silence_padding_seconds": 0.25,
    "protect_head_seconds": 2.0,
    "protect_tail_seconds": 1.5,
    "max_removed_ratio": 0.25,
    "min_segment_seconds": 1.5,
    "event_layout": "split",
    "punch_in": True,
    "punch_zoom": 1.10,
    "punch_seconds": 1.2,
    "punch_cooldown_seconds": 6.0,
    "max_punches": 4,
    "hook_punch": True,
    "hook_seconds": 1.5,
    "hook_zoom": 1.06,
    "speed_up_silence": False,
    "silence_speed": 1.6,
}

_MIN_SEGMENT_LENGTH = 1e-9


class MontageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Cut:
    """Один кусок итогового ролика, вырезанный из исходного клипа."""

    source_start: float
    source_end: float
    layout: str
    zoom: float
    speed: float
    reason: str

    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def output_duration(self) -> float:
        return self.source_duration / self.speed


@dataclass(frozen=True, slots=True)
class EditPlan:
    cuts: tuple[Cut, ...]
    source_duration: float
    output_duration: float
    removed_seconds: float
    cut_count: int

    def is_passthrough(self) -> bool:
        """True if this plan is a single unmodified cut spanning the whole clip."""
        if self.cut_count != 1:
            return False
        cut = self.cuts[0]
        return (
            math.isclose(cut.source_start, 0.0, abs_tol=1e-9)
            and math.isclose(cut.source_end, self.source_duration, abs_tol=1e-9)
            and math.isclose(cut.zoom, 1.0, abs_tol=1e-9)
            and math.isclose(cut.speed, 1.0, abs_tol=1e-9)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cuts": [
                {
                    "source_start": cut.source_start,
                    "source_end": cut.source_end,
                    "layout": cut.layout,
                    "zoom": cut.zoom,
                    "speed": cut.speed,
                    "reason": cut.reason,
                }
                for cut in self.cuts
            ],
            "source_duration": self.source_duration,
            "output_duration": self.output_duration,
            "removed_seconds": self.removed_seconds,
            "cut_count": self.cut_count,
        }

    def map_source_time(self, time: float, *, snap: str = "none") -> float | None:
        """Translate a source-clip timestamp into the rendered output timeline.

        Cutting silence shifts everything after each removed gap. Burned-in
        subtitles are authored against the source clip, so every word span has
        to travel through the same mapping or the captions drift by exactly the
        amount of footage that was removed before them.

        ``snap`` decides what happens to a timestamp that lands inside removed
        footage: ``"forward"`` moves it to the next surviving frame (right for a
        word start), ``"back"`` to the previous one (right for a word end), and
        ``"none"`` returns ``None``.
        """
        offset = 0.0
        for cut in self.cuts:
            if time < cut.source_start:
                if snap == "forward":
                    return offset
                return None if snap == "none" else offset
            if time <= cut.source_end:
                return offset + (time - cut.source_start) / cut.speed
            offset += cut.output_duration
        if snap == "back":
            return offset
        return None if snap == "none" else offset

    def remap_words(
        self, words: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return word spans rebased onto the output timeline, dropping cut-away ones."""
        remapped: list[dict[str, Any]] = []
        for word in words:
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if self.map_source_time(start) is None and self.map_source_time(end) is None:
                # The whole word lives inside removed footage.
                continue
            new_start = self.map_source_time(start, snap="forward")
            new_end = self.map_source_time(end, snap="back")
            if new_start is None or new_end is None or new_end <= new_start:
                continue
            item = dict(word)
            item["start"] = new_start
            item["end"] = new_end
            remapped.append(item)
        return remapped

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EditPlan:
        try:
            raw_cuts = data["cuts"]
            source_duration = float(data["source_duration"])
            output_duration = float(data["output_duration"])
            removed_seconds = float(data["removed_seconds"])
            cut_count = int(data["cut_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MontageError(f"Invalid edit plan payload: {exc}") from exc
        cuts: list[Cut] = []
        for item in raw_cuts:
            try:
                cuts.append(
                    Cut(
                        source_start=float(item["source_start"]),
                        source_end=float(item["source_end"]),
                        layout=str(item["layout"]),
                        zoom=float(item["zoom"]),
                        speed=float(item["speed"]),
                        reason=str(item["reason"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MontageError(f"Invalid cut payload: {exc}") from exc
        return cls(
            cuts=tuple(cuts),
            source_duration=source_duration,
            output_duration=output_duration,
            removed_seconds=removed_seconds,
            cut_count=cut_count,
        )


def _validated_duration(duration: Any) -> float:
    try:
        value = float(duration)
    except (TypeError, ValueError) as exc:
        raise MontageError(f"Invalid duration: {duration!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise MontageError(f"Duration must be finite and non-negative, got {duration!r}")
    return value


def passthrough_plan(duration: float, layout: str = "split") -> EditPlan:
    """Return a single-cut plan with no jump cuts, layout switches, zoom or speed change."""
    duration_value = _validated_duration(duration)
    resolved_layout = layout if layout in _LAYOUTS else "split"
    cut = Cut(
        source_start=0.0,
        source_end=duration_value,
        layout=resolved_layout,
        zoom=1.0,
        speed=1.0,
        reason="body",
    )
    return EditPlan(
        cuts=(cut,),
        source_duration=duration_value,
        output_duration=duration_value,
        removed_seconds=0.0,
        cut_count=1,
    )


def _coerce(key: str, raw: Any, default: Any) -> Any:
    if key in _BOOL_KEYS:
        return raw if isinstance(raw, bool) else default
    if key in _STR_KEYS:
        return raw if isinstance(raw, str) and raw in _LAYOUTS else default
    if key in _INT_KEYS:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _resolve_montage_config(
    config: Mapping[str, Any], template_motion: Mapping[str, Any] | None
) -> dict[str, Any]:
    montage_cfg = config.get("montage") if isinstance(config, Mapping) else None
    montage_cfg = montage_cfg if isinstance(montage_cfg, Mapping) else {}
    template_cfg = template_motion if isinstance(template_motion, Mapping) else {}
    resolved: dict[str, Any] = {}
    for key, default in _DEFAULTS.items():
        if key in template_cfg:
            raw = template_cfg[key]
        elif key in montage_cfg:
            raw = montage_cfg[key]
        else:
            raw = default
        resolved[key] = _coerce(key, raw, default)
    return resolved


def _word_spans(words: Sequence[Mapping[str, Any]], duration: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for item in words:
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end <= start:
            continue
        spans.append((start, end))
    spans.sort()
    return spans


def _snap_to_word_gap(t: float, word_spans: list[tuple[float, float]], duration: float) -> float:
    """Push a boundary that lands inside a word out to its nearest edge (never mid-phrase)."""
    t = max(0.0, min(duration, t))
    for start, end in word_spans:
        if start < t < end:
            return start if (t - start) <= (end - t) else end
    return t


def _event_window(
    recommendation: Mapping[str, Any], duration: float
) -> tuple[float, float] | None:
    if recommendation.get("gameplay_event_validated") is not True:
        return None
    try:
        event_time = float(recommendation["event_time"])
        event_end = float(recommendation["event_end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(event_time) and math.isfinite(event_end)):
        return None
    if not (0.0 <= event_time < event_end <= duration):
        return None
    return event_time, event_end


def _webcam_cut(recommendation: Mapping[str, Any], duration: float) -> float | None:
    try:
        cut_time = float(recommendation["webcam_cut_time"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(cut_time) or not (0.0 < cut_time < duration):
        return None
    return cut_time


def _make_layout_fn(
    base_layout: str,
    webcam_cut_time: float | None,
    event_window: tuple[float, float] | None,
    event_layout: str,
):
    def layout_at(t: float) -> str:
        # The gameplay event window wins over the webcam intro: by the time a
        # validated on-screen event happens, the webcam intro has necessarily ended.
        if event_window is not None and event_window[0] <= t < event_window[1]:
            return event_layout
        if webcam_cut_time is not None and t < webcam_cut_time:
            return "webcam_full"
        return base_layout

    return layout_at


def _merge_short_segments(
    segments: list[dict[str, Any]], min_segment_seconds: float
) -> list[dict[str, Any]]:
    if len(segments) <= 1:
        return segments
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        if seg["end"] - seg["start"] < min_segment_seconds:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    # The first segment could not look back; fold it forward if it is still short.
    if len(merged) > 1 and merged[0]["end"] - merged[0]["start"] < min_segment_seconds:
        merged[1]["start"] = merged[0]["start"]
        merged.pop(0)
    return merged


def _layout_segments(
    duration: float,
    boundary_times: list[float],
    layout_fn,
    word_spans: list[tuple[float, float]],
    min_segment_seconds: float,
) -> list[dict[str, Any]]:
    points = {0.0, duration}
    for t in boundary_times:
        points.add(_snap_to_word_gap(t, word_spans, duration))
    ordered = sorted(p for p in points if 0.0 <= p <= duration)
    raw: list[dict[str, Any]] = []
    for start, end in itertools.pairwise(ordered):
        if end - start <= _MIN_SEGMENT_LENGTH:
            continue
        mid = (start + end) / 2.0
        raw.append(
            {
                "start": start,
                "end": end,
                "layout": layout_fn(mid),
                "zoom": 1.0,
                "speed": 1.0,
                "reason": "body",
            }
        )
    if not raw:
        raw = [{"start": 0.0, "end": duration, "layout": layout_fn(duration / 2.0),
                "zoom": 1.0, "speed": 1.0, "reason": "body"}]
    return _merge_short_segments(raw, min_segment_seconds)


def _protected_regions(
    duration: float, cfg: Mapping[str, Any], event_window: tuple[float, float] | None
) -> list[tuple[float, float]]:
    regions = [
        (0.0, min(duration, cfg["protect_head_seconds"])),
        (max(0.0, duration - cfg["protect_tail_seconds"]), duration),
    ]
    if event_window is not None:
        regions.append(event_window)
    return regions


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _merge_intervals(
    activity: Sequence[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for item in activity:
        try:
            a, b = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        a = max(0.0, min(duration, a))
        b = max(0.0, min(duration, b))
        if b > a:
            cleaned.append((a, b))
    cleaned.sort()
    merged: list[tuple[float, float]] = []
    for a, b in cleaned:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _silence_gaps(
    duration: float, merged_activity: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in merged_activity:
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


def _eligible_gap_middles(
    duration: float,
    activity: Sequence[tuple[float, float]],
    cfg: Mapping[str, Any],
    protected: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged_activity = _merge_intervals(activity, duration)
    gaps = _silence_gaps(duration, merged_activity)
    candidates: list[tuple[float, float]] = []
    for g_start, g_end in gaps:
        if g_end - g_start <= cfg["max_silence_seconds"]:
            continue
        # Conservative: a gap that touches any protected region at all is left
        # untouched rather than partially trimmed, so the hook/payoff/event
        # window can never lose material to an off-by-a-second boundary.
        if any(_overlaps((g_start, g_end), region) for region in protected):
            continue
        removal_start = g_start + cfg["silence_padding_seconds"]
        removal_end = g_end - cfg["silence_padding_seconds"]
        if removal_end <= removal_start:
            continue
        candidates.append((removal_start, removal_end))
    return candidates


def _apply_ratio_budget(
    duration: float, candidates: list[tuple[float, float]], max_ratio: float
) -> list[tuple[float, float]]:
    budget = duration * max_ratio
    accepted: list[tuple[float, float]] = []
    total = 0.0
    for start, end in sorted(candidates):
        length = end - start
        if total + length > budget + 1e-9:
            continue
        accepted.append((start, end))
        total += length
    return accepted


def _remove_intervals_from_segments(
    segments: list[dict[str, Any]], removed: list[tuple[float, float]]
) -> list[dict[str, Any]]:
    if not removed:
        return segments
    result = list(segments)
    for r_start, r_end in removed:
        next_result: list[dict[str, Any]] = []
        for seg in result:
            s, e = seg["start"], seg["end"]
            if r_end <= s or r_start >= e:
                next_result.append(seg)
                continue
            if s < r_start:
                next_result.append({**seg, "end": r_start})
            if r_end < e:
                next_result.append({**seg, "start": r_end})
        result = next_result
    return [seg for seg in result if seg["end"] - seg["start"] > _MIN_SEGMENT_LENGTH]


def _split_segments_with_markers(
    segments: list[dict[str, Any]],
    markers: list[tuple[float, float]],
    *,
    speed: float,
    reason: str,
) -> list[dict[str, Any]]:
    if not markers:
        return segments
    result = list(segments)
    for m_start, m_end in markers:
        next_result: list[dict[str, Any]] = []
        for seg in result:
            s, e = seg["start"], seg["end"]
            if m_end <= s or m_start >= e:
                next_result.append(seg)
                continue
            lo, hi = max(s, m_start), min(e, m_end)
            if s < lo:
                next_result.append({**seg, "end": lo})
            next_result.append({**seg, "start": lo, "end": hi, "speed": speed, "reason": reason})
            if hi < e:
                next_result.append({**seg, "start": hi})
        result = next_result
    return [seg for seg in result if seg["end"] - seg["start"] > _MIN_SEGMENT_LENGTH]


def _peak_candidates(
    peaks: Sequence[Mapping[str, Any]], duration: float
) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    for item in peaks:
        try:
            t = float(item["time"])
            score = float(item.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(t) and math.isfinite(score)):
            continue
        if 0.0 <= t <= duration:
            candidates.append((score, t))
    return candidates


def _apply_single_punch(
    segments: list[dict[str, Any]], t: float, cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    owner = None
    for seg in segments:
        if seg["start"] <= t < seg["end"]:
            owner = seg
            break
    if owner is None:
        return segments
    half = cfg["punch_seconds"] / 2.0
    p_start = max(owner["start"], t - half)
    p_end = min(owner["end"], t + half)
    if p_end - p_start <= 1e-6:
        return segments
    result: list[dict[str, Any]] = []
    for seg in segments:
        if seg is not owner:
            result.append(seg)
            continue
        s, e = seg["start"], seg["end"]
        if s < p_start:
            result.append({**seg, "end": p_start})
        result.append(
            {
                **seg,
                "start": p_start,
                "end": p_end,
                "zoom": cfg["punch_zoom"],
                "reason": "punch",
            }
        )
        if p_end < e:
            result.append({**seg, "start": p_end})
    result.sort(key=lambda item: item["start"])
    return result


def _apply_punches(
    segments: list[dict[str, Any]],
    peaks: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    duration: float,
) -> list[dict[str, Any]]:
    ranked = sorted(_peak_candidates(peaks, duration), key=lambda pair: pair[0], reverse=True)
    chosen: list[float] = []
    for _score, t in ranked:
        if len(chosen) >= cfg["max_punches"]:
            break
        if any(abs(t - c) < cfg["punch_cooldown_seconds"] for c in chosen):
            continue
        chosen.append(t)
    for t in sorted(chosen):
        segments = _apply_single_punch(segments, t, cfg)
    return segments


def _apply_hook(segments: list[dict[str, Any]], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not segments:
        return segments
    first = segments[0]
    hook_end = min(first["end"], first["start"] + cfg["hook_seconds"])
    if hook_end <= first["start"]:
        return segments
    result: list[dict[str, Any]] = []
    if hook_end < first["end"]:
        result.append({**first, "end": hook_end, "zoom": cfg["hook_zoom"], "reason": "hook"})
        result.append({**first, "start": hook_end})
    else:
        result.append({**first, "zoom": cfg["hook_zoom"], "reason": "hook"})
    result.extend(segments[1:])
    return result


def _finalize(segments: list[dict[str, Any]], duration: float) -> EditPlan:
    ordered = sorted(segments, key=lambda item: item["start"])
    cuts = tuple(
        Cut(
            source_start=seg["start"],
            source_end=seg["end"],
            layout=seg["layout"],
            zoom=seg["zoom"],
            speed=seg["speed"],
            reason=seg["reason"],
        )
        for seg in ordered
    )
    source_total = sum(cut.source_duration for cut in cuts)
    output_total = sum(cut.output_duration for cut in cuts)
    return EditPlan(
        cuts=cuts,
        source_duration=duration,
        output_duration=output_total,
        removed_seconds=duration - source_total,
        cut_count=len(cuts),
    )


def plan_edit(
    *,
    duration: float,
    words: Sequence[Mapping[str, Any]],
    recommendation: Mapping[str, Any],
    config: Mapping[str, Any],
    peaks: Sequence[Mapping[str, Any]] | None = None,
    activity: Sequence[tuple[float, float]] | None = None,
    template_motion: Mapping[str, Any] | None = None,
) -> EditPlan:
    """Compute a montage plan (jump cuts, layout switches, punch-ins, hook) for one clip."""
    duration_value = _validated_duration(duration)
    rec = recommendation if isinstance(recommendation, Mapping) else {}
    raw_layout = rec.get("layout")
    base_layout = raw_layout if raw_layout in _LAYOUTS else "split"

    if duration_value <= 0.0:
        return passthrough_plan(duration_value, base_layout)

    cfg = _resolve_montage_config(config, template_motion)
    if not cfg["enabled"]:
        return passthrough_plan(duration_value, base_layout)

    word_spans = _word_spans(words, duration_value)
    event_window = _event_window(rec, duration_value)
    webcam_cut_time = _webcam_cut(rec, duration_value)
    layout_fn = _make_layout_fn(base_layout, webcam_cut_time, event_window, cfg["event_layout"])

    boundary_times: list[float] = []
    if webcam_cut_time is not None:
        boundary_times.append(webcam_cut_time)
    if event_window is not None:
        boundary_times.extend(event_window)

    segments = _layout_segments(
        duration_value, boundary_times, layout_fn, word_spans, cfg["min_segment_seconds"]
    )

    if activity is not None and cfg["trim_silence"]:
        protected = _protected_regions(duration_value, cfg, event_window)
        candidates = _eligible_gap_middles(duration_value, activity, cfg, protected)
        if cfg["speed_up_silence"]:
            segments = _split_segments_with_markers(
                segments, candidates, speed=cfg["silence_speed"], reason="silence-trim"
            )
        else:
            removed = _apply_ratio_budget(duration_value, candidates, cfg["max_removed_ratio"])
            segments = _remove_intervals_from_segments(segments, removed)

    if cfg["punch_in"] and peaks:
        segments = _apply_punches(segments, peaks, cfg, duration_value)

    if cfg["hook_punch"]:
        segments = _apply_hook(segments, cfg)

    if not segments:
        return passthrough_plan(duration_value, base_layout)

    plan = _finalize(segments, duration_value)
    LOGGER.info(
        "Planned %d cut(s) for a %.2fs clip, removed %.2fs",
        plan.cut_count,
        duration_value,
        plan.removed_seconds,
    )
    return plan
