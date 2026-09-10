"""Focal point tracking and FFmpeg crop filter dynamic expression generator."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, val))


def _cubic_ease_in_out(t: float) -> float:
    """Standard smooth cubic easeInOut (Hermite interpolation / smoothstep: 3t^2 - 2t^3)."""
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _catmull_rom_interpolate(
    p0: float, p1: float, p2: float, p3: float, t: float
) -> float:
    """Centripetal/uniform Catmull-Rom spline interpolation between p1 and p2 for t in [0, 1]."""
    t = _clamp(t, 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    val = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return _clamp(val, 0.0, 1.0)


class PointTracker:
    """Tracks and interpolates focal points over time."""

    def __init__(
        self,
        points: Sequence[dict[str, Any]],
        duration: float = 0.0,
        method: str = "catmull_rom",
    ) -> None:
        """Initialize tracker with a list of focal points and total duration.

        points format: [{"time": float, "focus_x": float, "focus_y": float}, ...]
        method: 'catmull_rom' or 'ease_in_out' / 'cubic'
        """
        self.duration = max(0.0, float(duration))
        self.method = method

        # Clean and sort points
        cleaned: list[dict[str, float]] = []
        for pt in points:
            try:
                t = float(pt["time"])
                x = _clamp(float(pt["focus_x"]))
                y = _clamp(float(pt["focus_y"]))
                cleaned.append({"time": t, "focus_x": x, "focus_y": y})
            except (KeyError, TypeError, ValueError):
                continue

        cleaned.sort(key=lambda p: p["time"])

        # Deduplicate points with identical timestamps (keep last)
        deduped: list[dict[str, float]] = []
        for pt in cleaned:
            if deduped and math.isclose(deduped[-1]["time"], pt["time"], abs_tol=1e-5):
                deduped[-1] = pt
            else:
                deduped.append(pt)

        if not deduped:
            # Default fallback: center point
            deduped = [{"time": 0.0, "focus_x": 0.5, "focus_y": 0.5}]

        self.points = deduped

    def is_static(self) -> bool:
        """Return True if tracker represents a static point."""
        if len(self.points) == 1:
            return True
        first_x = self.points[0]["focus_x"]
        first_y = self.points[0]["focus_y"]
        return all(
            math.isclose(p["focus_x"], first_x, abs_tol=1e-5)
            and math.isclose(p["focus_y"], first_y, abs_tol=1e-5)
            for p in self.points
        )

    def evaluate(self, t: float) -> tuple[float, float]:
        """Evaluate interpolated focus (x, y) at time t in [0.0, 1.0] coordinates."""
        if len(self.points) == 1 or t <= self.points[0]["time"]:
            return self.points[0]["focus_x"], self.points[0]["focus_y"]

        if t >= self.points[-1]["time"]:
            return self.points[-1]["focus_x"], self.points[-1]["focus_y"]

        # Find interval
        idx = 0
        while idx < len(self.points) - 1 and self.points[idx + 1]["time"] < t:
            idx += 1

        p1 = self.points[idx]
        p2 = self.points[idx + 1]

        t1, t2 = p1["time"], p2["time"]
        dt = t2 - t1
        if dt <= 1e-6:
            return p1["focus_x"], p1["focus_y"]

        norm_t = (t - t1) / dt

        if self.method in ("ease_in_out", "cubic"):
            ease = _cubic_ease_in_out(norm_t)
            x = p1["focus_x"] + (p2["focus_x"] - p1["focus_x"]) * ease
            y = p1["focus_y"] + (p2["focus_y"] - p1["focus_y"]) * ease
            return _clamp(x), _clamp(y)

        # Catmull-Rom spline
        p0 = self.points[idx - 1] if idx > 0 else p1
        p3 = self.points[idx + 2] if idx + 2 < len(self.points) else p2

        x = _catmull_rom_interpolate(
            p0["focus_x"], p1["focus_x"], p2["focus_x"], p3["focus_x"], norm_t
        )
        y = _catmull_rom_interpolate(
            p0["focus_y"], p1["focus_y"], p2["focus_y"], p3["focus_y"], norm_t
        )
        return _clamp(x), _clamp(y)

    def generate_lookup(
        self, fps: float = 30.0, duration: float | None = None
    ) -> list[dict[str, Any]]:
        """Generate a per-frame or per-timestamp lookup table of focus positions."""
        total_duration = duration if duration is not None else self.duration
        if total_duration <= 0 and self.points:
            total_duration = self.points[-1]["time"]
        total_duration = max(total_duration, 0.0)

        frames = max(1, math.ceil(total_duration * fps)) if total_duration > 0 else 1
        lookup: list[dict[str, Any]] = []

        for frame in range(frames):
            time_sec = frame / fps if fps > 0 else 0.0
            x, y = self.evaluate(time_sec)
            lookup.append({
                "frame": frame,
                "time": round(time_sec, 4),
                "focus_x": round(x, 6),
                "focus_y": round(y, 6),
            })
        return lookup

    def ffmpeg_x_expr(
        self, crop_w_expr: str = "out_w", in_w_expr: str = "in_w"
    ) -> str:
        """Generate dynamic FFmpeg eval expression for crop x parameter.

        Returns string suitable for x='...' in FFmpeg crop filter.
        The crop window will center on focus_x and clamp to [0, in_w - out_w].
        """
        if self.is_static():
            focus_x = self.points[0]["focus_x"]
            # Constant expression clamped between 0 and in_w - out_w
            return f"clip({focus_x:.4f}*{in_w_expr}-({crop_w_expr})/2,0,{in_w_expr}-{crop_w_expr})"

        # Build nested if(between(t, t1, t2), interp, ...) chain
        # For smooth interpolation across intervals, easeInOut (Hermite cubic)
        # works cleanly in FFmpeg expressions.
        last_x = self.points[-1]["focus_x"]
        expr = f"clip({last_x:.4f}*{in_w_expr}-({crop_w_expr})/2,0,{in_w_expr}-{crop_w_expr})"

        # Work backwards from last interval to first
        for i in range(len(self.points) - 2, -1, -1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            t1 = p1["time"]
            t2 = p2["time"]
            x1 = p1["focus_x"]
            x2 = p2["focus_x"]

            dt = t2 - t1
            if dt <= 1e-6:
                continue

            # normalized time u = (t - t1) / dt
            # Hermite cubic ease: 3*u^2 - 2*u^3 = u*u*(3 - 2*u)
            # x(t) = x1 + (x2 - x1) * (3*((t-t1)/dt)^2 - 2*((t-t1)/dt)^3)
            # Center of crop: x_center * in_w - out_w / 2
            u_expr = f"(t-{t1:.4f})/{dt:.4f}"
            ease_expr = f"(3*pow({u_expr},2)-2*pow({u_expr},3))"
            interp_focus = f"({x1:.4f}+({x2 - x1:.4f})*{ease_expr})"
            target_x = f"({interp_focus}*{in_w_expr}-({crop_w_expr})/2)"

            base_x = f"{x1:.4f}*{in_w_expr}-({crop_w_expr})/2"
            expr = f"if(lt(t,{t2:.4f}),if(lt(t,{t1:.4f}),{base_x},{target_x}),{expr})"

        return f"clip({expr},0,{in_w_expr}-{crop_w_expr})"

    def ffmpeg_y_expr(
        self, crop_h_expr: str = "out_h", in_h_expr: str = "in_h"
    ) -> str:
        """Generate dynamic FFmpeg eval expression for crop y parameter."""
        if self.is_static():
            focus_y = self.points[0]["focus_y"]
            return f"clip({focus_y:.4f}*{in_h_expr}-({crop_h_expr})/2,0,{in_h_expr}-{crop_h_expr})"

        last_y = self.points[-1]["focus_y"]
        expr = f"clip({last_y:.4f}*{in_h_expr}-({crop_h_expr})/2,0,{in_h_expr}-{crop_h_expr})"

        for i in range(len(self.points) - 2, -1, -1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            t1 = p1["time"]
            t2 = p2["time"]
            y1 = p1["focus_y"]
            y2 = p2["focus_y"]

            dt = t2 - t1
            if dt <= 1e-6:
                continue

            u_expr = f"(t-{t1:.4f})/{dt:.4f}"
            ease_expr = f"(3*pow({u_expr},2)-2*pow({u_expr},3))"
            interp_focus = f"({y1:.4f}+({y2 - y1:.4f})*{ease_expr})"
            target_y = f"({interp_focus}*{in_h_expr}-({crop_h_expr})/2)"

            base_y = f"{y1:.4f}*{in_h_expr}-({crop_h_expr})/2"
            expr = f"if(lt(t,{t2:.4f}),if(lt(t,{t1:.4f}),{base_y},{target_y}),{expr})"

        return f"clip({expr},0,{in_h_expr}-{crop_h_expr})"

    def ffmpeg_crop_filter(
        self,
        crop_width: int | str,
        crop_height: int | str,
        in_w: int | str = "in_w",
        in_h: int | str = "in_h",
    ) -> str:
        """Generate full FFmpeg crop filter string: crop=w:h:x:y."""
        x_expr = self.ffmpeg_x_expr(crop_w_expr=str(crop_width), in_w_expr=str(in_w))
        y_expr = self.ffmpeg_y_expr(crop_h_expr=str(crop_height), in_h_expr=str(in_h))
        return f"crop=w={crop_width}:h={crop_height}:x='{x_expr}':y='{y_expr}'"


def build_crop_filter(
    points: Sequence[dict[str, Any]],
    duration: float,
    crop_width: int | str,
    crop_height: int | str,
    method: str = "cubic",
) -> str:
    """Convenience function to generate FFmpeg crop filter string."""
    tracker = PointTracker(points, duration=duration, method=method)
    return tracker.ffmpeg_crop_filter(crop_width, crop_height)
