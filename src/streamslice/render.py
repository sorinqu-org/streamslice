from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .config import project_path
from .ffmpeg_render import render_with_ffmpeg, resolve_edit_plan, resolve_template
from .media import copy_for_remotion, probe
from .models import Candidate
from .process import ProcessError, require_binary, run
from .subtitles import build_subtitles_ass

LOGGER = logging.getLogger(__name__)


_LAYOUT_TIME_KEYS = ("focus_time", "event_time", "event_end", "webcam_cut_time")
_LAYOUTS = {"split", "webcam_full", "gameplay_full"}


def _normalized_crop(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        crop = {key: float(value[key]) for key in ("x", "y", "width", "height")}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in crop.values()):
        return None
    if (
        crop["width"] <= 0
        or crop["height"] <= 0
        or crop["x"] < 0
        or crop["y"] < 0
        or crop["x"] + crop["width"] > 1
        or crop["y"] + crop["height"] > 1
    ):
        return None
    return crop


def normalized_recommendation(candidate: Candidate) -> dict[str, Any]:
    """Return a clip-local, render-safe camera recommendation.

    Discovery sees the source chunk and therefore emits absolute timestamps.
    Remotion sees the cut clip and must receive timestamps relative to that cut.
    Layout analysis marks its own output as clip-local with ``time_base`` so a
    second normalization pass cannot subtract the candidate start a second time.
    """
    recommendation = dict(candidate.camera_layout_recommendation or {})
    if recommendation.get("layout") not in _LAYOUTS:
        recommendation["layout"] = "split"
    time_base = recommendation.get("time_base")
    for key in _LAYOUT_TIME_KEYS:
        if key not in recommendation:
            continue
        try:
            value = float(recommendation[key])
            # Legacy recommendations predate the explicit time_base field.
            # Discovery promised absolute source seconds, so an unmarked
            # value inside the candidate's source interval is absolute even
            # when it is also numerically smaller than clip duration.
            legacy_absolute = time_base != "clip" and (
                candidate.start_time <= value <= candidate.end_time
                or value > candidate.duration + 1
            )
            if time_base == "source" or legacy_absolute:
                value -= candidate.start_time
            if not math.isfinite(value):
                raise ValueError("non-finite layout timestamp")
            recommendation[key] = max(0.0, min(candidate.duration, value))
        except (TypeError, ValueError):
            recommendation.pop(key, None)

    # A gameplay event is an explicit visual-analysis result, not merely a
    # focus_time supplied by discovery.  Drop incomplete/invalid event windows
    # so downstream renderers cannot accidentally animate routine gameplay.
    event_time = recommendation.get("event_time")
    event_end = recommendation.get("event_end")
    event_valid = recommendation.get("gameplay_event_validated") is True
    if not (
        event_valid
        and isinstance(event_time, (int, float))
        and isinstance(event_end, (int, float))
        and math.isfinite(float(event_time))
        and math.isfinite(float(event_end))
        and 0 <= float(event_time) < float(event_end) <= candidate.duration
    ):
        recommendation.pop("event_time", None)
        recommendation.pop("event_end", None)
        recommendation.pop("gameplay_event_validated", None)
        recommendation.pop("gameplay_zoom", None)
        recommendation.pop("gameplay_crop_width", None)

    if recommendation.get("webcam_box_validated") is True:
        webcam_crop = _normalized_crop(
            recommendation.get("webcam_crop") or recommendation.get("webcam_box")
        )
        if webcam_crop is None:
            recommendation.pop("webcam_box_validated", None)
            recommendation.pop("webcam_box_confidence", None)
            recommendation.pop("webcam_crop", None)
            recommendation.pop("webcam_box", None)
        else:
            # Keep both spellings in the serialized contract for old queue
            # consumers; Remotion uses webcam_crop as the canonical field.
            recommendation["webcam_crop"] = webcam_crop
            recommendation["webcam_box"] = dict(webcam_crop)
    else:
        recommendation.pop("webcam_box_validated", None)
        recommendation.pop("webcam_box_confidence", None)
        recommendation.pop("webcam_crop", None)
        recommendation.pop("webcam_box", None)
    recommendation.pop("webcam_detected", None)
    recommendation["time_base"] = "clip"
    return recommendation


def plan_montage_and_subtitles(
    source_clip: Path,
    *,
    props: dict[str, Any],
    candidate: Candidate,
    clip_dir: Path,
    config: dict[str, Any],
    duration: float,
) -> dict[str, Any]:
    """Attach an edit plan to the props and burn subtitles onto its timeline.

    Trimming silence moves every later caption earlier by exactly the amount of
    footage removed before it. Planning and subtitle generation therefore have to
    happen together: the plan is stored in the props for the renderer, while the
    ``.ass`` file is written from word spans already rebased onto the cut
    timeline.
    """
    template = resolve_template(props, config)
    plan = resolve_edit_plan(
        source_clip,
        props=props,
        config=config,
        template=template,
        duration=duration,
    )
    props["template"] = template.name
    props["editPlan"] = plan.to_dict()
    words = list(props.get("words") or [])
    timeline_words = words if plan.is_passthrough() else plan.remap_words(words)
    if not plan.is_passthrough():
        LOGGER.info(
            "Montage: %s cuts, %.2fs trimmed, %s of %s captions kept",
            plan.cut_count,
            plan.removed_seconds,
            len(timeline_words),
            len(words),
        )
    build_subtitles_ass(
        timeline_words,
        Candidate(
            start_time=0.0,
            end_time=max(0.1, plan.output_duration),
            highlight_reason=candidate.highlight_reason,
            emotion_score=candidate.emotion_score,
            camera_layout_recommendation=props.get("layoutRecommendation"),
        ),
        clip_dir,
        config,
        title=props.get("title", ""),
    )
    return props


def prepare_render_props(
    *,
    source_name: str,
    candidate: Candidate,
    words: list[dict[str, Any]],
    overlay_masks: list[dict[str, Any]],
    props_path: Path,
    config: dict[str, Any],
    title: str = "",
) -> dict[str, Any]:
    render = config["render"]
    fallback_title = candidate.highlight_reason if hasattr(candidate, "highlight_reason") else ""
    props = {
        "source": source_name,
        "title": title or getattr(candidate, "title", "") or fallback_title,
        "durationInSeconds": candidate.duration,
        "renderFps": int(render.get("remotion_fps", render.get("fps", 60))),
        "renderVersion": int(render.get("render_version", 1)),
        "emotionScore": candidate.emotion_score,
        "layoutRecommendation": normalized_recommendation(candidate),
        "words": words,
        "overlayMasks": overlay_masks,
        "overlayBlurPx": int(config["overlay_cleanup"]["blur_px"]),
        "layout": config["layout"],
        "subtitleStyle": config["subtitles"],
    }
    props_path.write_text(json.dumps(props, ensure_ascii=False), encoding="utf-8")
    return props


def render_clip(
    source_clip: Path,
    candidate: Candidate,
    words: list[dict[str, Any]],
    overlay_masks: list[dict[str, Any]],
    output_path: Path,
    *,
    job_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    render = config["render"]
    engine = str(render.get("engine", "ffmpeg")).lower()
    props_path = output_path.parent / "remotion-props.json"
    props = prepare_render_props(
        source_name="source.mp4",
        candidate=candidate,
        words=words,
        overlay_masks=overlay_masks,
        props_path=props_path,
        config=config,
    )
    if engine == "ffmpeg":
        props = plan_montage_and_subtitles(
            source_clip,
            props=props,
            candidate=candidate,
            clip_dir=output_path.parent,
            config=config,
            duration=candidate.duration,
        )
        props_path.write_text(json.dumps(props, ensure_ascii=False), encoding="utf-8")
        return render_with_ffmpeg(
            source_clip=source_clip,
            props=props,
            ass_path=output_path.parent / "subtitles.ass",
            output_path=output_path,
            config=config,
            timeout_seconds=float(render.get("timeout_seconds", 300)),
        )

    remotion_dir = project_path(config, render["remotion_dir"])
    public_name = f"jobs/{job_id}/source.mp4"
    public_target = remotion_dir / "public" / public_name
    copy_for_remotion(source_clip, public_target)
    props["source"] = public_name
    props_path.write_text(json.dumps(props, ensure_ascii=False), encoding="utf-8")
    return _render_with_props(props_path, output_path, config)


def render_prepared_clip(
    clip_dir: Path,
    *,
    job_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    clip_dir = clip_dir.expanduser().resolve()
    LOGGER.info("[%s] [%s] Подготовка и старт рендера клипа...", job_id, clip_dir.name)
    source_clip = clip_dir / "source.mp4"
    props_path = clip_dir / "remotion-props.json"
    if not source_clip.is_file() or not props_path.is_file():
        raise ProcessError(f"Incomplete render bundle: {clip_dir}")
    props = json.loads(props_path.read_text(encoding="utf-8"))
    if not isinstance(props, dict):
        raise ProcessError(f"Invalid render props: {props_path}")

    output_path = clip_dir / f"{clip_dir.name}.mp4"
    engine = str(config.get("render", {}).get("engine", "ffmpeg")).lower()

    if engine == "ffmpeg":
        # A queued bundle carries props prepared on the host, but the montage plan
        # and subtitles are rebuilt here so a template or montage change on the
        # render machine takes effect without re-preparing the job.
        duration = float(props.get("durationInSeconds", 30.0))
        props = plan_montage_and_subtitles(
            source_clip,
            props=props,
            candidate=Candidate(
                start_time=0.0,
                end_time=max(0.1, duration),
                highlight_reason=props.get("title", ""),
                emotion_score=float(props.get("emotionScore", 5.0)),
                camera_layout_recommendation=props.get("layoutRecommendation"),
            ),
            clip_dir=clip_dir,
            config=config,
            duration=duration,
        )
        props_path.write_text(json.dumps(props, ensure_ascii=False), encoding="utf-8")
        info = render_with_ffmpeg(
            source_clip=source_clip,
            props=props,
            ass_path=clip_dir / "subtitles.ass",
            output_path=output_path,
            config=config,
            timeout_seconds=float(config.get("render", {}).get("timeout_seconds", 300)),
        )
    else:
        remotion_dir = project_path(config, config["render"]["remotion_dir"])
        public_name = f"jobs/{job_id}/{clip_dir.name}/source.mp4"
        copy_for_remotion(source_clip, remotion_dir / "public" / public_name)
        props["source"] = public_name
        local_props = clip_dir / "remotion-props.local.json"
        local_props.write_text(json.dumps(props, ensure_ascii=False), encoding="utf-8")
        info = _render_with_props(local_props, output_path, config)

    LOGGER.info(
        "[%s] [%s] Рендер клипа успешно завершен -> %s",
        job_id,
        clip_dir.name,
        output_path.name,
    )
    return info


def _render_with_props(
    props_path: Path,
    output_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    render = config["render"]
    remotion_dir = project_path(config, render["remotion_dir"])
    chromium = require_binary(render["chromium_executable"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(render.get("scale", 1.0))
    remotion_output = (
        output_path
        if abs(scale - 1.0) < 1e-6
        else output_path.with_name(output_path.stem + ".remotion.mp4")
    )
    remotion_fps = int(render.get("remotion_fps", 60))
    final_fps = int(render.get("fps", 60))
    remotion_fps_args = (
        ["--override-fps", str(remotion_fps)] if remotion_fps != final_fps else []
    )
    acceleration = str(render.get("hardware_acceleration", "if-possible"))
    quality_args = (
        ["--video-bitrate", str(render.get("video_bitrate", "8M"))]
        if acceleration != "disable"
        else ["--crf", str(render["crf"])]
    )
    chromium_flags = render.get(
        "chromium_options",
        "--ignore-gpu-blocklist,--enable-gpu-rasterization,--enable-zero-copy,--disable-gpu-sandbox",
    )
    LOGGER.info(
        "Running Remotion: props=%s -> output=%s (scale=%.2f, fps=%d, concurrency=%s)",
        props_path.name,
        remotion_output.name,
        scale,
        remotion_fps,
        render.get("concurrency"),
    )
    run(
        [
            require_binary("npm"),
            "run",
            "render",
            "--",
            "--props",
            str(props_path),
            "--output",
            str(remotion_output),
            "--browser-executable",
            chromium,
            "--concurrency",
            str(render["concurrency"]),
            *quality_args,
            "--hardware-acceleration",
            acceleration,
            "--chromium-options",
            chromium_flags,
            "--offthread-video-threads",
            str(render.get("offthread_video_threads", 4)),
            "--gl",
            str(render.get("gl_renderer", "angle-egl")),
            "--overwrite",
            "--scale",
            str(scale),
            *remotion_fps_args,
        ],
        cwd=remotion_dir,
        timeout=float(render["timeout_seconds"]),
        capture=False,
    )
    if remotion_output != output_path:
        LOGGER.info(
            "Upscaling video %s -> %s via FFmpeg...", remotion_output.name, output_path.name
        )
        _upscale_to_final(remotion_output, output_path, render)
    info = probe(output_path)
    video = next(item for item in info["streams"] if item.get("codec_type") == "video")
    audio = next((item for item in info["streams"] if item.get("codec_type") == "audio"), None)
    if (video.get("width"), video.get("height")) != (1080, 1920):
        raise ProcessError(f"Unexpected output resolution: {video}")
    numerator, denominator = map(int, str(video["r_frame_rate"]).split("/"))
    if not denominator or not math.isclose(numerator / denominator, 60, abs_tol=0.01):
        raise ProcessError(f"Unexpected output FPS: {video['r_frame_rate']}")
    if not audio:
        raise ProcessError(f"Rendered clip has no audio: {output_path}")
    return info


def _upscale_to_final(source: Path, target: Path, render: dict[str, Any]) -> None:
    prefix = [
        require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        "scale=1080:1920:flags=lanczos,fps=60",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
    ]
    suffix = ["-movflags", "+faststart", "-y", str(target)]
    encoder = str(render.get("upscale_encoder", "h264_nvenc"))
    try:
        run(
            [*prefix, "-c:v", encoder, "-preset", "p5", "-cq", "19", "-c:a", "copy", *suffix],
            timeout=600,
        )
    except Exception:
        run(
            [
                *prefix,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "copy",
                *suffix,
            ],
            timeout=600,
        )
