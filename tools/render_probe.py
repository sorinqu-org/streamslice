#!/usr/bin/env python3
"""Render one clip with explicit overrides and report the framing decisions.

A visual-composition change cannot be reviewed by reading the filtergraph. This
harness renders a real source clip, then prints the crop geometry it chose and
extracts still frames so the result can actually be looked at.

    PYTHONPATH=src python3 tools/render_probe.py \\
        --source remotion/public/jobs/<job>/clip-02/source.mp4 \\
        --out .work/probe/after --template classic-split
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from streamslice.config import load_config
from streamslice.ffmpeg_render import render_with_ffmpeg, resolve_edit_plan, resolve_template
from streamslice.media import probe
from streamslice.models import Candidate
from streamslice.process import require_binary, run
from streamslice.render import plan_montage_and_subtitles

DEFAULT_WORDS: list[dict[str, Any]] = [
    {"start": 0.4, "end": 0.9, "text": "смотри"},
    {"start": 1.0, "end": 1.6, "text": "что"},
    {"start": 1.7, "end": 2.4, "text": "происходит"},
    {"start": 6.0, "end": 6.5, "text": "нет"},
    {"start": 6.6, "end": 7.4, "text": "стоп"},
    {"start": 12.0, "end": 12.8, "text": "серьёзно"},
    {"start": 20.0, "end": 20.6, "text": "всё"},
    {"start": 20.7, "end": 21.5, "text": "конец"},
]


def build_props(source: Path, config: dict[str, Any], duration: float) -> dict[str, Any]:
    return {
        "source": source.name,
        "title": "ТЕСТОВЫЙ ПРОГОН РЕНДЕРА",
        "durationInSeconds": duration,
        "renderFps": int(config["render"].get("fps", 60)),
        "renderVersion": int(config["render"].get("render_version", 1)),
        "emotionScore": 9.0,
        "layoutRecommendation": {
            "layout": "split",
            "focus_x": 0.4,
            "focus_y": 0.5,
            "time_base": "clip",
        },
        "words": DEFAULT_WORDS,
        "overlayMasks": [],
        "overlayBlurPx": int(config["overlay_cleanup"]["blur_px"]),
        "layout": config["layout"],
        "subtitleStyle": config["subtitles"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--template")
    parser.add_argument("--layout", default="split")
    parser.add_argument(
        "--webcam",
        help="override the webcam region as x,y,w,h in 0..1 (e.g. 0,0,1,1 for a full-frame cam)",
    )
    parser.add_argument("--no-face", action="store_true", help="disable face tracking")
    parser.add_argument("--no-montage", action="store_true", help="disable montage cuts")
    parser.add_argument("--stills", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )

    config = load_config(Path(args.config))
    if args.template:
        config["render"]["template"] = args.template
    config.setdefault("face_tracking", {})["enabled"] = not args.no_face
    config.setdefault("montage", {})["enabled"] = not args.no_montage

    source = Path(args.source).resolve()
    clip_dir = Path(args.out).resolve()
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    clip_dir.mkdir(parents=True)
    local_source = clip_dir / "source.mp4"
    shutil.copy2(source, local_source)

    info = probe(local_source)
    duration = float(info.get("duration") or 0.0)
    props = build_props(local_source, config, duration)
    props["layoutRecommendation"]["layout"] = args.layout
    if args.webcam:
        x, y, width, height = (float(part) for part in args.webcam.split(","))
        box = {"x": x, "y": y, "width": width, "height": height}
        props["layoutRecommendation"]["webcam_crop"] = box
        props["layoutRecommendation"]["webcam_box"] = dict(box)
        props["layout"]["webcam"] = dict(box)

    candidate = Candidate(
        start_time=0.0,
        end_time=duration,
        highlight_reason=props["title"],
        emotion_score=9.0,
        camera_layout_recommendation=props["layoutRecommendation"],
    )
    template = resolve_template(props, config)
    plan = resolve_edit_plan(
        local_source, props=props, config=config, template=template, duration=duration
    )
    props = plan_montage_and_subtitles(
        local_source,
        props=props,
        candidate=candidate,
        clip_dir=clip_dir,
        config=config,
        duration=duration,
    )
    (clip_dir / "remotion-props.json").write_text(
        json.dumps(props, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output = clip_dir / "render.mp4"
    render_with_ffmpeg(
        source_clip=local_source,
        props=props,
        ass_path=clip_dir / "subtitles.ass",
        output_path=output,
        config=config,
        timeout_seconds=900.0,
    )

    out_info = probe(output)
    out_duration = float(out_info.get("duration") or 0.0)
    print(f"template     : {template.name}")
    print(f"source       : {duration:.2f}s -> output {out_duration:.2f}s")
    print(f"plan         : {plan.cut_count} cuts, {plan.removed_seconds:.2f}s trimmed")
    for cut in plan.cuts:
        print(
            f"  [{cut.source_start:6.2f} -> {cut.source_end:6.2f}] "
            f"{cut.layout:<14} zoom={cut.zoom:.3f} speed={cut.speed:.2f} {cut.reason}"
        )

    ffmpeg = require_binary("ffmpeg")
    for index in range(args.stills):
        timestamp = out_duration * (index + 0.5) / args.stills
        run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-ss", f"{timestamp:.3f}", "-i", str(output),
                "-frames:v", "1", "-y", str(clip_dir / f"still-{index:02d}.png"),
            ],
            timeout=120,
        )
    print(f"stills       : {clip_dir}/still-*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
