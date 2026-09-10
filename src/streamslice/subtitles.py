from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Candidate, Word
from .profanity import mask_profanity


def build_subtitles(
    words: list[Word | dict[str, Any]],
    candidate: Candidate,
    output_dir: Path,
    config: dict[str, Any],
    title: str = "",
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    previous_end = 0.0
    for item in words:
        w_start = float(item["start"] if isinstance(item, dict) else item.start)
        w_end = float(item["end"] if isinstance(item, dict) else item.end)
        w_text = str(item["text"] if isinstance(item, dict) else item.text)
        if w_end < candidate.start_time or w_start > candidate.end_time:
            continue
        start = max(0.0, w_start - candidate.start_time, previous_end)
        end = min(candidate.duration, w_end - candidate.start_time)
        end = max(start + 0.02, end)
        if start >= candidate.duration:
            continue
        end = min(candidate.duration, end)
        if config["subtitles"].get("mask_profanity", True):
            w_text = mask_profanity(w_text)
        selected.append({"start": round(start, 3), "end": round(end, 3), "text": w_text})
        previous_end = end
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "duration": candidate.duration,
        "words_per_group": int(config["subtitles"]["words_per_group"]),
        "words": selected,
    }
    (output_dir / "subtitles.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    layout_mode = (
        candidate.camera_layout_recommendation.get("layout", "split")
        if candidate.camera_layout_recommendation
        else "split"
    )
    title_start = (
        float(candidate.camera_layout_recommendation.get("webcam_cut_time", 0.0))
        if candidate.camera_layout_recommendation
        else 0.0
    )
    (output_dir / "subtitles.ass").write_text(
        _to_ass(
            selected,
            config,
            title=title,
            duration=candidate.duration,
            layout_mode=layout_mode,
            title_start_time=title_start,
        ),
        encoding="utf-8",
    )
    return selected


def build_subtitles_ass(
    words: list[Word | dict[str, Any]],
    candidate: Candidate,
    output_dir: Path,
    config: dict[str, Any],
    title: str = "",
) -> str:
    """Build subtitle files and return ASS content."""
    selected = build_subtitles(words, candidate, output_dir, config, title=title)
    layout_mode = (
        candidate.camera_layout_recommendation.get("layout", "split")
        if candidate.camera_layout_recommendation
        else "split"
    )
    title_start = (
        float(candidate.camera_layout_recommendation.get("webcam_cut_time", 0.0))
        if candidate.camera_layout_recommendation
        else 0.0
    )
    return _to_ass(
        selected,
        config,
        title=title,
        duration=candidate.duration,
        layout_mode=layout_mode,
        title_start_time=title_start,
    )


def _to_ass(
    words: list[dict[str, Any]],
    config: dict[str, Any],
    title: str = "",
    duration: float = 0.0,
    layout_mode: str = "split",
    title_start_time: float = 0.0,
) -> str:
    settings = config.get("subtitles", {})
    font_family = settings.get("font_family", "Montserrat")
    font_size = settings.get("font_size", 92)
    outline_width = settings.get("outline_width", 14)

    style_format = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding"
    )
    title_style = (
        "Style: Title,Montserrat,48,&H0015E8FF,&H00000000,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,10,4,8,30,30,710,1"
    )
    tiktok_style = (
        f"Style: TikTok,{font_family},{font_size},&H0015E8FF,&H00000000,&H00000000,"
        f"&H80000000,-1,0,0,0,100,100,0,0,1,{outline_width},4,2,70,70,240,1"
    )
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 2\n"
        "\n"
        "[V4+ Styles]\n"
        f"{style_format}\n"
        f"{title_style}\n"
        f"{tiktok_style}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events: list[str] = []

    # Title / Hook is only shown on split layout (never on fullscreen webcam)
    clean_title = str(title).strip()
    if clean_title and layout_mode != "webcam_full":
        masked_title = mask_profanity(clean_title)
        t_start = max(0.0, float(title_start_time))
        end_time = duration if duration > 0 else (words[-1]["end"] if words else 60.0)
        if end_time > t_start:
            events.append(
                f"Dialogue: 0,{_ass_time(t_start)},{_ass_time(end_time)},"
                f"Title,,0,0,0,,{masked_title}"
            )

    group_size = int(settings.get("words_per_group", 1))
    for index in range(0, len(words), group_size):
        group = words[index : index + group_size]
        if not group:
            continue
        raw_text = " ".join(mask_profanity(str(item["text"])) for item in group).replace("\n", " ")
        # Viral kinetic pop-in animation on word start: scale 115% down to 100% over 80ms
        animated_text = rf"{{\fscx115\fscy115\t(0,80,\fscx100\fscy100)}}{raw_text}"
        events.append(
            f"Dialogue: 1,{_ass_time(group[0]['start'])},{_ass_time(group[-1]['end'])},"
            f"TikTok,,0,0,0,,{animated_text}"
        )
    return header + "\n".join(events) + "\n"


def _ass_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, rem = divmod(centiseconds, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"
