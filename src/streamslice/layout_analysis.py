from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from .gemini import GeminiClient, GeminiError
from .models import Candidate
from .process import ProcessError, require_binary, run
from .render import normalized_recommendation

REACTION_WORDS = re.compile(r"^(ой|ох|а+|бля|блядь|сука|ебать|пиздец|ёб)", re.IGNORECASE)
LAYOUT_ANALYSIS_VERSION = 10
LOGGER = logging.getLogger(__name__)


def analyze_dynamic_layout(
    source_clip: Path,
    *,
    duration: float,
    words: list[dict[str, Any]],
    candidate: Candidate,
    output_dir: Path,
    config: dict[str, Any],
    client: GeminiClient,
) -> dict[str, Any]:
    recommendation = normalized_recommendation(candidate)
    requested_layout = recommendation.get("layout", "split")
    webcam_min_confidence = float(
        config["layout"].get("webcam_detection_min_confidence", 0.75)
    )

    result_path = output_dir / "layout-analysis.json"
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("analysis_version") == LAYOUT_ANALYSIS_VERSION:
            return _merge_recommendation(
                recommendation,
                payload.get("analysis", {}),
                duration,
                frame_times=payload.get("event_frame_times"),
                allow_gameplay_event=requested_layout != "webcam_full",
                webcam_min_confidence=webcam_min_confidence,
            )

    focus_x = float(recommendation.get("focus_x", 0.5))
    anchor = _reaction_anchor(words, duration, focus_x if math.isfinite(focus_x) else None)
    broad_times = (
        0.0,
        max(0.0, min(duration - 0.05, duration * 0.25)),
        duration * 0.5,
        max(0.0, min(duration - 0.05, duration * 0.75)),
        max(0.0, duration - 0.05)
    )
    event_frame_times = sorted(
        {
            max(0.0, min(duration - 0.05, anchor + offset))
            for offset in (-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5)
        }
    )
    frame_times = sorted(
        set(event_frame_times)
        | {
            max(0.0, min(duration - 0.05, timestamp))
            for timestamp in broad_times
        }
    )
    frame_dir = output_dir / "layout-frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    for index, timestamp in enumerate(frame_times):
        target = frame_dir / f"frame-{index:02d}-{timestamp:.2f}.jpg"
        try:
            run(
                [
                    require_binary("ffmpeg"),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    source_clip,
                    "-frames:v",
                    "1",
                    "-strict",
                    "-1",
                    "-vf",
                    "scale=1280:-2:force_original_aspect_ratio=decrease",
                    "-q:v",
                    "3",
                    "-y",
                    target,
                ],
                timeout=120,
            )
            if target.is_file() and target.stat().st_size > 0:
                images.append(target)
        except (ProcessError, OSError) as exc:
            LOGGER.warning("Could not extract frame at %.2fs: %s", timestamp, exc)

    event_frame_text = ", ".join(f"{item:.2f}" for item in event_frame_times)
    frame_times_text = ", ".join(f"{item:.2f}" for item in frame_times)
    prompt = (
        f"Кадры идут по времени {frame_times_text} секунд.\n"
        f"Плотная последовательность вокруг ключевого действия: {event_frame_text}.\n"
        "Остальные кадры показывают общий контекст экрана стрима.\n"
        "\n"
        "Твоя задача — профессионально скомпоновать кадр для вертикального "
        "Shorts/TikTok видео:\n"
        "\n"
        "1. ВЕБКАМЕРА:\n"
        "- Если стример на протяжении клипа сидит в углу (маленькая вебка) — укажи "
        "её точные координаты: x, y, width, height (0..1).\n"
        "- Если стример сначала сидит НА ПОЛНЫЙ ЭКРАН ПО ЦЕНТРУ, а потом "
        "уменьшается в угол, когда включает видео:\n"
        "  * Укажи webcam_cut_time: точная секунда, когда стример включает видео "
        "/ уменьшает вебку в угол!\n"
        "  * До webcam_cut_time клип будет полноэкранным лицом стримера по "
        "центру, а после — аккуратным сплитом!\n"
        "\n"
        "2. ОПРЕДЕЛЕНИЕ СМЫСЛОВОГО КОНТЕНТА:\n"
        "- Найди точное окно с видео / игрой / перепиской и укажи focus_x, "
        "focus_y (центр окна контента).\n"
        "- Если видео / игра начинается не сразу, а с определенной секунды, "
        "укажи webcam_cut_time.\n"
        "\n"
        "3. ПАРАМЕТРЫ:\n"
        "- layout: \"split\" (если есть контент и вебка) или \"webcam_full\" "
        "(если весь клип это чисто стример).\n"
        "- focus_x, focus_y: координаты смыслового центра контента.\n"
        "- webcam_cut_time: секунда переключения с полноэкранного стримера на "
        "сплит с видео (0.0 если сплит с первой секунды).\n"
        "\n"
        "Верни только JSON:\n"
        '{"webcam_box_validated":true,"webcam_box_confidence":0.95,\n'
        '  "webcam_box":{"x":0.01,"y":0.07,"width":0.23,"height":0.24},\n'
        '  "has_gameplay_event":true,"focus_x":0.50,"focus_y":0.50,\n'
        '  "webcam_cut_time":0.0,\n'
        f'  "event_time":0.0,"event_end":{duration:.2f},"gameplay_zoom":1.0,\n'
        '  "reason":"стример смотрит видео по центру экрана"}'
    )
    try:
        data, raw = client.chat_json(
            model=config["models"]["overlay_detection"],
            system=(
                "Ты режиссёр вертикальных игровых клипов. Отдельно проверяй вебкамеру "
                "и gameplay по кадрам. При сомнении возвращай webcam_box_validated=false "
                "или has_gameplay_event=false; ничего не выдумывай."
            ),
            prompt=prompt,
            image_paths=images,
            timeout=180,
            max_tokens=2048,
            temperature=0,
        )
    except GeminiError as exc:
        # Layout is an enrichment pass. A transient visual-model failure must
        # leave the deterministic config crop/layout intact, not lose a clip.
        LOGGER.warning("Dynamic layout analysis skipped: %s", exc)
        return recommendation
    result_path.write_text(
        json.dumps(
            {
                "analysis_version": LAYOUT_ANALYSIS_VERSION,
                "analysis": data,
                "raw": raw,
                "frame_times": frame_times,
                "event_frame_times": event_frame_times,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return _merge_recommendation(
        recommendation,
        data,
        duration,
        frame_times=event_frame_times,
        allow_gameplay_event=requested_layout != "webcam_full",
        webcam_min_confidence=webcam_min_confidence,
    )


def _reaction_anchor(
    words: list[dict[str, Any]],
    duration: float,
    focus_time: Any = None,
) -> float:
    try:
        parsed_focus = float(focus_time)
        if math.isfinite(parsed_focus):
            return max(0.0, min(duration - 0.1, parsed_focus))
    except (TypeError, ValueError):
        pass

    reaction_times: list[float] = []
    for item in words:
        text = str(item.get("text", "")).strip()
        if REACTION_WORDS.search(text):
            try:
                reaction_times.append(float(item.get("start", duration * 0.5)))
            except (TypeError, ValueError):
                continue
    if reaction_times:
        closest = min(reaction_times, key=lambda value: abs(value - duration * 0.5))
        return max(0.0, min(duration - 0.1, closest))
    return duration * 0.5


def _merge_recommendation(
    recommendation: dict[str, Any],
    analysis: Any,
    duration: float,
    *,
    frame_times: Any = None,
    allow_gameplay_event: bool = True,
    webcam_min_confidence: float = 0.75,
) -> dict[str, Any]:
    merged = dict(recommendation)
    merged["time_base"] = "clip"
    if not isinstance(analysis, dict):
        return _without_gameplay_event(merged)

    webcam = _validated_webcam(analysis, webcam_min_confidence)
    if webcam is not None:
        merged["webcam_box_validated"] = True
        merged["webcam_box_confidence"] = webcam["confidence"]
        merged["webcam_crop"] = dict(webcam["crop"])
        merged["webcam_box"] = dict(webcam["crop"])
    else:
        merged.pop("webcam_box_validated", None)
        merged.pop("webcam_box_confidence", None)
        merged.pop("webcam_crop", None)
        merged.pop("webcam_box", None)

    if not allow_gameplay_event or analysis.get("has_gameplay_event") is not True:
        return _without_gameplay_event(merged)

    parsed: dict[str, float] = {}
    for key in (
        "focus_x",
        "focus_y",
        "webcam_cut_time",
        "event_time",
        "event_end",
        "gameplay_zoom",
        "gameplay_crop_width",
    ):
        try:
            if key in analysis:
                value = float(analysis[key])
                if math.isfinite(value):
                    parsed[key] = value
        except (TypeError, ValueError):
            continue

    if "webcam_cut_time" in parsed:
        w_cut = parsed["webcam_cut_time"]
        if 0.5 <= w_cut <= duration - 0.5:
            merged["webcam_cut_time"] = round(w_cut, 3)

    event_time = parsed.get("event_time")
    event_end = parsed.get("event_end")
    if event_time is None or event_end is None or not (0 <= event_time < event_end <= duration):
        return _without_gameplay_event(merged)

    sample_times: list[float] = []
    if isinstance(frame_times, list):
        for value in frame_times:
            try:
                parsed_time = float(value)
                if math.isfinite(parsed_time):
                    sample_times.append(parsed_time)
            except (TypeError, ValueError):
                continue
    if sample_times and not (
        min(sample_times) - 0.05 <= event_time
        and event_end <= max(sample_times) + 0.05
    ):
        return _without_gameplay_event(merged)

    merged["event_time"] = event_time
    merged["event_end"] = event_end
    merged["focus_time"] = event_time
    merged["gameplay_event_validated"] = True
    if "focus_x" in parsed:
        merged["focus_x"] = max(0.0, min(0.98, parsed["focus_x"]))
    if "focus_y" in parsed:
        merged["focus_y"] = max(0.0, min(1.0, parsed["focus_y"]))
    if isinstance(analysis.get("focal_trajectory"), list):
        merged["focal_trajectory"] = analysis["focal_trajectory"]
    merged["gameplay_zoom"] = max(1.0, min(1.22, parsed.get("gameplay_zoom", 1.0)))
    merged["gameplay_crop_width"] = max(
        0.738, min(1.0, parsed.get("gameplay_crop_width", 0.98))
    )
    return merged


def _without_gameplay_event(recommendation: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(recommendation)
    for key in (
        "event_time",
        "event_end",
        "gameplay_event_validated",
        "gameplay_zoom",
        "gameplay_crop_width",
    ):
        cleaned.pop(key, None)
    return cleaned


def _validated_webcam(analysis: dict[str, Any], minimum_confidence: float) -> dict[str, Any] | None:
    if analysis.get("webcam_box_validated") is not True:
        return None
    try:
        confidence = float(analysis.get("webcam_box_confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or confidence < minimum_confidence:
        return None
    raw = analysis.get("webcam_crop") or analysis.get("webcam_box")
    if not isinstance(raw, dict):
        return None
    try:
        crop = {key: float(raw[key]) for key in ("x", "y", "width", "height")}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in crop.values()):
        return None
    if (
        crop["x"] < 0
        or crop["y"] < 0
        or crop["width"] < 0.05
        or crop["height"] < 0.05
        or crop["x"] + crop["width"] > 1
        or crop["y"] + crop["height"] > 1
        or crop["width"] * crop["height"] > 0.6
    ):
        return None
    return {"crop": crop, "confidence": confidence}
