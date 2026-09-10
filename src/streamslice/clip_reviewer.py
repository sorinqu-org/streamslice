"""Visual AI Director & Quality Control Reviewer before final render."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from .gemini import GeminiClient, GeminiError
from .media import probe
from .metadata import format_youtube_shorts_title
from .process import ProcessError, require_binary, run
from .profanity import mask_profanity

LOGGER = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = (
    "Ты главный визуальный арт-директор и контролёр качества (QC Director) вирусных вертикальных "
    "видео (TikTok, YouTube Shorts, Reels).\n"
    "Твоя задача — провести предрендеринговый визуальный аудит тестовых кадров клипа и точно "
    "скорректировать параметры кадрирования, композиции и заголовка-хука.\n"
    "\n"
    "КРИТЕРИИ И ЗАДАЧИ АУДИТА:\n"
    "1. ВЕБКАМЕРА И ЛИЦО:\n"
    "   - Определи, где находится вебкамера стримера, и задай точный bounding box `webcam_crop` "
    "(x, y, width, height в диапазоне 0..1).\n"
    "   - Убедись, что лицо и голова стримера не обрезаны границами кадра.\n"
    "2. ФОКУС КОНТЕНТА:\n"
    "   - Определи главный смысловой центр действия на экране (игрок, чат, видео на экране, "
    "ключевой объект) в координатах `focus_x`, `focus_y` (0..1).\n"
    "3. ПЕРЕКЛЮЧЕНИЕ С ПОЛНОГО ЭКРАНА (webcam_cut_time):\n"
    "   - Проверь, не начинал ли стример говорить на полный экран перед тем как включить "
    "видео/игру.\n"
    "   - Если клип начинается с полноэкранной вебки, а затем появляется игра/видео, укажи точную "
    "секунду переключения `webcam_cut_time` (в секундах, иначе 0.0).\n"
    "4. ТИП ЛЕЙАУТА:\n"
    "   - `layout`: \"split\" (вебка сверху + контент снизу) или \"webcam_full\" (если весь клип "
    "стример на весь экран).\n"
    "5. ХУК-ЗАГОЛОВОК:\n"
    "   - Оцени текущий заголовок клипа. Если он не отражает то, что реально происходит на кадрах, "
    "или может быть усилен — уточни `refined_title` (3-6 слов, КАПСОМ, без мата, мощный "
    "кликбейтный хук).\n"
    "\n"
    "Верни ТОЛЬКО валидный JSON следующего формата:\n"
    "{\n"
    "  \"webcam_crop\": {\"x\": 0.738, \"y\": 0.739, \"width\": 0.262, \"height\": 0.261},\n"
    "  \"focus_x\": 0.50,\n"
    "  \"focus_y\": 0.50,\n"
    "  \"webcam_cut_time\": 0.0,\n"
    "  \"layout\": \"split\",\n"
    "  \"refined_title\": \"УТОЧНЕННЫЙ ЗАГОЛОВОК ХУК\",\n"
    "  \"director_notes\": \"Краткий комментарий режиссёра о кадрировании\"\n"
    "}\n"
    ""
)


def extract_preview_frames(
    source_clip: Path,
    output_dir: Path,
    duration: float,
    sample_positions: list[float] | None = None,
) -> list[Path]:
    """Extract key test preview frames (10%, 30%, 50%, 80%) from source clip via fast seeking."""
    positions = sample_positions or [0.10, 0.30, 0.50, 0.80]
    frame_dir = output_dir / "preview-frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []

    for idx, pos in enumerate(positions):
        timestamp = max(0.0, min(duration - 0.05, duration * float(pos)))
        target = frame_dir / f"frame-{idx:02d}-{timestamp:.2f}.jpg"
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
                    str(source_clip),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=1280:-2:force_original_aspect_ratio=decrease",
                    "-q:v",
                    "3",
                    "-y",
                    str(target),
                ],
                timeout=60,
            )
            if target.is_file() and target.stat().st_size > 0:
                images.append(target)
        except (ProcessError, OSError) as exc:
            LOGGER.warning("Could not extract review frame at %.2fs: %s", timestamp, exc)

    return images


def _validate_webcam_crop(crop_data: Any) -> dict[str, float] | None:
    if not isinstance(crop_data, dict):
        return None
    try:
        x = float(crop_data.get("x", 0.0))
        y = float(crop_data.get("y", 0.0))
        w = float(crop_data.get("width", 0.0))
        h = float(crop_data.get("height", 0.0))
    except (TypeError, ValueError):
        return None

    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and math.isfinite(h)):
        return None

    if w < 0.05 or h < 0.05 or x < 0 or y < 0 or x + w > 1.05 or y + h > 1.05:
        return None

    return {
        "x": max(0.0, min(1.0 - min(w, 1.0), round(x, 4))),
        "y": max(0.0, min(1.0 - min(h, 1.0), round(y, 4))),
        "width": max(0.05, min(1.0, round(w, 4))),
        "height": max(0.05, min(1.0, round(h, 4))),
    }


def review_and_refine_clip(
    clip_dir: str | Path,
    config: dict[str, Any],
    client: GeminiClient,
) -> dict[str, Any]:
    """Visual AI Director & Quality Control Reviewer before final render.

    1. Extracts key test preview frames (10%, 30%, 50%, 80%).
    2. Sends frames to Gemini Vision with Director prompt checking:
       - Webcam framing (face/head not cut off)
       - Gameplay/content focus (focus_x, focus_y)
       - Webcam cut transition (webcam_cut_time)
       - Title hook accuracy & punchiness
    3. Refines remotion-props.json and metadata.json with updated parameters.
    """
    root = Path(clip_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Clip directory not found: {root}")

    # Check for cache
    review_cache = root / "visual-review.json"
    if review_cache.is_file():
        try:
            cached = json.loads(review_cache.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("reviewed"):
                LOGGER.info("[%s] Clip already passed visual review, using cache", root.name)
                return cached.get("refinements", {})
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            LOGGER.debug("Could not read visual-review.json cache for %s", root.name)

    # Read existing properties
    props_path = root / "remotion-props.json"
    props: dict[str, Any] = {}
    if props_path.is_file():
        try:
            props = json.loads(props_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            props = {}

    meta_path = root / "metadata.json"
    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            metadata = {}

    sel_path = root / "selection.json"
    selection: dict[str, Any] = {}
    if sel_path.is_file():
        try:
            selection = json.loads(sel_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            selection = {}

    source_clip = root / "source.mp4"
    duration = 30.0
    if source_clip.is_file():
        try:
            info = probe(source_clip)
            duration = float(info.get("duration", 30.0))
        except (ProcessError, OSError, KeyError, TypeError, ValueError):
            LOGGER.debug("Could not probe source clip duration for %s", root.name)
    if duration <= 0 or duration == 30.0:
        duration = float(props.get("durationInSeconds") or selection.get("duration") or 30.0)

    # Extract preview frames
    sample_positions = config.get("clip_reviewer", {}).get(
        "sample_positions", [0.10, 0.30, 0.50, 0.80]
    )
    images: list[Path] = []
    if source_clip.is_file():
        images = extract_preview_frames(source_clip, root, duration, sample_positions)

    current_title = str(
        props.get("title")
        or metadata.get("title")
        or selection.get("highlight_reason")
        or ""
    )
    current_rec = props.get("layoutRecommendation", {})
    context_summary = str(selection.get("context_summary") or "")
    highlight_reason = str(selection.get("highlight_reason") or "")

    prompt = (
        f"Перед тобой {len(images)} тестовых кадров клипа длительностью {duration:.1f} сек "
        "(взят на отметках 10%, 30%, 50%, 80%).\n"
        f"Текущий заголовок-хук: {current_title}\n"
        f"Причина выбора момента: {highlight_reason}\n"
        f"Контекст: {context_summary}\n"
        f"Текущие параметры компоновки: {json.dumps(current_rec, ensure_ascii=False)}\n"
        "\n"
        "Проведи визуальный аудит качества и композиции:\n"
        "1. Проверь вебкамеру: задай точный `webcam_crop` (x, y, width, height в диапазоне "
        "0..1), чтобы лицо стримера было в кадре и не срезалось.\n"
        "2. Проверь контент: задай `focus_x`, `focus_y` (0..1) центра главного "
        "действия.\n"
        "3. Проверь переход: если стример в начале сидит на весь экран и только потом "
        "включает контент, укажи `webcam_cut_time` (секунда переключения).\n"
        "4. Проверь формат: `layout` (\"split\" или \"webcam_full\").\n"
        "5. Проверь хук: если заголовок можно улучшить под кадры, укажи `refined_title` "
        "(3-6 слов, КАПСОМ, мощный хук без мата)."
    )

    model = (
        config.get("models", {}).get("visual_review")
        or config.get("models", {}).get("overlay_detection")
        or config.get("models", {}).get("curator")
        or "gemini-3.7-flash-high"
    )

    data: dict[str, Any] = {}
    raw = ""
    try:
        data, raw = client.chat_json(
            model=model,
            system=REVIEWER_SYSTEM_PROMPT,
            prompt=prompt,
            image_paths=images or None,
            timeout=120,
            max_tokens=2048,
            temperature=0.2,
        )
    except (GeminiError, OSError, TimeoutError) as exc:
        LOGGER.warning("[%s] Visual Director review call failed: %s", root.name, exc)
        data = {}

    # Extract refined parameters
    layout_val = str(data.get("layout", "")).strip().lower()
    refined_layout = (
        layout_val
        if layout_val in ("split", "webcam_full", "gameplay_full")
        else current_rec.get("layout", "split")
    )

    webcam_crop = _validate_webcam_crop(data.get("webcam_crop") or data.get("webcam_box"))
    if not webcam_crop and isinstance(current_rec.get("webcam_crop"), dict):
        webcam_crop = _validate_webcam_crop(current_rec["webcam_crop"])

    focus_x = None
    try:
        if "focus_x" in data:
            fx = float(data["focus_x"])
            if math.isfinite(fx):
                focus_x = max(0.0, min(1.0, fx))
    except (TypeError, ValueError):
        pass
    if focus_x is None:
        try:
            focus_x = float(current_rec.get("focus_x", 0.50))
        except (TypeError, ValueError):
            focus_x = 0.50

    focus_y = None
    try:
        if "focus_y" in data:
            fy = float(data["focus_y"])
            if math.isfinite(fy):
                focus_y = max(0.0, min(1.0, fy))
    except (TypeError, ValueError):
        pass
    if focus_y is None:
        try:
            focus_y = float(current_rec.get("focus_y", 0.50))
        except (TypeError, ValueError):
            focus_y = 0.50

    webcam_cut_time = 0.0
    try:
        if "webcam_cut_time" in data:
            w_cut = float(data["webcam_cut_time"])
            if math.isfinite(w_cut) and 0.5 <= w_cut <= duration - 0.5:
                webcam_cut_time = round(w_cut, 3)
    except (TypeError, ValueError):
        pass
    if webcam_cut_time == 0.0 and "webcam_cut_time" in current_rec:
        try:
            w_cut = float(current_rec["webcam_cut_time"])
            if 0.5 <= w_cut <= duration - 0.5:
                webcam_cut_time = round(w_cut, 3)
        except (TypeError, ValueError):
            pass

    refined_title = str(data.get("refined_title") or data.get("title") or "").strip()
    refined_title = mask_profanity(refined_title) if refined_title else current_title

    # Update remotion-props.json
    layout_rec = dict(current_rec)
    layout_rec["time_base"] = "clip"
    layout_rec["layout"] = refined_layout
    if webcam_crop:
        layout_rec["webcam_crop"] = webcam_crop
        layout_rec["webcam_box"] = dict(webcam_crop)
        layout_rec["webcam_box_validated"] = True
        layout_rec["webcam_box_confidence"] = 0.95
    layout_rec["focus_x"] = round(focus_x, 4)
    layout_rec["focus_y"] = round(focus_y, 4)
    layout_rec["webcam_cut_time"] = round(webcam_cut_time, 3)

    props["layoutRecommendation"] = layout_rec
    if refined_title:
        props["title"] = refined_title

    props_path.write_text(json.dumps(props, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update metadata.json
    if meta_path.is_file() and metadata:
        if refined_title:
            metadata["title"] = refined_title
            creator = metadata.get("creator", {})
            creator_login = str(
                creator.get("twitch_login") or creator.get("display_name") or "twitch"
            )
            creator_display = str(creator.get("display_name") or creator_login)
            hashtags = metadata.get("hashtags", [])
            metadata["youtube_title"] = format_youtube_shorts_title(
                creator_login, creator_display, hashtags
            )

            desc_file = root / "description.txt"
            if desc_file.is_file():
                desc_body = metadata.get("description", "")
                desc_file.write_text(
                    f"{refined_title}\n\n{desc_body}\n\n{' '.join(hashtags)}\n",
                    encoding="utf-8",
                )
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    refinements = {
        "webcam_crop": webcam_crop,
        "focus_x": focus_x,
        "focus_y": focus_y,
        "webcam_cut_time": webcam_cut_time,
        "layout": refined_layout,
        "title": refined_title,
    }

    # Save cache
    review_cache.write_text(
        json.dumps(
            {
                "reviewed": True,
                "reviewed_at_unix": time.time(),
                "refinements": refinements,
                "raw": raw,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info(
        "[%s] Visual Director review complete: layout=%s, webcam_cut=%.1fs, title=%s",
        root.name,
        refined_layout,
        webcam_cut_time,
        refined_title,
    )
    return refinements
