from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gemini import GeminiClient
from .process import require_binary, run

SYSTEM_PROMPT = """Ты анализируешь кадры стрима для автоматического монтажа.
Найди только посторонние рекламные и интерфейсные оверлеи, которые нужно скрыть.
Не отмечай игровой HUD, прицел, счётчики игроков, предметы игры, субтитры игры,
сам gameplay, лицо стримера или рамку вебкамеры. Верни только JSON."""


def detect_overlay_masks(
    source_clip: Path,
    *,
    duration: float,
    output_dir: Path,
    config: dict[str, Any],
    client: GeminiClient,
) -> list[dict[str, Any]]:
    settings = config["overlay_cleanup"]
    if not settings.get("enabled", True):
        return []
    result_path = output_dir / "overlay-analysis.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        # Older runs rejected Gemini's valid `widget` label. Re-run those
        # analyses once so a visible donation/promo widget is not left exposed.
        if '"kind": "widget"' not in str(payload.get("raw", "")):
            return list(payload.get("boxes", []))

    frame_dir = output_dir / "overlay-frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    for index, position in enumerate(settings["sample_positions"]):
        timestamp = max(0.0, min(duration - 0.05, duration * float(position)))
        target = frame_dir / f"frame-{index:02d}-{timestamp:.2f}.jpg"
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
                "-vf",
                "scale=1280:-2",
                "-q:v",
                "3",
                "-y",
                target,
            ],
            timeout=120,
        )
        images.append(target)

    prompt = """Перед тобой несколько кадров одного клипа в исходном формате 16:9.
Найди рекламные баннеры, спонсорские логотипы, донатные плашки и внешний чат,
которые наложены поверх gameplay и не являются частью игры.

Верни:
{"boxes":[
  {"x":0.0,"y":0.0,"width":0.12,"height":0.08,
   "kind":"ad_banner","confidence":0.98}
]}

Координаты нормализованы относительно всего исходного кадра: 0..1.
Объедини одинаковый постоянный оверлей на разных кадрах в один box.
kind должен быть одним из: ad_banner, sponsor_logo, donation_banner,
external_chat, promo, widget.
Не включай вебкамеру и игровые HUD-элементы. Если оверлеев нет, верни {"boxes":[]}."""
    data, raw = client.chat_json(
        model=config["models"]["overlay_detection"],
        system=SYSTEM_PROMPT,
        prompt=prompt,
        image_paths=images,
        timeout=180,
        max_tokens=4096,
        temperature=0,
    )
    boxes = _validate_boxes(data, config)
    result_path.write_text(
        json.dumps({"boxes": boxes, "raw": raw}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return boxes


def _validate_boxes(payload: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    settings = config["overlay_cleanup"]
    gameplay = config["layout"]["gameplay"]
    gameplay_right = float(gameplay["x"]) + float(gameplay["width"])
    minimum = float(settings["min_confidence"])
    padding = float(settings["padding"])
    allowed = {
        "ad_banner",
        "sponsor_logo",
        "donation_banner",
        "external_chat",
        "promo",
        "widget",
    }
    accepted: list[dict[str, Any]] = []
    items = payload.get("boxes", []) if isinstance(payload, dict) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            confidence = float(item.get("confidence", 0))
            kind = str(item.get("kind", "")).strip().casefold()
            x = float(item["x"])
            y = float(item["y"])
            width = float(item["width"])
            height = float(item["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if confidence < minimum or kind not in allowed or width <= 0 or height <= 0:
            continue
        x = max(float(gameplay["x"]), x - padding)
        y = max(0.0, y - padding)
        right = min(gameplay_right, x + width + padding * 2)
        bottom = min(1.0, y + height + padding * 2)
        if right <= x or bottom <= y:
            continue
        accepted.append(
            {
                "x": round(x, 5),
                "y": round(y, 5),
                "width": round(right - x, 5),
                "height": round(bottom - y, 5),
                "kind": kind,
                "confidence": round(confidence, 3),
            }
        )
        if len(accepted) >= int(settings["max_boxes"]):
            break
    return accepted
