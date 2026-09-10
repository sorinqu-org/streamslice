from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .gemini import GeminiClient

LOGGER = logging.getLogger(__name__)

SUPER_CURATOR_SYSTEM_PROMPT = (
    "Ты главный шеф-продюсер и эксперт по вирусным вертикальным видео (TikTok, Shorts, Reels).\n"
    "Твоя задача — отобрать ТОП 2-3 самых взрывных, смешных, удерживающих внимание и вирусных "
    "клипов из предложенного пула кандидатов со всех стримов и чанков в текущем окне.\n"
    "\n"
    "КРИТЕРИИ ОТБОРА:\n"
    "1. Вирусность и Хук: клип должен цеплять с первой секунды (неожиданность, крик, спор, угар, "
    "панчлайн).\n"
    "2. Разнообразие (Variety): отдавай предпочтение разнообразному контенту (разные стримеры, "
    "разные темы/ситуации: спор, мем, игровой фейл, музыкальный момент, сгорание), если качество "
    "сопоставимо.\n"
    "3. Цельность и контекст: зритель из рекомендаций должен сразу понять суть происходящего без "
    "долгой предыстории.\n"
    "4. Учет лора стримера: максимальный приоритет моментам, где стример проявляет свои ключевые "
    "триггерные черты (эмоции, токсичность, база, самоирония).\n"
    "\n"
    "Верни ТОЛЬКО валидный JSON с выбранными клипами."
)


def _gather_clip_info(job_dir: Path, clip_dir: Path) -> dict[str, Any] | None:
    """Gather all relevant metadata for a single clip in a job directory."""
    if not clip_dir.is_dir():
        return None

    clip_name = clip_dir.name  # e.g. "clip-01"

    # Read selection.json
    selection: dict[str, Any] = {}
    sel_path = clip_dir / "selection.json"
    if sel_path.is_file():
        try:
            selection = json.loads(sel_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            LOGGER.debug("Could not read selection.json for %s", clip_dir.name)

    # Read metadata.json
    metadata: dict[str, Any] = {}
    meta_path = clip_dir / "metadata.json"
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            LOGGER.debug("Could not read metadata.json for %s", clip_dir.name)

    # Extract creator info from metadata or parent manifest
    creator: dict[str, Any] = metadata.get("creator", {})
    if not creator:
        manifest_path = job_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                creator = manifest_data.get("creator", {})
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                LOGGER.debug("Could not read manifest.json for %s", job_dir.name)

    streamer_name = (
        creator.get("display_name")
        or creator.get("twitch_login")
        or creator.get("source_key")
        or job_dir.name.split("_")[0]
    )
    character_lore = creator.get("character_lore", "")

    start_time = float(selection.get("start_time", 0.0))
    end_time = float(selection.get("end_time", 0.0))
    duration = end_time - start_time if end_time > start_time else 0.0

    emotion_score = float(selection.get("emotion_score", 0.0))
    quality_score = float(selection.get("quality_score", 0.0))
    context_summary = str(selection.get("context_summary", ""))
    highlight_reason = str(selection.get("highlight_reason", ""))
    title = str(metadata.get("title", ""))

    return {
        "job_dir": job_dir,
        "clip_name": clip_name,
        "metadata": metadata,
        "selection": selection,
        # Flattened fields for comparison
        "streamer_name": streamer_name,
        "character_lore": character_lore,
        "title": title,
        "duration": duration,
        "emotion_score": emotion_score,
        "quality_score": quality_score,
        "context_summary": context_summary,
        "highlight_reason": highlight_reason,
    }


def select_top_clips(
    batch_jobs: list[Path],
    config: dict[str, Any],
    client: GeminiClient,
    max_clips: int = 3,
    min_clips: int = 2,
) -> list[dict[str, Any]]:
    """Cross-streamer & cross-chunk master curator for selecting top clips across a batch.

    Scans all `clip-*` in all job directories of the batch.
    Gathers metadata: clip duration, emotion_score, quality_score, context_summary,
    highlight_reason, title, streamer name, character_lore.
    If total clips <= max_clips, returns all of them.
    Otherwise, queries Gemini with a cross-streamer comparison prompt to select the
    top 2-3 most viral, funny, and engaging clips with variety.

    Returns:
        List of chosen clip references:
        [{"job_dir": Path, "clip_name": str, "metadata": dict, "selection": dict}, ...]
    """
    candidates: list[dict[str, Any]] = []

    for job_path in batch_jobs:
        job_dir = Path(job_path).expanduser().resolve()
        if not job_dir.is_dir():
            continue

        for clip_dir in sorted(job_dir.glob("clip-*")):
            clip_info = _gather_clip_info(job_dir, clip_dir)
            if clip_info:
                candidates.append(clip_info)

    if not candidates:
        LOGGER.warning("No clips found across batch jobs: %s", batch_jobs)
        return []

    # If total candidate clips <= max_clips, return all of them
    if len(candidates) <= max_clips:
        LOGGER.info(
            "Batch candidate count (%d) <= max_clips (%d); returning all candidates",
            len(candidates),
            max_clips,
        )
        return [
            {
                "job_dir": item["job_dir"],
                "clip_name": item["clip_name"],
                "metadata": item["metadata"],
                "selection": item["selection"],
            }
            for item in candidates
        ]

    # Otherwise, query Gemini to compare and select the top 2-3 clips
    candidate_prompts = []
    for idx, c in enumerate(candidates):
        candidate_prompts.append(
            {
                "candidate_id": idx,
                "job_id": c["job_dir"].name,
                "clip_name": c["clip_name"],
                "streamer_name": c["streamer_name"],
                "character_lore": c["character_lore"][:200] if c["character_lore"] else "",
                "title": c["title"],
                "duration_seconds": round(c["duration"], 1),
                "emotion_score": c["emotion_score"],
                "quality_score": c["quality_score"],
                "highlight_reason": c["highlight_reason"],
                "context_summary": c["context_summary"],
            }
        )

    target_count = min(max_clips, len(candidates))
    min_count = min(min_clips, target_count)

    prompt = (
        f"Ниже представлен общий пул кандидатов в клипы ({len(candidates)} шт.) со стримов в "
        "текущем временном окне:\n"
        f"{json.dumps(candidate_prompts, ensure_ascii=False, indent=2)}\n"
        "\n"
        f"Твоя задача — отобрать от {min_count} до {target_count} САМЫХ МОЩНЫХ, ВИРУСНЫХ И "
        "СМЕШНЫХ клипов из этого пула.\n"
        "\n"
        "ТРЕБОВАНИЯ:\n"
        f"1. Выбери от {min_count} до {target_count} лучших кандидатов (укажи их "
        "`candidate_id`).\n"
        "2. Сохраняй разнообразие (variety): если есть сильные моменты от разных "
        "стримеров, возьми лучшее от каждого, а не 3 клипа одного стримера, если только "
        "один стример не выдал абсолютно феноменальный контент.\n"
        "3. Оценивай взрывной потенциал для Shorts/TikTok/Reels, юмор, силу хука и "
        "законченность панчлайна.\n"
        "\n"
        "Верни JSON следующего формата:\n"
        "{\n"
        '  "selected_candidates": [\n'
        "    {\n"
        '      "candidate_id": 0,\n'
        '      "curator_comment": "Краткое обоснование почему этот клип победил в общем '
        'зачете"\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    model = config.get("models", {}).get("curator", "gemini-3.7-flash-high")
    data, _raw = client.chat_json(
        model=model,
        system=SUPER_CURATOR_SYSTEM_PROMPT,
        prompt=prompt,
        timeout=180,
        max_tokens=4096,
        temperature=0.2,
    )

    selected_ids: list[int] = []
    if isinstance(data, dict):
        raw_selections = data.get("selected_candidates", [])
        if isinstance(raw_selections, list):
            for item in raw_selections:
                if isinstance(item, dict) and "candidate_id" in item:
                    try:
                        cid = int(item["candidate_id"])
                        if 0 <= cid < len(candidates) and cid not in selected_ids:
                            selected_ids.append(cid)
                    except (ValueError, TypeError):
                        pass
                elif (
                    isinstance(item, int)
                    and 0 <= item < len(candidates)
                    and item not in selected_ids
                ):
                    selected_ids.append(item)

    # Fallback ranking if model didn't select enough valid clips
    if len(selected_ids) < min_count:
        LOGGER.warning(
            "SuperCurator selected %d clips (expected %d-%d), supplementing with heuristic ranking",
            len(selected_ids),
            min_count,
            target_count,
        )
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda i: (
                candidates[i]["quality_score"] * 0.5
                + candidates[i]["emotion_score"] * 0.5
            ),
            reverse=True,
        )
        for idx in ranked_indices:
            if idx not in selected_ids:
                selected_ids.append(idx)
            if len(selected_ids) >= target_count:
                break

    # Limit to max_clips
    final_indices = selected_ids[:max_clips]

    result: list[dict[str, Any]] = []
    for idx in final_indices:
        item = candidates[idx]
        result.append(
            {
                "job_dir": item["job_dir"],
                "clip_name": item["clip_name"],
                "metadata": item["metadata"],
                "selection": item["selection"],
            }
        )

    LOGGER.info(
        "SuperCurator selected %d clips out of %d candidates across %d jobs",
        len(result),
        len(candidates),
        len(batch_jobs),
    )
    return result
