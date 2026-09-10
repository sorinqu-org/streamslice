from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gemini import GeminiClient
from .models import Candidate

SYSTEM_PROMPT = (
    "Ты главный продюсер вирусных нарезок в TikTok/Reels/Shorts.\n"
    "Твоя цель — выбрать только топ-контент с взрывным удержанием аудитории.\n"
    "Оценивай силу хука, динамику, мемный потенциал и законченность панчлайна.\n"
    "Верни только JSON."
)

CURATION_VERSION = 7


def _editorial_score(candidate: Candidate) -> float:
    return (
        candidate.quality_score * 0.50
        + candidate.context_score * 0.30
        + candidate.emotion_score * 0.20
    )


def curate(
    candidates: list[Candidate],
    *,
    duration: float,
    work_dir: Path,
    config: dict[str, Any],
    client: GeminiClient,
    creator_identity: dict[str, Any] | None = None,
) -> list[Candidate]:
    output = work_dir / "selection.json"
    if output.is_file():
        data = json.loads(output.read_text(encoding="utf-8"))
        if data.get("curation_version") == CURATION_VERSION:
            return [Candidate.from_dict(item) for item in data["highlights"]]
    settings = config["selection"]
    target_count = min(int(settings["final_count"]), len(candidates))
    if target_count == 0:
        output.write_text(
            json.dumps(
                {"curation_version": CURATION_VERSION, "highlights": []},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return []
    ranked = sorted(candidates, key=_editorial_score, reverse=True)
    pool = ranked[: int(settings["max_candidates_for_curator"])]
    min_desired = 1 if target_count == 1 else min(target_count, 2)
    creator_info = ""
    if creator_identity and creator_identity.get("character_lore"):
        creator_info = (
            "\nКонтекст о стримере и что обожает его комьюнити:\n"
            f"{creator_identity['character_lore']}\n"
        )

    display_name = creator_identity.get("display_name", "t2x2") if creator_identity else "t2x2"
    prompt = (
        "Кандидаты:\n"
        f"{json.dumps([item.to_dict() for item in pool], ensure_ascii=False)}\n"
        f"{creator_info}\n"
        f"Твоя задача — отобрать от {min_desired} до {target_count} САМЫХ ЛУЧШИХ И "
        "ВИРУСНЫХ моментов из предложенного списка.\n"
        f"Оставь start_time и end_time в пределах 0..{duration:.3f}.\n"
        "Длительность каждого: 20–90 секунд.\n"
        "\n"
        "ВАЖНЫЕ ПРАВИЛА ВЫБОРА:\n"
        "- Выбирай ТОЛЬКО самые мощные, смешные, эмоциональные и вирусные моменты "
        f"(максимум {target_count}).\n"
        f"- Учитывай специфику стримера ({display_name}) и то, что реально "
        "вирусится у его аудитории: сгорания, споры с чатом, донаты, нелепые "
        "обещания, абсурдные теории, фейлы.\n"
        f"- Обязательно выбери от {min_desired} до {target_count} моментов.\n"
        "- Предпочитай клипы с высоким потенциалом обсуждения и шер-рейта в "
        "комментариях.\n"
        "- Для camera_layout_recommendation: используй split (по умолчанию) или "
        "webcam_full (если стример эмоционально рассказывает/реагирует).\n"
        "\n"
        'Верни объект {"highlights":[...]}. В каждом элементе обязательны:\n'
        "start_time, end_time, highlight_reason, emotion_score, "
        "camera_layout_recommendation."
    )
    data, raw = client.chat_json(
        model=config["models"]["curator"],
        system=SYSTEM_PROMPT,
        prompt=prompt,
        timeout=180,
        max_tokens=8192,
        temperature=0.15,
    )
    (work_dir / "curator-raw.txt").write_text(raw, encoding="utf-8")
    items = data.get("highlights", []) if isinstance(data, dict) else []
    selected: list[Candidate] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            candidate = Candidate.from_dict(item)
        except (KeyError, TypeError, ValueError):
            continue
        match = min(
            pool,
            key=lambda value: abs(value.start_time - candidate.start_time)
            + abs(value.end_time - candidate.end_time),
        )
        # The curator chooses an existing reviewed candidate. Its response must
        # not erase the scores/evidence produced by the stricter visual critic.
        candidate = match
        if _valid(candidate, selected, duration, settings):
            selected.append(candidate)
        if len(selected) == target_count:
            break
    for candidate in ranked:
        if len(selected) == target_count:
            break
        if _valid(candidate, selected, duration, settings):
            selected.append(candidate)
    selected.sort(key=lambda item: item.start_time)
    output.write_text(
        json.dumps(
            {
                "curation_version": CURATION_VERSION,
                "highlights": [item.to_dict() for item in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected


def _valid(
    candidate: Candidate,
    selected: list[Candidate],
    duration: float,
    settings: dict[str, Any] | None = None,
) -> bool:
    if candidate.start_time < 0 or candidate.end_time > duration:
        return False
    if candidate.duration < 20 or candidate.duration > 90:
        return False
    if settings and settings.get("strict_context", True):
        if not candidate.self_contained:
            return False
        if candidate.context_score < float(settings.get("min_context_score", 7.0)):
            return False
        if not candidate.context_summary:
            return False
        if candidate.quality_score < float(settings.get("min_quality_score", 7.5)):
            return False
        if not candidate.quality_summary:
            return False
    for current in selected:
        overlap = min(candidate.end_time, current.end_time) - max(
            candidate.start_time, current.start_time
        )
        if overlap > 1:
            return False
    return True
