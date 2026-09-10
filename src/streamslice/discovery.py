from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .gemini import GeminiClient, GeminiError
from .models import AudioPeak, Candidate, Word

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты главный продюсер вирусных игровых нарезок для TikTok, YouTube Shorts и Reels.\n"
    "Твоя цель — находить моменты с максимальным удержанием (Watch Time) и "
    "виральностью (Shareability).\n"
    "Ищи только те эпизоды, которые цепляют зрителя с первых 2-3 секунд (сильный "
    "Hook) и имеют взрывную развязку (Payoff).\n"
    "Верни только JSON."
)

DISCOVERY_VERSION = 8


def discover_candidates(
    words: list[Word],
    peaks: list[AudioPeak],
    *,
    duration: float,
    work_dir: Path,
    config: dict[str, Any],
    client: GeminiClient,
    creator_identity: dict[str, Any] | None = None,
) -> list[Candidate]:
    output = work_dir / "candidates.json"
    if output.is_file():
        data = json.loads(output.read_text(encoding="utf-8"))
        if data.get("discovery_version") == DISCOVERY_VERSION:
            return [Candidate.from_dict(item) for item in data["candidates"]]

    settings = config["selection"]
    block = float(settings["analysis_window_seconds"])
    overlap = float(settings["analysis_overlap_seconds"])
    windows: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < duration:
        windows.append((cursor, min(duration, cursor + block)))
        cursor += max(1.0, block - overlap)

    creator_info = ""
    if creator_identity and creator_identity.get("character_lore"):
        creator_info = (
            "\nКонтекст о стримере и предпочтениях аудитории:\n"
            f"{creator_identity['character_lore']}\n"
        )

    def worker(index: int, start: float, end: float) -> tuple[int, list[Candidate], str]:
        block_words = [word for word in words if word.end >= start and word.start <= end]
        block_peaks = [peak for peak in peaks if start <= peak.time <= end]
        transcript = _format_transcript(block_words, max_chars=45000)
        peak_text = json.dumps(
            [{"time": round(item.time, 2), "score": item.score} for item in block_peaks],
            ensure_ascii=False,
        )
        prompt = (
            f"Диапазон исходного видео: {start:.3f}–{end:.3f} секунд.{creator_info}\n"
            "Транскрипт с абсолютными таймкодами:\n"
            f"{transcript}\n"
            "\n"
            "Пики аудио:\n"
            f"{peak_text}\n"
            "\n"
            "Найди до 10 вирусных моментов. При отборе руководствуйся правилами "
            "виральности для TikTok/Shorts:\n"
            "1. ХУК (0-3 сек): Начало фрагмента должно мгновенно создавать вопрос "
            "(«Что происходит?», «Сможет ли он?», «О чем он спорит?»).\n"
            "2. ЭСКАЛАЦИЯ: Напряжение, кринж, спор с чатом, нелепый фейл или "
            "нарастающий абсурд ситуации.\n"
            "3. РАЗВЯЗКА / КУЛЬМИНАЦИЯ (Payoff): Неожиданный финал, скример, "
            "осознание ошибки, жесткий троллинг или панчлайн.\n"
            "4. САМОДОСТАТОЧНОСТЬ: Любой незнакомый человек из ленты TikTok должен "
            "мгновенно понять юмор или драму без контекста всего стрима.\n"
            "\n"
            "Каждый итоговый диапазон обязан:\n"
            "- длиться от 18 до 75 секунд (оптимально для 100% удержания и "
            "повторных просмотров);\n"
            "- захватывать короткую завязку (3–8 сек) перед взрывом/событием;\n"
            "- заканчиваться сразу после кульминации/реакции (без долгих пост-пауз);\n"
            "- использовать абсолютные секунды исходного видео.\n"
            "\n"
            "Не добавляй:\n"
            "- унылое молчаливое прохождение, рутинный фарм, скучное чтение "
            "донатов без панчлайна;\n"
            "- бессвязные выкрики и мат без причины события;\n"
            "- моменты с оборванным началом или развязкой.\n"
            "\n"
            "Для каждого кандидата оцени:\n"
            "- context_score: 0..10 (понятность с нуля);\n"
            "- context_summary: «завязка → взрыв/событие → развязка»;\n"
            "- quality_score: 0..10 (вирусный потенциал и желание отправить другу);\n"
            "- quality_summary: в чем главный мем или триггер удержания.\n"
            "\n"
            "Шкала quality_score:\n"
            "- 0–4: заполнитель, рутина или реакция без события;\n"
            "- 5–6: понятный, но проходной момент;\n"
            "- 7–8: есть сильный хук и конкретная развязка;\n"
            "- 9–10: редкий момент с мгновенно понятным событием, сильной "
            "эскалацией и\n"
            "  запоминающимся итогом. Не ставь 9–10 только за крик или мат.\n"
            "\n"
            "Для camera_layout_recommendation соблюдай правило:\n"
            "- webcam_full — только если сильная реакция стримера является "
            "главным событием,\n"
            "  а gameplay/просматриваемое видео в этот момент не несёт "
            "отдельной важной информации;\n"
            "- split — если интересны и реакция стримера, и происходящее на "
            "gameplay/видео;\n"
            "- gameplay_full — если главным событием является gameplay/видео, "
            "а реакция вторична.\n"
            "\n"
            "Верни:\n"
            '{"candidates":[{\n'
            '  "start_time": 100.0,\n'
            '  "end_time": 145.0,\n'
            '  "highlight_reason": "конкретная причина",\n'
            '  "emotion_score": 8.2,\n'
            '  "context_score": 8.5,\n'
            '  "context_summary": "стример замечает угрозу, пытается уйти и '
            'пугается скримера",\n'
            '  "self_contained": true,\n'
            '  "quality_score": 8.0,\n'
            '  "quality_summary": "есть ясный скример и сильная реакция",\n'
            '  "camera_layout_recommendation": {\n'
            '    "layout": "split|webcam_full|gameplay_full",\n'
            '    "focus_x": 0.5,\n'
            '    "focus_y": 0.5,\n'
            '    "focus_time": 22.0\n'
            "  }\n"
            "}]}"
        )
        data, raw = client.chat_json(
            model=config["models"]["analysis"],
            system=SYSTEM_PROMPT,
            prompt=prompt,
            timeout=180,
            max_tokens=8192,
            temperature=0.35,
        )
        items = data.get("candidates", []) if isinstance(data, dict) else []
        candidates = _validate_candidates(items, duration, settings)
        return index, candidates, raw

    results: list[tuple[int, list[Candidate], str]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=int(settings["parallel_requests"])) as pool:
        futures = {
            pool.submit(worker, index, start, end): (index, start, end)
            for index, (start, end) in enumerate(windows)
        }
        for future in as_completed(futures):
            index, start, end = futures[future]
            try:
                results.append(future.result())
            except GeminiError as exc:
                failures.append(
                    {
                        "index": index,
                        "start_time": start,
                        "end_time": end,
                        "error": str(exc),
                    }
                )
                LOGGER.warning(
                    "Candidate discovery skipped window %s (%.3f-%.3f): %s",
                    index,
                    start,
                    end,
                    exc,
                )

    if windows and not results:
        first_error = failures[0]["error"] if failures else "no completed windows"
        raise GeminiError(
            f"Candidate discovery failed for all {len(windows)} windows: {first_error}"
        )

    raw_dir = work_dir / "analysis-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[Candidate] = []
    for index, items, raw in sorted(results):
        (raw_dir / f"window-{index:03d}.txt").write_text(raw, encoding="utf-8")
        candidates.extend(items)
    candidates.extend(_peak_fallback(peaks, words, duration, settings))
    candidates = _deduplicate(candidates)
    output.write_text(
        json.dumps(
            {
                "discovery_version": DISCOVERY_VERSION,
                "failures": failures,
                "candidates": [item.to_dict() for item in candidates],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("Candidate discovery produced %s moments", len(candidates))
    return candidates


def _format_transcript(words: list[Word], max_chars: int) -> str:
    lines: list[str] = []
    bucket: list[str] = []
    bucket_start: float | None = None
    last_end = 0.0
    for word in words:
        if bucket_start is None:
            bucket_start = word.start
        bucket.append(word.text)
        last_end = word.end
        if last_end - bucket_start >= 8 or len(bucket) >= 24:
            lines.append(f"[{bucket_start:.2f}-{last_end:.2f}] {' '.join(bucket)}")
            bucket = []
            bucket_start = None
        if sum(map(len, lines)) > max_chars:
            break
    if bucket and bucket_start is not None:
        lines.append(f"[{bucket_start:.2f}-{last_end:.2f}] {' '.join(bucket)}")
    return "\n".join(lines)


def _validate_candidates(
    items: list[Any],
    duration: float,
    settings: dict[str, Any],
) -> list[Candidate]:
    result: list[Candidate] = []
    minimum = float(settings["min_duration_seconds"])
    maximum = float(settings["max_duration_seconds"])
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            candidate = Candidate.from_dict(item)
        except (KeyError, TypeError, ValueError):
            continue
        candidate.camera_layout_recommendation["time_base"] = "source"
        candidate.start_time = max(0.0, candidate.start_time)
        candidate.end_time = min(duration, candidate.end_time)
        if minimum <= candidate.duration <= maximum and candidate.highlight_reason:
            result.append(candidate)
    return result


def _peak_fallback(
    peaks: list[AudioPeak],
    words: list[Word],
    duration: float,
    settings: dict[str, Any],
) -> list[Candidate]:
    fallback: list[Candidate] = []
    for peak in sorted(peaks, key=lambda item: item.score, reverse=True)[:20]:
        start = max(0.0, peak.time - 18)
        end = min(duration, peak.time + 22)
        if end - start < float(settings["min_duration_seconds"]):
            continue
        nearby = [word.text for word in words if start <= word.start <= end][:20]
        fallback.append(
            Candidate(
                start_time=start,
                end_time=end,
                highlight_reason=f"Аудиовсплеск: {' '.join(nearby)}".strip(),
                emotion_score=max(4.0, min(9.5, 5.5 + peak.score / 2)),
                camera_layout_recommendation={
                    "layout": "webcam_full" if peak.crest > 5 else "split",
                    "focus_x": 0.5,
                    "focus_y": 0.5,
                    "focus_time": peak.time - start,
                    "time_base": "clip",
                },
            )
        )
    return fallback


def _editorial_score(candidate: Candidate) -> float:
    return (
        candidate.quality_score * 0.45
        + candidate.context_score * 0.35
        + candidate.emotion_score * 0.20
    )


def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    accepted: list[Candidate] = []
    ranked = sorted(candidates, key=_editorial_score, reverse=True)
    for candidate in ranked:
        duplicate = False
        for current in accepted:
            overlap = max(
                0.0,
                min(candidate.end_time, current.end_time)
                - max(candidate.start_time, current.start_time),
            )
            if overlap / min(candidate.duration, current.duration) > 0.65:
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)
    return sorted(accepted, key=lambda item: item.start_time)
