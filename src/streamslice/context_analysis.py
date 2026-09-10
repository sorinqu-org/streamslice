from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .gemini import GeminiClient, GeminiError
from .models import Candidate, Word
from .process import require_binary, run

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты главный аналитик вирусного контента для коротких видео (TikTok/Reels).
Оценивай каждый кандидат по жестким критериям виральности:
1. Hook & Retention (есть ли интересная завязка в первые 3 секунды?)
2. Clarity (понятна ли ситуация зрителю с улицы?)
3. Punchline / Payoff (есть ли сильный финал?)
4. Shareability (захотят ли скинуть друзьям?)
Отклоняй проходные и скучные моменты."""
CONTEXT_REVIEW_VERSION = 6


def review_candidate_context(
    source: Path,
    words: list[Word],
    candidates: list[Candidate],
    *,
    work_dir: Path,
    config: dict[str, Any],
    client: GeminiClient,
    creator_identity: dict[str, Any] | None = None,
) -> list[Candidate]:
    settings = config["selection"]
    if not settings.get("strict_context", True):
        return candidates

    review_limit = int(settings.get("context_review_limit", 24))
    editorial = sorted(
        candidates,
        key=lambda item: (
            item.quality_score * 0.45
            + item.context_score * 0.35
            + item.emotion_score * 0.20
        ),
        reverse=True,
    )
    emotion_quota = min(max(2, review_limit // 4), review_limit)
    loud = sorted(candidates, key=lambda item: item.emotion_score, reverse=True)
    review_pool: list[Candidate] = []
    for candidate in editorial[: max(0, review_limit - emotion_quota)] + loud:
        if candidate not in review_pool:
            review_pool.append(candidate)
        if len(review_pool) >= review_limit:
            break
    signature = hashlib.sha256(
        json.dumps(
            [item.to_dict() for item in review_pool],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = work_dir / "context-review.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            payload.get("context_review_version") == CONTEXT_REVIEW_VERSION
            and payload.get("candidate_signature") == signature
        ):
            return [Candidate.from_dict(item) for item in payload.get("approved", [])]

    frame_root = work_dir / "context-frames" / signature[:16]
    frame_root.mkdir(parents=True, exist_ok=True)

    creator_info = ""
    if creator_identity and creator_identity.get("character_lore"):
        creator_info = (
            "\nКонтекст о стримере и что ценит аудитория:\n"
            f"{creator_identity['character_lore']}\n"
        )

    def worker(index: int, candidate: Candidate) -> tuple[int, Candidate, dict[str, Any], str]:
        images = _extract_context_frames(source, candidate, frame_root / f"{index:03d}")
        transcript = _candidate_transcript(words, candidate)
        prompt = f"""Проверь, понятен и интересен ли этот клип для TikTok/Shorts.{creator_info}
Диапазон: {candidate.start_time:.3f}–{candidate.end_time:.3f} секунд.
Причина-кандидат: {candidate.highlight_reason}
Транскрипт внутри диапазона:
{transcript or "[стример почти не говорит]"}

Приложены четыре кадра по хронологии: начало, развитие, событие и завершение.
Визуально понятный клип можно одобрить даже при коротком транскрипте.

Отклони сцену, если:
- она начинается с реакции, но причина реакции осталась до начала клипа;
- это внутренняя шутка или ссылка на предыдущие события без объяснения;
- непонятно, чего стример добивается или что именно его напугало/рассмешило;
- клип обрывается до результата или развязки;
- внутри есть только громкий звук, крик или ругань без понятного события.

Отдельно оцени качество момента. Не пытайся «спасти» слабый эпизод красивым
описанием — оценки должны опираться только на кадры и транскрипт:
- 0–4: обычный разговор, рутинный геймплей, случайный звук или слабая реакция;
- 5–6: что-то происходит, но нет сильного панчлайна, страха или неожиданности;
- 7–8: ясный смешной/страшный/неожиданный момент с хорошей развязкой;
- 9–10: редкий сильный момент, который хочется досмотреть и переслать.
Не ставь высокий quality_score только за громкость или ругань.

Верни только JSON:
{{
  "self_contained": true,
  "context_score": 8.5,
  "context_summary": "конкретная завязка → событие → итог",
  "hook_score": 8.0,
  "payoff_score": 8.5,
  "shareability_score": 8.0,
  "visual_clarity_score": 7.5,
  "quality_score": 8.0,
  "quality_summary": "почему момент сильнее обычного фрагмента",
  "evidence": ["конкретная реплика или действие", "конкретный итог"],
  "redundancy_key": "краткое название эпизода для поиска дублей",
  "rejection_reason": ""
}}"""
        data, raw = client.chat_json(
            model=config["models"]["curator"],
            system=(
                "Ты строгий контекст-редактор коротких видео. "
                "Не одобряй эмоциональный фрагмент, если зритель не поймёт причину реакции."
            ),
            prompt=prompt,
            image_paths=images,
            timeout=180,
            max_tokens=2048,
            temperature=0,
        )
        if not isinstance(data, dict):
            data = {}
        candidate.self_contained = bool(data.get("self_contained", False))
        try:
            candidate.context_score = max(
                0.0, min(10.0, float(data.get("context_score", 0)))
            )
        except (TypeError, ValueError):
            candidate.context_score = 0.0
        candidate.context_summary = str(data.get("context_summary", "")).strip()
        component_scores: list[float] = []
        component_keys = (
            "hook_score",
            "payoff_score",
            "shareability_score",
            "visual_clarity_score",
        )
        for key in component_keys:
            try:
                component_scores.append(max(0.0, min(10.0, float(data.get(key, 0)))))
            except (TypeError, ValueError):
                component_scores.append(0.0)
        for key, score in zip(component_keys, component_scores, strict=True):
            data[key] = score
        try:
            reported_quality = max(
                0.0, min(10.0, float(data.get("quality_score", 0)))
            )
        except (TypeError, ValueError):
            reported_quality = 0.0
        # A high generic quality claim cannot hide a missing hook or payoff.
        component_quality = sum(component_scores) / len(component_scores)
        candidate.quality_score = min(reported_quality, component_quality)
        candidate.quality_summary = str(data.get("quality_summary", "")).strip()
        return index, candidate, data, raw

    completed: list[tuple[int, Candidate, dict[str, Any], str]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=int(settings.get("context_parallel_requests", 3))
    ) as pool:
        futures = {
            pool.submit(worker, index, candidate): (index, candidate)
            for index, candidate in enumerate(review_pool)
        }
        for future in as_completed(futures):
            index, candidate = futures[future]
            try:
                completed.append(future.result())
            except GeminiError as exc:
                failure = {
                    "index": index,
                    "start_time": candidate.start_time,
                    "end_time": candidate.end_time,
                    "error": str(exc),
                }
                failures.append(failure)
                LOGGER.warning(
                    "Context review skipped candidate %s (%.3f-%.3f): %s",
                    index,
                    candidate.start_time,
                    candidate.end_time,
                    exc,
                )

    if review_pool and not completed:
        first_error = failures[0]["error"] if failures else "no completed reviews"
        raise GeminiError(
            f"Context review failed for all {len(review_pool)} candidates: {first_error}"
        )

    threshold = float(settings.get("min_context_score", 7.0))
    quality_threshold = float(settings.get("min_quality_score", 7.5))
    ordered = sorted(completed)
    approved = [
        candidate
        for _, candidate, analysis, _ in ordered
        if candidate.self_contained
        and candidate.context_score >= threshold
        and bool(candidate.context_summary)
        and candidate.quality_score >= quality_threshold
        and bool(candidate.quality_summary)
        and isinstance(analysis.get("evidence"), list)
        and len(analysis["evidence"]) >= 2
        and float(analysis.get("hook_score", 0)) >= 6.5
        and float(analysis.get("payoff_score", 0)) >= 6.5
        and float(analysis.get("shareability_score", 0)) >= 6.5
    ]
    cache_path.write_text(
        json.dumps(
            {
                "context_review_version": CONTEXT_REVIEW_VERSION,
                "candidate_signature": signature,
                "threshold": threshold,
                "failures": failures,
                "reviews": [
                    {
                        "candidate": candidate.to_dict(),
                        "analysis": analysis,
                        "raw": raw,
                    }
                    for _, candidate, analysis, raw in ordered
                ],
                "approved": [item.to_dict() for item in approved],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return approved


def _extract_context_frames(
    source: Path,
    candidate: Candidate,
    target_dir: Path,
) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    offsets = (0.08, 0.34, 0.66, 0.92)
    images: list[Path] = []
    for index, ratio in enumerate(offsets):
        timestamp = candidate.start_time + candidate.duration * ratio
        target = target_dir / f"frame-{index:02d}-{timestamp:.2f}.jpg"
        if not target.is_file():
            run(
                [
                    require_binary("ffmpeg"),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    source,
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=960:-2",
                    "-q:v",
                    "4",
                    "-y",
                    target,
                ],
                timeout=120,
            )
        images.append(target)
    return images


def _candidate_transcript(words: list[Word], candidate: Candidate) -> str:
    selected = [
        word
        for word in words
        if word.end >= candidate.start_time and word.start <= candidate.end_time
    ]
    lines: list[str] = []
    bucket: list[str] = []
    bucket_start: float | None = None
    last_end = candidate.start_time
    for word in selected:
        if bucket_start is None:
            bucket_start = word.start
        bucket.append(word.text)
        last_end = word.end
        if last_end - bucket_start >= 6 or len(bucket) >= 20:
            lines.append(f"[{bucket_start:.2f}-{last_end:.2f}] {' '.join(bucket)}")
            bucket = []
            bucket_start = None
    if bucket and bucket_start is not None:
        lines.append(f"[{bucket_start:.2f}-{last_end:.2f}] {' '.join(bucket)}")
    return "\n".join(lines)
