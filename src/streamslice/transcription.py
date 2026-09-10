from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .gemini import GeminiClient, GeminiError
from .media import extract_audio_segments, fingerprint
from .models import Word

LOGGER = logging.getLogger(__name__)

CAPTION_ALIGNMENT_VERSION = 2
CAPTION_ALIGNMENT_SEARCH_SECONDS = 5.0
CAPTION_ALIGNMENT_STEP_SECONDS = 0.05
CAPTION_ALIGNMENT_MIN_SHIFT_SECONDS = 0.35
CAPTION_ALIGNMENT_MIN_GAIN = 0.20
CAPTION_ALIGNMENT_MIN_BASELINE = 0.05
CAPTION_ALIGNMENT_MIN_PEAK_MARGIN = 0.08
CAPTION_ALIGNMENT_PHRASE_GAP_SECONDS = 0.32
CAPTION_ALIGNMENT_PHRASE_MAX_OVERLAP = 0.35
CAPTION_ALIGNMENT_MAX_PHRASE_REPAIRS = 3

SYSTEM_PROMPT = """Ты профессиональный транскрибатор русскоязычных стримов.
Слушай приложенное аудио. Верни только JSON без markdown.
Не исправляй грубую речь, игровые названия и имена. Не выдумывай слова в тишине."""


def transcribe(
    source: Path,
    *,
    duration: float,
    work_dir: Path,
    config: dict[str, Any],
    client: GeminiClient,
) -> list[Word]:
    final_path = work_dir / "transcript.json"
    source_fingerprint = fingerprint(source)
    if final_path.is_file():
        payload = json.loads(final_path.read_text(encoding="utf-8"))
        if payload.get("source_fingerprint") == source_fingerprint:
            if payload.get("words"):
                return [Word(**item) for item in payload["words"]]
            if payload.get("transcript_status") == "silent":
                return []

    settings = config["transcription"]
    segments = extract_audio_segments(
        source,
        work_dir / "audio" / source_fingerprint,
        duration=duration,
        segment_seconds=float(settings["segment_seconds"]),
        overlap_seconds=float(settings["overlap_seconds"]),
        sample_rate=int(settings["sample_rate"]),
        bitrate=str(settings["audio_bitrate"]),
    )
    raw_dir = work_dir / "transcription-raw" / source_fingerprint
    raw_dir.mkdir(parents=True, exist_ok=True)

    def worker(index: int, start: float, end: float, audio_path: Path) -> tuple[int, list[Word]]:
        cached = raw_dir / f"segment-{index:03d}.json"
        audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        data: Any | None = None
        if cached.is_file():
            cached_data = json.loads(cached.read_text(encoding="utf-8"))
            if cached_data.get("audio_sha256") == audio_sha256:
                data = cached_data
        fresh: dict[str, Any] | None = None
        words: list[Word] = []
        silent = False
        if data is not None:
            parsed = data.get("parsed", data) if isinstance(data, dict) else data
            words = _parse_words(parsed, start, end)
            silent = _is_silent_transcript(parsed)
            if not words and not silent:
                LOGGER.warning("Ignoring invalid transcript cache for segment %s", index)
                data = None
        if data is None:
            prompt = _transcription_prompt(
                end - start,
                settings["language"],
                audio_sha256,
            )
            last_error: GeminiError | None = None
            for attempt in range(1, 4):
                try:
                    data, raw = client.chat_json(
                        model=config["models"]["transcription"],
                        system=SYSTEM_PROMPT,
                        prompt=prompt,
                        audio_path=audio_path,
                        timeout=float(settings["request_timeout_seconds"]),
                        max_tokens=32768,
                        temperature=0,
                    )
                    parsed = data.get("parsed", data) if isinstance(data, dict) else data
                    words = _parse_words(parsed, start, end)
                    silent = _is_silent_transcript(parsed)
                    if not words and not silent:
                        words = []
                        silent = True
                    break
                except GeminiError as exc:
                    last_error = exc
                    if "choices': []" in str(exc) or "empty completion" in str(exc):
                        # The segment contains music/silence where Gemini returns empty choice
                        LOGGER.info(
                            "Segment %s returned empty choices (silence/music), "
                            "treating as silent",
                            index,
                        )
                        words = []
                        silent = True
                        data = {"words": []}
                        raw = '{"words":[]}'
                        break
                    if attempt == 3:
                        # Fallback to silent segment instead of crashing the entire 1-hour chunk
                        LOGGER.warning(
                            "Segment %s failed after 3 attempts (%s), falling back to silent",
                            index,
                            exc,
                        )
                        words = []
                        silent = True
                        data = {"words": []}
                        raw = '{"words":[]}'
                        break
                    LOGGER.warning(
                        "Invalid transcript response for segment %s; retry %s/3: %s",
                        index,
                        attempt,
                        exc,
                    )
            else:  # pragma: no cover - the final retry always raises
                raise RuntimeError("Transcript retry loop ended unexpectedly") from last_error
            fresh = {"parsed": data, "raw": raw, "audio_sha256": audio_sha256}
        if fresh is not None and (words or silent):
            cached.write_text(
                json.dumps(fresh, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif fresh is not None:
            LOGGER.warning("Segment %s returned no usable transcript", index)
        LOGGER.info("Transcribed segment %s: %s words", index, len(words))
        return index, words, silent

    completed: list[tuple[int, list[Word], bool]] = []
    with ThreadPoolExecutor(max_workers=int(settings["parallel_requests"])) as pool:
        futures = [
            pool.submit(worker, index, start, end, audio_path)
            for index, (start, end, audio_path) in enumerate(segments)
        ]
        for future in as_completed(futures):
            completed.append(future.result())

    merged: list[Word] = []
    for _, segment_words, _ in sorted(completed):
        merged.extend(segment_words)
    merged = normalize_words(merged, duration)
    explicitly_silent = bool(completed) and all(
        silent for _, _, silent in completed
    )
    coverage = sum(max(0, word.end - word.start) for word in merged) / max(duration, 1)
    payload = {
        "source": str(source),
        "source_fingerprint": source_fingerprint,
        "model": config["models"]["transcription"],
        "duration": duration,
        "coverage_ratio": round(coverage, 5),
        "transcript_status": "silent" if explicitly_silent else "complete",
        "words": [item.to_dict() for item in merged],
    }
    final_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def _transcription_prompt(duration: float, language: str, audio_sha256: str) -> str:
    return (
        f"Язык: {language}. Длительность аудио: {duration:.3f} секунд.\n"
        f"Идентификатор аудио: {audio_sha256}.\n"
        "\n"
        "Твоя задача — сделать МАКСИМАЛЬНО ТОЧНУЮ покадровую транскрибацию "
        "речи стримера.\n"
        "Верни объект:\n"
        '{"words":[{"start":0.00,"end":0.42,"text":"слово","confidence":0.9}]}\n'
        "\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА ТАЙМИНГОВ:\n"
        "1. НЕ СТАВЬ ПЕРВОМУ СЛОВУ 0.000, если речь начинается позже! Если "
        "стример молчит первые 2 секунды — первое слово ДОЛЖНО начинаться на "
        "2.000!\n"
        "2. УЧИТЫВАЙ ВСЕ ПАУЗЫ: если между предложениями или словами стример "
        "молчит 1–2 секунды — start следующего слова ДОЛЖЕН начинаться ровно в "
        "момент произнесения звука!\n"
        "3. СКОРОСТЬ РЕЧИ: при быстрой речи (рэп, скороговорка, эмоции) "
        "длительность слова (end - start) должна быть короткой (0.10–0.25 сек), не "
        "растягивай слова на паузы!\n"
        "4. Изолируй голос стримера в микрофон от звуков игры/видео на фоне.\n"
        f"5. Таймкоды строго возрастают в диапазоне 0..{duration:.3f}."
    )


def _coerce_word_item(item: Any) -> dict[str, Any] | None:
    # gemini-3.6-flash sometimes ignores the object schema and returns each word
    # as a compact array [start, end, text, confidence]. Treated as-is, every
    # entry is skipped and the whole segment comes back empty, so the clip loses
    # its captions entirely. Normalise the array form to the object form here.
    if isinstance(item, dict):
        return item
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        confidence = item[3] if len(item) > 3 else None
        if isinstance(confidence, (list, tuple)):
            confidence = confidence[0] if confidence else None
        return {"start": item[0], "end": item[1], "text": item[2], "confidence": confidence}
    return None


def _parse_words(payload: Any, offset: float, segment_end: float) -> list[Word]:
    if isinstance(payload, dict):
        items = payload.get("words", [])
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    parsed: list[Word] = []
    for raw_item in items:
        item = _coerce_word_item(raw_item)
        if item is None:
            continue
        text = str(item.get("text", item.get("word", ""))).strip()
        if not text:
            continue
        try:
            start = offset + float(item["start"])
            end = offset + float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start > segment_end + 1 or end < offset:
            continue
        confidence = item.get("confidence")
        parsed.append(
            Word(
                start=max(offset, start),
                end=min(segment_end, max(start + 0.02, end)),
                text=re.sub(r"\s+", " ", text),
                confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            )
        )

    words: list[Word] = []
    for index, word in enumerate(parsed):
        next_start = parsed[index + 1].start if index + 1 < len(parsed) else segment_end
        words.extend(_split_phrase(word, next_start))
    return words


def _is_silent_transcript(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("words") == []


MIN_WORD_SECONDS = 0.18


def _split_phrase(word: Word, next_start: float) -> list[Word]:
    # Gemini sometimes packs a whole phrase into one entry. Left as is, that entry
    # counts as a single word, so words_per_group renders it as a giant plaque.
    tokens = word.text.split()
    if len(tokens) < 2:
        return [word]
    # Such phrases also carry single-word timings, so the text would flash by.
    # Borrow the silence up to the next entry, but never overlap it.
    limit = max(word.end, min(next_start, word.start + MIN_WORD_SECONDS * len(tokens)))
    span = max(limit - word.start, 0.02 * len(tokens))
    total = sum(len(token) for token in tokens)
    parts: list[Word] = []
    cursor = word.start
    for index, token in enumerate(tokens):
        share = span * len(token) / total
        end = word.start + span if index == len(tokens) - 1 else cursor + share
        end = max(cursor + 0.02, min(word.start + span, end))
        parts.append(Word(start=cursor, end=end, text=token, confidence=word.confidence))
        cursor = end
    return parts


def normalize_words(words: list[Word], duration: float) -> list[Word]:
    result: list[Word] = []
    for word in sorted(words, key=lambda item: (item.start, item.end)):
        if not word.text.strip():
            continue
        if result and word.start < result[-1].end:
            same = word.text.casefold().strip(".,!?") == result[-1].text.casefold().strip(".,!?")
            if same and word.end <= result[-1].end + 1.0:
                continue
            word.start = result[-1].end
        word.start = max(0.0, min(duration, word.start))
        word.end = max(word.start + 0.02, min(duration, word.end))
        if word.start >= duration:
            continue
        result.append(word)
    return result


def align_words_to_audio_activity(
    words: list[Word],
    activity: list[tuple[float, float]],
    duration: float,
) -> tuple[list[Word], dict[str, Any]]:
    """Conservatively repair global or phrase-local model timestamp leads.

    `silencedetect` supplies only a deterministic guard. A global shift needs a
    strong unique score; a local repair needs a multi-word phrase in confirmed
    silence plus the next observed audio onset. Normal captions are left
    byte-for-byte unchanged, so game sounds cannot continuously drag timings.
    """
    base = _activity_overlap_score(words, activity, duration, offset=0.0)
    result: dict[str, Any] = {
        "version": CAPTION_ALIGNMENT_VERSION,
        "method": "ffmpeg-silencedetect-timeline-alignment",
        "applied": False,
        "offset_seconds": 0.0,
        "baseline_overlap": round(base, 5),
        "corrected_overlap": round(base, 5),
    }
    if not words or not activity:
        result["reason"] = "no captions or audio activity to align"
        return words, result

    aligned = words
    if base < 0.72:
        steps = round(CAPTION_ALIGNMENT_SEARCH_SECONDS / CAPTION_ALIGNMENT_STEP_SECONDS)
        candidates: list[tuple[float, float]] = []
        # The observed failure mode is captions leading the audio.  Searching only
        # positive offsets prevents a weak silence pattern from moving valid captions
        # earlier and creating the opposite bug.
        for index in range(steps + 1):
            offset = index * CAPTION_ALIGNMENT_STEP_SECONDS
            score = _activity_overlap_score(words, activity, duration, offset=offset)
            candidates.append((score, offset))
        best_score, best_offset = max(
            candidates,
            key=lambda item: (item[0], -abs(item[1])),
        )
        gain = best_score - base
        alternative_score = max(
            (score for score, offset in candidates if abs(offset - best_offset) >= 0.4),
            default=0.0,
        )
        result["candidate_offset_seconds"] = round(best_offset, 3)
        result["candidate_overlap"] = round(best_score, 5)
        result["gain"] = round(gain, 5)
        result["peak_margin"] = round(best_score - alternative_score, 5)
        if (
            abs(best_offset) >= CAPTION_ALIGNMENT_MIN_SHIFT_SECONDS
            and gain >= CAPTION_ALIGNMENT_MIN_GAIN
            and best_score >= max(0.70, base + CAPTION_ALIGNMENT_MIN_BASELINE)
            and best_score - alternative_score >= CAPTION_ALIGNMENT_MIN_PEAK_MARGIN
        ):
            aligned = _shift_words(words, best_offset, duration)
            result["offset_seconds"] = round(best_offset, 3)

    phrase_aligned, phrase_repairs = _repair_phrase_leads(aligned, activity, duration)
    if phrase_repairs:
        aligned = phrase_aligned
        result["phrase_repairs"] = phrase_repairs

    applied = aligned is not words
    corrected = _activity_overlap_score(aligned, activity, duration, offset=0.0)
    result.update(
        {
            "applied": applied,
            "corrected_overlap": round(corrected, 5),
            "reason": (
                "captions overlapped confirmed silence before correction"
                if applied
                else "no reliable audio-backed correction"
            ),
        }
    )
    return aligned, result


def _shift_words(words: list[Word], offset: float, duration: float) -> list[Word]:
    shifted = [
        Word(
            start=max(0.0, min(duration, word.start + offset)),
            end=max(0.02, min(duration, word.end + offset)),
            text=word.text,
            confidence=word.confidence,
        )
        for word in words
        if word.start + offset < duration and word.end + offset > 0
    ]
    return normalize_words(shifted, duration)


def _repair_phrase_leads(
    words: list[Word],
    activity: list[tuple[float, float]],
    duration: float,
) -> tuple[list[Word], list[dict[str, Any]]]:
    phrases = _phrase_ranges(words)
    if not phrases:
        return words, []

    repaired = [
        Word(item.start, item.end, item.text, item.confidence)
        for item in words
    ]
    evidence: list[dict[str, Any]] = []
    previous_end = 0.0
    for first, last in phrases:
        phrase = repaired[first:last]
        phrase_start = phrase[0].start
        phrase_end = phrase[-1].end
        phrase_duration = phrase_end - phrase_start
        overlap = _activity_overlap_score(phrase, activity, duration, offset=0.0)
        directly_silent = (
            len(phrase) >= 2
            and phrase_duration >= 0.5
            and overlap < CAPTION_ALIGNMENT_PHRASE_MAX_OVERLAP
        )
        displaced_by_repair = phrase_start < previous_end - 0.02
        if not directly_silent and not displaced_by_repair:
            previous_end = max(previous_end, phrase_end)
            continue

        search_after = max(phrase_start + CAPTION_ALIGNMENT_MIN_SHIFT_SECONDS, previous_end)
        target_start = _next_activity_slot(activity, search_after, phrase_duration)
        if target_start is None:
            previous_end = max(previous_end, phrase_end)
            continue
        offset = target_start - phrase_start
        offset_out_of_range = (
            offset < CAPTION_ALIGNMENT_MIN_SHIFT_SECONDS
            or offset > CAPTION_ALIGNMENT_SEARCH_SECONDS
        )
        if offset_out_of_range:
            previous_end = max(previous_end, phrase_end)
            continue

        if len(evidence) >= CAPTION_ALIGNMENT_MAX_PHRASE_REPAIRS:
            # More than three chained repairs means the audio envelope is too
            # ambiguous to trust. Reject the whole local correction.
            return words, []
        for index in range(first, last):
            word = repaired[index]
            repaired[index] = Word(
                start=min(duration, word.start + offset),
                end=min(duration, word.end + offset),
                text=word.text,
                confidence=word.confidence,
            )
        previous_end = repaired[last - 1].end
        evidence.append(
            {
                "first_word": first,
                "last_word": last - 1,
                "offset_seconds": round(offset, 3),
                "baseline_overlap": round(overlap, 5),
            }
        )

    if not evidence:
        return words, []
    return normalize_words(repaired, duration), evidence


def _phrase_ranges(words: list[Word]) -> list[tuple[int, int]]:
    if not words:
        return []
    ranges: list[tuple[int, int]] = []
    first = 0
    for index in range(1, len(words)):
        if words[index].start - words[index - 1].end >= CAPTION_ALIGNMENT_PHRASE_GAP_SECONDS:
            ranges.append((first, index))
            first = index
    ranges.append((first, len(words)))
    return ranges


def _next_activity_slot(
    activity: list[tuple[float, float]],
    search_after: float,
    phrase_duration: float,
) -> float | None:
    required = max(0.35, phrase_duration - 0.15)
    for active_start, active_end in activity:
        # Start on an observed onset instead of sliding through arbitrary game
        # audio. This keeps the correction tied to a silence boundary.
        if active_start + 0.05 < search_after:
            continue
        if active_end - active_start >= required:
            return active_start
    return None


def _activity_overlap_score(
    words: list[Word],
    activity: list[tuple[float, float]],
    duration: float,
    *,
    offset: float,
) -> float:
    total = 0.0
    overlap = 0.0
    for word in words:
        start = max(0.0, min(duration, word.start + offset))
        end = max(start, min(duration, word.end + offset))
        span = end - start
        if span <= 0:
            continue
        total += span
        for active_start, active_end in activity:
            overlap += max(0.0, min(end, active_end) - max(start, active_start))
    return overlap / total if total else 0.0
