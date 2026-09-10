from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .gemini import GeminiClient
from .models import Candidate
from .profanity import mask_profanity

SYSTEM_PROMPT = (
    "Ты главный редактор и эксперт по виральным Shorts/TikTok/Reels.\n"
    "Твоя задача — создавать цепляющие хуки-заголовки и кликбейтные (но честные) "
    "описания, которые удерживают зрителя с первых секунд.\n"
    "Заголовок (title) должен быть коротким панчем (3-6 слов, КАПСОМ), который "
    "служит визуальным хуком над видео.\n"
    "Верни только JSON."
)
METADATA_VERSION = 5

SLOP_PATTERNS = (
    r"\bпогруз\w*",
    r"\bнастоящ\w+\s+шедевр",
    r"\bв мире игр\b",
    r"\bневероятн\w+\s+путешеств",
    r"\bприготовьтесь\b",
    r"\bэто не просто\b",
    r"\bуникальн\w+\s+опыт",
    r"\bнельзя пропустить\b",
)


def generate_metadata(
    candidate: Candidate,
    transcript: str,
    output_dir: Path,
    config: dict[str, Any],
    client: GeminiClient,
    creator_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("metadata_version") == METADATA_VERSION:
            return cached
    creator = creator_identity or {
        "display_name": "t2x2",
        "aliases": ["t2x2"],
        "preferred_mentions": ["t2x2", "Тоха"],
        "real_name": "",
    }
    prompt = (
        f"Причина выбора: {candidate.highlight_reason}\n"
        f"Оценка эмоции: {candidate.emotion_score}/10\n"
        f"Контекст: {candidate.context_summary}\n"
        f"Автор ролика: {json.dumps(creator, ensure_ascii=False)}\n"
        "Реплики:\n"
        f"{transcript[:12000]}\n"
        "\n"
        "Верни JSON:\n"
        '{"title":"КОРОТКИЙ ВЗРЫВНОЙ ХУК-ЗАГОЛОВОК (3-6 СЛОВ, КАПС)",\n'
        '  "description":"1–2 живых интригующих предложения о происходящем без штампов",\n'
        '  "hashtags":["#тема_ролика1","#тема_ролика2","#тема_ролика3","#fyp"]}\n'
        "\n"
        "Правила генерации хэштегов:\n"
        "- Придумай 3–5 УНИКАЛЬНЫХ, специфичных хэштегов под конкретное событие в клипе "
        "(например: #майнкрафт, #кс2, #рофл, #тильт, #донат, #база, #бан, #смешно, "
        "#паника, #пранк).\n"
        "- Хэштеги должны точно отражать суть видео и различаться от клипа к клипу!\n"
        "- Запрещен тег #viral."
    )
    data, raw = client.chat_json(
        model=config["models"]["metadata"],
        system=SYSTEM_PROMPT,
        prompt=prompt,
        timeout=120,
        max_tokens=2048,
        temperature=0.55,
    )
    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    if (
        not title
        or not description
        or _contains_slop(title + " " + description)
        or _contains_generic_creator_term(title + " " + description)
    ):
        repair = f"""Перепиши текст без штампов и верни тот же JSON.
Текущий ответ: {json.dumps(data, ensure_ascii=False)}
Используй конкретную сцену: {candidate.highlight_reason}
Называй автора как {creator.get("display_name", "t2x2")}.
Запрещены слова «стример», «стримерша» и «автор стрима»."""
        data, raw = client.chat_json(
            model=config["models"]["metadata"],
            system=SYSTEM_PROMPT,
            prompt=repair,
            timeout=120,
            max_tokens=2048,
            temperature=0.35,
        )
        title = str(data.get("title", "")).strip()
        description = str(data.get("description", "")).strip()
    if _contains_generic_creator_term(title + " " + description):
        replacement = str(creator.get("display_name", "t2x2")).strip() or "t2x2"
        title = re.sub(r"\bстример(?:ша|ом|у|а|е|ы|ов)?\b", replacement, title, flags=re.IGNORECASE)
        description = re.sub(
            r"\bстример(?:ша|ом|у|а|е|ы|ов)?\b",
            replacement,
            description,
            flags=re.IGNORECASE,
        )
    title = mask_profanity(title)
    description = mask_profanity(description)
    hashtags = _normalize_hashtags(data.get("hashtags", []), candidate.highlight_reason)

    creator_login = str(creator.get("twitch_login") or creator.get("display_name") or "twitch")
    creator_display = str(creator.get("display_name") or creator_login)
    creator_aliases = creator.get("aliases") or []

    yt_title = format_youtube_shorts_title(creator_login, creator_display, hashtags)
    yt_tags = format_youtube_tags(creator_login, creator_display, hashtags, creator_aliases)
    default_source_url = "https://twitch.tv/" + creator_login
    is_gaming = any(
        k in candidate.highlight_reason.lower()
        for k in ("майнкрафт", "minecraft", "игра", "cs", "игры")
    )
    category = "Gaming" if is_gaming else "Entertainment"

    result = {
        "metadata_version": METADATA_VERSION,
        "creator": creator,
        "title": title,
        "description": description,
        "hashtags": hashtags,
        "youtube_title": yt_title,
        "youtube_description": (
            f"{description}\n\n"
            f"Twitch: {creator.get('source_url', default_source_url)}\n\n"
            f"{' '.join(hashtags)}"
        ),
        "youtube_tags": yt_tags,
        "category": category,
    }
    metadata_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "description.txt").write_text(
        f"{title}\n\n{description}\n\n{' '.join(hashtags)}\n", encoding="utf-8"
    )
    (output_dir / "metadata-raw.txt").write_text(raw, encoding="utf-8")
    return result


def _contains_slop(text: str) -> bool:
    lowered = text.casefold()
    return any(re.search(pattern, lowered) for pattern in SLOP_PATTERNS)


def _contains_generic_creator_term(text: str) -> bool:
    return bool(re.search(r"\bстример(?:ша|ом|у|а|е|ы|ов)?\b", text.casefold()))


def format_youtube_shorts_title(
    creator_login: str,
    creator_name: str = "",
    topic_tags: list[str] | None = None,
) -> str:
    """Format YouTube Shorts title as: twitch: login #login #name #fyp #tag1 #tag2...

    Exactly 5-6 tags, no #viral.
    """
    login = str(creator_login).strip().lower()
    name = str(creator_name).strip().lower()

    tags = [f"#{login}"]
    if name and name != login:
        clean_name = re.sub(r"[^\wа-яА-ЯёЁ]", "", name)
        if clean_name and f"#{clean_name}" not in tags:
            tags.append(f"#{clean_name}")
    tags.append("#fyp")

    for t in (topic_tags or []):
        t_clean = re.sub(r"[^\wа-яА-ЯёЁ#]", "", str(t)).lower()
        if not t_clean.startswith("#"):
            t_clean = "#" + t_clean
        if t_clean in ("#viral", "#fyp", f"#{login}"):
            continue
        if t_clean not in tags:
            tags.append(t_clean)
        if len(tags) >= 6:
            break

    # If fewer than 5 tags, add topic-relevant tags
    fallback_tags = ["#стрим", "#моменты", "#игры", "#юмор"]
    for fb in fallback_tags:
        if len(tags) >= 5:
            break
        if fb not in tags:
            tags.append(fb)

    title = f"twitch: {login} {' '.join(tags)}"
    return title[:100]


def format_youtube_tags(
    creator_login: str,
    creator_name: str = "",
    topic_tags: list[str] | None = None,
    aliases: list[str] | None = None,
) -> str:
    """Format comma-separated YouTube tags for Studio upload (no #, up to 500 chars)."""
    seen: set[str] = set()
    tags: list[str] = []

    def add_tag(raw: str) -> None:
        clean = re.sub(r"[^\wа-яА-ЯёЁ\s]", "", str(raw)).strip().lower()
        if clean and clean not in seen and len(clean) > 1 and clean != "viral":
            seen.add(clean)
            tags.append(clean)

    add_tag(creator_login)
    if creator_name:
        add_tag(creator_name)
    for a in (aliases or []):
        add_tag(a)
    for base in ("twitch", "твич", "стрим", "нарезка", "fyp", "моменты"):
        add_tag(base)
    for t in (topic_tags or []):
        add_tag(t)

    result = ", ".join(tags)
    return result[:490]


def _normalize_hashtags(values: Any, reason: str) -> list[str]:
    result: list[str] = []
    for value in values if isinstance(values, list) else []:
        tag = re.sub(r"[^\wа-яА-ЯёЁ#]", "", str(value))
        if not tag:
            continue
        if mask_profanity(tag) != tag:
            continue
        if not tag.startswith("#"):
            tag = "#" + tag
        tag_lower = tag.casefold()
        if tag_lower != "#viral" and tag_lower not in {item.casefold() for item in result}:
            result.append(tag)

    if "#fyp" not in {item.casefold() for item in result}:
        result.append("#fyp")

    # Extract topic tags from highlight reason
    reason_lower = reason.casefold()
    mentions_minecraft = "майнкрафт" in reason_lower or "minecraft" in reason_lower
    if mentions_minecraft and "#майнкрафт" not in result:
        result.insert(0, "#майнкрафт")
    mentions_cs = "кс" in reason_lower or "cs2" in reason_lower or "кс2" in reason_lower
    if mentions_cs and "#кс2" not in result:
        result.insert(0, "#кс2")
    if "донат" in reason_lower and "#донат" not in result:
        result.append("#донат")
    mentions_funny = "рофл" in reason_lower or "смешн" in reason_lower
    if mentions_funny and "#рофл" not in result:
        result.append("#рофл")

    fallback = ["#стрим", "#twitch", "#игры", "#юмор"]
    for tag in fallback:
        if len(result) >= 5:
            break
        if tag.casefold() not in {item.casefold() for item in result}:
            result.append(tag)

    return result[:6]
