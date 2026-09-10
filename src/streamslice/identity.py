from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .gemini import GeminiClient

LOGGER = logging.getLogger(__name__)

IDENTITY_VERSION = 2


def resolve_creator_identity(
    source: Path,
    *,
    config: dict[str, Any],
    client: GeminiClient,
    work_dir: Path,
) -> dict[str, Any]:
    cache_path = work_dir / "creator-profile.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("identity_version") == IDENTITY_VERSION:
            return dict(payload["identity"])

    sync_root = Path(config["sync"]["local_dir"]).expanduser().resolve()
    try:
        relative = source.resolve().relative_to(sync_root)
        streamer_key = relative.parts[0]
    except ValueError:
        streamer_key = source.parent.parent.name
    profiles = config.get("identity", {}).get("profiles", {})
    configured = profiles.get(streamer_key, {}) if isinstance(profiles, dict) else {}
    if not isinstance(configured, dict):
        configured = {}

    # The model normalizes the configured public facts, but it is not allowed
    # to invent a name. Known profiles therefore remain deterministic offline.
    identity = {
        "source_key": streamer_key,
        "twitch_login": str(configured.get("twitch_login", streamer_key)),
        "display_name": str(configured.get("display_name", streamer_key)),
        "aliases": list(configured.get("aliases", [streamer_key])),
        "preferred_mentions": list(
            configured.get("preferred_mentions", configured.get("aliases", [streamer_key]))
        ),
        "real_name": str(configured.get("real_name", "")),
        "character_lore": str(configured.get("character_lore", "")),
        "pronoun_hint": str(configured.get("pronoun_hint", "")),
        "source_url": str(configured.get("source_url", "")),
        "confidence": 1.0 if configured else 0.5,
    }

    if config.get("identity", {}).get("model_normalize", True):
        prompt = f"""Определи, как корректно называть автора ролика в заголовке и описании.
Источник: {streamer_key}
Проверенные настройки профиля:
{json.dumps(identity, ensure_ascii=False)}

Верни только JSON с теми же полями. Не придумывай новые факты и не меняй
display_name на слово «стример». Если профиль неизвестен, оставь display_name
равным source_key. Для t2x2 используй «t2x2», для stintik — «stint».
"""
        try:
            data, raw = client.chat_json(
                model=config["models"]["metadata"],
                system="Ты редактор профилей авторов. Нормализуй только заданные факты.",
                prompt=prompt,
                timeout=120,
                max_tokens=1024,
                temperature=0,
            )
            if isinstance(data, dict):
                display_name = str(data.get("display_name", "")).strip()
                allowed_names = {
                    identity["display_name"].casefold(),
                    *(str(item).casefold() for item in identity["aliases"]),
                }
                if (
                    display_name
                    and display_name.casefold() in allowed_names
                    and not re.search(r"\bстример\b", display_name, re.IGNORECASE)
                ):
                    identity["display_name"] = display_name
                aliases = data.get("aliases")
                if isinstance(aliases, list) and aliases:
                    normalized_aliases = [
                        str(item).strip()
                        for item in aliases
                        if str(item).strip().casefold() in allowed_names
                    ]
                    if normalized_aliases:
                        identity["aliases"] = normalized_aliases
                mentions = data.get("preferred_mentions")
                if isinstance(mentions, list) and mentions:
                    allowed_mentions = {
                        str(item).casefold() for item in identity["preferred_mentions"]
                    }
                    normalized_mentions = [
                        str(item).strip()
                        for item in mentions
                        if str(item).strip().casefold() in allowed_mentions
                    ]
                    if normalized_mentions:
                        identity["preferred_mentions"] = normalized_mentions
                identity["model_raw"] = raw
        except Exception as exc:
            # Best-effort model normalization: the deterministic profile mapping
            # above is always a safe fallback when the proxy is busy or errors.
            LOGGER.debug("Creator identity normalization failed: %s", exc, exc_info=True)

    cache_path.write_text(
        json.dumps(
            {"identity_version": IDENTITY_VERSION, "identity": identity},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return identity
