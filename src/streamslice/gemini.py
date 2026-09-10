from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .proxy import api_key

LOGGER = logging.getLogger(__name__)

DEFAULT_PROVIDER = "cliproxy"


class GeminiError(RuntimeError):
    pass


def _provider_settings(config: dict[str, Any], provider_name: str) -> dict[str, Any]:
    providers = config.get("providers") or {}
    settings = dict(providers.get(provider_name) or {})
    if provider_name == DEFAULT_PROVIDER:
        proxy = config["proxy"]
        settings.setdefault("base_url", proxy["base_url"])
        settings.setdefault("api_key_env", proxy["api_key_env"])
    return settings


class GeminiClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        runtime = config["runtime"]
        self.retries = int(runtime["retries"])
        self.initial_delay = float(runtime["retry_initial_seconds"])
        self.max_delay = float(runtime["retry_max_seconds"])

    def _resolve_target(
        self, model: Any
    ) -> tuple[str, dict[str, str], bool, dict[str, Any]]:
        """Resolve a ``models.*`` config entry into endpoint, headers, JSON mode.

        An entry is either a plain string (routed to the default CLIProxyAPI
        provider, exactly like before) or a mapping such as::

            {provider: openrouter, model: stealth/ox-alpha}
        """
        if isinstance(model, dict):
            provider_name = str(model.get("provider") or DEFAULT_PROVIDER).strip()
            model_name = str(model.get("model") or "").strip()
            if not model_name:
                raise GeminiError(f"Model entry must define 'model': {model!r}")
        else:
            provider_name = DEFAULT_PROVIDER
            model_name = str(model).strip()
        settings = _provider_settings(self.config, provider_name)
        base_url = str(settings.get("base_url") or "").strip()
        if not base_url:
            raise GeminiError(
                f"Provider '{provider_name}' has no 'base_url' in config providers"
            )
        endpoint = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if provider_name == DEFAULT_PROVIDER:
            headers["Authorization"] = f"Bearer {api_key(self.config)}"
        else:
            env_name = str(settings.get("api_key_env") or "").strip()
            secret = os.environ.get(env_name, "").strip() if env_name else ""
            if not secret:
                secret = str(settings.get("api_key") or "").strip()
            if not secret:
                hint = f"Set {env_name} or providers.{provider_name}.api_key" if env_name else (
                    f"Set providers.{provider_name}.api_key"
                )
                raise GeminiError(f"{hint} before running StreamSlice")
            headers["Authorization"] = f"Bearer {secret}"
            referer = str(settings.get("http_referer") or "").strip()
            title = str(settings.get("title") or "").strip()
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
        json_mode = bool(settings.get("json_mode", True))
        return endpoint, headers, json_mode, settings

    def chat_json(
        self,
        *,
        model: Any,
        system: str,
        prompt: str,
        timeout: float = 180,
        audio_path: Path | None = None,
        image_paths: list[Path] | None = None,
        max_tokens: int = 16384,
        temperature: float = 0.2,
    ) -> tuple[Any, str]:
        endpoint, headers, json_mode, provider_settings = self._resolve_target(model)
        model_name = self._model_name(model)
        # Reasoning models spend completion tokens on hidden reasoning before
        # the visible content; a small call-site budget would leave the answer
        # truncated (content=None). Providers can declare a floor instead of
        # touching every call site.
        floor = int(provider_settings.get("min_max_tokens") or 0)
        if floor > 0:
            max_tokens = max(int(max_tokens), floor)
        content: str | list[dict[str, Any]]
        if audio_path:
            audio = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio, "format": "mp3"}},
            ]
        elif image_paths:
            content = [{"type": "text", "text": prompt}]
            for image_path in image_paths:
                suffix = image_path.suffix.casefold()
                mime = "image/png" if suffix == ".png" else "image/jpeg"
                encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded_image}"},
                    }
                )
        else:
            content = prompt
        body: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if not json_mode:
            body.pop("response_format")
        return self._request(endpoint, headers, body, timeout)

    @staticmethod
    def _model_name(model: Any) -> str:
        if isinstance(model, dict):
            return str(model.get("model") or "").strip()
        return str(model).strip()

    def _request(
        self,
        endpoint: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout: float,
    ) -> tuple[Any, str]:
        delay = self.initial_delay
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                endpoint,
                data=encoded,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                    if response.status >= 400:
                        raise GeminiError(f"HTTP {response.status}: {payload[-2000:]!r}")
                    raw = json.loads(payload)
                    text = _response_text(raw)
                    return extract_json(text), text
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = GeminiError(f"HTTP {exc.code}: {detail[-2000:]}")
                # Some OpenRouter models reject strict JSON mode; fall back to
                # plain prompting (extract_json still salvages the payload).
                if (
                    exc.code in (400, 404, 422)
                    and "response_format" in body
                    and (
                        "response_format" in detail
                        or "json_object" in detail
                        or "json schema" in detail.lower()
                    )
                ):
                    LOGGER.warning(
                        "Provider rejected response_format=json_object; retrying without JSON mode"
                    )
                    body.pop("response_format")
                    continue
                retryable = exc.code in (408, 409, 425, 429) or exc.code >= 500
                if not retryable:
                    raise last_error from exc
            except (
                GeminiError,
                OSError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                last_error = exc
            if attempt >= self.retries:
                break
            sleep_for = min(self.max_delay, delay) * random.uniform(0.8, 1.2)
            LOGGER.warning(
                "Gemini request failed, retry %s in %.1fs: %s", attempt + 1, sleep_for, last_error
            )
            time.sleep(sleep_for)
            delay *= 2
        raise GeminiError(f"Gemini request failed after retries: {last_error}")


def _response_text(raw: Any) -> str:
    try:
        choices = raw["choices"]
        if not choices:
            raise GeminiError(f"Unexpected proxy response: {raw}")
        text = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"Unexpected proxy response: {raw}") from exc
    if not isinstance(text, str) or not text.strip():
        raise GeminiError(f"Unexpected proxy response: {raw}")
    return text


def _decode_full(candidate: str, decoder: json.JSONDecoder) -> Any | None:
    for index, character in enumerate(candidate):
        if character not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        remainder = candidate[index + end:].strip(" \t\r\n`")
        # Accept a value that consumes essentially all of the text, or the real
        # container object (the one carrying "words"). The "words" guard matters:
        # when the container itself is malformed (e.g. a stray "...}" suffix), the
        # scan would otherwise stop on the first inner word object — a lone dict
        # that parses cleanly — and the clip would silently keep a single word.
        if not remainder:
            return value
        if isinstance(value, dict) and "words" in value:
            return value
    return None


def _insert_missing_field_commas(candidate: str) -> str:
    # gemini-3.6-flash sometimes drops the comma between two fields, e.g.
    # "text":"Пидень,""confidence":0.85 — two adjacent quotes with no separator.
    # That single defect makes the whole object, and thus the segment, unparseable
    # and the clip loses every caption. Re-insert the comma before a known key.
    return re.sub(
        r'""(?=(?:start|end|text|word|confidence)"\s*:)',
        '","',
        candidate,
    )


def _next_element_start(candidate: str, index: int) -> int:
    # Resume salvage at the next word element, whether words are objects ({...})
    # or compact arrays ([...]).
    return min(
        (
            pos
            for pos in (candidate.find("{", index + 1), candidate.find("[", index + 1))
            if pos != -1
        ),
        default=-1,
    )


def _salvage_words_array(candidate: str) -> list[Any] | None:
    # When the container is beyond repair, recover by decoding the "words" array
    # one element at a time and keeping every entry that still parses. This handles
    # a stray suffix after the array as well as a single corrupt element mid-list.
    match = re.search(r'"words"\s*:\s*\[', candidate)
    if not match:
        return None
    decoder = json.JSONDecoder()
    items: list[Any] = []
    index = match.end()
    length = len(candidate)
    while index < length:
        while index < length and candidate[index] in " \t\r\n,":
            index += 1
        if index >= length or candidate[index] == "]":
            break
        try:
            value, index = decoder.raw_decode(candidate, index)
        except json.JSONDecodeError:
            following = _next_element_start(candidate, index)
            if following == -1:
                break
            index = following
            continue
        items.append(value)
    return items or None


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    candidates = [cleaned]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE
        )
    )
    decoder = json.JSONDecoder()
    for candidate in candidates:
        value = _decode_full(candidate, decoder)
        if value is not None:
            return value
        repaired = re.sub(
            r'((?:"start"|"end")\s*:\s*-?\d+(?:\.\d+)?)[A-Za-z_]+(?=\s*[,}])',
            r"\1",
            candidate,
        )
        repaired = _insert_missing_field_commas(repaired)
        if repaired != candidate:
            value = _decode_full(repaired, decoder)
            if value is not None:
                LOGGER.warning("Recovered JSON after repairing malformed fields")
                return value
        salvaged = _salvage_words_array(candidate)
        if not salvaged:
            salvaged = _salvage_words_array(_insert_missing_field_commas(candidate))
        if salvaged:
            LOGGER.warning("Recovered %s words from malformed transcript JSON", len(salvaged))
            return {"words": salvaged}
    raise GeminiError(f"Model did not return usable JSON: {text[:500]}")
