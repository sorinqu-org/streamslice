from __future__ import annotations

import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from streamslice.gemini import GeminiClient, GeminiError, extract_json
from streamslice.proxy import configured_proxy_models


class FakeResponse:
    status = 200

    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _client_config(*, retries: int = 1) -> dict:
    return {
        "proxy": {
            "base_url": "http://proxy.test/v1",
            "api_key_env": "TEST_PROXY_API_KEY",
        },
        "runtime": {
            "retries": retries,
            "retry_initial_seconds": 0,
            "retry_max_seconds": 0,
        },
    }


class GeminiClientTests(unittest.TestCase):
    def test_retries_empty_choices_inside_request_budget(self) -> None:
        responses = [
            FakeResponse({"choices": []}),
            FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]}),
        ]
        client = GeminiClient(_client_config(retries=1))
        with (
            patch("streamslice.gemini.api_key", return_value="test-key"),
            patch("streamslice.gemini.urllib.request.urlopen", side_effect=responses) as urlopen,
            patch("streamslice.gemini.time.sleep") as sleep,
        ):
            data, text = client.chat_json(model="test", system="system", prompt="prompt")

        self.assertEqual(data, {"ok": True})
        self.assertEqual(text, '{"ok": true}')
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    def test_invalid_success_responses_fail_after_retry_budget(self) -> None:
        responses = [
            FakeResponse({"choices": []}),
            FakeResponse({"choices": [{"message": {"content": ""}}]}),
        ]
        client = GeminiClient(_client_config(retries=1))
        with (
            patch("streamslice.gemini.api_key", return_value="test-key"),
            patch("streamslice.gemini.urllib.request.urlopen", side_effect=responses) as urlopen,
            patch("streamslice.gemini.time.sleep"),
            self.assertRaisesRegex(GeminiError, "failed after retries"),
        ):
            client.chat_json(model="test", system="system", prompt="prompt")

        self.assertEqual(urlopen.call_count, 2)

    def test_retries_malformed_json_inside_request_budget(self) -> None:
        responses = [
            FakeResponse({"choices": [{"message": {"content": "not json"}}]}),
            FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]}),
        ]
        client = GeminiClient(_client_config(retries=1))
        with (
            patch("streamslice.gemini.api_key", return_value="test-key"),
            patch("streamslice.gemini.urllib.request.urlopen", side_effect=responses) as urlopen,
            patch("streamslice.gemini.time.sleep") as sleep,
        ):
            data, text = client.chat_json(model="test", system="system", prompt="prompt")

        self.assertEqual(data, {"ok": True})
        self.assertEqual(text, '{"ok": true}')
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()


class ExtractJsonTests(unittest.TestCase):
    def test_extracts_json_after_model_preamble(self) -> None:
        self.assertEqual(
            extract_json('Reasoning first. ```json\n{"words": []}\n```'),
            {"words": []},
        )

    def test_extracts_json_without_closing_fence(self) -> None:
        self.assertEqual(
            extract_json('Analysis. ```json\n{"words": [{"text": "да"}]}'),
            {"words": [{"text": "да"}]},
        )

    def test_repairs_invalid_suffix_on_timestamp(self) -> None:
        self.assertEqual(
            extract_json('{"words":[{"start":0.34decay,"end":0.65,"text":"да"}]}'),
            {"words": [{"start": 0.34, "end": 0.65, "text": "да"}]},
        )

    def test_rejects_text_without_json(self) -> None:
        with self.assertRaises(GeminiError):
            extract_json("No structured response")

    def test_prefers_full_object_over_leading_word_array(self) -> None:
        # A standalone word array must not shadow the real object payload.
        text = '```json\n{"words":[[1.0,1.4,"привет",[0.9]],[1.4,1.8,"мир"]]}\n```'
        self.assertEqual(
            extract_json(text),
            {"words": [[1.0, 1.4, "привет", [0.9]], [1.4, 1.8, "мир"]]},
        )

    def test_salvages_words_from_doubled_bracket(self) -> None:
        # gemini-3.6-flash sometimes writes "[[0.9]]" for the confidence, which
        # unbalances the object. The salvage path keeps every intact word.
        text = (
            '```json\n{"words":[[1.0,1.4,"иди",[0.9]],'
            '[1.4,1.8,"сюда",[[0.9]]],[1.8,2.2,"блядь",[0.9]]]}\n```'
        )
        value = extract_json(text)
        self.assertIn("words", value)
        texts = [item[2] for item in value["words"]]
        self.assertIn("иди", texts)
        self.assertIn("блядь", texts)
        self.assertGreaterEqual(len(value["words"]), 2)

    def test_recovers_words_after_stray_suffix(self) -> None:
        # A valid word array followed by a stray "...}" must not collapse to the
        # first inner word object. The container carries no closing, so the scan
        # would otherwise return one word and the clip would lose its captions.
        text = (
            '{"words":[{"start":0.0,"end":0.2,"text":"Ом-то,"},'
            '{"start":0.24,"end":0.56,"text":"кстати,"},'
            '{"start":0.6,"end":0.9,"text":"нет?"}]...}'
        )
        value = extract_json(text)
        self.assertEqual(len(value["words"]), 3)
        self.assertEqual(value["words"][0]["text"], "Ом-то,")

    def test_repairs_missing_comma_between_fields(self) -> None:
        # Two adjacent quotes with no comma ("text":"x""confidence":0.9) break the
        # whole object. Re-inserting the separator restores every word.
        text = (
            '{"words":[{"start":0.0,"end":0.44,"text":"Пидень,""confidence":0.85},'
            '{"start":0.56,"end":0.74,"text":"ну,""confidence":0.95},'
            '{"start":0.74,"end":1.2,"text":"блядь!","confidence":0.93}]}'
        )
        value = extract_json(text)
        self.assertEqual([item["text"] for item in value["words"]], ["Пидень,", "ну,", "блядь!"])


def _openrouter_config(*, retries: int = 1, inline_key: str | None = "or-key") -> dict:
    config = _client_config(retries=retries)
    provider: dict = {
        "base_url": "https://openrouter.test/api/v1",
        "api_key_env": "TEST_OPENROUTER_API_KEY",
    }
    if inline_key is not None:
        provider["api_key"] = inline_key
    config["providers"] = {"openrouter": provider}
    return config


OPENROUTER_MODEL = {"provider": "openrouter", "model": "stealth/ox-alpha"}


class MultiProviderRoutingTests(unittest.TestCase):
    def test_openrouter_model_routes_to_provider_endpoint(self) -> None:
        responses = [FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})]
        client = GeminiClient(_openrouter_config())
        with (
            patch("streamslice.gemini.api_key", return_value="proxy-key"),
            patch(
                "streamslice.gemini.urllib.request.urlopen", side_effect=responses
            ) as urlopen,
            patch("streamslice.gemini.time.sleep"),
        ):
            data, _ = client.chat_json(model=OPENROUTER_MODEL, system="s", prompt="p")

        self.assertEqual(data, {"ok": True})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.test/api/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer or-key")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "stealth/ox-alpha")
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_plain_string_model_still_uses_cliproxy(self) -> None:
        responses = [FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})]
        client = GeminiClient(_openrouter_config())
        with (
            patch("streamslice.gemini.api_key", return_value="proxy-key"),
            patch(
                "streamslice.gemini.urllib.request.urlopen", side_effect=responses
            ) as urlopen,
            patch("streamslice.gemini.time.sleep"),
        ):
            client.chat_json(model="gemini-3.7-flash-high", system="s", prompt="p")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://proxy.test/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer proxy-key")

    def test_env_var_overrides_inline_openrouter_key(self) -> None:
        responses = [FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})]
        client = GeminiClient(_openrouter_config())
        with (
            patch.dict(os.environ, {"TEST_OPENROUTER_API_KEY": "env-key"}),
            patch("streamslice.gemini.api_key", return_value="proxy-key"),
            patch(
                "streamslice.gemini.urllib.request.urlopen", side_effect=responses
            ) as urlopen,
            patch("streamslice.gemini.time.sleep"),
        ):
            client.chat_json(model=OPENROUTER_MODEL, system="s", prompt="p")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer env-key")

    def test_missing_openrouter_key_raises(self) -> None:
        client = GeminiClient(_openrouter_config(inline_key=None))
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("streamslice.gemini.api_key", return_value="proxy-key"),
            self.assertRaisesRegex(GeminiError, "TEST_OPENROUTER_API_KEY"),
        ):
            client.chat_json(model=OPENROUTER_MODEL, system="s", prompt="p")

    def test_falls_back_from_json_mode_on_400(self) -> None:
        error = urllib.error.HTTPError(
            url="https://openrouter.test/api/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"response_format is not supported"}}'),
        )
        responses = [
            error,
            FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]}),
        ]
        client = GeminiClient(_openrouter_config(retries=1))
        with (
            patch("streamslice.gemini.api_key", return_value="proxy-key"),
            patch(
                "streamslice.gemini.urllib.request.urlopen", side_effect=responses
            ) as urlopen,
            patch("streamslice.gemini.time.sleep"),
        ):
            data, _ = client.chat_json(model=OPENROUTER_MODEL, system="s", prompt="p")

        self.assertEqual(data, {"ok": True})
        first_body = json.loads(urlopen.call_args_list[0].args[0].data)
        second_body = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertIn("response_format", first_body)
        self.assertNotIn("response_format", second_body)

    def test_json_mode_disabled_by_provider_setting(self) -> None:
        config = _openrouter_config()
        config["providers"]["openrouter"]["json_mode"] = False
        responses = [FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})]
        client = GeminiClient(config)
        with (
            patch("streamslice.gemini.api_key", return_value="proxy-key"),
            patch(
                "streamslice.gemini.urllib.request.urlopen", side_effect=responses
            ) as urlopen,
            patch("streamslice.gemini.time.sleep"),
        ):
            client.chat_json(model=OPENROUTER_MODEL, system="s", prompt="p")

        body = json.loads(urlopen.call_args.args[0].data)
        self.assertNotIn("response_format", body)


class EnsureProxyRoutingTests(unittest.TestCase):
    def test_only_cliproxy_models_are_validated(self) -> None:
        config = {
            "models": {
                "transcription": "gemini-3.7-flash-high",
                "analysis": {"provider": "openrouter", "model": "stealth/ox-alpha"},
                "curator": {"provider": "cliproxy", "model": "gemini-3.7-flash"},
            }
        }
        self.assertEqual(
            configured_proxy_models(config),
            {"gemini-3.7-flash-high", "gemini-3.7-flash"},
        )

    def test_empty_models_validate_nothing(self) -> None:
        config = {
            "models": {
                "analysis": {"provider": "openrouter", "model": "stealth/ox-alpha"},
            }
        }
        self.assertEqual(configured_proxy_models(config), set())
