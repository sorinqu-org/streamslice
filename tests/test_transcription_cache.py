from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from streamslice.media import fingerprint
from streamslice.transcription import transcribe


class StubClient:
    """Answers with a fixed payload and counts how often it was asked."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def chat_json(self, **_: object) -> tuple[dict, str]:
        self.calls += 1
        return self.payload, json.dumps(self.payload, ensure_ascii=False)


def _config() -> dict:
    return {
        "models": {"transcription": "stub-model"},
        "transcription": {
            "segment_seconds": 600,
            "overlap_seconds": 0,
            "sample_rate": 16000,
            "audio_bitrate": "32k",
            "language": "ru",
            "parallel_requests": 1,
            "request_timeout_seconds": 30,
        },
    }


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required to build the sample clip")
class EmptyTranscriptCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.work_dir = root / "work"
        self.work_dir.mkdir()
        self.source = root / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
                "-shortest", "-y", str(self.source),
            ],
            check=True,
        )

    def _transcribe(self, client: StubClient) -> list:
        return transcribe(
            self.source,
            duration=2.0,
            work_dir=self.work_dir,
            config=_config(),
            client=client,
        )

    def test_valid_empty_transcript_is_cached_as_silence(self) -> None:
        client = StubClient({"words": []})
        self.assertEqual(self._transcribe(client), [])
        payload = json.loads((self.work_dir / "transcript.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["transcript_status"], "silent")
        self.assertEqual(payload["words"], [])

    def test_valid_empty_answer_is_reused_without_new_request(self) -> None:
        client = StubClient({"words": []})
        self.assertEqual(self._transcribe(client), [])
        calls_after_first = client.calls
        self.assertEqual(self._transcribe(client), [])
        self.assertEqual(client.calls, calls_after_first)

    def test_stale_empty_transcript_is_ignored(self) -> None:
        # Older empty cache files have no explicit silent status and remain suspect.
        (self.work_dir / "transcript.json").write_text(
            json.dumps(
                {"source_fingerprint": fingerprint(self.source), "words": []},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        client = StubClient({"words": [{"start": 0.1, "end": 0.5, "text": "привет"}]})
        words = self._transcribe(client)
        self.assertEqual([item.text for item in words], ["привет"])
        self.assertGreater(client.calls, 0)

    def test_invalid_segment_cache_is_not_treated_as_silence(self) -> None:
        silent_client = StubClient({"words": []})
        self.assertEqual(self._transcribe(silent_client), [])
        (self.work_dir / "transcript.json").unlink()
        segment_cache = next((self.work_dir / "transcription-raw").glob("*/segment-000.json"))
        cached = json.loads(segment_cache.read_text(encoding="utf-8"))
        cached["parsed"] = {}
        segment_cache.write_text(json.dumps(cached), encoding="utf-8")

        speaking_client = StubClient(
            {"words": [{"start": 0.1, "end": 0.5, "text": "привет"}]}
        )
        words = self._transcribe(speaking_client)

        self.assertEqual([item.text for item in words], ["привет"])
        self.assertGreater(speaking_client.calls, 0)
        payload = json.loads((self.work_dir / "transcript.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["transcript_status"], "complete")

    def test_filled_transcript_is_cached_and_reused(self) -> None:
        client = StubClient({"words": [{"start": 0.1, "end": 0.5, "text": "привет"}]})
        first = self._transcribe(client)
        calls_after_first = client.calls
        second = self._transcribe(client)
        self.assertEqual([item.text for item in first], ["привет"])
        self.assertEqual([item.text for item in second], ["привет"])
        self.assertEqual(client.calls, calls_after_first)
