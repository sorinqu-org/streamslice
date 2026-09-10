from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamslice.context_analysis import review_candidate_context
from streamslice.discovery import discover_candidates
from streamslice.gemini import GeminiError
from streamslice.models import Candidate


def _candidate(start: float, reason: str) -> Candidate:
    return Candidate(
        start_time=start,
        end_time=start + 30,
        highlight_reason=reason,
        emotion_score=8,
        quality_score=8,
        context_score=8,
    )


class SelectiveContextClient:
    def __init__(self, *, fail_all: bool = False) -> None:
        self.fail_all = fail_all

    def chat_json(self, **kwargs: object) -> tuple[dict, str]:
        prompt = str(kwargs["prompt"])
        if self.fail_all or "Диапазон: 0.000" in prompt:
            raise GeminiError("temporary empty completion")
        data = {
            "self_contained": True,
            "context_score": 8.5,
            "context_summary": "завязка, событие и итог понятны",
            "hook_score": 8,
            "payoff_score": 8,
            "shareability_score": 8,
            "visual_clarity_score": 8,
            "quality_score": 8,
            "quality_summary": "есть ясный хук и развязка",
            "evidence": ["завязка", "итог"],
        }
        return data, json.dumps(data, ensure_ascii=False)


class SelectiveDiscoveryClient:
    def __init__(self, *, fail_all: bool = False) -> None:
        self.fail_all = fail_all

    def chat_json(self, **kwargs: object) -> tuple[dict, str]:
        prompt = str(kwargs["prompt"])
        if self.fail_all or "0.000–60.000" in prompt:
            raise GeminiError("temporary empty completion")
        data = {
            "candidates": [
                {
                    "start_time": 65,
                    "end_time": 95,
                    "highlight_reason": "понятный момент",
                    "emotion_score": 8,
                    "context_score": 8,
                    "context_summary": "завязка, событие и итог",
                    "self_contained": True,
                    "quality_score": 8,
                    "quality_summary": "сильная развязка",
                }
            ]
        }
        return data, json.dumps(data, ensure_ascii=False)


class ParallelAnalysisFailureTests(unittest.TestCase):
    def test_context_review_skips_one_model_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            config = {
                "models": {"curator": "stub"},
                "selection": {
                    "strict_context": True,
                    "context_review_limit": 2,
                    "context_parallel_requests": 2,
                    "min_context_score": 7,
                    "min_quality_score": 7.5,
                },
            }
            with patch("streamslice.context_analysis._extract_context_frames", return_value=[]):
                approved = review_candidate_context(
                    Path("unused.mp4"),
                    [],
                    [_candidate(0, "fails"), _candidate(40, "works")],
                    work_dir=work_dir,
                    config=config,
                    client=SelectiveContextClient(),
                )

            self.assertEqual([item.highlight_reason for item in approved], ["works"])
            cache = json.loads((work_dir / "context-review.json").read_text(encoding="utf-8"))
            self.assertEqual(len(cache["failures"]), 1)

    def test_context_review_raises_when_every_candidate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "models": {"curator": "stub"},
                "selection": {
                    "strict_context": True,
                    "context_review_limit": 2,
                    "context_parallel_requests": 2,
                    "min_context_score": 7,
                    "min_quality_score": 7.5,
                },
            }
            with (
                patch("streamslice.context_analysis._extract_context_frames", return_value=[]),
                self.assertRaisesRegex(GeminiError, "failed for all"),
            ):
                review_candidate_context(
                    Path("unused.mp4"),
                    [],
                    [_candidate(0, "first"), _candidate(40, "second")],
                    work_dir=Path(temporary),
                    config=config,
                    client=SelectiveContextClient(fail_all=True),
                )

    def test_discovery_skips_one_model_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "models": {"analysis": "stub"},
                "selection": {
                    "analysis_window_seconds": 60,
                    "analysis_overlap_seconds": 0,
                    "parallel_requests": 2,
                    "min_duration_seconds": 20,
                    "max_duration_seconds": 90,
                },
            }
            candidates = discover_candidates(
                [],
                [],
                duration=120,
                work_dir=Path(temporary),
                config=config,
                client=SelectiveDiscoveryClient(),
            )

            self.assertEqual([item.highlight_reason for item in candidates], ["понятный момент"])
            cache = json.loads((Path(temporary) / "candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(len(cache["failures"]), 1)

    def test_discovery_raises_when_every_window_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "models": {"analysis": "stub"},
                "selection": {
                    "analysis_window_seconds": 60,
                    "analysis_overlap_seconds": 0,
                    "parallel_requests": 2,
                    "min_duration_seconds": 20,
                    "max_duration_seconds": 90,
                },
            }
            with self.assertRaisesRegex(GeminiError, "failed for all"):
                discover_candidates(
                    [],
                    [],
                    duration=120,
                    work_dir=Path(temporary),
                    config=config,
                    client=SelectiveDiscoveryClient(fail_all=True),
                )
