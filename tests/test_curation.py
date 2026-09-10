import unittest

from streamslice.curation import _editorial_score, _valid
from streamslice.models import Candidate


class CurationTests(unittest.TestCase):
    def test_duration_and_overlap(self) -> None:
        first = Candidate(10, 40, "a", 8)
        second = Candidate(41, 70, "b", 7)
        overlap = Candidate(30, 60, "c", 9)
        self.assertTrue(_valid(first, [], 100))
        self.assertTrue(_valid(second, [first], 100))
        self.assertFalse(_valid(overlap, [first], 100))

    def test_strict_context_rejects_unexplained_scene(self) -> None:
        settings = {"strict_context": True, "min_context_score": 7.0}
        unclear = Candidate(10, 40, "громкий крик", 9)
        clear = Candidate(
            41,
            70,
            "понятная сцена",
            8,
            context_score=8.0,
            context_summary="завязка → событие → итог",
            self_contained=True,
            quality_score=8.0,
            quality_summary="сильный и ясный момент",
        )
        self.assertFalse(_valid(unclear, [], 100, settings))
        self.assertTrue(_valid(clear, [], 100, settings))

    def test_editorial_score_prefers_quality_over_raw_loudness(self) -> None:
        loud_but_weak = Candidate(
            10,
            40,
            "только крик",
            10,
            context_score=7,
            quality_score=5,
        )
        strong_story = Candidate(
            41,
            75,
            "завязка и сильная развязка",
            7,
            context_score=9,
            quality_score=9,
        )
        self.assertGreater(_editorial_score(strong_story), _editorial_score(loud_but_weak))
