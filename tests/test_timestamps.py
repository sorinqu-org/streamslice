import itertools
import unittest

from streamslice.models import Word
from streamslice.transcription import align_words_to_audio_activity, normalize_words


class TimestampTests(unittest.TestCase):
    def test_overlap_is_monotonic(self) -> None:
        words = [
            Word(0.0, 0.5, "раз"),
            Word(0.45, 0.8, "два"),
            Word(0.7, 1.0, "три"),
        ]
        normalized = normalize_words(words, 2.0)
        self.assertEqual(len(normalized), 3)
        for previous, current in itertools.pairwise(normalized):
            self.assertGreaterEqual(current.start, previous.end)
            self.assertGreater(current.end, current.start)

    def test_duplicate_overlap_word_is_removed(self) -> None:
        words = [Word(1.0, 1.5, "тест"), Word(1.2, 1.6, "тест")]
        self.assertEqual(len(normalize_words(words, 3.0)), 1)

    def test_alignment_repairs_large_global_lead_into_silence(self) -> None:
        words = [
            Word(1.0, 1.4, "one"),
            Word(2.0, 2.4, "two"),
            Word(3.0, 3.4, "three"),
        ]
        activity = [(3.0, 3.5), (4.0, 4.5), (5.0, 5.5), (7.0, 7.1)]

        aligned, evidence = align_words_to_audio_activity(words, activity, 8.0)

        self.assertTrue(evidence["applied"])
        self.assertAlmostEqual(evidence["offset_seconds"], 2.0, places=2)
        self.assertEqual([round(item.start, 2) for item in aligned], [3.0, 4.0, 5.0])

    def test_alignment_does_not_shift_normal_captions(self) -> None:
        words = [Word(1.0, 1.4, "one"), Word(2.0, 2.4, "two")]
        activity = [(0.8, 1.5), (1.8, 2.5)]

        aligned, evidence = align_words_to_audio_activity(words, activity, 5.0)

        self.assertFalse(evidence["applied"])
        self.assertIs(aligned, words)

    def test_alignment_rejects_ambiguous_game_audio(self) -> None:
        words = [Word(1.0, 1.5, "one"), Word(2.0, 2.5, "two")]
        activity = [(0.0, 5.0)]

        aligned, evidence = align_words_to_audio_activity(words, activity, 5.0)

        self.assertFalse(evidence["applied"])
        self.assertIs(aligned, words)

    def test_alignment_repairs_two_locally_compressed_phrases(self) -> None:
        words = [
            Word(0.2, 0.6, "a"),
            Word(0.7, 1.0, "b"),
            Word(1.2, 1.5, "c"),
            Word(2.0, 2.4, "d"),
            Word(3.8, 4.1, "earlier"),
            Word(5.1, 5.3, "bad"),
            Word(5.3, 5.7, "phrase"),
            Word(7.5, 7.8, "next"),
            Word(7.8, 8.2, "phrase"),
            Word(15.0, 15.4, "good"),
        ]
        activity = [(0.0, 3.0), (3.7, 4.5), (7.94, 9.66), (12.05, 14.85)]

        aligned, evidence = align_words_to_audio_activity(words, activity, 16.0)

        self.assertTrue(evidence["applied"])
        self.assertEqual(len(evidence["phrase_repairs"]), 2)
        self.assertAlmostEqual(aligned[5].start, 7.94, places=2)
        self.assertAlmostEqual(aligned[7].start, 12.05, places=2)
