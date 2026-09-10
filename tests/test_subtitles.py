import itertools
import json
import tempfile
import unittest
from pathlib import Path

from streamslice.models import Candidate, Word
from streamslice.subtitles import build_subtitles, build_subtitles_ass
from streamslice.transcription import _parse_words


class SubtitleTests(unittest.TestCase):
    def test_clip_local_timestamps_and_files(self) -> None:
        words = [Word(9, 10.2, "до"), Word(10.1, 10.5, "привет"), Word(11, 11.4, "мир")]
        candidate = Candidate(10, 40, "test", 8)
        config = {
            "subtitles": {
                "font_family": "Montserrat",
                "font_size": 92,
                "outline_width": 14,
                "words_per_group": 1,
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            values = build_subtitles(words, candidate, output, config)
            self.assertEqual(values[0]["start"], 0.0)
            self.assertTrue((output / "subtitles.json").is_file())
            self.assertTrue((output / "subtitles.ass").is_file())
            ass_content = (output / "subtitles.ass").read_text(encoding="utf-8")
            self.assertIn(
                "Style: TikTok,Montserrat,92,&H0015E8FF,&H00000000,&H00000000,&H80000000,"
                "-1,0,0,0,100,100,0,0,1,14,4,2,70,70,240,1",
                ass_content,
            )
            self.assertIn(r"{\fscx115\fscy115\t(0,80,\fscx100\fscy100)}", ass_content)

    def test_title_hook_header_and_build_subtitles_ass(self) -> None:
        words = [Word(0.0, 1.0, "хайлайт")]
        candidate = Candidate(0.0, 10.0, "test", 8)
        config = {
            "subtitles": {
                "font_family": "Montserrat",
                "font_size": 92,
                "outline_width": 14,
                "words_per_group": 1,
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            ass_content = build_subtitles_ass(
                words, candidate, output, config, title="Тоха жестко затащил"
            )
            self.assertIn(
                "Style: Title,Montserrat,48,&H0015E8FF,&H00000000,&H00000000,&H80000000,"
                "-1,0,0,0,100,100,0,0,1,10,4,8,30,30,710,1",
                ass_content,
            )
            self.assertIn(
                "Dialogue: 0,0:00:00.00,0:00:10.00,Title,,0,0,0,,Тоха жестко затащил",
                ass_content,
            )
            self.assertIn(
                r"Dialogue: 1,0:00:00.00,0:00:01.00,TikTok,,0,0,0,,"
                r"{\fscx115\fscy115\t(0,80,\fscx100\fscy100)}хайлайт",
                ass_content,
            )

    def test_profanity_is_masked_in_json_and_ass(self) -> None:
        words = [Word(0.0, 0.6, "блять")]
        candidate = Candidate(0, 2, "test", 8)
        config = {
            "subtitles": {
                "font_family": "Montserrat",
                "font_size": 92,
                "outline_width": 14,
                "words_per_group": 1,
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            build_subtitles(words, candidate, output, config, title="блять заголовок")
            payload = json.loads((output / "subtitles.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["words"][0]["text"], "б***ь")
            ass_content = (output / "subtitles.ass").read_text(encoding="utf-8")
            self.assertIn("б***ь", ass_content)
            self.assertIn("б***ь заголовок", ass_content)


class PhraseSplitTests(unittest.TestCase):
    def test_single_word_entry_is_kept_as_is(self) -> None:
        payload = {"words": [{"start": 1.0, "end": 1.4, "text": "привет", "confidence": 0.9}]}
        words = _parse_words(payload, 0.0, 60.0)
        self.assertEqual([item.text for item in words], ["привет"])
        self.assertAlmostEqual(words[0].start, 1.0)
        self.assertAlmostEqual(words[0].end, 1.4)

    def test_phrase_entry_is_split_into_words(self) -> None:
        payload = {"words": [{"start": 2.44, "end": 2.76, "text": "А как ты догадаешься"}]}
        words = _parse_words(payload, 0.0, 60.0)
        self.assertEqual([item.text for item in words], ["А", "как", "ты", "догадаешься"])
        # Timings stay ordered, non-overlapping and readable.
        for previous, current in itertools.pairwise(words):
            self.assertLessEqual(previous.end, current.start)
        for item in words:
            self.assertGreater(item.end, item.start)
        self.assertGreater(words[-1].end - words[0].start, 0.32)

    def test_array_shaped_words_are_parsed(self) -> None:
        payload = {
            "words": [
                [1.66, 2.0, "Понятно.", [0.9]],
                [2.04, 2.56, "Я думаю,", 0.9],
                [2.56, 2.78, "что вы"],
            ]
        }
        words = _parse_words(payload, 0.0, 60.0)
        self.assertEqual([item.text for item in words], ["Понятно.", "Я", "думаю,", "что", "вы"])
        self.assertAlmostEqual(words[0].start, 1.66)
        self.assertEqual(words[0].confidence, 0.9)

    def test_bare_array_payload_is_parsed(self) -> None:
        payload = [[0.5, 0.9, "привет", [0.8]]]
        words = _parse_words(payload, 0.0, 60.0)
        self.assertEqual([item.text for item in words], ["привет"])

    def test_split_never_overlaps_the_next_entry(self) -> None:
        payload = {
            "words": [
                {"start": 1.0, "end": 1.2, "text": "первая длинная фраза тут"},
                {"start": 1.5, "end": 1.9, "text": "потом"},
            ]
        }
        words = _parse_words(payload, 0.0, 60.0)
        following = next(item for item in words if item.text == "потом")
        for item in words:
            if item.text == "потом":
                continue
            self.assertLessEqual(item.end, following.start)
