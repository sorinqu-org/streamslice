import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from streamslice.session_batcher import (
    extract_chunk_number,
    extract_streamer_name,
    group_jobs_into_batches,
    parse_timestamp_from_string,
)
from streamslice.super_curator import _gather_clip_info, select_top_clips


class SessionBatcherTests(unittest.TestCase):
    def test_parse_timestamp_from_string(self) -> None:
        dt = parse_timestamp_from_string("t2x2_2026-08-25_14-01-58_chunk_1_b5d058cb3b9dc6fe")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 8)
        self.assertEqual(dt.day, 25)
        self.assertEqual(dt.hour, 14)
        self.assertEqual(dt.minute, 1)
        self.assertEqual(dt.second, 58)

    def test_extract_streamer_and_chunk(self) -> None:
        name = "mazellovvv_2026-08-24_16-01-25_chunk_5_809415e46ac9561e"
        streamer = extract_streamer_name(name)
        chunk = extract_chunk_number(name)
        self.assertEqual(streamer, "mazellovvv")
        self.assertEqual(chunk, 5)

    def test_group_jobs_into_batches_time_proximity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # Create 3 jobs close to each other (10 minutes apart) and 1 job 2 hours later
            job1 = root / "t2x2_2026-08-25_14-00-00_chunk_1_aaa"
            job2 = root / "stintik_2026-08-25_14-10-00_chunk_1_bbb"
            job3 = root / "t2x2_2026-08-25_14-25-00_chunk_2_ccc"
            job4 = root / "mazellovvv_2026-08-25_17-00-00_chunk_1_ddd"

            for j in (job1, job2, job3, job4):
                j.mkdir(parents=True)

            batches = group_jobs_into_batches([job4, job1, job3, job2], batch_window_seconds=1800.0)
            self.assertEqual(len(batches), 2)
            # First batch should have job1, job2, job3
            self.assertEqual(len(batches[0]), 3)
            self.assertEqual(batches[0], [job1, job2, job3])
            # Second batch should have job4
            self.assertEqual(len(batches[1]), 1)
            self.assertEqual(batches[1], [job4])

    def test_group_jobs_with_created_at_unix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job1 = root / "job_a"
            job2 = root / "job_b"
            job1.mkdir()
            job2.mkdir()

            (job1 / "render-job.json").write_text(
                json.dumps({"created_at_unix": 1000.0, "source": "/path/t2x2/chunk_1.mp4"}),
                encoding="utf-8",
            )
            (job2 / "render-job.json").write_text(
                json.dumps({"created_at_unix": 2000.0, "source": "/path/stintik/chunk_1.mp4"}),
                encoding="utf-8",
            )

            batches = group_jobs_into_batches([job1, job2], batch_window_seconds=1800.0)
            self.assertEqual(len(batches), 1)
            self.assertEqual(batches[0], [job1, job2])


class SuperCuratorTests(unittest.TestCase):
    def test_gather_clip_info(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "t2x2_job"
            clip_dir = job_dir / "clip-01"
            clip_dir.mkdir(parents=True)

            (clip_dir / "selection.json").write_text(
                json.dumps(
                    {
                        "start_time": 10.0,
                        "end_time": 50.0,
                        "emotion_score": 9.0,
                        "quality_score": 8.5,
                        "context_summary": "Summary of context",
                        "highlight_reason": "Very funny moment",
                    }
                ),
                encoding="utf-8",
            )
            (clip_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "title": "ХУК ЗАГОЛОВОК",
                        "creator": {
                            "display_name": "t2x2",
                            "character_lore": "Тоха стример",
                        },
                    }
                ),
                encoding="utf-8",
            )

            info = _gather_clip_info(job_dir, clip_dir)
            self.assertIsNotNone(info)
            self.assertEqual(info["clip_name"], "clip-01")
            self.assertEqual(info["duration"], 40.0)
            self.assertEqual(info["emotion_score"], 9.0)
            self.assertEqual(info["streamer_name"], "t2x2")
            self.assertEqual(info["title"], "ХУК ЗАГОЛОВОК")

    def test_select_top_clips_under_max(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir) / "job_1"
            clip1 = job_dir / "clip-01"
            clip2 = job_dir / "clip-02"
            clip1.mkdir(parents=True)
            clip2.mkdir(parents=True)

            mock_client = MagicMock()
            config = {"models": {"curator": "gemini-3.7-flash-high"}}

            selected = select_top_clips([job_dir], config, mock_client, max_clips=3, min_clips=2)
            self.assertEqual(len(selected), 2)
            mock_client.chat_json.assert_not_called()

    def test_select_top_clips_with_gemini(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job1 = Path(temp_dir) / "job_1"
            job2 = Path(temp_dir) / "job_2"
            for j in (job1, job2):
                for c in ("clip-01", "clip-02"):
                    cd = j / c
                    cd.mkdir(parents=True)
                    (cd / "selection.json").write_text(
                        json.dumps(
                            {
                                "start_time": 0,
                                "end_time": 30,
                                "emotion_score": 8.0,
                                "quality_score": 8.0,
                            }
                        ),
                        encoding="utf-8",
                    )

            # Total 4 clips > max_clips (3)
            mock_client = MagicMock()
            mock_client.chat_json.return_value = (
                {
                    "selected_candidates": [
                        {"candidate_id": 0, "curator_comment": "Top clip 1"},
                        {"candidate_id": 2, "curator_comment": "Top clip 2"},
                        {"candidate_id": 3, "curator_comment": "Top clip 3"},
                    ]
                },
                "raw_response",
            )
            config = {"models": {"curator": "gemini-3.7-flash-high"}}

            selected = select_top_clips([job1, job2], config, mock_client, max_clips=3, min_clips=2)
            self.assertEqual(len(selected), 3)
            mock_client.chat_json.assert_called_once()
            self.assertEqual(selected[0]["clip_name"], "clip-01")


if __name__ == "__main__":
    unittest.main()
