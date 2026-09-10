import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamslice.clip_reviewer import (
    _validate_webcam_crop,
    extract_preview_frames,
    review_and_refine_clip,
)


class ClipReviewerTests(unittest.TestCase):
    def test_validate_webcam_crop(self) -> None:
        valid = {"x": 0.738, "y": 0.739, "width": 0.262, "height": 0.261}
        res = _validate_webcam_crop(valid)
        self.assertIsNotNone(res)
        self.assertEqual(res["x"], 0.738)
        self.assertEqual(res["y"], 0.739)

        invalid_bounds = {"x": -0.1, "y": 0.5, "width": 0.5, "height": 0.5}
        self.assertIsNone(_validate_webcam_crop(invalid_bounds))

        invalid_too_small = {"x": 0.1, "y": 0.1, "width": 0.01, "height": 0.01}
        self.assertIsNone(_validate_webcam_crop(invalid_too_small))

    def test_extract_preview_frames_calls_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"dummy")

            def mock_run(cmd, **_kwargs):
                out_path = Path(cmd[-1])
                out_path.write_bytes(b"image")

            with (
                patch("streamslice.clip_reviewer.require_binary", return_value="ffmpeg"),
                patch("streamslice.clip_reviewer.run", side_effect=mock_run),
            ):
                frames = extract_preview_frames(source, root, duration=40.0)

            self.assertEqual(len(frames), 4)
            for f in frames:
                self.assertTrue(f.is_file())

    def test_review_and_refine_clip_updates_props_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_dir = Path(temp_dir) / "clip-01"
            clip_dir.mkdir(parents=True)

            (clip_dir / "remotion-props.json").write_text(
                json.dumps(
                    {
                        "title": "СТАРЫЙ ЗАГОЛОВОК",
                        "durationInSeconds": 30.0,
                        "layoutRecommendation": {
                            "layout": "split",
                            "focus_x": 0.5,
                            "focus_y": 0.5,
                            "webcam_crop": {"x": 0.7, "y": 0.7, "width": 0.25, "height": 0.25},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (clip_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "title": "СТАРЫЙ ЗАГОЛОВОК",
                        "creator": {"twitch_login": "t2x2", "display_name": "t2x2"},
                        "hashtags": ["#t2x2", "#fyp"],
                    }
                ),
                encoding="utf-8",
            )
            (clip_dir / "selection.json").write_text(
                json.dumps(
                    {
                        "start_time": 0.0,
                        "end_time": 30.0,
                        "highlight_reason": "Смешной момент",
                        "context_summary": "Стример играет в игру",
                    }
                ),
                encoding="utf-8",
            )
            (clip_dir / "source.mp4").write_bytes(b"dummy")

            mock_client = MagicMock()
            mock_client.chat_json.return_value = (
                {
                    "webcam_crop": {"x": 0.75, "y": 0.75, "width": 0.24, "height": 0.24},
                    "focus_x": 0.48,
                    "focus_y": 0.52,
                    "webcam_cut_time": 5.2,
                    "layout": "split",
                    "refined_title": "НОВЫЙ ВЗРЫВНОЙ ХУК",
                    "director_notes": "Уточнено кадрирование лица и фокус на чате",
                },
                "raw_response",
            )

            config = {
                "models": {"visual_review": "gemini-3.7-flash-high"},
                "clip_reviewer": {"sample_positions": [0.10, 0.30, 0.50, 0.80]},
            }

            with (
                patch("streamslice.clip_reviewer.probe", return_value={"duration": 30.0}),
                patch("streamslice.clip_reviewer.extract_preview_frames", return_value=[]),
            ):
                refinements = review_and_refine_clip(clip_dir, config, mock_client)

            self.assertEqual(refinements["title"], "НОВЫЙ ВЗРЫВНОЙ ХУК")
            self.assertEqual(refinements["layout"], "split")
            self.assertEqual(refinements["webcam_cut_time"], 5.2)
            self.assertEqual(refinements["focus_x"], 0.48)
            self.assertEqual(refinements["focus_y"], 0.52)

            updated_props = json.loads(
                (clip_dir / "remotion-props.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(updated_props["title"], "НОВЫЙ ВЗРЫВНОЙ ХУК")
            self.assertEqual(updated_props["layoutRecommendation"]["webcam_cut_time"], 5.2)
            self.assertEqual(updated_props["layoutRecommendation"]["focus_x"], 0.48)
            self.assertEqual(
                updated_props["layoutRecommendation"]["webcam_crop"],
                {"x": 0.75, "y": 0.75, "width": 0.24, "height": 0.24},
            )

            updated_meta = json.loads((clip_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(updated_meta["title"], "НОВЫЙ ВЗРЫВНОЙ ХУК")
            self.assertIn("#t2x2", updated_meta["youtube_title"])

            # Verify cached run doesn't call Gemini again
            mock_client.chat_json.reset_mock()
            refinements_cached = review_and_refine_clip(clip_dir, config, mock_client)
            mock_client.chat_json.assert_not_called()
            self.assertEqual(refinements_cached["title"], "НОВЫЙ ВЗРЫВНОЙ ХУК")


if __name__ == "__main__":
    unittest.main()
