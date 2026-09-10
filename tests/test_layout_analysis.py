import json
import tempfile
import unittest
from pathlib import Path

from streamslice.layout_analysis import (
    LAYOUT_ANALYSIS_VERSION,
    _merge_recommendation,
    _reaction_anchor,
    analyze_dynamic_layout,
)
from streamslice.models import Candidate


class FailingClient:
    def chat_json(self, **kwargs):  # pragma: no cover - cache must prevent this call
        raise AssertionError(f"unexpected Gemini call: {kwargs}")


class LayoutAnalysisTests(unittest.TestCase):
    def test_focus_time_is_localized_before_cached_analysis_is_merged(self) -> None:
        candidate = Candidate(
            start_time=100.0,
            end_time=140.0,
            highlight_reason="reaction",
            emotion_score=7.0,
            camera_layout_recommendation={
                "layout": "webcam_full",
                "focus_time": 121.0,
            },
        )
        analysis = {
            "webcam_box_validated": True,
            "webcam_box_confidence": 0.93,
            "webcam_box": {"x": 0.02, "y": 0.70, "width": 0.25, "height": 0.28},
            "has_gameplay_event": True,
            "event_time": 20.0,
            "event_end": 21.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "layout-analysis.json").write_text(
                json.dumps(
                    {
                        "analysis_version": LAYOUT_ANALYSIS_VERSION,
                        "analysis": analysis,
                        "frame_times": [0.0, 20.0, 21.0, 39.95],
                    }
                ),
                encoding="utf-8",
            )
            recommendation = analyze_dynamic_layout(
                output_dir / "missing-source.mp4",
                duration=40.0,
                words=[],
                candidate=candidate,
                output_dir=output_dir,
                config={
                    "layout": {
                        "fullscreen_emotion_threshold": 8.5,
                        "webcam_detection_min_confidence": 0.75,
                    }
                },
                client=FailingClient(),
            )

        self.assertEqual(recommendation["layout"], "webcam_full")
        self.assertEqual(recommendation["focus_time"], 21.0)
        self.assertEqual(recommendation["time_base"], "clip")
        self.assertTrue(recommendation["webcam_box_validated"])
        self.assertEqual(recommendation["webcam_crop"]["x"], 0.02)
        self.assertNotIn("event_time", recommendation)
        self.assertNotIn("gameplay_zoom", recommendation)

    def test_webcam_crop_is_independent_from_gameplay_event(self) -> None:
        merged = _merge_recommendation(
            {"layout": "split", "time_base": "clip", "focus_time": 12.0},
            {
                "webcam_box_validated": True,
                "webcam_box_confidence": 0.91,
                "webcam_crop": {"x": 0.01, "y": 0.68, "width": 0.27, "height": 0.31},
                "has_gameplay_event": False,
            },
            40.0,
        )

        self.assertEqual(merged["webcam_crop"]["x"], 0.01)
        self.assertTrue(merged["webcam_box_validated"])
        self.assertNotIn("event_time", merged)
        self.assertNotIn("gameplay_zoom", merged)

    def test_validated_gameplay_event_must_be_inside_sampled_frames(self) -> None:
        analysis = {
            "has_gameplay_event": True,
            "event_time": 10.0,
            "event_end": 11.0,
            "focus_x": 0.62,
            "focus_y": 0.40,
            "gameplay_zoom": 1.5,
            "gameplay_crop_width": 0.80,
        }
        accepted = _merge_recommendation(
            {"layout": "split", "time_base": "clip"},
            analysis,
            30.0,
            frame_times=[9.0, 10.0, 11.0],
        )
        rejected = _merge_recommendation(
            {"layout": "split", "time_base": "clip"},
            analysis,
            30.0,
            frame_times=[15.0, 16.0, 17.0],
        )

        self.assertTrue(accepted["gameplay_event_validated"])
        self.assertEqual(accepted["focus_time"], 10.0)
        self.assertEqual(accepted["gameplay_zoom"], 1.22)
        self.assertEqual(accepted["gameplay_crop_width"], 0.80)
        self.assertNotIn("event_time", rejected)
        self.assertNotIn("gameplay_event_validated", rejected)

    def test_low_confidence_or_invalid_webcam_box_is_rejected(self) -> None:
        for analysis in (
            {
                "webcam_box_validated": True,
                "webcam_box_confidence": 0.50,
                "webcam_box": {"x": 0.0, "y": 0.7, "width": 0.2, "height": 0.3},
            },
            {
                "webcam_box_validated": True,
                "webcam_box_confidence": 0.95,
                "webcam_box": {"x": 0.9, "y": 0.7, "width": 0.2, "height": 0.3},
            },
        ):
            with self.subTest(analysis=analysis):
                merged = _merge_recommendation(
                    {"layout": "split", "time_base": "clip"},
                    analysis,
                    30.0,
                )
                self.assertNotIn("webcam_crop", merged)
                self.assertNotIn("webcam_box_validated", merged)

    def test_local_focus_time_is_preferred_over_first_profanity(self) -> None:
        words = [
            {"start": 1.0, "end": 1.3, "text": "бля"},
            {"start": 18.0, "end": 18.3, "text": "ой"},
        ]
        self.assertEqual(_reaction_anchor(words, 30.0, 19.0), 19.0)


if __name__ == "__main__":
    unittest.main()
