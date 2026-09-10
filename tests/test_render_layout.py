import unittest

from streamslice.models import Candidate
from streamslice.render import normalized_recommendation


class RenderLayoutTests(unittest.TestCase):
    def test_absolute_layout_times_become_clip_local_once(self) -> None:
        candidate = Candidate(
            start_time=100.0,
            end_time=140.0,
            highlight_reason="event",
            emotion_score=9.0,
            camera_layout_recommendation={
                "layout": "split",
                "focus_time": 120.0,
                "event_time": 121.0,
                "event_end": 123.0,
                "gameplay_event_validated": True,
                "gameplay_zoom": 1.10,
            },
        )

        first = normalized_recommendation(candidate)
        candidate.camera_layout_recommendation = first
        second = normalized_recommendation(candidate)

        self.assertEqual(first["focus_time"], 20.0)
        self.assertEqual(first["event_time"], 21.0)
        self.assertEqual(first["event_end"], 23.0)
        self.assertEqual(second, first)

    def test_legacy_absolute_time_is_localized_even_when_smaller_than_duration(self) -> None:
        candidate = Candidate(
            start_time=20.0,
            end_time=80.0,
            highlight_reason="early source event",
            emotion_score=7.0,
            camera_layout_recommendation={"layout": "split", "focus_time": 45.0},
        )

        recommendation = normalized_recommendation(candidate)

        self.assertEqual(recommendation["focus_time"], 25.0)
        self.assertEqual(recommendation["time_base"], "clip")

    def test_unvalidated_event_cannot_carry_zoom_or_crop(self) -> None:
        candidate = Candidate(
            start_time=0.0,
            end_time=30.0,
            highlight_reason="routine gameplay",
            emotion_score=5.0,
            camera_layout_recommendation={
                "layout": "split",
                "focus_time": 12.0,
                "event_time": 11.0,
                "event_end": 13.0,
                "gameplay_zoom": 1.18,
                "gameplay_crop_width": 0.85,
            },
        )

        recommendation = normalized_recommendation(candidate)

        self.assertEqual(recommendation["focus_time"], 12.0)
        self.assertNotIn("event_time", recommendation)
        self.assertNotIn("gameplay_zoom", recommendation)
        self.assertNotIn("gameplay_crop_width", recommendation)

    def test_only_validated_webcam_crop_reaches_remotion(self) -> None:
        valid = Candidate(
            start_time=0.0,
            end_time=30.0,
            highlight_reason="face reaction",
            emotion_score=8.0,
            camera_layout_recommendation={
                "layout": "webcam_full",
                "webcam_box_validated": True,
                "webcam_box": {"x": 0.02, "y": 0.70, "width": 0.25, "height": 0.28},
            },
        )
        invalid = Candidate(
            start_time=0.0,
            end_time=30.0,
            highlight_reason="bad crop",
            emotion_score=8.0,
            camera_layout_recommendation={
                "layout": "webcam_full",
                "webcam_box_validated": True,
                "webcam_box": {"x": 0.90, "y": 0.70, "width": 0.25, "height": 0.28},
            },
        )

        valid_recommendation = normalized_recommendation(valid)
        invalid_recommendation = normalized_recommendation(invalid)

        self.assertEqual(valid_recommendation["webcam_crop"]["x"], 0.02)
        self.assertEqual(valid_recommendation["webcam_box"], valid_recommendation["webcam_crop"])
        self.assertNotIn("webcam_crop", invalid_recommendation)
        self.assertNotIn("webcam_box_validated", invalid_recommendation)

if __name__ == "__main__":
    unittest.main()
