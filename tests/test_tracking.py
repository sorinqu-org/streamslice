import unittest

from streamslice.tracking import PointTracker, build_crop_filter


class TrackingTests(unittest.TestCase):
    def test_static_single_point(self) -> None:
        points = [{"time": 0.0, "focus_x": 0.75, "focus_y": 0.25}]
        tracker = PointTracker(points, duration=10.0)
        self.assertTrue(tracker.is_static())
        x, y = tracker.evaluate(5.0)
        self.assertAlmostEqual(x, 0.75)
        self.assertAlmostEqual(y, 0.25)

        x_expr = tracker.ffmpeg_x_expr(crop_w_expr="1080", in_w_expr="1920")
        self.assertIn("0.7500*1920-(1080)/2", x_expr)
        crop_filter = tracker.ffmpeg_crop_filter(1080, 1920)
        self.assertTrue(crop_filter.startswith("crop=w=1080:h=1920:"))

    def test_static_multiple_identical_points(self) -> None:
        points = [
            {"time": 0.0, "focus_x": 0.5, "focus_y": 0.5},
            {"time": 5.0, "focus_x": 0.5, "focus_y": 0.5},
        ]
        tracker = PointTracker(points, duration=5.0)
        self.assertTrue(tracker.is_static())

    def test_cubic_ease_in_out_interpolation(self) -> None:
        points = [
            {"time": 0.0, "focus_x": 0.2, "focus_y": 0.2},
            {"time": 10.0, "focus_x": 0.8, "focus_y": 0.8},
        ]
        tracker = PointTracker(points, duration=10.0, method="cubic")
        self.assertFalse(tracker.is_static())

        # Start and end
        x0, y0 = tracker.evaluate(0.0)
        self.assertAlmostEqual(x0, 0.2)
        self.assertAlmostEqual(y0, 0.2)

        x10, y10 = tracker.evaluate(10.0)
        self.assertAlmostEqual(x10, 0.8)
        self.assertAlmostEqual(y10, 0.8)

        # Midpoint at t=5 should be exactly 0.5
        x5, y5 = tracker.evaluate(5.0)
        self.assertAlmostEqual(x5, 0.5)
        self.assertAlmostEqual(y5, 0.5)

        # Out-of-bounds clamped
        x_before, _ = tracker.evaluate(-2.0)
        self.assertAlmostEqual(x_before, 0.2)
        x_after, _ = tracker.evaluate(15.0)
        self.assertAlmostEqual(x_after, 0.8)

    def test_catmull_rom_spline(self) -> None:
        points = [
            {"time": 0.0, "focus_x": 0.1, "focus_y": 0.1},
            {"time": 2.0, "focus_x": 0.4, "focus_y": 0.4},
            {"time": 4.0, "focus_x": 0.9, "focus_y": 0.9},
            {"time": 6.0, "focus_x": 0.5, "focus_y": 0.5},
        ]
        tracker = PointTracker(points, duration=6.0, method="catmull_rom")
        for t in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            x, y = tracker.evaluate(t)
            self.assertTrue(0.0 <= x <= 1.0)
            self.assertTrue(0.0 <= y <= 1.0)

    def test_generate_lookup(self) -> None:
        points = [
            {"time": 0.0, "focus_x": 0.0, "focus_y": 0.0},
            {"time": 1.0, "focus_x": 1.0, "focus_y": 1.0},
        ]
        tracker = PointTracker(points, duration=1.0)
        lookup = tracker.generate_lookup(fps=10.0)
        self.assertEqual(len(lookup), 10)
        self.assertEqual(lookup[0]["frame"], 0)
        self.assertAlmostEqual(lookup[0]["focus_x"], 0.0)
        self.assertAlmostEqual(lookup[-1]["focus_x"], 1.0, delta=0.15)

    def test_ffmpeg_expressions(self) -> None:
        points = [
            {"time": 0.0, "focus_x": 0.2, "focus_y": 0.3},
            {"time": 4.0, "focus_x": 0.8, "focus_y": 0.7},
        ]
        tracker = PointTracker(points, duration=4.0)
        x_expr = tracker.ffmpeg_x_expr("607", "1920")
        crop_filter = tracker.ffmpeg_crop_filter(607, 1080)

        self.assertIn("clip(", x_expr)
        self.assertIn("pow((t-0.0000)/4.0000,2)", x_expr)
        self.assertIn("crop=w=607:h=1080:x='", crop_filter)
        self.assertIn(":y='", crop_filter)

        helper_filter = build_crop_filter(points, duration=4.0, crop_width=607, crop_height=1080)
        self.assertEqual(helper_filter, crop_filter)

    def test_invalid_and_empty_points_fallback(self) -> None:
        tracker = PointTracker([], duration=0.0)
        self.assertTrue(tracker.is_static())
        x, y = tracker.evaluate(1.0)
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.5)
