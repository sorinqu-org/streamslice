import math
import time
import unittest
from pathlib import Path

from streamslice.face_tracking import (
    FaceSample,
    FaceTrack,
    smooth_samples,
    track_face,
)
from streamslice.tracking import PointTracker

try:
    import cv2  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

REAL_CLIP = (
    Path(__file__).resolve().parents[1]
    / "remotion"
    / "public"
    / "jobs"
    / "t2x2_2026-08-22_14-04-39_chunk_6_e3d978d92d6ab1dc"
    / "clip-02"
    / "source.mp4"
)


def _make_sample(t: float, cx: float, cy: float, w: float = 0.1, h: float = 0.1) -> FaceSample:
    return FaceSample(time=t, x=cx - w / 2, y=cy - h / 2, width=w, height=h, confidence=0.9)


class SmoothSamplesTests(unittest.TestCase):
    def test_deadband_suppresses_jitter(self) -> None:
        samples = [
            _make_sample(0.0, 0.500, 0.500),
            _make_sample(0.25, 0.505, 0.500),
            _make_sample(0.50, 0.498, 0.500),
            _make_sample(0.75, 0.503, 0.500),
            _make_sample(1.00, 0.501, 0.500),
        ]
        smoothed = smooth_samples(
            samples, median_window=1, deadband=0.05, ema_alpha=1.0, max_velocity=100.0
        )
        xs = [s.center_x for s in smoothed]
        self.assertTrue(all(math.isclose(x, xs[0], abs_tol=1e-9) for x in xs))

    def test_velocity_limit_prevents_jump(self) -> None:
        samples = [
            _make_sample(0.0, 0.1, 0.5),
            _make_sample(1.0, 0.9, 0.5),
        ]
        smoothed = smooth_samples(
            samples, median_window=1, deadband=0.0, ema_alpha=1.0, max_velocity=0.3
        )
        delta = abs(smoothed[1].center_x - smoothed[0].center_x)
        self.assertLessEqual(delta, 0.3 + 1e-9)

    def test_median_rejects_single_outlier(self) -> None:
        samples = [
            _make_sample(0.0, 0.5, 0.5),
            _make_sample(0.25, 0.5, 0.5),
            _make_sample(0.50, 0.95, 0.5),  # single-frame outlier
            _make_sample(0.75, 0.5, 0.5),
            _make_sample(1.00, 0.5, 0.5),
        ]
        smoothed = smooth_samples(
            samples, median_window=5, deadband=0.0, ema_alpha=1.0, max_velocity=100.0
        )
        outlier_x = smoothed[2].center_x
        self.assertLess(abs(outlier_x - 0.5), 0.2)


class RegionCoordinateTests(unittest.TestCase):
    def test_region_to_full_frame_conversion(self) -> None:
        from streamslice.face_tracking import _bbox_region_to_full

        # region covers right-bottom quadrant of the frame
        region = {"x": 0.5, "y": 0.5, "width": 0.5, "height": 0.5}
        # bbox at the center of that region, half its size
        full = _bbox_region_to_full(
            0.25,
            0.25,
            0.5,
            0.5,
            region_x=region["x"],
            region_y=region["y"],
            region_width=region["width"],
            region_height=region["height"],
        )
        self.assertAlmostEqual(full[0], 0.625)
        self.assertAlmostEqual(full[1], 0.625)
        self.assertAlmostEqual(full[2], 0.25)
        self.assertAlmostEqual(full[3], 0.25)


class FaceTrackGeometryTests(unittest.TestCase):
    def test_bbox_at_interpolates_between_samples(self) -> None:
        samples = (
            FaceSample(time=0.0, x=0.1, y=0.1, width=0.1, height=0.1, confidence=0.9),
            FaceSample(time=1.0, x=0.3, y=0.3, width=0.1, height=0.1, confidence=0.9),
        )
        track = FaceTrack(
            samples=samples,
            detector="yunet",
            coverage=1.0,
            sample_fps=1.0,
            duration=1.0,
        )
        bbox = track.bbox_at(0.5)
        assert bbox is not None
        self.assertAlmostEqual(bbox[0], 0.2)
        self.assertAlmostEqual(bbox[1], 0.2)

    def test_bbox_at_returns_none_outside_track(self) -> None:
        samples = (FaceSample(time=1.0, x=0.1, y=0.1, width=0.1, height=0.1, confidence=0.9),)
        track = FaceTrack(
            samples=samples,
            detector="yunet",
            coverage=1.0,
            sample_fps=1.0,
            duration=2.0,
        )
        self.assertIsNone(track.bbox_at(0.0))
        self.assertIsNone(track.bbox_at(5.0))

    def test_max_bbox_unions_all_samples(self) -> None:
        samples = (
            FaceSample(time=0.0, x=0.1, y=0.2, width=0.1, height=0.1, confidence=0.9),
            FaceSample(time=1.0, x=0.5, y=0.1, width=0.1, height=0.2, confidence=0.9),
        )
        track = FaceTrack(
            samples=samples,
            detector="yunet",
            coverage=1.0,
            sample_fps=1.0,
            duration=1.0,
        )
        bbox = track.max_bbox()
        assert bbox is not None
        x, y, w, h = bbox
        self.assertAlmostEqual(x, 0.1)
        self.assertAlmostEqual(y, 0.1)
        self.assertAlmostEqual(x + w, 0.6)
        self.assertAlmostEqual(y + h, 0.3)

    def test_points_feed_point_tracker(self) -> None:
        samples = (
            FaceSample(time=0.0, x=0.1, y=0.1, width=0.1, height=0.1, confidence=0.9),
            FaceSample(time=1.0, x=0.3, y=0.3, width=0.1, height=0.1, confidence=0.9),
            FaceSample(time=2.0, x=0.5, y=0.5, width=0.1, height=0.1, confidence=0.9),
        )
        track = FaceTrack(
            samples=samples,
            detector="yunet",
            coverage=1.0,
            sample_fps=1.0,
            duration=2.0,
        )
        tracker = PointTracker(track.points(), duration=track.duration)
        x, y = tracker.evaluate(0.0)
        self.assertAlmostEqual(x, 0.15)
        self.assertAlmostEqual(y, 0.15)
        x2, y2 = tracker.evaluate(2.0)
        self.assertAlmostEqual(x2, 0.55)
        self.assertAlmostEqual(y2, 0.55)

    def test_to_dict_from_dict_round_trip(self) -> None:
        samples = (
            FaceSample(time=0.0, x=0.1, y=0.1, width=0.1, height=0.1, confidence=0.9),
            FaceSample(time=0.5, x=0.2, y=0.2, width=0.1, height=0.1, confidence=0.8),
        )
        track = FaceTrack(
            samples=samples,
            detector="yunet",
            coverage=0.75,
            sample_fps=4.0,
            duration=10.0,
        )
        restored = FaceTrack.from_dict(track.to_dict())
        self.assertEqual(restored.detector, track.detector)
        self.assertAlmostEqual(restored.coverage, track.coverage)
        self.assertAlmostEqual(restored.sample_fps, track.sample_fps)
        self.assertAlmostEqual(restored.duration, track.duration)
        self.assertEqual(len(restored.samples), len(track.samples))
        for a, b in zip(restored.samples, track.samples, strict=True):
            self.assertAlmostEqual(a.time, b.time)
            self.assertAlmostEqual(a.x, b.x)
            self.assertAlmostEqual(a.confidence, b.confidence)

    def test_is_usable_thresholds_on_coverage_and_detector(self) -> None:
        samples = (FaceSample(time=0.0, x=0.1, y=0.1, width=0.1, height=0.1, confidence=0.9),)
        usable = FaceTrack(
            samples=samples,
            detector="yunet",
            coverage=0.5,
            sample_fps=4.0,
            duration=1.0,
        )
        self.assertTrue(usable.is_usable(min_coverage=0.4))
        self.assertFalse(usable.is_usable(min_coverage=0.6))

        no_detector = FaceTrack(
            samples=samples,
            detector="none",
            coverage=0.9,
            sample_fps=4.0,
            duration=1.0,
        )
        self.assertFalse(no_detector.is_usable())


class TrackFaceDegradationTests(unittest.TestCase):
    def test_missing_model_returns_none_detector_without_raising(self) -> None:
        config = {"face_tracking": {"model_path": "/nonexistent/model.onnx"}}
        track = track_face(
            REAL_CLIP if REAL_CLIP.is_file() else Path("does-not-matter.mp4"),
            config=config,
            duration=1.0,
        )
        self.assertEqual(track.detector, "none")
        self.assertEqual(track.coverage, 0.0)
        self.assertEqual(track.samples, ())

    def test_disabled_returns_none_detector(self) -> None:
        config = {"face_tracking": {"enabled": False}}
        track = track_face(Path("does-not-matter.mp4"), config=config, duration=5.0)
        self.assertEqual(track.detector, "none")
        self.assertEqual(track.samples, ())

    def test_empty_config_section_uses_defaults_and_degrades_gracefully(self) -> None:
        # No cv2/model assumptions here beyond "must not raise"; if the real model
        # and cv2 are present and the fake path can't be probed, it raises
        # FaceTrackingError, which is also acceptable behavior to assert on.
        from streamslice.face_tracking import FaceTrackingError

        try:
            track = track_face(Path("does-not-exist.mp4"), config={}, duration=1.0)
        except FaceTrackingError:
            return
        self.assertIn(track.detector, ("none", "yunet"))


@unittest.skipUnless(HAS_CV2, "cv2 not available")
@unittest.skipUnless(REAL_CLIP.is_file(), f"missing test clip: {REAL_CLIP}")
class TrackFaceIntegrationTests(unittest.TestCase):
    def test_finds_dominant_face_in_webcam_region(self) -> None:
        region = {"x": 0.738, "y": 0.739, "width": 0.262, "height": 0.261}
        start = time.time()
        track = track_face(REAL_CLIP, config={}, region=region)
        elapsed = time.time() - start

        self.assertEqual(track.detector, "yunet")
        self.assertGreater(track.coverage, 0.5)

        centers_x = [s.center_x for s in track.samples]
        centers_y = [s.center_y for s in track.samples]
        avg_cx = sum(centers_x) / len(centers_x)
        avg_cy = sum(centers_y) / len(centers_y)

        print(
            f"\n[integration] samples={len(track.samples)} coverage={track.coverage:.3f} "
            f"avg_center=({avg_cx:.4f},{avg_cy:.4f}) elapsed={elapsed:.2f}s"
        )

        self.assertLess(abs(avg_cx - 0.835), 0.06)
        self.assertLess(abs(avg_cy - 0.821), 0.06)


if __name__ == "__main__":
    unittest.main()
