import itertools
import unittest

from streamslice.montage import Cut, EditPlan, MontageError, passthrough_plan, plan_edit

_LAYOUTS = {"split", "webcam_full", "gameplay_full"}


def _words(duration: float, gap: float = 0.4, word_len: float = 0.3) -> list[dict[str, object]]:
    """Evenly spaced one-word utterances covering the whole clip."""
    words = []
    t = 0.0
    idx = 0
    while t + word_len < duration:
        words.append({"start": t, "end": t + word_len, "text": f"word{idx}"})
        t += word_len + gap
        idx += 1
    return words


def _activity_with_gaps(
    duration: float, gaps: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Build speech-activity intervals that are silent exactly on the given gaps."""
    gaps = sorted(gaps)
    activity: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in gaps:
        if start > cursor:
            activity.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        activity.append((cursor, duration))
    return activity


def _base_recommendation(layout: str = "split") -> dict[str, object]:
    return {"layout": layout}


class InvariantsMixin:
    def assert_plan_invariants(self, plan: EditPlan, duration: float) -> None:
        self.assertGreater(plan.cut_count, 0)
        self.assertEqual(plan.cut_count, len(plan.cuts))
        prev_end = None
        source_total = 0.0
        output_total = 0.0
        for cut in plan.cuts:
            self.assertLess(cut.source_start, cut.source_end)
            self.assertGreaterEqual(cut.source_start, -1e-9)
            self.assertLessEqual(cut.source_end, duration + 1e-9)
            if prev_end is not None:
                self.assertGreaterEqual(cut.source_start, prev_end - 1e-9)
            prev_end = cut.source_end
            self.assertIn(cut.layout, _LAYOUTS)
            self.assertGreaterEqual(cut.zoom, 1.0)
            self.assertGreater(cut.speed, 0.0)
            source_total += cut.source_duration
            output_total += cut.output_duration
        self.assertAlmostEqual(plan.output_duration, output_total, delta=1e-6)
        self.assertAlmostEqual(plan.removed_seconds, duration - source_total, delta=1e-6)
        self.assertAlmostEqual(plan.source_duration, duration, delta=1e-9)


class PropertyInvariantTests(InvariantsMixin, unittest.TestCase):
    def test_invariants_hold_across_synthetic_inputs(self) -> None:
        duration = 45.0
        words = _words(duration)
        scenarios: list[dict[str, object]] = [
            {},
            {"activity": _activity_with_gaps(duration, [(10.0, 14.0)])},
            {"activity": _activity_with_gaps(duration, [(0.5, 1.5)])},
            {"activity": _activity_with_gaps(duration, [(43.0, 44.5)])},
            {"activity": _activity_with_gaps(duration, [(20.0, 24.0)]),
             "recommendation": {"layout": "split", "event_time": 20.0, "event_end": 24.0,
                                 "gameplay_event_validated": True}},
            {"recommendation": {"layout": "split", "webcam_cut_time": 5.0}},
            {"peaks": [{"time": 6.0, "score": 5.0}, {"time": 30.0, "score": 4.0}]},
            {"config": {"montage": {"hook_punch": True}}},
            {"config": {"montage": {"speed_up_silence": True}},
             "activity": _activity_with_gaps(duration, [(10.0, 14.0)])},
            {"config": {"montage": {"enabled": False}}},
            {"activity": _activity_with_gaps(
                duration, [(5.0, 9.0), (15.0, 19.0), (25.0, 29.0), (35.0, 39.0)]
            )},
        ]
        for index, extra in enumerate(scenarios):
            with self.subTest(scenario=index):
                plan = plan_edit(
                    duration=duration,
                    words=words,
                    recommendation=extra.get("recommendation", _base_recommendation()),
                    config=extra.get("config", {}),
                    peaks=extra.get("peaks"),
                    activity=extra.get("activity"),
                )
                self.assert_plan_invariants(plan, duration)


class SilenceTrimTests(InvariantsMixin, unittest.TestCase):
    def test_long_mid_clip_gap_is_cut_with_padding(self) -> None:
        duration = 30.0
        words = _words(duration)
        activity = _activity_with_gaps(duration, [(10.0, 14.0)])
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={},
            activity=activity,
        )
        self.assert_plan_invariants(plan, duration)
        self.assertGreater(plan.removed_seconds, 0.0)
        # Padding (0.25s default) must survive on each side of the removed gap.
        for cut in plan.cuts:
            self.assertFalse(10.25 < cut.source_start < 13.75)
            self.assertFalse(10.25 < cut.source_end < 13.75)
        covered = sorted((cut.source_start, cut.source_end) for cut in plan.cuts)
        self.assertTrue(
            any(s <= 10.25 <= e for s, e in covered)
            or any(abs(e - 10.25) < 1e-6 for s, e in covered),
        )

    def test_gap_in_protected_head_is_not_cut(self) -> None:
        duration = 30.0
        words = _words(duration)
        activity = _activity_with_gaps(duration, [(0.2, 1.8)])
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={},
            activity=activity,
        )
        self.assertAlmostEqual(plan.removed_seconds, 0.0, delta=1e-6)

    def test_gap_in_protected_tail_is_not_cut(self) -> None:
        duration = 30.0
        words = _words(duration)
        activity = _activity_with_gaps(duration, [(28.7, 29.9)])
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={},
            activity=activity,
        )
        self.assertAlmostEqual(plan.removed_seconds, 0.0, delta=1e-6)

    def test_gap_inside_event_window_is_not_cut(self) -> None:
        duration = 30.0
        words = _words(duration)
        activity = _activity_with_gaps(duration, [(15.5, 17.5)])
        recommendation = {
            "layout": "split",
            "event_time": 15.0,
            "event_end": 18.0,
            "gameplay_event_validated": True,
        }
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=recommendation,
            config={},
            activity=activity,
        )
        self.assertAlmostEqual(plan.removed_seconds, 0.0, delta=1e-6)

    def test_max_removed_ratio_limits_total_cut(self) -> None:
        duration = 60.0
        words = _words(duration)
        gaps = [(5.0 + i * 5.0, 5.0 + i * 5.0 + 3.0) for i in range(10)]
        activity = _activity_with_gaps(duration, gaps)
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={},
            activity=activity,
        )
        self.assert_plan_invariants(plan, duration)
        self.assertLessEqual(plan.removed_seconds, duration * 0.25 + 1e-6)

    def test_no_activity_means_no_trim_stage(self) -> None:
        duration = 30.0
        words = _words(duration)
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={},
            activity=None,
        )
        self.assertAlmostEqual(plan.removed_seconds, 0.0, delta=1e-6)


class LayoutSwitchTests(InvariantsMixin, unittest.TestCase):
    def test_webcam_cut_time_splits_layout(self) -> None:
        duration = 30.0
        words = _words(duration)
        recommendation = {"layout": "split", "webcam_cut_time": 5.0}
        plan = plan_edit(
            duration=duration, words=words, recommendation=recommendation, config={}
        )
        self.assert_plan_invariants(plan, duration)
        # Boundaries snap to the nearest word gap, so check layout by sampling
        # well clear of the nominal cut point rather than the raw timestamp.
        layout_before = next(c.layout for c in plan.cuts if c.source_start <= 2.0 < c.source_end)
        layout_after = next(c.layout for c in plan.cuts if c.source_start <= 20.0 < c.source_end)
        self.assertEqual(layout_before, "webcam_full")
        self.assertEqual(layout_after, "split")

    def test_event_window_gets_event_layout(self) -> None:
        duration = 45.0
        words = _words(duration)
        recommendation = {
            "layout": "split",
            "event_time": 20.0,
            "event_end": 24.0,
            "gameplay_event_validated": True,
        }
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=recommendation,
            config={"montage": {"event_layout": "gameplay_full"}},
        )
        self.assert_plan_invariants(plan, duration)
        matching = [c for c in plan.cuts if c.source_start < 22.0 < c.source_end]
        self.assertTrue(matching)
        self.assertEqual(matching[0].layout, "gameplay_full")

    def test_short_segment_merges_with_neighbor(self) -> None:
        duration = 20.0
        words = _words(duration)
        # webcam_cut_time very close to event start forces a sub-min-segment slice.
        recommendation = {
            "layout": "split",
            "webcam_cut_time": 9.0,
            "event_time": 9.5,
            "event_end": 12.0,
            "gameplay_event_validated": True,
        }
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=recommendation,
            config={"montage": {"min_segment_seconds": 1.5}},
        )
        self.assert_plan_invariants(plan, duration)
        for cut in plan.cuts:
            self.assertGreaterEqual(cut.source_duration, 0.0)


class PunchInTests(InvariantsMixin, unittest.TestCase):
    def test_punch_respects_cooldown_and_max_count(self) -> None:
        duration = 60.0
        words = _words(duration)
        peaks = [
            {"time": float(t), "score": 10.0 - i * 0.01} for i, t in enumerate(range(2, 58, 2))
        ]
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={"montage": {"max_punches": 4, "punch_cooldown_seconds": 6.0}},
            peaks=peaks,
        )
        self.assert_plan_invariants(plan, duration)
        punch_cuts = sorted(
            (c.source_start + c.source_end) / 2.0 for c in plan.cuts if c.reason == "punch"
        )
        self.assertLessEqual(len(punch_cuts), 4)
        for a, b in itertools.pairwise(punch_cuts):
            self.assertGreaterEqual(b - a, 6.0 - 1e-6)
        for cut in plan.cuts:
            if cut.reason == "punch":
                self.assertGreaterEqual(cut.zoom, 1.0)
                self.assertLessEqual(cut.zoom, 1.10 + 1e-9)

    def test_punch_does_not_cross_layout_boundary(self) -> None:
        duration = 30.0
        words = _words(duration)
        recommendation = {"layout": "split", "webcam_cut_time": 10.0}
        # First find the actual (word-snapped) boundary with no punches, then
        # place a peak exactly on it: the punch must still land fully inside
        # a single layout segment, never straddling the switch.
        baseline = plan_edit(
            duration=duration, words=words, recommendation=recommendation, config={}
        )
        boundary = next(
            after.source_start
            for before, after in itertools.pairwise(baseline.cuts)
            if before.layout == "webcam_full" and after.layout == "split"
        )
        peaks = [{"time": boundary, "score": 9.0}]
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=recommendation,
            config={"montage": {"punch_seconds": 2.0}},
            peaks=peaks,
        )
        self.assert_plan_invariants(plan, duration)
        for cut in plan.cuts:
            if cut.reason == "punch":
                self.assertTrue(
                    cut.source_end <= boundary + 1e-6 or cut.source_start >= boundary - 1e-6
                )


class HookTests(InvariantsMixin, unittest.TestCase):
    def test_first_cut_gets_hook_zoom_and_reason(self) -> None:
        duration = 30.0
        words = _words(duration)
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={"montage": {"hook_punch": True, "hook_zoom": 1.06, "hook_seconds": 1.5}},
        )
        self.assert_plan_invariants(plan, duration)
        first = plan.cuts[0]
        self.assertEqual(first.reason, "hook")
        self.assertAlmostEqual(first.zoom, 1.06, delta=1e-9)
        self.assertLessEqual(first.source_duration, 1.5 + 1e-6)

    def test_hook_disabled_leaves_first_cut_untouched(self) -> None:
        duration = 30.0
        words = _words(duration)
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={"montage": {"hook_punch": False}},
        )
        self.assertNotEqual(plan.cuts[0].reason, "hook")


class SpeedUpSilenceTests(InvariantsMixin, unittest.TestCase):
    def test_speed_up_marks_speed_instead_of_cutting(self) -> None:
        duration = 30.0
        words = _words(duration)
        activity = _activity_with_gaps(duration, [(10.0, 14.0)])
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config={"montage": {"speed_up_silence": True, "silence_speed": 1.6}},
            activity=activity,
        )
        self.assert_plan_invariants(plan, duration)
        self.assertAlmostEqual(plan.removed_seconds, 0.0, delta=1e-6)
        sped = [c for c in plan.cuts if c.reason == "silence-trim"]
        self.assertTrue(sped)
        for cut in sped:
            self.assertAlmostEqual(cut.speed, 1.6, delta=1e-9)
        expected_output = sum(c.source_duration / c.speed for c in plan.cuts)
        self.assertAlmostEqual(plan.output_duration, expected_output, delta=1e-6)
        self.assertLess(plan.output_duration, plan.source_duration)


class PassthroughTests(unittest.TestCase):
    def test_all_flags_disabled_is_passthrough(self) -> None:
        duration = 30.0
        words = _words(duration)
        config = {
            "montage": {
                "trim_silence": False,
                "punch_in": False,
                "hook_punch": False,
                "speed_up_silence": False,
            }
        }
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation=_base_recommendation(),
            config=config,
            activity=_activity_with_gaps(duration, [(10.0, 14.0)]),
            peaks=[{"time": 5.0, "score": 9.0}],
        )
        self.assertTrue(plan.is_passthrough())

    def test_montage_disabled_entirely_is_passthrough(self) -> None:
        duration = 20.0
        plan = plan_edit(
            duration=duration,
            words=[],
            recommendation=_base_recommendation(),
            config={"montage": {"enabled": False}},
        )
        self.assertTrue(plan.is_passthrough())

    def test_helper_produces_passthrough(self) -> None:
        plan = passthrough_plan(42.0, layout="webcam_full")
        self.assertTrue(plan.is_passthrough())
        self.assertEqual(plan.cuts[0].layout, "webcam_full")
        self.assertEqual(plan.cut_count, 1)
        self.assertAlmostEqual(plan.output_duration, 42.0)
        self.assertAlmostEqual(plan.removed_seconds, 0.0)

    def test_invalid_layout_falls_back_to_split(self) -> None:
        plan = passthrough_plan(10.0, layout="nonsense")
        self.assertEqual(plan.cuts[0].layout, "split")


class SerializationTests(unittest.TestCase):
    def test_round_trip_to_dict_from_dict(self) -> None:
        duration = 30.0
        words = _words(duration)
        activity = _activity_with_gaps(duration, [(10.0, 14.0)])
        plan = plan_edit(
            duration=duration,
            words=words,
            recommendation={"layout": "split", "webcam_cut_time": 5.0},
            config={},
            activity=activity,
            peaks=[{"time": 20.0, "score": 8.0}],
        )
        payload = plan.to_dict()
        restored = EditPlan.from_dict(payload)
        self.assertEqual(restored, plan)

    def test_from_dict_rejects_malformed_payload(self) -> None:
        with self.assertRaises(MontageError):
            EditPlan.from_dict({"cuts": "not-a-list", "source_duration": 1, "output_duration": 1,
                                 "removed_seconds": 0, "cut_count": 1})

    def test_from_dict_rejects_missing_keys(self) -> None:
        with self.assertRaises(MontageError):
            EditPlan.from_dict({"cuts": []})


class DegenerateCaseTests(InvariantsMixin, unittest.TestCase):
    def test_zero_duration(self) -> None:
        plan = plan_edit(
            duration=0.0,
            words=[],
            recommendation=_base_recommendation(),
            config={},
        )
        self.assertTrue(plan.is_passthrough())
        self.assertEqual(plan.source_duration, 0.0)

    def test_empty_words(self) -> None:
        duration = 15.0
        plan = plan_edit(
            duration=duration,
            words=[],
            recommendation=_base_recommendation(),
            config={},
            activity=[(0.0, 15.0)],
        )
        self.assert_plan_invariants(plan, duration)

    def test_activity_none(self) -> None:
        duration = 15.0
        plan = plan_edit(
            duration=duration,
            words=_words(duration),
            recommendation=_base_recommendation(),
            config={},
            activity=None,
        )
        self.assert_plan_invariants(plan, duration)
        self.assertAlmostEqual(plan.removed_seconds, 0.0, delta=1e-6)

    def test_activity_covers_whole_clip_without_gaps(self) -> None:
        duration = 15.0
        plan = plan_edit(
            duration=duration,
            words=_words(duration),
            recommendation=_base_recommendation(),
            config={},
            activity=[(0.0, duration)],
        )
        self.assert_plan_invariants(plan, duration)
        self.assertAlmostEqual(plan.removed_seconds, 0.0, delta=1e-6)

    def test_single_solid_silence(self) -> None:
        duration = 15.0
        plan = plan_edit(
            duration=duration,
            words=[],
            recommendation=_base_recommendation(),
            config={},
            activity=[],
        )
        self.assert_plan_invariants(plan, duration)


class CutPropertyTests(unittest.TestCase):
    def test_source_and_output_duration_properties(self) -> None:
        cut = Cut(
            source_start=1.0, source_end=4.0, layout="split", zoom=1.05, speed=2.0, reason="body",
        )
        self.assertAlmostEqual(cut.source_duration, 3.0)
        self.assertAlmostEqual(cut.output_duration, 1.5)


if __name__ == "__main__":
    unittest.main()
