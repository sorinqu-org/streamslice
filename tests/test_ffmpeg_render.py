import unittest

from streamslice.ffmpeg_render import (
    _escape_filter_value,
    _simplify_trajectory,
    build_ffmpeg_filtergraph,
)
from streamslice.montage import Cut, EditPlan, passthrough_plan
from streamslice.templates import load_template


def _plan(cuts: list[Cut], duration: float) -> EditPlan:
    output = sum(cut.output_duration for cut in cuts)
    source = sum(cut.source_duration for cut in cuts)
    return EditPlan(
        cuts=tuple(cuts),
        source_duration=duration,
        output_duration=output,
        removed_seconds=duration - source,
        cut_count=len(cuts),
    )


class EscapeFilterValueTests(unittest.TestCase):
    def test_backslash_is_escaped_not_replaced(self) -> None:
        # An earlier version turned a literal backslash into "/", which silently
        # rewrote the path instead of escaping it.
        self.assertEqual(_escape_filter_value("a\\b"), "a\\\\b")

    def test_filtergraph_separators_are_escaped(self) -> None:
        for char in ("'", ":", ",", ";", "[", "]", "="):
            with self.subTest(char=char):
                self.assertEqual(_escape_filter_value(f"a{char}b"), f"a\\{char}b")

    def test_plain_name_is_untouched(self) -> None:
        self.assertEqual(_escape_filter_value("subtitles.ass"), "subtitles.ass")


class FiltergraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.template = load_template("classic-split", {})
        self.props = {
            "layoutRecommendation": {"layout": "split", "focus_x": 0.5, "focus_y": 0.5},
            "layout": {
                "webcam": {"x": 0.738, "y": 0.739, "width": 0.262, "height": 0.261},
                "gameplay": {"x": 0.0, "y": 0.0, "width": 0.738, "height": 1.0},
            },
            "overlayMasks": [],
        }

    def _build(self, plan: EditPlan, *, has_audio: bool = True) -> str:
        return build_ffmpeg_filtergraph(
            props=self.props,
            ass_name="subtitles.ass",
            source_duration=plan.source_duration,
            template=self.template,
            plan=plan,
            has_audio=has_audio,
        )

    def test_single_cut_graph_has_no_concat(self) -> None:
        graph = self._build(passthrough_plan(30.0, "split"))
        self.assertNotIn("concat=", graph)
        self.assertIn("[outv]", graph)
        self.assertIn("subtitles=filename='subtitles.ass'", graph)

    def test_multi_cut_graph_concatenates_video_and_audio(self) -> None:
        plan = _plan(
            [
                Cut(0.0, 5.0, "split", 1.0, 1.0, "hook"),
                Cut(8.0, 20.0, "split", 1.0, 1.0, "body"),
            ],
            30.0,
        )
        graph = self._build(plan)
        self.assertIn("concat=n=2:v=1:a=1[vcat][acat]", graph)
        self.assertIn("atrim=start=8.0000:end=20.0000", graph)

    def test_silent_source_produces_no_audio_stream(self) -> None:
        plan = _plan(
            [
                Cut(0.0, 5.0, "split", 1.0, 1.0, "hook"),
                Cut(8.0, 20.0, "split", 1.0, 1.0, "body"),
            ],
            30.0,
        )
        graph = self._build(plan, has_audio=False)
        self.assertIn("concat=n=2:v=1:a=0[vcat]", graph)
        self.assertNotIn("[0:a]", graph)
        self.assertNotIn("acat", graph)

    def test_speed_change_adjusts_both_video_and_audio(self) -> None:
        plan = _plan([Cut(0.0, 10.0, "split", 1.0, 1.6, "silence-trim")], 10.0)
        graph = self._build(plan)
        self.assertIn("setpts=PTS/1.600000", graph)
        self.assertIn("atempo=1.600000", graph)

    def test_extreme_speed_chains_atempo_stages(self) -> None:
        # A single atempo only spans 0.5x..2x, so 3x has to be split.
        plan = _plan([Cut(0.0, 10.0, "split", 1.0, 3.0, "silence-trim")], 10.0)
        graph = self._build(plan)
        self.assertGreaterEqual(graph.count("atempo="), 2)

    def test_bands_are_stacked_for_split_layout(self) -> None:
        graph = self._build(passthrough_plan(30.0, "split"))
        self.assertIn("vstack=inputs=2", graph)

    def test_full_layout_has_no_stack(self) -> None:
        graph = self._build(passthrough_plan(30.0, "webcam_full"))
        self.assertNotIn("vstack", graph)

    def test_crop_dimensions_are_even(self) -> None:
        graph = self._build(passthrough_plan(30.0, "split"))
        import re

        sizes = re.findall(r"crop=w=(\d+):h=(\d+)", graph)
        self.assertTrue(sizes)
        for width, height in sizes:
            self.assertEqual(int(width) % 2, 0, f"odd crop width {width}")
            self.assertEqual(int(height) % 2, 0, f"odd crop height {height}")

    def test_expressions_stay_within_ffmpeg_parser_limits(self) -> None:
        # A 4 fps face track over 30s once produced an 18 KB expression with 252
        # nested if() calls, which FFmpeg refused to configure.
        import re

        trajectory = [
            {"time": index / 4.0, "focus_x": 0.5 + 0.2 * (index % 3), "focus_y": 0.5}
            for index in range(126)
        ]
        self.props["layoutRecommendation"]["focal_trajectory"] = trajectory
        graph = self._build(passthrough_plan(31.5, "gameplay_full"))
        for expression in re.findall(r"x='([^']*)'", graph):
            self.assertLess(len(expression), 4000, "crop expression is too long for FFmpeg")
            self.assertLess(expression.count("if("), 60, "too many nested if() calls")


class SimplifyTrajectoryTests(unittest.TestCase):
    def test_endpoints_are_preserved(self) -> None:
        points = [
            {"time": float(index), "focus_x": index / 10.0, "focus_y": 0.5} for index in range(10)
        ]
        simplified = _simplify_trajectory(points, max_points=4)
        self.assertEqual(simplified[0], points[0])
        self.assertEqual(simplified[-1], points[-1])

    def test_budget_is_respected(self) -> None:
        points = [
            {"time": index / 4.0, "focus_x": 0.5 + 0.3 * (index % 2), "focus_y": 0.5}
            for index in range(200)
        ]
        self.assertLessEqual(len(_simplify_trajectory(points, max_points=24)), 24)

    def test_straight_line_collapses_to_endpoints(self) -> None:
        points = [
            {"time": float(index), "focus_x": 0.1 * index, "focus_y": 0.1 * index}
            for index in range(12)
        ]
        self.assertEqual(len(_simplify_trajectory(points, tolerance=0.01)), 2)

    def test_real_movement_survives(self) -> None:
        # A sharp excursion in the middle must not be smoothed away entirely.
        points = [{"time": float(index), "focus_x": 0.2, "focus_y": 0.5} for index in range(6)]
        points[3] = {"time": 3.0, "focus_x": 0.9, "focus_y": 0.5}
        simplified = _simplify_trajectory(points, max_points=24, tolerance=0.004)
        self.assertIn(0.9, [point["focus_x"] for point in simplified])

    def test_short_input_is_returned_unchanged(self) -> None:
        points = [{"time": 0.0, "focus_x": 0.5, "focus_y": 0.5}]
        self.assertEqual(_simplify_trajectory(points), points)


if __name__ == "__main__":
    unittest.main()
