import unittest
from pathlib import Path

from streamslice.config import load_config
from streamslice.overlay_analysis import _validate_boxes

ROOT = Path(__file__).resolve().parents[1]


class OverlayAnalysisTests(unittest.TestCase):
    def test_widget_is_masked_but_box_outside_gameplay_is_clipped(self) -> None:
        config = load_config(ROOT / "config/test.yaml")
        boxes = _validate_boxes(
            {
                "boxes": [
                    {
                        "x": 0.0,
                        "y": 0.0,
                        "width": 0.105,
                        "height": 0.075,
                        "kind": "widget",
                        "confidence": 0.96,
                    },
                    {
                        "x": 0.85,
                        "y": 0.3,
                        "width": 0.15,
                        "height": 0.3,
                        "kind": "external_chat",
                        "confidence": 0.99,
                    },
                ]
            },
            config,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["kind"], "widget")
        self.assertLessEqual(
            boxes[0]["x"] + boxes[0]["width"],
            config["layout"]["gameplay"]["x"] + config["layout"]["gameplay"]["width"],
        )
