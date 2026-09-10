import unittest
from pathlib import Path

from streamslice.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_test_config_extends_default(self) -> None:
        config = load_config(ROOT / "config/test.yaml")
        self.assertFalse(config["sync"]["enabled"])
        self.assertEqual(config["selection"]["final_count"], 2)
        self.assertEqual(
            (config["render"]["width"], config["render"]["height"], config["render"]["fps"]),
            (1080, 1920, 60),
        )

