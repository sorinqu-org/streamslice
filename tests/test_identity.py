import tempfile
import unittest
from pathlib import Path

from streamslice.identity import resolve_creator_identity


class IdentityTests(unittest.TestCase):
    def test_t2x2_profile_uses_specific_mentions(self) -> None:
        config = {
            "sync": {"local_dir": "/home/yuwye/streams"},
            "identity": {
                "model_normalize": False,
                "profiles": {
                    "t2x2": {
                        "twitch_login": "t2x2",
                        "display_name": "t2x2",
                        "aliases": ["t2x2", "Тоха"],
                        "preferred_mentions": ["t2x2", "Тоха"],
                        "real_name": "Антон",
                    }
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            identity = resolve_creator_identity(
                Path("/home/yuwye/streams/t2x2/session/chunk_1.mp4"),
                config=config,
                client=object(),
                work_dir=Path(directory),
            )
        self.assertEqual(identity["display_name"], "t2x2")
        self.assertEqual(identity["preferred_mentions"], ["t2x2", "Тоха"])
        self.assertNotIn("стример", identity["preferred_mentions"])
