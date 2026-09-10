import unittest

from streamslice.sync import _select_chunk_files


class SyncTests(unittest.TestCase):
    def test_selects_requested_chunk_numbers(self) -> None:
        files = ["chunk_1.mp4", "chunk_02.mp4", "chunk_3.mp4", "notes.txt"]
        self.assertEqual(
            _select_chunk_files(files, [1, 2]),
            {1: "chunk_1.mp4", 2: "chunk_02.mp4"},
        )
