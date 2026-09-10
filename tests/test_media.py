import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamslice.media import cut_source_clip, detect_audio_activity, probe
from streamslice.models import Candidate


class MediaTests(unittest.TestCase):
    @patch("streamslice.media.probe")
    @patch("streamslice.media.run")
    @patch("streamslice.media.require_binary", return_value="/usr/bin/ffmpeg")
    def test_cut_source_clip_uses_fast_input_seek(
        self,
        _require_binary,
        run,
        probe,
    ) -> None:
        run.return_value = subprocess.CompletedProcess([], 0, "", "")
        probe.return_value = {
            "duration": 60.0,
            "start_time": 0.0,
            "streams": [
                {"codec_type": "video", "start_time": 0.0},
                {"codec_type": "audio", "start_time": 0.0},
            ],
        }
        candidate = Candidate(
            start_time=3498.0,
            end_time=3558.0,
            highlight_reason="test",
            emotion_score=8.0,
        )
        source = Path("/tmp/hour-chunk.mp4")
        target = Path("/tmp/source-clip.mp4")
        config = {
            "runtime": {
                "duration_tolerance_seconds": 0.5,
                "ffmpeg_crf": 23,
                "ffmpeg_preset": "ultrafast",
                "source_clip_timeout_seconds": 1800,
            }
        }

        cut_source_clip(source, candidate, target, config)

        command = run.call_args.args[0]
        seek_index = command.index("-ss")
        input_index = command.index("-i")
        self.assertLess(seek_index, input_index)
        self.assertEqual(command[seek_index + 1], "3498.000")
        self.assertEqual(command[input_index + 1], source)
        self.assertEqual(run.call_args.kwargs["timeout"], 1800)
        self.assertIn("setpts=PTS-STARTPTS,fps=60", command)
        self.assertIn("aresample=async=1:first_pts=0", command)
        self.assertIn("-avoid_negative_ts", command)

    @patch("streamslice.media.run")
    @patch("streamslice.media.require_binary", return_value="/usr/bin/ffmpeg")
    def test_detect_audio_activity_returns_complement_of_silence(
        self,
        _require_binary,
        run,
    ) -> None:
        run.return_value = subprocess.CompletedProcess(
            [],
            0,
            "",
            "\n".join(
                [
                    "silence_start: 0",
                    "silence_end: 1.250 | silence_duration: 1.250",
                    "silence_start: 3.500",
                    "silence_end: 4.000 | silence_duration: 0.500",
                ]
            ),
        )

        activity = detect_audio_activity(Path("/tmp/clip.mp4"), duration=5.0)

        self.assertEqual(activity, [(1.25, 3.5), (4.0, 5.0)])


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg and ffprobe are required for the timestamp integration test",
)
class TimestampNormalizationIntegrationTests(unittest.TestCase):
    def test_cut_normalizes_nonzero_input_pts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "offset-input.mp4"
            target = root / "normalized-cut.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=96x54:rate=30:duration=5",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=5",
                    "-vf",
                    "setpts=PTS+3/TB",
                    "-af",
                    "asetpts=PTS+3/TB",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-copyts",
                    "-y",
                    str(source),
                ],
                check=True,
            )
            source_info = probe(source)
            self.assertGreater(float(source_info["start_time"] or 0), 2.0)

            config = {
                "runtime": {
                    "duration_tolerance_seconds": 0.5,
                    "ffmpeg_crf": 23,
                    "ffmpeg_preset": "ultrafast",
                    "source_clip_timeout_seconds": 120,
                }
            }
            info = cut_source_clip(
                source,
                Candidate(0.5, 3.5, "test", 8.0),
                target,
                config,
            )

            self.assertLess(abs(float(info["start_time"] or 0)), 0.05)
            for stream in info["streams"]:
                if stream.get("codec_type") in {"video", "audio"}:
                    self.assertLess(abs(float(stream["start_time"] or 0)), 0.05)


if __name__ == "__main__":
    unittest.main()
