import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from streamslice.process import ProcessError
from streamslice.render_queue import (
    _queue_lock,
    _safe_child,
    _validate_job_id,
    sync_and_render_queue,
)


class RenderQueueTests(unittest.TestCase):
    def test_safe_child_rejects_escape(self) -> None:
        root = Path("/tmp/render-job").resolve()
        with self.assertRaises(ProcessError):
            _safe_child(root, "../other/source.mp4")

    def test_safe_child_accepts_clip_asset(self) -> None:
        root = Path("/tmp/render-job").resolve()
        self.assertEqual(
            _safe_child(root, "clip-01/source.mp4"),
            root / "clip-01/source.mp4",
        )

    def test_job_id_rejects_dot_segments(self) -> None:
        for value in (".", ".."):
            with self.subTest(value=value), self.assertRaises(ProcessError):
                _validate_job_id(value)

    def test_upload_is_verified_before_remote_job_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "incoming" / "job-1"
            job_dir.mkdir(parents=True)
            (job_dir / "render-job.json").write_text(
                json.dumps({"job_id": "job-1"}), encoding="utf-8"
            )
            config = self._queue_config(root)
            config["queue"]["max_parallel_jobs"] = 2

            with (
                patch("streamslice.render_queue.require_binary", return_value="rclone"),
                patch(
                    "streamslice.render_queue.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as run,
                patch("streamslice.render_queue.render_job", return_value=job_dir) as render,
            ):
                rendered = sync_and_render_queue(config)

            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(rendered, [job_dir])
            self.assertTrue((job_dir / ".result-uploaded").is_file())
            self.assertTrue((job_dir / ".result-verified").is_file())
            self.assertTrue((job_dir / ".remote-job-removed").is_file())
            self.assertEqual(render.call_args.kwargs["max_parallel"], 2)
            self.assertEqual([command[1] for command in commands], [
                "lsd", "copy", "mkdir", "purge"
            ])
            self.assertEqual(commands[3][2], "gdrive:queue/job-1")

    def test_failed_cleanup_retries_without_rerendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "incoming" / "job-2"
            job_dir.mkdir(parents=True)
            (job_dir / "render-job.json").write_text(
                json.dumps({"job_id": "job-2"}), encoding="utf-8"
            )
            config = self._queue_config(root)

            def fail_purge(command, **_kwargs):
                if command[1] == "purge":
                    raise ProcessError("temporary purge failure")
                return subprocess.CompletedProcess([], 0, "", "")

            with (
                patch("streamslice.render_queue.require_binary", return_value="rclone"),
                patch("streamslice.render_queue.run", side_effect=fail_purge),
                patch("streamslice.render_queue.render_job", return_value=job_dir),
                self.assertRaises(ProcessError),
            ):
                sync_and_render_queue(config)

            self.assertTrue((job_dir / ".result-uploaded").is_file())
            self.assertTrue((job_dir / ".result-verified").is_file())
            self.assertFalse((job_dir / ".remote-job-removed").exists())

            with (
                patch("streamslice.render_queue.require_binary", return_value="rclone"),
                patch(
                    "streamslice.render_queue.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as run,
                patch("streamslice.render_queue.render_job") as render,
            ):
                sync_and_render_queue(config)

            render.assert_not_called()
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(
                [command[1] for command in commands],
                ["lsd", "copy", "mkdir", "purge"],
            )
            self.assertTrue((job_dir / ".remote-job-removed").is_file())

    def test_legacy_upload_marker_is_verified_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "incoming" / "job-legacy"
            job_dir.mkdir(parents=True)
            (job_dir / "render-job.json").write_text(
                json.dumps({"job_id": "job-legacy"}), encoding="utf-8"
            )
            (job_dir / ".result-uploaded").write_text("legacy\n", encoding="utf-8")
            config = self._queue_config(root)

            with (
                patch("streamslice.render_queue.require_binary", return_value="rclone"),
                patch(
                    "streamslice.render_queue.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as run,
                patch("streamslice.render_queue.render_job") as render,
            ):
                sync_and_render_queue(config)

            render.assert_not_called()
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(
                [command[1] for command in commands],
                ["lsd", "copy", "check", "mkdir", "purge"],
            )
            self.assertIn(".result-uploaded", commands[2])
            self.assertTrue((job_dir / ".result-verified").is_file())
            self.assertTrue((job_dir / ".remote-job-removed").is_file())

    def test_legacy_metadata_difference_uses_asset_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "incoming" / "job-legacy-diff"
            job_dir.mkdir(parents=True)
            (job_dir / "render-job.json").write_text(
                json.dumps({"job_id": "job-legacy-diff"}), encoding="utf-8"
            )
            (job_dir / ".result-uploaded").write_text("legacy\n", encoding="utf-8")
            config = self._queue_config(root)
            checks = 0

            def legacy_check(command, **_kwargs):
                nonlocal checks
                if command[1] == "check":
                    checks += 1
                    if checks == 1:
                        raise ProcessError("legacy metadata differs")
                    self.assertIn("manifest.json", command)
                return subprocess.CompletedProcess([], 0, "", "")

            with (
                patch("streamslice.render_queue.require_binary", return_value="rclone"),
                patch("streamslice.render_queue.run", side_effect=legacy_check) as run,
                patch("streamslice.render_queue.render_job") as render,
            ):
                sync_and_render_queue(config)

            render.assert_not_called()
            self.assertEqual(checks, 2)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(
                [command[1] for command in commands],
                ["lsd", "copy", "check", "check", "mkdir", "purge"],
            )
            self.assertTrue((job_dir / ".result-verified").is_file())
            self.assertTrue((job_dir / ".remote-job-removed").is_file())

    def test_manifest_job_id_must_match_queue_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "incoming" / "job-a"
            job_dir.mkdir(parents=True)
            (job_dir / "render-job.json").write_text(
                json.dumps({"job_id": "job-b"}), encoding="utf-8"
            )
            config = self._queue_config(root)

            with (
                patch("streamslice.render_queue.require_binary", return_value="rclone"),
                patch(
                    "streamslice.render_queue.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as run,
                patch("streamslice.render_queue.render_job") as render,
                self.assertRaises(ProcessError),
            ):
                sync_and_render_queue(config)

            render.assert_not_called()
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual([command[1] for command in commands], ["lsd", "copy"])

    def test_missing_remote_queue_finishes_verified_local_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "incoming" / "job-local"
            job_dir.mkdir(parents=True)
            (job_dir / "render-job.json").write_text(
                json.dumps({"job_id": "job-local"}), encoding="utf-8"
            )
            (job_dir / ".result-uploaded").write_text("legacy\n", encoding="utf-8")
            config = self._queue_config(root)

            def missing_queue(command, **_kwargs):
                if command[1] == "lsd":
                    raise ProcessError("directory not found")
                return subprocess.CompletedProcess([], 0, "", "")

            with (
                patch("streamslice.render_queue.require_binary", return_value="rclone"),
                patch("streamslice.render_queue.run", side_effect=missing_queue) as run,
                patch("streamslice.render_queue.render_job") as render,
            ):
                sync_and_render_queue(config)

            render.assert_not_called()
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual([command[1] for command in commands], ["lsd", "check"])
            self.assertTrue((job_dir / ".result-verified").is_file())
            self.assertTrue((job_dir / ".remote-job-removed").is_file())

    def test_queue_lock_rejects_overlapping_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _queue_lock(root), self.assertRaises(ProcessError), _queue_lock(root):
                self.fail("second lock acquisition unexpectedly succeeded")

    def test_upload_prepared_job_returns_none_when_manifest_has_no_clips(self) -> None:
        from streamslice.render_queue import upload_prepared_job
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "empty-job"
            job_dir.mkdir()
            (job_dir / "manifest.json").write_text(
                json.dumps({"status": "complete_without_renders", "clips": []}),
                encoding="utf-8",
            )
            config = {"queue": {"enabled": True, "remote_jobs": "gdrive:test"}}
            res = upload_prepared_job(job_dir, config)
            self.assertIsNone(res)

    def test_sync_and_render_queue_with_super_curator_and_visual_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "incoming" / "t2x2_2026-08-25_14-00-00_chunk_1_test"
            job_dir.mkdir(parents=True)
            clip1 = job_dir / "clip-01"
            clip1.mkdir()
            (clip1 / "source.mp4").write_bytes(b"dummy")
            (clip1 / "remotion-props.json").write_text(
                json.dumps({"title": "CLIP 1"}), encoding="utf-8",
            )
            (job_dir / "render-job.json").write_text(
                json.dumps({
                    "job_id": job_dir.name,
                    "clips": [
                        {"index": 1, "directory": "clip-01", "output": "clip-01/clip-01.mp4"},
                    ],
                }),
                encoding="utf-8",
            )
            config = self._queue_config(root)
            config["queue"]["super_curator_enabled"] = True
            config["queue"]["visual_review_enabled"] = True

            mock_client = MagicMock()
            mock_review = MagicMock()
            mock_review.return_value = {"title": "REVIEWED CLIP 1"}

            with (
                patch("streamslice.render_queue.require_binary", return_value="rclone"),
                patch(
                    "streamslice.render_queue.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ),
                patch("streamslice.render_queue.GeminiClient", return_value=mock_client),
                patch("streamslice.render_queue.review_and_refine_clip", mock_review),
                patch("streamslice.render_queue.render_job", return_value=job_dir) as render,
            ):
                rendered = sync_and_render_queue(config)

            self.assertEqual(rendered, [job_dir])
            mock_review.assert_called_once_with(clip1.resolve(), config, mock_client)
            render.assert_called_once()

    @staticmethod
    def _queue_config(root: Path) -> dict[str, object]:
        return {
            "queue": {
                "enabled": True,
                "remote_jobs": "gdrive:queue",
                "remote_results": "gdrive:results",
                "local_root": str(root),
                "rclone_config": "/tmp/rclone.conf",
                "sync_timeout_seconds": 30,
                "max_parallel_renders": 2,
                "max_parallel_jobs": 1,
            }
        }
