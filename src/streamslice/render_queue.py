from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import re
import shutil
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .clip_reviewer import review_and_refine_clip
from .gemini import GeminiClient
from .media import probe
from .process import ProcessError, require_binary, run
from .render import render_prepared_clip
from .session_batcher import group_jobs_into_batches
from .super_curator import select_top_clips
from .youtube_errors import YoutubeAuthRequired, YoutubeUploadError

try:
    from .youtube_api_uploader import upload_shorts_api
except ImportError:
    # The host queues jobs without the Google API client installed; uploading is
    # the render machine's job. Aliasing the exceptions to Exception here (as an
    # earlier version did) made every `except YoutubeAuthRequired` catch
    # everything, so they now come from a dependency-free module instead.
    upload_shorts_api = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
RESULT_UPLOADED_MARKER = ".result-uploaded"
RESULT_VERIFIED_MARKER = ".result-verified"
REMOTE_JOB_REMOVED_MARKER = ".remote-job-removed"


def render_job(
    job_dir: str | Path,
    config: dict[str, Any],
    *,
    max_parallel: int | None = None,
) -> Path:
    root = Path(job_dir).expanduser().resolve()
    job_path = root / "render-job.json"
    if not job_path.is_file():
        raise ProcessError(f"Render job manifest is missing: {job_path}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if int(job.get("job_version", 0)) != 1:
        raise ProcessError(f"Unsupported render job version: {job.get('job_version')}")
    job_id = _validate_job_id(job.get("job_id", ""))
    clips = job.get("clips", [])
    if not isinstance(clips, list):
        raise ProcessError(f"Render job clips must be a list: {job_path}")
    if not clips:
        LOGGER.info("[%s] No selected clips in job, finishing without rendering.", job_id)
        done = {
            "job_version": 1,
            "job_id": job_id,
            "render_version": job.get("render_version"),
            "completed_at_unix": time.time(),
            "clips": [],
        }
        _write_json(root / "DONE.json", done)
        (root / "FAILED.json").unlink(missing_ok=True)
        _mark_manifest_complete(root, done)
        final_output = _materialize_output(root, job_id, config)
        done["local_output"] = str(final_output)
        _write_json(root / "DONE.json", done)
        shutil.copy2(root / "DONE.json", final_output / "DONE.json")
        return root

    tasks: list[tuple[dict[str, Any], Path]] = []
    for item in clips:
        if not isinstance(item, dict):
            raise ProcessError(f"Invalid clip entry in {job_path}")
        directory = _safe_child(root, str(item.get("directory", "")))
        source = _safe_child(root, str(item.get("source", "")))
        if source.parent != directory or not source.is_file():
            raise ProcessError(f"Missing prepared source clip: {source}")
        expected_hash = str(item.get("source_sha256", "")).strip().casefold()
        if expected_hash and _sha256(source) != expected_hash:
            raise ProcessError(f"Prepared source checksum mismatch: {source}")
        tasks.append((item, directory))

    parallel = max_parallel or int(config.get("queue", {}).get("max_parallel_renders", 2))
    parallel = max(1, min(parallel, len(tasks)))
    LOGGER.info(
        "[%s] Starting job render: %d clips (parallelism: %d)",
        job_id,
        len(tasks),
        parallel,
    )
    completed: list[dict[str, Any]] = []
    errors: list[str] = []

    def worker(item: dict[str, Any], directory: Path) -> dict[str, Any]:
        output = _safe_child(root, str(item["output"]))
        try:
            info = probe(output) if output.is_file() else None
        except (ProcessError, OSError, json.JSONDecodeError, KeyError, ValueError):
            # Existing output is corrupt or unreadable; drop it and re-render.
            output.unlink(missing_ok=True)
            info = None
        if info is None:
            LOGGER.info(
                "[%s] Starting clip render %s (index %s)...",
                job_id,
                directory.name,
                item.get("index"),
            )
            info = render_prepared_clip(directory, job_id=job_id, config=config)
        else:
            LOGGER.info(
                "[%s] Clip %s already exists and is valid, skipping.", job_id, directory.name
            )
        return {
            "index": int(item["index"]),
            "output": str(output.relative_to(root)),
            "output_sha256": _sha256(output),
            "probe": info,
        }

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(worker, item, directory): int(item["index"])
            for item, directory in tasks
        }
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                LOGGER.warning("Clip %s render failed: %s", futures[future], exc, exc_info=True)
                errors.append(f"clip {futures[future]}: {exc}")

    if errors:
        failed = {
            "job_id": job_id,
            "failed_at_unix": time.time(),
            "errors": errors,
        }
        _write_json(root / "FAILED.json", failed)
        LOGGER.error("[%s] Errors while rendering job: %s", job_id, "; ".join(errors))
        raise ProcessError("; ".join(errors))

    done = {
        "job_version": 1,
        "job_id": job_id,
        "render_version": job.get("render_version"),
        "completed_at_unix": time.time(),
        "clips": sorted(completed, key=lambda item: item["index"]),
    }
    _write_json(root / "DONE.json", done)
    (root / "FAILED.json").unlink(missing_ok=True)
    _mark_manifest_complete(root, done)
    final_output = _materialize_output(root, job_id, config)
    done["local_output"] = str(final_output)
    _write_json(root / "DONE.json", done)
    shutil.copy2(root / "DONE.json", final_output / "DONE.json")
    LOGGER.info("[%s] Job successfully rendered and saved to %s", job_id, final_output)
    return root


def _is_job_fully_rendered(job_dir: Path) -> bool:
    """Return False if any clip declared in render-job.json is missing on disk."""
    job_path = job_dir / "render-job.json"
    if not job_path.is_file():
        return True
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        clips = job.get("clips", [])
        if not isinstance(clips, list) or not clips:
            return True
        for item in clips:
            if not isinstance(item, dict):
                continue
            out_rel = str(item.get("output", ""))
            if not out_rel:
                continue
            out_file = job_dir / out_rel
            if not out_file.is_file() or out_file.stat().st_size == 0:
                return False
        return True
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return True


def sync_and_render_queue(config: dict[str, Any]) -> list[Path]:
    settings = config.get("queue", {})
    if not settings.get("enabled", False):
        raise ProcessError("queue.enabled must be true")
    local_root = Path(settings["local_root"]).expanduser().resolve()
    incoming = local_root / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    with _queue_lock(local_root):
        return _sync_and_render_queue(config, settings, incoming)


def _sync_and_render_queue(
    config: dict[str, Any],
    settings: dict[str, Any],
    incoming: Path,
) -> list[Path]:
    rclone = require_binary("rclone")
    common = _rclone_common(settings)
    queue_available = True
    LOGGER.info("Checking queue availability on Google Drive (%s)...", settings["remote_jobs"])
    try:
        run(
            [rclone, "lsd", str(settings["remote_jobs"]), *common],
            timeout=120,
            capture=True,
        )
    except ProcessError as exc:
        if "directory not found" in str(exc).casefold():
            LOGGER.info("Remote queue does not exist yet: %s", settings["remote_jobs"])
            queue_available = False
        else:
            raise
    if queue_available:
        LOGGER.info(
            "Syncing incoming queue: %s -> %s...", settings["remote_jobs"], incoming
        )
        run(
            [
                rclone,
                "copy",
                str(settings["remote_jobs"]),
                incoming,
                "--create-empty-src-dirs",
                *common,
            ],
            timeout=float(settings.get("sync_timeout_seconds", 7200)),
            capture=False,
        )
        LOGGER.info("Queue sync with Google Drive finished.")

    pending: list[Path] = []
    for path in sorted(incoming.glob("*/render-job.json")):
        job_dir = path.parent
        result_marker = job_dir / RESULT_UPLOADED_MARKER
        removed_marker = job_dir / REMOTE_JOB_REMOVED_MARKER
        # Only skip if the job is truly fully rendered AND marked complete
        if result_marker.is_file() and removed_marker.is_file() and _is_job_fully_rendered(job_dir):
            continue
        pending.append(job_dir)

    if not pending:
        LOGGER.info("Queue is empty: no jobs to render.")
        return []

    LOGGER.info(
        "Found jobs to render (%d): %s",
        len(pending),
        ", ".join(p.name for p in pending),
    )
    max_parallel = max(1, int(settings.get("max_parallel_renders", 2)))
    max_jobs = max(1, int(settings.get("max_parallel_jobs", 1)))
    per_job = max_parallel

    batch_window_seconds = float(settings.get("batch_window_minutes", 30)) * 60.0
    super_curator_enabled = bool(settings.get("super_curator_enabled", True))
    visual_review_enabled = bool(settings.get("visual_review_enabled", True))
    max_batch_clips = int(settings.get("max_batch_clips", 3))
    min_batch_clips = int(settings.get("min_batch_clips", 2))

    # Group pending jobs into session batches
    batches = group_jobs_into_batches(pending, batch_window_seconds=batch_window_seconds)
    LOGGER.info(
        "Formed session batches: %d (window: %.0f min)",
        len(batches),
        batch_window_seconds / 60.0,
    )

    client: GeminiClient | None = None
    if super_curator_enabled or visual_review_enabled:
        try:
            client = GeminiClient(config)
        except (OSError, TimeoutError, ValueError, KeyError) as exc:
            LOGGER.warning("Could not initialize GeminiClient for queue: %s", exc)

    def process_and_upload(job_dir: Path) -> Path:
        job = json.loads((job_dir / "render-job.json").read_text(encoding="utf-8"))
        job_id = _validate_job_id(job.get("job_id", ""))
        if job_dir.name != job_id:
            raise ProcessError(
                f"Render job directory does not match job id: {job_dir.name!r} != {job_id!r}"
            )
        result_marker = job_dir / RESULT_UPLOADED_MARKER
        verified_marker = job_dir / RESULT_VERIFIED_MARKER
        removed_marker = job_dir / REMOTE_JOB_REMOVED_MARKER
        remote = str(settings["remote_results"]).rstrip("/") + f"/{job_id}"
        filters = _result_filters()
        if result_marker.is_file() and _is_job_fully_rendered(job_dir):
            if not verified_marker.is_file():
                try:
                    _verify_remote_result(
                        job_dir,
                        remote,
                        rclone,
                        common,
                        filters,
                        settings,
                    )
                except ProcessError:
                    # Older uploads rewrote top-level bookkeeping after the
                    # result was uploaded. Verify all rendered assets again,
                    # excluding only those mutable bookkeeping files.
                    LOGGER.warning(
                        "Strict result check failed for legacy job %s; "
                        "retrying without mutable top-level metadata",
                        job_id,
                    )
                    _verify_remote_result(
                        job_dir,
                        remote,
                        rclone,
                        common,
                        _legacy_result_filters(),
                        settings,
                    )
                verified_marker.write_text(f"{time.time():.6f}\n", encoding="utf-8")
            if queue_available:
                _remove_remote_job(settings, job_id, rclone, common)
            removed_marker.write_text(f"{time.time():.6f}\n", encoding="utf-8")
            return job_dir

        render_job(job_dir, config, max_parallel=per_job)
        result_marker.write_text(f"{time.time():.6f}\n", encoding="utf-8")
        verified_marker.write_text(f"{time.time():.6f}\n", encoding="utf-8")
        if queue_available:
            LOGGER.info("[%s] Removing source job from queue on Google Drive...", job_id)
            _remove_remote_job(settings, job_id, rclone, common)
        removed_marker.write_text(f"{time.time():.6f}\n", encoding="utf-8")
        LOGGER.info(
            "[%s] Job successfully finished and saved locally (cloud upload disabled)!", job_id
        )
        return job_dir

    rendered: list[Path] = []
    errors: list[str] = []

    for batch_idx, batch_jobs in enumerate(batches, start=1):
        LOGGER.info(
            "=== Processing batch %d/%d (%d jobs: %s) ===",
            batch_idx,
            len(batches),
            len(batch_jobs),
            ", ".join(j.name for j in batch_jobs),
        )

        # Count total clips across this batch
        total_batch_clips: list[tuple[Path, Path]] = []
        for j_dir in batch_jobs:
            for c_dir in sorted(j_dir.glob("clip-*")):
                if c_dir.is_dir():
                    total_batch_clips.append((j_dir, c_dir))

        selected_clip_dirs: set[Path] = set()
        if super_curator_enabled and len(total_batch_clips) > max_batch_clips and client:
            LOGGER.info(
                "Total clips in batch (%d) > max_batch_clips (%d), running SuperCurator...",
                len(total_batch_clips),
                max_batch_clips,
            )
            try:
                top_clips = select_top_clips(
                    batch_jobs=batch_jobs,
                    config=config,
                    client=client,
                    max_clips=max_batch_clips,
                    min_clips=min_batch_clips,
                )
                for item in top_clips:
                    c_path = Path(item["job_dir"]) / item["clip_name"]
                    selected_clip_dirs.add(c_path.resolve())
                LOGGER.info(
                    "SuperCurator selected %d top clips to render in batch: %s",
                    len(selected_clip_dirs),
                    ", ".join(p.parent.name + "/" + p.name for p in selected_clip_dirs),
                )
            except (OSError, TimeoutError, ValueError, KeyError) as exc:
                LOGGER.warning("SuperCurator failed: %s; falling back to all batch clips", exc)
                selected_clip_dirs = {c_dir.resolve() for _, c_dir in total_batch_clips}
        else:
            selected_clip_dirs = {c_dir.resolve() for _, c_dir in total_batch_clips}

        # Filter jobs: adjust render-job.json if some clips in job were not selected.
        # If a job has no selected clips, we update render-job.json to have empty
        # clips or skip rendering.
        for j_dir in batch_jobs:
            job_file = j_dir / "render-job.json"
            if job_file.is_file():
                try:
                    job_data = json.loads(job_file.read_text(encoding="utf-8"))
                    original_clips = job_data.get("clips", [])
                    filtered_clips = [
                        c
                        for c in original_clips
                        if (_safe_child(j_dir, str(c.get("directory", "")))).resolve()
                        in selected_clip_dirs
                    ]
                    # If some clips were filtered out by super curator
                    if len(filtered_clips) != len(original_clips):
                        job_data["clips"] = filtered_clips
                        _write_json(job_file, job_data)
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    LOGGER.warning("Could not adjust render-job.json for %s: %s", j_dir.name, exc)

        # Visual Review & Refine for selected clips before render
        if visual_review_enabled and client:
            for clip_path in sorted(selected_clip_dirs):
                if clip_path.is_dir():
                    try:
                        LOGGER.info(
                            "[%s/%s] Starting Visual Director & QC Review...",
                            clip_path.parent.name,
                            clip_path.name,
                        )
                        review_and_refine_clip(clip_path, config, client)
                    except (
                        OSError,
                        TimeoutError,
                        ProcessError,
                        ValueError,
                        KeyError,
                    ) as exc:
                        LOGGER.warning("Visual review failed for %s: %s", clip_path, exc)

        # Render and process jobs in this batch
        batch_workers = min(max_jobs, len(batch_jobs))
        if batch_workers:
            with ThreadPoolExecutor(max_workers=batch_workers) as pool:
                futures = {
                    pool.submit(process_and_upload, job_dir): job_dir
                    for job_dir in batch_jobs
                }
                for future in as_completed(futures):
                    try:
                        rendered.append(future.result())
                    except Exception as exc:
                        LOGGER.warning(
                            "Job %s failed: %s", futures[future].name, exc, exc_info=True
                        )
                        errors.append(f"{futures[future].name}: {exc}")

    if errors:
        raise ProcessError("; ".join(errors))
    return rendered


def _remove_remote_job(
    settings: dict[str, Any],
    job_id: str,
    rclone: str,
    common: list[str],
) -> None:
    job_id = _validate_job_id(job_id)
    remote_jobs = str(settings["remote_jobs"]).rstrip("/")
    remote = f"{remote_jobs}/{job_id}"
    timeout = float(settings.get("sync_timeout_seconds", 7200))
    # mkdir makes purge idempotent when an earlier run already removed the job.
    run([rclone, "mkdir", remote, *common], timeout=timeout, capture=False)
    run([rclone, "purge", remote, *common], timeout=timeout, capture=False)


def _verify_remote_result(
    job_dir: Path,
    remote: str,
    rclone: str,
    common: list[str],
    filters: list[str],
    settings: dict[str, Any],
) -> None:
    run(
        [rclone, "check", job_dir, remote, "--one-way", *filters, *common],
        timeout=float(settings.get("sync_timeout_seconds", 7200)),
        capture=False,
    )


def _result_filters() -> list[str]:
    return [
        "--exclude",
        "clip-*/source.mp4",
        "--exclude",
        "clip-*/remotion-props.local.json",
        "--exclude",
        "**/*-frames/**",
        "--exclude",
        "**/clip-transcript/**",
        "--exclude",
        ".result-uploaded",
        "--exclude",
        ".result-verified",
        "--exclude",
        ".remote-job-removed",
    ]


def _legacy_result_filters() -> list[str]:
    return [
        *_result_filters(),
        "--exclude",
        "manifest.json",
        "--exclude",
        "render-job.json",
        "--exclude",
        "DONE.json",
    ]


def upload_prepared_job(
    job_dir: str | Path,
    config: dict[str, Any],
) -> str | None:
    root = Path(job_dir).expanduser().resolve()
    job_path = root / "render-job.json"
    if not job_path.is_file():
        manifest_path = root / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                no_renders = manifest.get("status") == "complete_without_renders"
                if no_renders or not manifest.get("clips"):
                    LOGGER.info(
                        "No clips selected for %s; skipping remote queue upload", root.name
                    )
                    return None
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                LOGGER.debug(
                    "Could not read manifest.json for %s, treating as missing job", root.name
                )
        raise ProcessError(f"Prepared render job is missing: {job_path}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_id = _validate_job_id(job.get("job_id", ""))
    settings = config.get("queue", {})
    if not settings.get("enabled", False):
        raise ProcessError("queue.enabled must be true")
    remote = str(settings["remote_jobs"]).rstrip("/") + f"/{job_id}"
    common = _rclone_common(settings)
    filters = [
        "--exclude",
        "**/*-frames/**",
        "--exclude",
        "**/clip-transcript/**",
        "--exclude",
        "**/metadata-raw.txt",
    ]
    rclone = require_binary("rclone")
    run(
        [rclone, "copy", root, remote, *filters, *common],
        timeout=float(settings.get("sync_timeout_seconds", 7200)),
        capture=False,
    )
    run(
        [rclone, "check", root, remote, "--one-way", *filters, *common],
        timeout=float(settings.get("sync_timeout_seconds", 7200)),
        capture=False,
    )
    _write_json(
        root / "bundle-upload.json",
        {"job_id": job_id, "remote": remote, "uploaded_at_unix": time.time()},
    )
    return remote


def _rclone_common(settings: dict[str, Any]) -> list[str]:
    result = [
        "--transfers",
        str(settings.get("transfers", 4)),
        "--checkers",
        str(settings.get("checkers", 8)),
        "--retries",
        "8",
        "--low-level-retries",
        "20",
        "--no-update-modtime",
        "--use-mmap",
        "--buffer-size",
        "64Mi",
        "--drive-chunk-size",
        str(settings.get("drive_chunk_size", "64Mi")),
        "--stats",
        "10s",
        "--stats-one-line",
    ]
    config_path = str(settings.get("rclone_config", "")).strip()
    if config_path:
        result.extend(["--config", config_path])
    return result


@contextmanager
def _queue_lock(local_root: Path) -> Iterator[None]:
    lock_path = local_root / ".render-queue.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProcessError(f"Render queue is already running: {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_child(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ProcessError(f"Unsafe relative job path: {relative!r}")
    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise ProcessError(f"Job path escapes its root: {relative!r}")
    return target


def _validate_job_id(value: Any) -> str:
    job_id = str(value).strip()
    if not JOB_ID_RE.fullmatch(job_id) or job_id in {".", ".."}:
        raise ProcessError(f"Unsafe render job id: {job_id!r}")
    return job_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mark_manifest_complete(root: Path, done: dict[str, Any]) -> None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_index = {int(item["index"]): item for item in done["clips"]}
    for record in manifest.get("clips", []):
        rendered = by_index.get(int(record.get("index", 0)))
        if rendered:
            record["status"] = "complete"
            record["video"] = rendered["output"]
            record["probe"] = rendered["probe"]
    manifest["status"] = "complete"
    manifest["render_completed_at_unix"] = done["completed_at_unix"]
    _write_json(manifest_path, manifest)


def _materialize_output(root: Path, job_id: str, config: dict[str, Any]) -> Path:
    destination = Path(config["output"]["root"]).expanduser().resolve() / job_id
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "manifest.json",
        "render-job.json",
        "DONE.json",
        "transcript.json",
        "audio-peaks.json",
        "selection.json",
        "candidates.json",
    ):
        source = root / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for clip_dir in sorted(root.glob("clip-*")):
        if not clip_dir.is_dir():
            continue
        target = destination / clip_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            f"{clip_dir.name}.mp4",
            "subtitles.json",
            "subtitles.ass",
            "metadata.json",
            "description.txt",
            "overlay-analysis.json",
            "layout-analysis.json",
            "caption-source.json",
            "remotion-props.json",
            "selection.json",
            ".youtube_uploaded",
        ):
            source = clip_dir / name
            if source.is_file():
                shutil.copy2(source, target / name)

    # Optional automated YouTube Shorts publication
    yt_cfg = config.get("youtube", {})
    if yt_cfg.get("auto_upload", False):
        _upload_destination_clips_to_youtube(destination, config)

    return destination


def _upload_destination_clips_to_youtube(destination: Path, config: dict[str, Any]) -> None:
    """Publish finished clips to YouTube Shorts automatically, spaced to avoid shadowbans."""
    yt_cfg = config.get("youtube", {})
    interval_seconds = max(0, int(yt_cfg.get("interval_minutes", 12)) * 60)

    pending_clips = []
    for clip_dir in sorted(destination.glob("clip-*")):
        if not clip_dir.is_dir():
            continue
        video_path = clip_dir / f"{clip_dir.name}.mp4"
        meta_path = clip_dir / "metadata.json"
        uploaded_marker = clip_dir / ".youtube_uploaded"
        if video_path.is_file() and meta_path.is_file() and not uploaded_marker.is_file():
            pending_clips.append((clip_dir, video_path, meta_path, uploaded_marker))

    for idx, (clip_dir, video_path, meta_path, uploaded_marker) in enumerate(pending_clips):
        if idx > 0 and interval_seconds > 0:
            LOGGER.info(
                "[YouTube API] Waiting %d min before publishing next clip "
                "(shadowban protection)...",
                interval_seconds // 60,
            )
            time.sleep(interval_seconds)

        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            LOGGER.info(
                "[YouTube API] [%d/%d] Publishing %s...", idx + 1, len(pending_clips), clip_dir.name
            )
            url = upload_shorts_api(
                video_path=video_path,
                metadata=metadata,
                config=config,
                interactive=False,
            )
            uploaded_marker.write_text(f"{url}\n", encoding="utf-8")
            LOGGER.info("[YouTube API] Successfully published %s -> %s", clip_dir.name, url)
        except YoutubeAuthRequired as exc:
            LOGGER.error("[YouTube] Authorization required: %s", exc)
            break
        except (
            OSError,
            TimeoutError,
            json.JSONDecodeError,
            YoutubeUploadError,
            KeyError,
            ValueError,
        ) as exc:
            LOGGER.error("[YouTube] Failed to publish %s: %s", clip_dir.name, exc)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
