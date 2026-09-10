from __future__ import annotations

import contextlib
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from .config import project_path
from .process import require_binary, run

LOGGER = logging.getLogger(__name__)


class UnsafeCleanupError(RuntimeError):
    pass


def sync_streams(config: dict[str, Any], *, confirm_cleanup: bool) -> None:
    settings = config["sync"]
    if not settings["enabled"]:
        LOGGER.info("Sync is disabled")
        return
    local_dir = Path(settings["local_dir"]).expanduser().resolve()
    if settings["clean_before_copy"]:
        if not confirm_cleanup:
            raise UnsafeCleanupError("Pass --confirm-cleanup to delete old local videos and caches")
        clean_stream_dir(local_dir, settings)
    local_dir.mkdir(parents=True, exist_ok=True)
    log_file = project_path(config, settings["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            require_binary("rclone"),
            "copy",
            settings["remote"],
            local_dir,
            "--transfers",
            str(settings["transfers"]),
            "--checkers",
            str(settings["checkers"]),
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
            "--fast-list",
            "--create-empty-src-dirs",
            "--log-level",
            "INFO",
            "--log-file",
            log_file,
        ],
        capture=False,
    )


def sync_latest_chunks(
    config: dict[str, Any],
    *,
    streamer: str,
    chunk_numbers: list[int],
    confirm_cleanup: bool,
) -> list[Path]:
    settings = config["sync"]
    if not settings["enabled"]:
        raise RuntimeError("Sync must be enabled for run-latest")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", streamer):
        raise ValueError(f"Unsafe streamer name: {streamer!r}")
    requested = sorted({int(value) for value in chunk_numbers})
    if not requested or any(value < 1 for value in requested):
        raise ValueError("Chunk numbers must be positive integers")

    rclone = require_binary("rclone")
    remote_root = str(settings["remote"]).rstrip("/")
    streamer_remote = f"{remote_root}/{streamer}"
    directory_result = run(
        [rclone, "lsf", streamer_remote, "--dirs-only"],
        timeout=120,
    )
    directories = [
        line.strip().rstrip("/")
        for line in directory_result.stdout.splitlines()
        if line.strip()
    ]
    if not directories:
        raise RuntimeError(f"No stream directories found at {streamer_remote}")
    latest = max(directories)
    latest_remote = f"{streamer_remote}/{latest}"
    file_result = run(
        [rclone, "lsf", latest_remote, "--files-only", "--include", "*.mp4"],
        timeout=120,
    )
    remote_files = [line.strip() for line in file_result.stdout.splitlines() if line.strip()]
    selected = _select_chunk_files(remote_files, requested)
    missing = [value for value in requested if value not in selected]
    if missing:
        raise RuntimeError(
            f"Latest stream {streamer}/{latest} has no chunks: "
            + ", ".join(map(str, missing))
        )

    local_dir = Path(settings["local_dir"]).expanduser().resolve()
    if settings["clean_before_copy"]:
        if not confirm_cleanup:
            raise UnsafeCleanupError("Pass --confirm-cleanup to delete old local videos and caches")
        clean_stream_dir(local_dir, settings)
    target_dir = local_dir / streamer / latest
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = project_path(config, settings["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for number in requested:
        remote_name = selected[number]
        target = target_dir / Path(remote_name).name
        run(
            [
                rclone,
                "copyto",
                f"{latest_remote}/{remote_name}",
                target,
                "--no-update-modtime",
                "--use-mmap",
                "--buffer-size",
                "64Mi",
                "--drive-chunk-size",
                str(settings.get("drive_chunk_size", "64Mi")),
                "--retries",
                "8",
                "--low-level-retries",
                "20",
                "--multi-thread-streams",
                str(settings.get("multi_thread_streams", 4)),
                "--multi-thread-chunk-size",
                str(settings.get("multi_thread_chunk_size", "64Mi")),
                "--log-level",
                "INFO",
                "--log-file",
                log_file,
            ],
            capture=False,
        )
        downloaded.append(target)
    LOGGER.info(
        "Downloaded latest %s stream %s chunks %s",
        streamer,
        latest,
        ", ".join(map(str, requested)),
    )
    return downloaded


def _select_chunk_files(files: list[str], requested: list[int]) -> dict[int, str]:
    selected: dict[int, str] = {}
    for name in files:
        match = re.search(r"(?:^|/)chunk[_-]?0*(\d+)\.mp4$", name, re.IGNORECASE)
        if not match:
            continue
        number = int(match.group(1))
        if number in requested:
            selected[number] = name
    return selected


def clean_stream_dir(local_dir: Path, settings: dict[str, Any]) -> list[Path]:
    resolved = local_dir.resolve()
    home = Path.home().resolve()
    if resolved in (Path("/"), home) or len(resolved.parts) < 4:
        raise UnsafeCleanupError(f"Refusing broad cleanup target: {resolved}")
    if not resolved.exists():
        return []
    extensions = {str(value).casefold() for value in settings["video_extensions"]}
    cache_names = set(settings["cache_names"])
    removed: list[Path] = []
    for path in list(resolved.rglob("*")):
        if path.is_file() and (path.suffix.casefold() in extensions or path.suffix == ".partial"):
            path.unlink()
            removed.append(path)
        elif path.is_dir() and path.name in cache_names:
            shutil.rmtree(path)
            removed.append(path)
    for directory in sorted(
        [item for item in resolved.rglob("*") if item.is_dir()],
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        with contextlib.suppress(OSError):
            directory.rmdir()
    LOGGER.info("Removed %s old videos/cache paths from %s", len(removed), resolved)
    return removed
