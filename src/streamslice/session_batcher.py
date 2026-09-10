from __future__ import annotations

import contextlib
import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Matches timestamps like 2026-08-25_14-01-58 or 2026-08-25T14:01:58
TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})[_T](\d{2})-(\d{2})-(\d{2})|"
    r"(\d{4}-\d{2}-\d{2})[_T](\d{2}):(\d{2}):(\d{2})"
)
CHUNK_RE = re.compile(r"chunk_(\d+)", re.IGNORECASE)


def parse_job_metadata(job_dir: str | Path) -> dict[str, Any]:
    """Extract streamer, timestamp, chunk number, job_id, created_at_unix from a job dir."""
    root = Path(job_dir).expanduser().resolve()
    dir_name = root.name

    job_data: dict[str, Any] = {}
    job_manifest = root / "render-job.json"
    if job_manifest.is_file():
        try:
            loaded = json.loads(job_manifest.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                job_data = loaded
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not read render-job.json in %s", root)

    manifest_data: dict[str, Any] = {}
    manifest_file = root / "manifest.json"
    if manifest_file.is_file():
        try:
            loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest_data = loaded
        except (OSError, json.JSONDecodeError):
            pass

    # 1. Job ID
    job_id = str(job_data.get("job_id") or manifest_data.get("stream_id") or dir_name)

    # 2. Created at timestamp (Unix float)
    created_at_unix: float | None = None
    if "created_at_unix" in job_data:
        with contextlib.suppress(TypeError, ValueError):
            created_at_unix = float(job_data["created_at_unix"])
    elif "started_at_unix" in manifest_data:
        with contextlib.suppress(TypeError, ValueError):
            created_at_unix = float(manifest_data["started_at_unix"])

    # Fallback to parsing from directory name or source path if created_at_unix is not in json
    parsed_dt = parse_timestamp_from_string(dir_name)
    if parsed_dt is None and "source" in job_data:
        parsed_dt = parse_timestamp_from_string(str(job_data["source"]))
    if parsed_dt is None and "source" in manifest_data:
        parsed_dt = parse_timestamp_from_string(str(manifest_data["source"]))

    if created_at_unix is None:
        if parsed_dt is not None:
            created_at_unix = parsed_dt.timestamp()
        else:
            try:
                created_at_unix = root.stat().st_mtime
            except OSError:
                created_at_unix = 0.0

    # 3. Streamer name
    streamer_name = extract_streamer_name(
        dir_name=dir_name,
        job_data=job_data,
        manifest_data=manifest_data,
    )

    # 4. Chunk number
    chunk_num = extract_chunk_number(
        dir_name=dir_name,
        job_data=job_data,
        manifest_data=manifest_data,
    )

    return {
        "job_dir": root,
        "job_id": job_id,
        "streamer_name": streamer_name,
        "timestamp": created_at_unix,
        "chunk_number": chunk_num,
        "chunk_datetime": parsed_dt,
        "job_data": job_data,
        "manifest_data": manifest_data,
    }


def parse_timestamp_from_string(text: str) -> datetime.datetime | None:
    """Parse date and time from text format YYYY-MM-DD_HH-MM-SS or YYYY-MM-DD_HH:MM:SS."""
    match = TIMESTAMP_RE.search(text)
    if not match:
        return None
    groups = match.groups()
    if groups[0] is not None:
        date_str, h, m, s = groups[0], groups[1], groups[2], groups[3]
    else:
        date_str, h, m, s = groups[4], groups[5], groups[6], groups[7]

    try:
        y, mo, d = (int(x) for x in date_str.split("-"))
        return datetime.datetime(y, mo, d, int(h), int(m), int(s), tzinfo=datetime.UTC)
    except (ValueError, TypeError):
        return None


def extract_streamer_name(
    dir_name: str,
    job_data: dict[str, Any] | None = None,
    manifest_data: dict[str, Any] | None = None,
) -> str:
    """Extract streamer identifier from metadata or directory name."""
    if manifest_data and isinstance(manifest_data.get("creator"), dict):
        creator = manifest_data["creator"]
        login = (
            creator.get("twitch_login")
            or creator.get("display_name")
            or creator.get("source_key")
        )
        if login:
            return str(login).strip()

    # Try parsing from source path e.g. /root/.../t2x2/2026-08-25_14-01-58/chunk_1.mp4
    for data in (job_data, manifest_data):
        if data and "source" in data:
            source_p = Path(str(data["source"]))
            # parent is date folder, parent.parent is streamer name
            if len(source_p.parts) >= 3 and source_p.parent.name:
                parent_streamer = source_p.parent.parent.name
                if parent_streamer and not TIMESTAMP_RE.search(parent_streamer):
                    return parent_streamer

    # Try parsing prefix from directory name e.g. "t2x2_2026-08-25_14-01-58_chunk_1_..."
    match = TIMESTAMP_RE.search(dir_name)
    if match:
        prefix = dir_name[: match.start()].rstrip("_-")
        if prefix:
            return prefix

    parts = dir_name.split("_")
    return parts[0] if parts else "unknown"


def extract_chunk_number(
    dir_name: str,
    job_data: dict[str, Any] | None = None,
    manifest_data: dict[str, Any] | None = None,
) -> int | None:
    """Extract chunk number from dir name or source."""
    match = CHUNK_RE.search(dir_name)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            pass

    for data in (job_data, manifest_data):
        if data and "source" in data:
            src_match = CHUNK_RE.search(str(data["source"]))
            if src_match:
                try:
                    return int(src_match.group(1))
                except (ValueError, TypeError):
                    pass
    return None


def group_jobs_into_batches(
    job_dirs: list[Path],
    batch_window_seconds: float = 1800.0,
) -> list[list[Path]]:
    """Groups pending render jobs into unified session batches based on timestamp proximity.

    If time difference between consecutive jobs is <= batch_window_seconds
    (default 30 minutes = 1800s), they form a single batch.

    Args:
        job_dirs: List of job directory paths.
        batch_window_seconds: Max time difference in seconds between consecutive
            jobs to group them into the same batch. Default is 1800.0 (30 mins).

    Returns:
        List of batches, where each batch is a list of Path objects sorted by timestamp.
    """
    valid_dirs = [Path(p).expanduser().resolve() for p in job_dirs if Path(p).is_dir()]
    if not valid_dirs:
        return []

    # Parse metadata for all directories
    parsed_jobs: list[tuple[float, Path]] = []
    for directory in valid_dirs:
        meta = parse_job_metadata(directory)
        ts = meta["timestamp"]
        parsed_jobs.append((ts, directory))

    # Sort all jobs chronologically
    parsed_jobs.sort(key=lambda item: (item[0], item[1].name))

    batches: list[list[Path]] = []
    current_batch: list[Path] = [parsed_jobs[0][1]]
    prev_time = parsed_jobs[0][0]

    for ts, job_path in parsed_jobs[1:]:
        diff = ts - prev_time
        if diff <= batch_window_seconds:
            current_batch.append(job_path)
        else:
            batches.append(current_batch)
            current_batch = [job_path]
        prev_time = ts

    if current_batch:
        batches.append(current_batch)

    return batches
