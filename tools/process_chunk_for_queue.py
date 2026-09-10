#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from streamslice.config import load_config
from streamslice.pipeline import process_video
from streamslice.render_queue import upload_prepared_job

LOGGER = logging.getLogger(__name__)


def _load_env_file(path: str = "/etc/streamslice.env") -> None:
    # systemd injects /etc/streamslice.env via EnvironmentFile, but a manual run
    # (python tools/process_chunk_for_queue.py ...) does not, so CLIPROXY_API_KEY is
    # empty and every Gemini call dies with "Set CLIPROXY_API_KEY before running".
    # Load the same file here so proxy auth works no matter how the tool is started.
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _cleanup_after_upload(chunk: Path, bundle: Path) -> None:
    # Once upload_prepared_job has copied the render bundle to Google Drive AND
    # verified it with `rclone check`, the host is done with this chunk: rendering
    # happens off-host from the Drive copy, so both the local bundle (its heavy
    # clip-*/source.mp4 and remotion props) and the raw chunk are dead weight.
    # Deleting them here is what keeps the 40 GB disk from filling one chunk at a
    # time -- the accumulation that previously snowballed into the empty-chunk storm.
    if bundle.is_dir():
        shutil.rmtree(bundle, ignore_errors=True)
        LOGGER.info("Removed local render bundle after upload: %s", bundle)
    try:
        chunk.unlink()
        LOGGER.info("Removed source chunk after upload: %s", chunk)
    except FileNotFoundError:
        pass
    except OSError:
        LOGGER.warning("Could not remove source chunk %s", chunk, exc_info=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare one recorder chunk and upload its portable Remotion job."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        _load_env_file()
        config = load_config(args.config)
        if config["render"].get("enabled", True):
            raise RuntimeError("Remote chunk processor requires render.enabled=false")
        chunk = Path(args.input)
        output = process_video(chunk, config)
        remote = upload_prepared_job(output, config)
        if remote:
            LOGGER.info("Prepared job uploaded to %s", remote)
        else:
            LOGGER.info(
                "No render job created (no clips); cleaning up chunk %s", chunk.name
            )
        _cleanup_after_upload(chunk, output)
        return 0
    except Exception:
        LOGGER.exception("Chunk processing failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
