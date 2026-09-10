from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .pipeline import doctor, process_downloaded, process_video
from .render_queue import render_job, sync_and_render_queue
from .sync import sync_latest_chunks, sync_streams
from .youtube_api_uploader import (
    YoutubeAuthRequired,
    YoutubeUploadError,
    authorize_all_projects_interactive,
    upload_shorts_api,
)

_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="streamslice")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    process = commands.add_parser("process")
    process.add_argument("--input", required=True)
    sync = commands.add_parser("sync")
    sync.add_argument("--confirm-cleanup", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--confirm-cleanup", action="store_true")
    run_latest = commands.add_parser("run-latest")
    run_latest.add_argument("--streamer", required=True)
    run_latest.add_argument("--chunks", nargs="+", type=int, default=[1, 2])
    run_latest.add_argument("--confirm-cleanup", action="store_true")
    render_job_parser = commands.add_parser("render-job")
    render_job_parser.add_argument("--input", required=True)
    render_job_parser.add_argument("--parallel", type=int)
    commands.add_parser("render-queue")

    # YouTube commands
    yt_upload = commands.add_parser("youtube-upload")
    yt_upload.add_argument("--video", required=True, help="Path to video file (.mp4)")
    yt_upload.add_argument("--metadata", required=True, help="Path to metadata JSON file")
    yt_upload.add_argument(
        "--debug", action="store_true", help="Run browser with visible UI (headless=False)"
    )

    yt_login = commands.add_parser("youtube-login")
    yt_login.add_argument("--cookies", help="Target cookies file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            print(json.dumps(doctor(config), ensure_ascii=False, indent=2))
        elif args.command == "process":
            output = process_video(Path(args.input), config)
            print(output)
        elif args.command == "sync":
            sync_streams(config, confirm_cleanup=args.confirm_cleanup)
        elif args.command == "run":
            sync_streams(config, confirm_cleanup=args.confirm_cleanup)
            for output in process_downloaded(config):
                print(output)
        elif args.command == "run-latest":
            sources = sync_latest_chunks(
                config,
                streamer=args.streamer,
                chunk_numbers=args.chunks,
                confirm_cleanup=args.confirm_cleanup,
            )
            for source in sources:
                print(process_video(source, config))
        elif args.command == "render-job":
            print(render_job(args.input, config, max_parallel=args.parallel))
        elif args.command == "render-queue":
            for output in sync_and_render_queue(config):
                print(output)
        elif args.command == "youtube-upload":
            video_file = Path(args.video).expanduser().resolve()
            meta_file = Path(args.metadata).expanduser().resolve()
            if not video_file.is_file():
                raise FileNotFoundError(f"Video file not found: {video_file}")
            if not meta_file.is_file():
                raise FileNotFoundError(f"Metadata file not found: {meta_file}")
            meta_dict = json.loads(meta_file.read_text(encoding="utf-8"))
            url = upload_shorts_api(
                video_path=video_file,
                metadata=meta_dict,
                config=config,
                interactive=True,
            )
            print(f"Uploaded: {url}")
        elif args.command == "youtube-login":
            authorize_all_projects_interactive()
        return 0
    except YoutubeAuthRequired as exc:
        logging.getLogger(__name__).error("%s", exc)
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    except (ConfigError, OSError, RuntimeError, ValueError, YoutubeUploadError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
