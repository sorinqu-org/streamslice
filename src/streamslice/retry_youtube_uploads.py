"""Scan output folders and upload unuploaded clips to YouTube Shorts with cooldowns."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from streamslice.config import load_config
from streamslice.youtube_api_uploader import (
    YoutubeAuthRequired,
    YoutubeUploadError,
    upload_shorts_api,
)

LOGGER = logging.getLogger("streamslice.youtube_sync")

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")


def _get_clip_timestamp(chunk_dir: Path, video_path: Path) -> float:
    """Extract creation timestamp from directory name or fallback to file mtime."""
    match = DATE_PATTERN.search(chunk_dir.name)
    if match:
        date_str = match.group(1)
        time_str = match.group(2).replace("-", ":")
        try:
            dt = datetime.datetime.fromisoformat(f"{date_str}T{time_str}")
            return dt.timestamp()
        except ValueError:
            pass
    try:
        return video_path.stat().st_mtime
    except OSError:
        return time.time()


def retry_failed_uploads(
    output_root: Path,
    config: dict[str, Any],
    interval_minutes: int | None = None,
    streamer_filter: str | None = None,
    max_age_days: float | None = 2.0,
) -> int:
    """Find all rendered clips missing .youtube_uploaded and attempt publishing."""
    yt_cfg = config.get("youtube", {})
    interval_m = (
        interval_minutes
        if interval_minutes is not None
        else int(yt_cfg.get("interval_minutes", 12))
    )
    interval_sec = max(0, interval_m * 60)

    if max_age_days is None:
        max_age_days = float(yt_cfg.get("max_age_days", 2.0))

    now = time.time()
    max_age_sec = max_age_days * 86400.0 if max_age_days > 0 else float("inf")

    output_root = output_root.expanduser().resolve()
    if not output_root.is_dir():
        LOGGER.error("Output directory does not exist: %s", output_root)
        return 0

    pending: list[tuple[Path, Path, Path, Path]] = []
    skipped_old = 0

    for chunk_dir in sorted(output_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        if streamer_filter and not chunk_dir.name.startswith(streamer_filter):
            continue
        for clip_dir in sorted(chunk_dir.glob("clip-*")):
            if not clip_dir.is_dir():
                continue
            video_path = clip_dir / f"{clip_dir.name}.mp4"
            meta_path = clip_dir / "metadata.json"
            marker_path = clip_dir / ".youtube_uploaded"
            if video_path.is_file() and meta_path.is_file() and not marker_path.is_file():
                clip_ts = _get_clip_timestamp(chunk_dir, video_path)
                age_sec = now - clip_ts
                if age_sec > max_age_sec:
                    skipped_old += 1
                    continue
                pending.append((chunk_dir, clip_dir, video_path, meta_path))

    if skipped_old > 0:
        print(f"ℹ️  Пропущено старых клипов (давнее {max_age_days:.1f} дн.): "
              f"{skipped_old}")

    if not pending:
        print("✅ Все актуальные срендеренные клипы уже опубликованы на YouTube!")
        return 0

    print("\n=======================================================")
    print(f"Найдено актуальных клипов для публикации: {len(pending)}")
    print(f"Максимальный возраст видео: {max_age_days} дн.")
    print(f"Интервал между публикациями: {interval_m} минут")
    print("=======================================================\n")

    uploaded_count = 0
    for idx, (chunk_dir, clip_dir, video_path, meta_path) in enumerate(pending, start=1):
        if uploaded_count > 0 and interval_sec > 0:
            print(
                f"\n⏳ Ожидание {interval_m} минут перед следующей публикацией "
                "(защита от теневого бана)..."
            )
            time.sleep(interval_sec)

        marker_path = clip_dir / ".youtube_uploaded"
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            title = metadata.get("youtube_title") or metadata.get("title") or clip_dir.name
            print(
                f"[{idx}/{len(pending)}] Публикация {chunk_dir.name}/{clip_dir.name} "
                f"-> '{title}'..."
            )

            url = upload_shorts_api(
                video_path=video_path,
                metadata=metadata,
                config=config,
                interactive=False,
            )
            marker_path.write_text(f"{url}\n", encoding="utf-8")
            print(f"✅ Успешно опубликован -> {url}")
            uploaded_count += 1

        except YoutubeAuthRequired as exc:
            print(f"\n❌ Требуется авторизация YouTube: {exc}")
            print("Запустите: PYTHONPATH=src .venv/bin/python -m streamslice.cli youtube-login\n")
            break
        except (OSError, TimeoutError, YoutubeUploadError, json.JSONDecodeError, ValueError) as exc:
            err_msg = str(exc)
            if "uploadLimitExceeded" in err_msg or "number of videos" in err_msg:
                print(
                    "\n⚠️  [YouTube] Достигнут суточный лимит загрузок самого "
                    "YouTube-канала (сброс через 24ч)."
                )
                print(
                    "Опубликованные клипы сохранены, остальные можно загрузить позже "
                    "этой же командой."
                )
                break
            print(f"❌ Ошибка публикации {clip_dir.name}: {exc}")

    print(f"\nИтог: опубликовано {uploaded_count} из {len(pending)} клипов.\n")
    return uploaded_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload unuploaded Shorts to YouTube.")
    parser.add_argument("--config", default="config/local-render.yaml", help="Path to config file")
    parser.add_argument("--output-root", default="~/tiktoks/output", help="Root output directory")
    parser.add_argument(
        "--interval", type=int, default=None, help="Interval in minutes between uploads"
    )
    parser.add_argument(
        "--streamer", help="Filter by streamer login (e.g. t2x2, stintik, mazellovvv)"
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=2.0,
        help="Skip clips older than this number of days (default: 2.0, 0 to disable)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config)
    output_dir = Path(args.output_root).expanduser().resolve()
    retry_failed_uploads(
        output_dir,
        cfg,
        interval_minutes=args.interval,
        streamer_filter=args.streamer,
        max_age_days=args.max_age_days,
    )


if __name__ == "__main__":
    main()
