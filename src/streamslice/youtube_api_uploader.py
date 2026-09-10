"""Official YouTube Data API v3 uploader with OAuth2, token refresh, and quota rotation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from .metadata import format_youtube_shorts_title, format_youtube_tags
from .youtube_errors import YoutubeAuthRequired, YoutubeUploadError

__all__ = [
    "YoutubeAuthRequired",
    "YoutubeUploadError",
    "authorize_all_projects_interactive",
    "upload_shorts_api",
]

LOGGER = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "streamslice"
DEFAULT_SECRETS_PATH = DEFAULT_CONFIG_DIR / "client_secrets.json"
DEFAULT_TOKEN_PATH = DEFAULT_CONFIG_DIR / "youtube_token.json"


def get_secrets_pool() -> list[tuple[Path, Path]]:
    """Return list of (secrets_path, token_path) pairs for all configured Google Cloud projects."""
    pool: list[tuple[Path, Path]] = []
    secrets_dir = DEFAULT_CONFIG_DIR / "secrets"

    if secrets_dir.is_dir():
        for sec in sorted(secrets_dir.glob("client_secrets_*.json")):
            idx = sec.stem.replace("client_secrets_", "")
            tok = DEFAULT_CONFIG_DIR / f"youtube_token_{idx}.json"
            pool.append((sec, tok))

    if not pool:
        sec = DEFAULT_SECRETS_PATH
        tok = DEFAULT_TOKEN_PATH
        if not sec.is_file():
            cwd_sec = Path("client_secrets.json").resolve()
            if cwd_sec.is_file():
                sec = cwd_sec
        pool.append((sec, tok))

    return pool


def get_credentials(
    secrets_path: Path,
    token_path: Path,
    interactive: bool = True,
) -> Credentials:
    """Load cached OAuth credentials or run interactive authorization flow."""
    sec_path = secrets_path.expanduser().resolve()
    tok_path = token_path.expanduser().resolve()

    creds: Credentials | None = None
    if tok_path.is_file():
        try:
            creds = Credentials.from_authorized_user_file(str(tok_path), SCOPES)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Could not load cached token from %s: %s", tok_path.name, exc)
            creds = None

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            LOGGER.info("Refreshing expired YouTube OAuth token for %s...", tok_path.name)
            creds.refresh(Request())
            tok_path.parent.mkdir(parents=True, exist_ok=True)
            tok_path.write_text(creds.to_json(), encoding="utf-8")
            return creds
        except (GoogleAuthError, OSError, TimeoutError) as exc:
            LOGGER.warning("Token refresh failed: %s, falling back to new login...", exc)

    if not sec_path.is_file():
        raise YoutubeAuthRequired(
            f"client_secrets file not found: {sec_path}.\n"
            "Download an OAuth 2.0 Client ID (Desktop App) from Google Cloud Console "
            f"and save it to {sec_path}"
        )

    if not interactive:
        raise YoutubeAuthRequired(
            f"Authorization required for {sec_path.name}. Run: "
            "python3 -m streamslice.cli youtube-login"
        )

    LOGGER.info("Launching browser for YouTube OAuth authorization [%s]...", sec_path.name)
    print(f"\n[StreamSlice] Authorizing YouTube for project: {sec_path.name}")
    flow = InstalledAppFlow.from_client_secrets_file(str(sec_path), SCOPES)
    creds = flow.run_local_server(port=0)

    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tok_path.write_text(creds.to_json(), encoding="utf-8")
    LOGGER.info("Successfully authorized and saved token to %s", tok_path)
    return creds


def authorize_all_projects_interactive() -> list[Path]:
    """Authorize all projects in the pool interactively."""
    pool = get_secrets_pool()
    authorized: list[Path] = []
    print(f"Found Google Cloud projects: {len(pool)}")
    for sec_path, tok_path in pool:
        print(f"\n---> Authorizing project {sec_path.name} -> {tok_path.name}")
        creds = get_credentials(sec_path, tok_path, interactive=True)
        if creds and creds.valid:
            authorized.append(tok_path)
    print(f"\nAll {len(authorized)} projects successfully authorized and ready to publish!")
    return authorized


def upload_shorts_api(
    video_path: str | Path,
    metadata: dict[str, Any],
    config: dict[str, Any] | None = None,
    interactive: bool = False,
) -> str:
    """Upload video to YouTube Shorts with automatic quota rotation across configured projects."""
    src_video = Path(video_path).resolve()
    if not src_video.is_file():
        raise FileNotFoundError(f"Video not found: {src_video}")

    cfg = config or {}
    yt_cfg = cfg.get("youtube", {})

    pool = get_secrets_pool()
    last_error: Exception | None = None

    # Prepare Title & Tags
    creator = metadata.get("creator", {})
    creator_login = str(creator.get("twitch_login") or creator.get("display_name") or "twitch")
    creator_display = str(creator.get("display_name") or creator_login)
    hashtags = metadata.get("hashtags", [])

    title = (
        metadata.get("youtube_title")
        or format_youtube_shorts_title(creator_login, creator_display, hashtags)
    )
    title = title[:100]

    default_source_url = "https://twitch.tv/" + creator_login
    description = metadata.get("youtube_description") or (
        f"{metadata.get('description', '')}\n\n"
        f"Twitch: {creator.get('source_url', default_source_url)}\n\n"
        f"{' '.join(hashtags)}"
    )

    tags_str = metadata.get("youtube_tags") or format_youtube_tags(
        creator_login, creator_display, hashtags
    )
    tags_list = [t.strip() for t in tags_str.split(",") if t.strip()]

    category = metadata.get("category", "Gaming")
    category_id = "20" if category == "Gaming" else "24"

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags_list[:30],
            "categoryId": category_id,
            "defaultLanguage": "ru",
            "defaultAudioLanguage": "ru",
        },
        "status": {
            "privacyStatus": yt_cfg.get("visibility", "public"),
            "selfDeclaredMadeForKids": False,
            "embeddable": True,
            "publicStatsViewable": True,
        },
    }

    for project_idx, (sec_path, tok_path) in enumerate(pool, start=1):
        try:
            creds = get_credentials(
                secrets_path=sec_path, token_path=tok_path, interactive=interactive
            )
            youtube = build("youtube", "v3", credentials=creds)

            media = MediaFileUpload(
                str(src_video),
                mimetype="video/mp4",
                resumable=True,
                chunksize=10 * 1024 * 1024,
            )

            LOGGER.info(
                "[YouTube API] [Project %d/%d] Uploading %s ('%s')...",
                project_idx,
                len(pool),
                src_video.name,
                title,
            )
            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    LOGGER.info("[YouTube API] Progress: %d%%", int(status.progress() * 100))

            video_id = response.get("id")
            if not video_id:
                raise YoutubeUploadError(f"Upload completed but no video ID returned: {response}")

            shorts_url = f"https://youtube.com/shorts/{video_id}"
            LOGGER.info(
                "[YouTube API] Successfully published (%s) -> %s", sec_path.name, shorts_url
            )
            return shorts_url

        except HttpError as exc:
            err_str = str(exc).lower()
            if "uploadlimitexceeded" in err_str or "number of videos they may upload" in err_str:
                LOGGER.warning(
                    "[YouTube API] Daily upload limit for this YouTube channel reached. "
                    "YouTube limits new channels (resets after 24h). To lift the limit, "
                    "verify your phone number and advanced features in studio.youtube.com "
                    "-> Settings -> Channel -> Feature eligibility."
                )
                last_error = exc
                break

            # Check for project quota exceeded -> rotate to next project
            quota_exceeded = exc.resp.status in (403, 429) or "quota" in err_str
            if quota_exceeded and "uploadlimitexceeded" not in err_str:
                LOGGER.warning(
                    "[YouTube API] Quota for project %s exhausted, switching to next project...",
                    sec_path.name,
                )
                last_error = exc
                continue

            LOGGER.error("[YouTube API] Upload error via %s: %s", sec_path.name, exc)
            last_error = exc
            break
        except YoutubeAuthRequired:
            raise
        except (OSError, TimeoutError, GoogleAuthError) as exc:
            LOGGER.warning("[YouTube API] Upload failure via %s: %s", sec_path.name, exc)
            last_error = exc
            continue

    raise YoutubeUploadError(f"Could not upload video via any available project: {last_error}")
