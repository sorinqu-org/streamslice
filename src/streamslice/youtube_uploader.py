"""YouTube Studio automated uploader for YouTube Shorts using Playwright."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page, async_playwright

from .audio_cleanup import remove_music_if_blocked

LOGGER = logging.getLogger(__name__)

DEFAULT_COOKIES_PATH = Path.home() / ".config" / "streamslice" / "youtube_cookies.json"

# Multi-language selectors for YouTube Studio UI (Russian and English)
SELECTOR_UPLOAD_DIALOG_ATTACHED = "ytcp-uploads-dialog"
SELECTOR_UPLOAD_DIALOG_VISIBLE = (
    "#dialog.ytcp-uploads-dialog, ytcp-uploads-dialog #dialog, "
    "div#textbox[contenteditable='true'], ytcp-video-title, "
    "#textbox[aria-label*='title' i], #textbox[aria-label*='название' i], div#textbox"
)
SELECTOR_TITLE_INPUT = (
    '#textbox[aria-label*="title" i], #textbox[aria-label*="название" i], '
    '#title-textarea #textbox, ytcp-social-suggestions-textbox#title-textarea #textbox, '
    'div#textbox[contenteditable="true"], div#textbox'
)
SELECTOR_DESC_INPUT = (
    'ytcp-video-description div#textbox[contenteditable="true"], '
    'div#description-container div#textbox, #description-textarea #textbox, '
    'ytcp-social-suggestions-textbox#description-textarea #textbox'
)
SELECTOR_SHOW_MORE = (
    'ytcp-button#toggle-button, [aria-label*="Развернуть" i], [aria-label*="Show more" i], '
    'ytcp-button:has-text("Show more"), ytcp-button:has-text("Развернуть"), '
    'ytcp-button:has-text("Ещё")'
)
SELECTOR_NOT_FOR_KIDS = (
    'tp-yt-paper-radio-button[name="NOT_MADE_FOR_KIDS"], '
    'tp-yt-paper-radio-button:has-text("не для детей"), '
    'tp-yt-paper-radio-button:has-text("Not made for kids")'
)
SELECTOR_NEXT_BUTTON = (
    'ytcp-button#next-button, button#next-button, '
    'ytcp-button:has-text("Next"), ytcp-button:has-text("Далее"), '
    '[aria-label*="Next" i], [aria-label*="Далее" i]'
)
SELECTOR_CHECKS_CONTAINER = (
    "ytcp-uploads-checks, #checks-container, ytcp-check-status, ytcp-uploads-dialog"
)
SELECTOR_PUBLIC_RADIO = (
    'tp-yt-paper-radio-button[name="PUBLIC"], '
    'tp-yt-paper-radio-button:has-text("Открытый доступ"), '
    'tp-yt-paper-radio-button:has-text("Public")'
)
SELECTOR_PUBLISH_BUTTON = (
    'ytcp-button#done-button, button#done-button, '
    'ytcp-button:has-text("Publish"), ytcp-button:has-text("Опубликовать"), '
    'ytcp-button:has-text("Save"), ytcp-button:has-text("Сохранить"), '
    '[aria-label*="Publish" i], [aria-label*="Опубликовать" i], '
    '[aria-label*="Save" i], [aria-label*="Сохранить" i]'
)


class YoutubeAuthRequired(Exception):
    """Raised when YouTube authentication cookies are missing, invalid, or expired."""


class YoutubeUploadError(Exception):
    """Raised when an error occurs during YouTube upload process."""


def get_cookies_path(config: dict[str, Any] | None = None) -> Path:
    if config:
        cfg_cookies = config.get("youtube", {}).get("cookies_file")
        if cfg_cookies:
            return Path(cfg_cookies).expanduser().resolve()
    return DEFAULT_COOKIES_PATH.resolve()


def normalize_cookie(cookie: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a raw cookie dictionary into a clean Playwright-compatible cookie format."""
    if not isinstance(cookie, dict):
        return None

    name = str(cookie.get("name", "")).strip()
    if not name:
        return None
    value = str(cookie.get("value", ""))

    norm: dict[str, Any] = {
        "name": name,
        "value": value,
    }

    # Domain / Path
    raw_domain = cookie.get("domain")
    if raw_domain and isinstance(raw_domain, str) and raw_domain.strip():
        norm["domain"] = raw_domain.strip()
    elif cookie.get("url"):
        norm["url"] = str(cookie["url"]).strip()
    else:
        norm["domain"] = ".youtube.com"

    raw_path = cookie.get("path")
    if raw_path and isinstance(raw_path, str) and raw_path.strip():
        norm["path"] = raw_path.strip()
    else:
        norm["path"] = "/"

    # Secure / HttpOnly
    if "secure" in cookie:
        norm["secure"] = bool(cookie["secure"])
    if "httpOnly" in cookie:
        norm["httpOnly"] = bool(cookie["httpOnly"])

    # Expires / ExpirationDate
    raw_exp = cookie.get("expires", cookie.get("expirationDate"))
    if raw_exp is not None:
        try:
            exp_val = float(raw_exp)
            if exp_val > 0:
                norm["expires"] = exp_val
        except (ValueError, TypeError):
            pass

    # SameSite: strictly "Strict" | "Lax" | "None" (case-sensitive) or omitted
    raw_same_site = cookie.get("sameSite")
    if raw_same_site and isinstance(raw_same_site, str):
        ss_lower = raw_same_site.strip().lower()
        if ss_lower == "strict":
            norm["sameSite"] = "Strict"
        elif ss_lower == "lax":
            norm["sameSite"] = "Lax"
        elif ss_lower in ("none", "no_restriction"):
            norm["sameSite"] = "None"
        # If "unspecified" or unrecognized, omit sameSite

    return norm


def normalize_cookies(cookies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize cookies for Playwright, covering both .youtube.com and .google.com domains."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for c in cookies:
        if isinstance(c, dict):
            normalized = normalize_cookie(c)
            if normalized is not None:
                key = (normalized["name"], normalized.get("domain", ""))
                if key not in seen:
                    seen.add(key)
                    result.append(normalized)

                # Replicate YouTube session cookies to .google.com so
                # accounts.google.com authentication succeeds
                domain = normalized.get("domain", "")
                if "youtube.com" in domain:
                    google_cookie = dict(normalized)
                    google_cookie["domain"] = ".google.com"
                    g_key = (google_cookie["name"], ".google.com")
                    if g_key not in seen:
                        seen.add(g_key)
                        result.append(google_cookie)

    return result


def load_cookies(cookies_path: Path) -> list[dict[str, Any]]:
    # Also check fallback text format cookies.txt in the same folder or current dir
    candidates = [
        cookies_path,
        cookies_path.with_name("cookies.txt"),
        cookies_path.with_name("youtube_cookies.txt"),
        Path("cookies.txt").resolve(),
    ]
    actual_path: Path | None = None
    for cand in candidates:
        if cand.is_file() and cand.stat().st_size > 0:
            actual_path = cand
            break

    if not actual_path:
        raise YoutubeAuthRequired(
            "YoutubeAuthRequired: Cookies missing or expired. Run 'streamslice "
            f"youtube-login' or export cookies to {cookies_path} (JSON) or "
            "~/.config/streamslice/cookies.txt"
        )

    raw = actual_path.read_text(encoding="utf-8").strip()

    # 1. Try parsing JSON format (from Cookie-Editor / EditThisCookie)
    if raw.startswith("[") and raw.endswith("]"):
        try:
            data = json.loads(raw)
            if isinstance(data, list) and len(data) > 0:
                normalized = normalize_cookies(data)
                if normalized:
                    return normalized
        except json.JSONDecodeError:
            LOGGER.debug("Cookies file is not valid JSON, falling back to Netscape format")

    # 2. Parse Netscape format (from Get cookies.txt LOCALLY / curl)
    parsed_netscape: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        http_only = False
        if line.startswith("#HttpOnly_"):
            http_only = True
            line = line[len("#HttpOnly_"):].strip()
        elif line.startswith("#"):
            continue

        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) >= 7:
            domain, _include_subdomains, path, secure, expires, name, value = parts[:7]
            try:
                exp_float = float(expires) if expires else -1.0
            except ValueError:
                exp_float = -1.0
            cookie_dict: dict[str, Any] = {
                "name": name,
                "value": value,
                "domain": domain,
                "path": path,
                "secure": secure.upper() == "TRUE",
                "httpOnly": http_only,
            }
            if exp_float > 0:
                cookie_dict["expires"] = exp_float
            parsed_netscape.append(cookie_dict)

    if parsed_netscape:
        normalized = normalize_cookies(parsed_netscape)
        if normalized:
            return normalized

    raise YoutubeAuthRequired(
        f"YoutubeAuthRequired: Could not recognize cookies format in {actual_path}. "
        "Make sure you exported a valid JSON or Netscape cookies file."
    )


def save_cookies(cookies: list[dict[str, Any]], cookies_path: Path) -> None:
    cookies_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_cookies(cookies)
    cookies_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved cookies to %s", cookies_path)


async def youtube_login_interactive(cookies_path: Path | None = None) -> None:
    """Launch a visible browser window for the user to log into YouTube Studio and save cookies."""
    target_path = (cookies_path or DEFAULT_COOKIES_PATH).expanduser().resolve()
    LOGGER.info("Opening browser for YouTube authentication...")
    print("\n[StreamSlice] Opening browser for YouTube login.")
    print("Please log into your Google / YouTube Studio account in the opened window.")
    print("Once logged in and YouTube Studio is visible, cookies will be captured automatically...")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        await page.goto("https://studio.youtube.com", wait_until="networkidle")

        print("Waiting for login to complete (navigating to studio.youtube.com)...")
        # Wait until URL is studio.youtube.com and contains channel or dashboard, or up to 5 minutes
        logged_in = False
        for _ in range(150):  # 150 * 2 = 300s = 5 minutes
            await asyncio.sleep(2)
            url = page.url
            if "studio.youtube.com" in url and (
                "channel" in url or "video" in url or "dashboard" in url
            ):
                logged_in = True
                break
            # Check if avatar button or studio header is present
            try:
                if await page.locator("button#avatar-btn, ytcp-app, #create-icon").count() > 0:
                    logged_in = True
                    break
            except PlaywrightError:
                LOGGER.debug("Avatar button check failed, retrying")

        if not logged_in:
            await browser.close()
            raise YoutubeAuthRequired("Login timed out or was not completed.")

        cookies = await context.cookies()
        save_cookies(cookies, target_path)
        print(f"Successfully authenticated and saved cookies to {target_path}!\n")
        await browser.close()


def normalize_metadata(raw_metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize title, description, tags, category from metadata dict."""
    # 1. Title
    title = (
        raw_metadata.get("youtube_title")
        or raw_metadata.get("title")
        or ""
    ).strip()

    # 2. Description
    description = (
        raw_metadata.get("youtube_description")
        or raw_metadata.get("description")
        or ""
    ).strip()

    # 3. Tags
    raw_tags = (
        raw_metadata.get("youtube_tags")
        or raw_metadata.get("hashtags")
        or raw_metadata.get("tags")
        or []
    )
    if isinstance(raw_tags, str):
        tags_list = [t.strip().lstrip("#") for t in raw_tags.split(",") if t.strip()]
    elif isinstance(raw_tags, list):
        tags_list = [str(t).strip().lstrip("#") for t in raw_tags if str(t).strip()]
    else:
        tags_list = []

    # If hashtags are provided, we can also append them to description if not present
    hashtags = raw_metadata.get("hashtags", [])
    if isinstance(hashtags, list) and hashtags:
        formatted_hashtags = " ".join(
            f"#{tag.lstrip('#')}" for tag in hashtags if str(tag).strip()
        )
        if formatted_hashtags and formatted_hashtags not in description:
            description = f"{description}\n\n{formatted_hashtags}".strip()

    category = raw_metadata.get("category", "Gaming")
    game_title = raw_metadata.get("game_title", "")
    language = raw_metadata.get("language", "Russian")

    return {
        "title": title[:100],  # YouTube title limit is 100 chars
        "description": description[:5000],
        "tags": ", ".join(tags_list)[:500],
        "category": category,
        "game_title": game_title,
        "language": language,
    }


class YouTubeUploader:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        headless: bool = True,
        timeout: float = 120.0,
    ) -> None:
        self.config = config or {}
        self.headless = headless
        self.timeout = timeout
        self.cookies_path = get_cookies_path(self.config)

    async def upload_shorts(
        self,
        video_path: str | Path,
        metadata: dict[str, Any],
        max_copyright_retries: int = 1,
    ) -> str:
        """Upload video as YouTube Shorts, handle copyright checks and Demucs cleanup if flagged."""
        current_video_path = Path(video_path).resolve()
        if not current_video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {current_video_path}")

        meta = normalize_metadata(metadata)
        cookies = load_cookies(self.cookies_path)
        formatted_cookies = normalize_cookies(cookies)

        for attempt in range(max_copyright_retries + 1):
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=self.headless,
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                )
                context = await browser.new_context(
                    viewport={"width": 1440, "height": 900},
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                )
                await context.add_cookies(formatted_cookies)
                page = await context.new_page()

                try:
                    upload_result = await self._perform_upload(page, current_video_path, meta)

                    if upload_result["copyright_issue"]:
                        if attempt < max_copyright_retries:
                            LOGGER.warning(
                                "YouTube Copyright Check FAILED: Copyright claim detected "
                                "on audio/music. Cancelling draft and cleaning audio with Demucs..."
                            )
                            # Close / cancel current upload draft
                            await self._cancel_or_delete_draft(page)
                            await browser.close()

                            # Audio separation & cleanup
                            clean_path = current_video_path.with_name(
                                f"{current_video_path.stem}_nocopyright{current_video_path.suffix}"
                            )
                            LOGGER.info(
                                "Stripping copyrighted music using Demucs to %s...", clean_path
                            )
                            remove_music_if_blocked(current_video_path, clean_path)
                            current_video_path = clean_path
                            LOGGER.info("Retrying YouTube Shorts upload with clean audio...")
                            continue
                        else:
                            raise YoutubeUploadError(
                                "YouTube Copyright Check FAILED: "
                                "Copyright claim persists after cleanup."
                            )

                    shorts_url = upload_result["url"]
                    LOGGER.info("YouTube Shorts successfully uploaded: %s", shorts_url)
                    await browser.close()
                    return shorts_url

                except Exception:
                    await browser.close()
                    raise

        raise YoutubeUploadError("Upload failed unexpectedly.")

    async def _perform_upload(
        self,
        page: Page,
        video_path: Path,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform the steps in YouTube Studio dialog."""
        LOGGER.info("Navigating to YouTube Studio...")
        await page.goto(
            "https://studio.youtube.com", wait_until="domcontentloaded", timeout=60000
        )
        await asyncio.sleep(3)

        auth_required_message = (
            "YoutubeAuthRequired: Cookies missing or expired. Run 'streamslice "
            f"youtube-login' or export cookies to {self.cookies_path}"
        )

        # Check authentication
        if "accounts.google.com" in page.url or "signin" in page.url:
            raise YoutubeAuthRequired(auth_required_message)

        # 1. Click CREATE -> Upload videos or click upload button directly
        create_btn = page.locator(
            "#create-icon, button#create-icon, [aria-label*='Create' i], [aria-label*='Создать' i]"
        ).first
        upload_entry = page.locator(
            "tp-yt-paper-item:has-text('Upload videos'), "
            "tp-yt-paper-item:has-text('Добавить видео')"
        ).first

        if await create_btn.is_visible(timeout=10000):
            await create_btn.click()
            await asyncio.sleep(1)
            if await upload_entry.is_visible(timeout=5000):
                await upload_entry.click()
        else:
            # Maybe direct upload button on empty channel dashboard
            direct_upload = page.locator(
                "#upload-button, [aria-label*='Upload' i], [aria-label*='Загрузить' i]"
            ).first
            if await direct_upload.is_visible(timeout=5000):
                await direct_upload.click()
            else:
                # Check for channel / account error
                raise YoutubeAuthRequired(auth_required_message)

        # 2. File input chooser
        file_input = page.locator("input[type='file']").first
        await file_input.wait_for(state="attached", timeout=30000)
        LOGGER.info("Uploading file %s ...", video_path)
        await file_input.set_input_files(str(video_path))

        # Wait for the upload modal to load:
        # Note: <ytcp-uploads-dialog> custom element host evaluates as hidden
        # (display: contents or shadow boundary)
        # Wait for attached state on ytcp-uploads-dialog AND visible inner dialog elements
        await page.wait_for_selector(
            SELECTOR_UPLOAD_DIALOG_ATTACHED, state="attached", timeout=60000
        )
        await page.wait_for_selector(
            SELECTOR_UPLOAD_DIALOG_VISIBLE, state="visible", timeout=60000
        )
        LOGGER.info("Upload dialog opened. Filling details...")
        await asyncio.sleep(2)

        # Also wait for title input / textbox with timeout
        title_input = page.locator(SELECTOR_TITLE_INPUT).first
        await title_input.wait_for(state="visible", timeout=30000)

        # 3. Fill Details:
        # Title
        if meta["title"] and await title_input.is_visible(timeout=10000):
            await title_input.click()
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Backspace")
            await title_input.fill(meta["title"])
            LOGGER.info("Set title: %s", meta["title"])

        # Description
        if meta["description"]:
            desc_input = page.locator(SELECTOR_DESC_INPUT).first
            if await desc_input.is_visible(timeout=10000):
                await desc_input.click()
                await page.keyboard.press("Control+A")
                await page.keyboard.press("Backspace")
                await desc_input.fill(meta["description"])
                LOGGER.info("Set description.")

        # Audience (COPPA): 'No, it's not made for kids'
        not_for_kids_radio = page.locator(SELECTOR_NOT_FOR_KIDS).first
        if await not_for_kids_radio.is_visible(timeout=10000):
            await not_for_kids_radio.click()

        # Age restriction: 'No, don't restrict my video to viewers over 18 only'
        age_restriction_toggle = page.locator(
            "ytcp-button:has-text('Age restriction'), "
            "ytcp-button:has-text('Возрастные ограничения')"
        ).first
        if await age_restriction_toggle.is_visible(timeout=3000):
            await age_restriction_toggle.click()
            await asyncio.sleep(0.5)

        no_age_restrict = page.locator(
            'tp-yt-paper-radio-button[name="NOT_RESTRICT_AGE"], '
            'tp-yt-paper-radio-button:has-text("don\'t restrict my video"), '
            'tp-yt-paper-radio-button:has-text("не ограничивать"), '
            'tp-yt-paper-radio-button:has-text("Нет, видео подходит")'
        ).first
        if await no_age_restrict.is_visible(timeout=3000):
            await no_age_restrict.click()

        # Expand "Show more" / "Развернуть" / "Ещё"
        show_more_btn = page.locator(SELECTOR_SHOW_MORE).first
        if await show_more_btn.is_visible(timeout=5000):
            await show_more_btn.click()
            await asyncio.sleep(1)

        # Paid promotion: Unchecked
        paid_promo = page.locator(
            "ytcp-checkbox-lit[name='PAID_PROMOTION'], "
            "ytcp-checkbox-lit:has-text('paid promotion'), "
            "ytcp-checkbox-lit:has-text('прямая реклама')"
        ).first
        if await paid_promo.is_visible(timeout=3000):
            paid_promo_checked = await paid_promo.get_attribute("aria-checked") == "true" or (
                "checked" in (await paid_promo.get_attribute("class") or "")
            )
            if paid_promo_checked:
                await paid_promo.click()

        # Altered content / AI: Select 'No' / 'Нет'
        ai_radio_no = page.locator(
            "tp-yt-paper-radio-button[name='ALTERED_CONTENT_NOT_USED'], "
            "ytcp-radio-button[name='NO'], "
            "tp-yt-paper-radio-button:has-text('No'):has-text('content'), "
            "tp-yt-paper-radio-button:has-text('Нет')"
        ).first
        if await ai_radio_no.is_visible(timeout=3000):
            await ai_radio_no.click()

        # Automatic chapters: Unchecked
        auto_chapters = page.locator(
            "ytcp-checkbox-lit[name='AUTO_CHAPTERS'], "
            "ytcp-checkbox-lit:has-text('Automatic chapters'), "
            "ytcp-checkbox-lit:has-text('Автоматическая разбивка на эпизоды')"
        ).first
        if await auto_chapters.is_visible(timeout=3000):
            auto_chapters_checked = await auto_chapters.get_attribute("aria-checked") == "true" or (
                "checked" in (await auto_chapters.get_attribute("class") or "")
            )
            if auto_chapters_checked:
                await auto_chapters.click()

        # Tags
        if meta["tags"]:
            tags_input = page.locator(
                "#tags-container input, input[aria-label*='Tags' i], "
                "input[aria-label*='Теги' i], #tags-container #text-input"
            ).first
            if await tags_input.is_visible(timeout=5000):
                await tags_input.fill(meta["tags"])
                await page.keyboard.press("Enter")
                LOGGER.info("Filled tags: %s", meta["tags"])

        # Next -> Step 2 (Video Elements)
        next_button = page.locator(SELECTOR_NEXT_BUTTON).first
        await next_button.click()
        await asyncio.sleep(2)

        # Next -> Step 3 (Checks)
        await next_button.click()
        await asyncio.sleep(2)

        # 5. Checks (Copyright / Content ID)
        LOGGER.info("Monitoring YouTube Content ID / Copyright checks...")
        copyright_detected = False

        try:
            checks_container = page.locator(SELECTOR_CHECKS_CONTAINER).first
            await checks_container.wait_for(state="attached", timeout=10000)
        except PlaywrightError:
            LOGGER.debug("Checks container did not attach in time, continuing anyway")

        # Wait for checks to complete or report status
        # Checks tab shows progress or 'Checks complete. No issues found'
        for _ in range(60):  # Check for up to 2 minutes
            dialog_text = ""
            try:
                dialog_elem = page.locator(
                    "ytcp-uploads-dialog, #dialog.ytcp-uploads-dialog, ytcp-uploads-dialog #dialog"
                ).first
                if await dialog_elem.count() > 0:
                    dialog_text = await dialog_elem.inner_text()
            except PlaywrightError:
                LOGGER.debug("Could not read upload dialog text, retrying")

            # Check for copyright warning
            if any(
                phrase in dialog_text.lower()
                for phrase in (
                    "в этом видео есть контент, на который заявлены права",
                    "copyright claim",
                    "video is blocked",
                    "заявлены права",
                    "найдена жалоба",
                    "copyright-protected content found",
                    "impact on video: blocked",
                    "видео заблокировано",
                )
            ):
                LOGGER.warning("YouTube Copyright Check detected violation in text!")
                copyright_detected = True
                break

            # Check if checks completed without issues
            if any(
                phrase in dialog_text.lower()
                for phrase in (
                    "checks complete. no issues found",
                    "проверка завершена. нарушений нет",
                    "no issues found",
                    "нарушений не найдено",
                    "проверка завершена",
                    "checks complete",
                )
            ):
                LOGGER.info("YouTube Checks complete: No copyright issues detected.")
                break

            # Also check if next button is enabled
            try:
                if await next_button.is_enabled():
                    LOGGER.debug("Next button is enabled during checks.")
            except PlaywrightError:
                LOGGER.debug("Could not query next button state, retrying")

            await asyncio.sleep(2)

        if copyright_detected:
            return {"copyright_issue": True, "url": ""}

        # Next -> Step 4 (Visibility)
        if await next_button.is_visible(timeout=5000):
            await next_button.click()
            await asyncio.sleep(2)

        # 6. Visibility: Select 'Public' ('Открытый доступ')
        public_radio = page.locator(SELECTOR_PUBLIC_RADIO).first
        await public_radio.wait_for(state="visible", timeout=15000)
        await public_radio.click()
        LOGGER.info("Selected Visibility: Public")

        # Get the video link before publishing
        video_url_anchor = page.locator(
            "a.ytcp-video-info, a.ytcp-uploads-review, span.ytcp-video-info a, "
            "a[href*='youtu.be'], a[href*='youtube.com/shorts']"
        ).first
        video_url = ""
        if await video_url_anchor.is_visible(timeout=5000):
            video_url = await video_url_anchor.get_attribute("href") or ""

        # Publish button
        publish_btn = page.locator(SELECTOR_PUBLISH_BUTTON).first
        await publish_btn.wait_for(state="visible", timeout=15000)
        await publish_btn.click()
        LOGGER.info("Clicked Publish!")

        # Wait for the published dialog / confirmation
        await asyncio.sleep(5)

        # If link wasn't captured earlier, look in success dialog
        if not video_url:
            post_link = page.locator("a[href*='youtu.be'], a[href*='youtube.com/shorts']").first
            if await post_link.is_visible(timeout=10000):
                video_url = await post_link.get_attribute("href") or ""

        # Format URL as shorts link
        shorts_url = _format_as_shorts_url(video_url)
        return {"copyright_issue": False, "url": shorts_url}

    async def _cancel_or_delete_draft(self, page: Page) -> None:
        """Cancel current upload or close draft modal."""
        try:
            close_btn = page.locator(
                "ytcp-uploads-dialog #close-button, "
                "ytcp-uploads-dialog [aria-label*='Close' i], "
                "ytcp-uploads-dialog [aria-label*='Закрыть' i], #close-button"
            ).first
            if await close_btn.is_visible(timeout=3000):
                await close_btn.click()
                await asyncio.sleep(1)

            # If confirmation popup asks to save as draft or cancel
            discard_btn = page.locator(
                "ytcp-button:has-text('Cancel upload'), "
                "ytcp-button:has-text('Отменить загрузку'), "
                "ytcp-button:has-text('Discard'), "
                "ytcp-button:has-text('Удалить черновик')"
            ).first
            if await discard_btn.is_visible(timeout=3000):
                await discard_btn.click()
                await asyncio.sleep(1)
        except Exception as exc:  # noqa: BLE001 - best-effort draft cleanup, must not raise
            LOGGER.debug("Error while closing draft: %s", exc)


def _format_as_shorts_url(url: str) -> str:
    """Format any youtube link (youtu.be/ID or watch?v=ID) into https://youtube.com/shorts/ID."""
    if not url:
        return ""
    if "youtube.com/shorts/" in url:
        return url
    video_id = ""
    match_short = re.search(r"youtu\.be/([a-zA-Z0-9_-]+)", url)
    if match_short:
        video_id = match_short.group(1)
    else:
        match_watch = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        if match_watch:
            video_id = match_watch.group(1)
    if video_id:
        return f"https://youtube.com/shorts/{video_id}"
    return url


def upload_video_sync(
    video_path: str | Path,
    metadata: dict[str, Any],
    config: dict[str, Any] | None = None,
    headless: bool = True,
) -> str:
    """Synchronous entry point to upload YouTube shorts."""
    uploader = YouTubeUploader(config=config, headless=headless)
    return asyncio.run(uploader.upload_shorts(video_path, metadata))

