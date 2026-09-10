import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from streamslice.youtube_uploader import (
    DEFAULT_COOKIES_PATH,
    SELECTOR_DESC_INPUT,
    SELECTOR_NEXT_BUTTON,
    SELECTOR_NOT_FOR_KIDS,
    SELECTOR_PUBLIC_RADIO,
    SELECTOR_PUBLISH_BUTTON,
    SELECTOR_SHOW_MORE,
    SELECTOR_TITLE_INPUT,
    SELECTOR_UPLOAD_DIALOG_ATTACHED,
    SELECTOR_UPLOAD_DIALOG_VISIBLE,
    YoutubeAuthRequired,
    YouTubeUploader,
    _format_as_shorts_url,
    get_cookies_path,
    load_cookies,
    normalize_cookie,
    normalize_metadata,
    save_cookies,
)


class TestYouTubeUploader(unittest.TestCase):
    def test_get_cookies_path_default(self) -> None:
        path = get_cookies_path(None)
        self.assertEqual(path, DEFAULT_COOKIES_PATH.resolve())

    def test_get_cookies_path_custom(self) -> None:
        custom_p = "/tmp/custom_youtube_cookies.json"
        config = {"youtube": {"cookies_file": custom_p}}
        path = get_cookies_path(config)
        self.assertEqual(path, Path(custom_p).resolve())

    def test_load_cookies_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "missing.json"
            with self.assertRaises(YoutubeAuthRequired) as ctx:
                load_cookies(missing_path)
            self.assertIn("Cookies missing or expired", str(ctx.exception))
            self.assertIn("streamslice youtube-login", str(ctx.exception))

    def test_load_cookies_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = Path(tmp_dir) / "invalid.json"
            invalid_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(YoutubeAuthRequired):
                load_cookies(invalid_path)

    def test_save_and_load_cookies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cookie_path = Path(tmp_dir) / "cookies.json"
            sample_cookies = [{"name": "SSID", "value": "12345", "domain": ".youtube.com"}]
            save_cookies(sample_cookies, cookie_path)
            loaded = load_cookies(cookie_path)
            self.assertGreaterEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["name"], "SSID")
            self.assertEqual(loaded[0]["value"], "12345")
            self.assertEqual(loaded[0]["domain"], ".youtube.com")

    def test_normalize_cookie_samesite_and_domains(self) -> None:
        # Check domain preservation and sameSite normalization
        c1 = {
            "name": "SID",
            "value": "xyz",
            "domain": ".youtube.com",
            "sameSite": "no_restriction",
            "secure": True,
            "httpOnly": True,
            "expirationDate": 1750000000,
        }
        n1 = normalize_cookie(c1)
        self.assertIsNotNone(n1)
        self.assertEqual(n1["name"], "SID")
        self.assertEqual(n1["domain"], ".youtube.com")
        self.assertEqual(n1["sameSite"], "None")
        self.assertTrue(n1["secure"])
        self.assertTrue(n1["httpOnly"])
        self.assertEqual(n1["expires"], 1750000000)

        # check strict / lax / invalid sameSite
        c2 = {"name": "HSID", "value": "123", "domain": "youtube.com", "sameSite": "strict"}
        n2 = normalize_cookie(c2)
        self.assertEqual(n2["sameSite"], "Strict")
        self.assertEqual(n2["domain"], "youtube.com")

        c3 = {"name": "APISID", "value": "123", "domain": ".google.com", "sameSite": "lax"}
        n3 = normalize_cookie(c3)
        self.assertEqual(n3["sameSite"], "Lax")
        self.assertEqual(n3["domain"], ".google.com")

        c4 = {"name": "SAPISID", "value": "123", "sameSite": "unspecified"}
        n4 = normalize_cookie(c4)
        self.assertNotIn("sameSite", n4)
        self.assertEqual(n4["domain"], ".youtube.com")

    def test_load_cookies_netscape_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            netscape_path = Path(tmp_dir) / "cookies.txt"
            netscape_content = (
                "# Netscape HTTP Cookie File\n"
                "#HttpOnly_.youtube.com\tTRUE\t/\tTRUE\t1750000000\tSSID\tabc123\n"
                ".google.com\tTRUE\t/\tFALSE\t0\tPREF\tf1=50000000\n"
            )
            netscape_path.write_text(netscape_content, encoding="utf-8")
            loaded = load_cookies(netscape_path)
            self.assertGreaterEqual(len(loaded), 2)
            names = [item["name"] for item in loaded]
            self.assertIn("SSID", names)
            self.assertIn("PREF", names)

    def test_selectors_multilingual_coverage(self) -> None:
        self.assertIn("название", SELECTOR_TITLE_INPUT.lower())
        self.assertIn("title", SELECTOR_TITLE_INPUT.lower())
        self.assertIn("contenteditable", SELECTOR_TITLE_INPUT)

        self.assertIn("description", SELECTOR_DESC_INPUT.lower())
        self.assertIn("contenteditable", SELECTOR_DESC_INPUT)

        self.assertIn("развернуть", SELECTOR_SHOW_MORE.lower())
        self.assertIn("show more", SELECTOR_SHOW_MORE.lower())
        self.assertIn("ещё", SELECTOR_SHOW_MORE.lower())

        self.assertIn("NOT_MADE_FOR_KIDS", SELECTOR_NOT_FOR_KIDS)
        self.assertIn("не для детей", SELECTOR_NOT_FOR_KIDS.lower())
        self.assertIn("not made for kids", SELECTOR_NOT_FOR_KIDS.lower())

        self.assertIn("next-button", SELECTOR_NEXT_BUTTON)
        self.assertIn("далее", SELECTOR_NEXT_BUTTON.lower())
        self.assertIn("next", SELECTOR_NEXT_BUTTON.lower())

        self.assertIn("public", SELECTOR_PUBLIC_RADIO.lower())
        self.assertIn("открытый доступ", SELECTOR_PUBLIC_RADIO.lower())

        self.assertIn("done-button", SELECTOR_PUBLISH_BUTTON)
        self.assertIn("publish", SELECTOR_PUBLISH_BUTTON.lower())
        self.assertIn("опубликовать", SELECTOR_PUBLISH_BUTTON.lower())
        self.assertIn("сохранить", SELECTOR_PUBLISH_BUTTON.lower())

        self.assertEqual(SELECTOR_UPLOAD_DIALOG_ATTACHED, "ytcp-uploads-dialog")
        self.assertIn("#dialog.ytcp-uploads-dialog", SELECTOR_UPLOAD_DIALOG_VISIBLE)

    def test_normalize_metadata(self) -> None:
        raw = {
            "youtube_title": "twitch: t2x2 #t2x2 #тоха #fyp #стрим",
            "youtube_description": "Описание ролика с лучшими моментами",
            "youtube_tags": "t2x2, тоха, т2х2, стрим, fyp, twitch",
            "category": "Gaming",
        }
        normalized = normalize_metadata(raw)
        self.assertEqual(normalized["title"], "twitch: t2x2 #t2x2 #тоха #fyp #стрим")
        self.assertEqual(normalized["description"], "Описание ролика с лучшими моментами")
        self.assertIn("t2x2", normalized["tags"])
        self.assertIn("тоха", normalized["tags"])
        self.assertEqual(normalized["category"], "Gaming")

    def test_normalize_metadata_from_standard_fields(self) -> None:
        raw = {
            "title": "ТОХА СГОРЕЛ НА БОССЕ",
            "description": "Стрим t2x2",
            "hashtags": ["#t2x2", "#fyp", "#viral"],
        }
        normalized = normalize_metadata(raw)
        self.assertEqual(normalized["title"], "ТОХА СГОРЕЛ НА БОССЕ")
        self.assertIn("#t2x2", normalized["description"])
        self.assertIn("t2x2", normalized["tags"])

    def test_format_as_shorts_url(self) -> None:
        self.assertEqual(
            _format_as_shorts_url("https://youtu.be/dQw4w9WgXcQ"),
            "https://youtube.com/shorts/dQw4w9WgXcQ",
        )
        self.assertEqual(
            _format_as_shorts_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "https://youtube.com/shorts/dQw4w9WgXcQ",
        )
        self.assertEqual(
            _format_as_shorts_url("https://youtube.com/shorts/dQw4w9WgXcQ"),
            "https://youtube.com/shorts/dQw4w9WgXcQ",
        )

    def test_upload_shorts_missing_video(self) -> None:
        uploader = YouTubeUploader()
        with self.assertRaises(FileNotFoundError):
            asyncio.run(
                uploader.upload_shorts("/non_existent_video_path_xyz.mp4", {"title": "Test"})
            )

    @patch("streamslice.youtube_uploader.async_playwright")
    @patch("streamslice.youtube_uploader.load_cookies")
    def test_upload_shorts_success(self, mock_load_cookies, mock_playwright) -> None:
        mock_load_cookies.return_value = [{"name": "SSID", "value": "val"}]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
            video_path = Path(temp_video.name)
            uploader = YouTubeUploader()

            with patch.object(
                uploader,
                "_perform_upload",
                new=AsyncMock(return_value={"copyright_issue": False, "url": "https://youtube.com/shorts/abc123"}),
            ):
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()
                mock_browser.new_context.return_value = mock_context
                mock_context.new_page.return_value = mock_page

                mock_p_instance = AsyncMock()
                mock_p_instance.chromium.launch.return_value = mock_browser
                mock_playwright.return_value.__aenter__.return_value = mock_p_instance

                res = asyncio.run(
                    uploader.upload_shorts(
                        video_path,
                        {
                            "youtube_title": "Test Title",
                            "youtube_description": "Test Desc",
                            "youtube_tags": "test, tags",
                        },
                    )
                )
                self.assertEqual(res, "https://youtube.com/shorts/abc123")

    @patch("streamslice.youtube_uploader.async_playwright")
    @patch("streamslice.youtube_uploader.load_cookies")
    @patch("streamslice.youtube_uploader.remove_music_if_blocked")
    def test_upload_shorts_copyright_retry_demucs(
        self, mock_remove_music, mock_load_cookies, mock_playwright
    ) -> None:
        mock_load_cookies.return_value = [{"name": "SSID", "value": "val"}]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
            video_path = Path(temp_video.name)
            uploader = YouTubeUploader()

            # First attempt: copyright issue -> demucs cleanup -> Second attempt: success
            mock_perform = AsyncMock(
                side_effect=[
                    {"copyright_issue": True, "url": ""},
                    {"copyright_issue": False, "url": "https://youtube.com/shorts/clean123"},
                ]
            )

            with (
                patch.object(uploader, "_perform_upload", new=mock_perform),
                patch.object(uploader, "_cancel_or_delete_draft", new=AsyncMock()),
            ):
                mock_browser = AsyncMock()
                mock_context = AsyncMock()
                mock_page = AsyncMock()
                mock_browser.new_context.return_value = mock_context
                mock_context.new_page.return_value = mock_page

                mock_p_instance = AsyncMock()
                mock_p_instance.chromium.launch.return_value = mock_browser
                mock_playwright.return_value.__aenter__.return_value = mock_p_instance

                # Make mock remove_music create the cleaned file
                def side_effect_demucs(src, dst):
                    Path(dst).write_bytes(b"clean_video")
                    return Path(dst)

                mock_remove_music.side_effect = side_effect_demucs

                res = asyncio.run(
                    uploader.upload_shorts(
                        video_path,
                        {"title": "Test With Music"},
                        max_copyright_retries=1,
                    )
                )
                self.assertEqual(res, "https://youtube.com/shorts/clean123")
                self.assertEqual(mock_perform.call_count, 2)
                self.assertEqual(mock_remove_music.call_count, 1)

    @patch("streamslice.youtube_uploader.async_playwright")
    @patch("streamslice.youtube_uploader.load_cookies")
    def test_perform_upload_mock_flow(self, mock_load_cookies, mock_playwright) -> None:
        mock_load_cookies.return_value = [
            {
                "name": "LOGIN_INFO",
                "value": "xyz",
                "domain": ".youtube.com",
                "sameSite": "no_restriction",
            },
        ]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
            video_path = Path(temp_video.name)
            uploader = YouTubeUploader()

            # Note: page.locator() is synchronous in Playwright async API
            mock_locator = MagicMock()
            mock_locator.first = mock_locator
            mock_locator.is_visible = AsyncMock(return_value=True)
            mock_locator.click = AsyncMock()
            mock_locator.fill = AsyncMock()
            mock_locator.wait_for = AsyncMock()
            mock_locator.set_input_files = AsyncMock()
            mock_locator.count = AsyncMock(return_value=1)
            mock_locator.get_attribute = AsyncMock(return_value="https://youtu.be/abcd1234efg")
            mock_locator.inner_text = AsyncMock(return_value="Checks complete. No issues found")
            mock_locator.is_enabled = AsyncMock(return_value=True)

            mock_page = MagicMock()
            mock_page.url = "https://studio.youtube.com/channel/UC123"
            mock_page.goto = AsyncMock()
            mock_page.locator.return_value = mock_locator
            mock_page.wait_for_selector = AsyncMock()
            mock_page.keyboard = MagicMock()
            mock_page.keyboard.press = AsyncMock()

            meta = {
                "title": "Short Title",
                "description": "Short Desc",
                "tags": "short, test",
            }

            res = asyncio.run(uploader._perform_upload(mock_page, video_path, meta))
            self.assertFalse(res["copyright_issue"])
            self.assertEqual(res["url"], "https://youtube.com/shorts/abcd1234efg")


if __name__ == "__main__":
    unittest.main()
