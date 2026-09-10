"""Upload exceptions, kept apart from the uploaders and their heavy dependencies.

Both uploaders once defined these classes themselves, so the same name referred
to two unrelated types and an ``except`` written against one silently failed to
catch the other. They live here so there is exactly one of each.

This module deliberately imports nothing: the CLI catches these exceptions on
every command, and the host that only queues jobs has neither the Google API
client nor Playwright installed.
"""
from __future__ import annotations


class YoutubeAuthRequired(Exception):
    """Credentials are missing or expired and interactive login is needed."""


class YoutubeUploadError(Exception):
    """The upload itself failed."""
