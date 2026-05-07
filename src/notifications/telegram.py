"""Telegram Bot API client (sendPhoto via URL).

Talks directly to https://api.telegram.org over HTTPS so we are not coupled
to a wrapper library that may lag the Bot API. The Bot API method we use
(sendPhoto) is stable and has been part of the API since 1.0.

Security: the bot token sits inside the request URL path. We must NEVER let
httpx's default INFO request log or its exception strings leak the token to
disk — those would write the token into pipeline.log on every send.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# httpx logs every request URL at INFO ("HTTP Request: POST
# https://api.telegram.org/bot<TOKEN>/sendPhoto ..."). Suppress at the source
# so any pipeline running at INFO doesn't persist the token to disk.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_API_BASE = "https://api.telegram.org"


@dataclass(frozen=True)
class TelegramMessage:
    """A message ready to be sent."""

    chat_id: str
    photo_url: str
    caption: str


class TelegramNotifier:
    """Synchronous Telegram client for sending a photo with a caption.

    The photo is passed as a URL — Telegram fetches it itself, so this works
    as long as `photo_url` is publicly reachable and < 5 MB.
    """

    def __init__(self, bot_token: str, timeout: float = 10.0):
        self._token = bot_token
        self._url = f"{_API_BASE}/bot{bot_token}/sendPhoto"
        self._client = httpx.Client(timeout=timeout)

    def _scrub(self, s: str) -> str:
        # httpx exception strings often include the request URL, which has
        # the bot token in the path. Strip it before any logging.
        return s.replace(self._token, "<bot_token>") if self._token else s

    def send_photo(self, msg: TelegramMessage) -> bool:
        """Send a photo. Returns True on success, False otherwise."""
        # Note: sendPhoto ignores `disable_web_page_preview` (that field is for
        # sendMessage); when a photo is attached, link previews aren't generated
        # for URLs in the caption anyway.
        try:
            resp = self._client.post(
                self._url,
                data={
                    "chat_id": msg.chat_id,
                    "photo": msg.photo_url,
                    "caption": msg.caption,
                },
            )
        except httpx.HTTPError as e:
            logger.warning(
                "Telegram send_photo %s: %s",
                e.__class__.__name__,
                self._scrub(str(e)),
            )
            return False

        if resp.status_code == 200:
            logger.info(
                "Telegram sendPhoto OK (chat=%s, caption=%r)",
                msg.chat_id, msg.caption.splitlines()[0][:80],
            )
            return True

        # Surface Telegram's error description when possible — these are
        # usually informative (e.g. "wrong file identifier", "Forbidden:
        # bot was blocked by the user"). Scrub defensively in case any
        # response body echoes the URL back.
        try:
            payload = resp.json()
            desc = payload.get("description", resp.text[:200])
        except Exception:
            desc = resp.text[:200]
        logger.warning(
            "Telegram send_photo HTTP %d: %s",
            resp.status_code, self._scrub(desc),
        )
        return False

    def close(self) -> None:
        self._client.close()
