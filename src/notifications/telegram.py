"""Telegram Bot API client (sendPhoto via URL).

Talks directly to https://api.telegram.org over HTTPS so we are not coupled
to a wrapper library that may lag the Bot API. The Bot API method we use
(sendPhoto) is stable and has been part of the API since 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

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

    def __init__(self, bot_token: str, timeout: float = 15.0):
        self._url = f"{_API_BASE}/bot{bot_token}/sendPhoto"
        self._client = httpx.Client(timeout=timeout)

    def send_photo(self, msg: TelegramMessage) -> bool:
        """Send a photo. Returns True on success, False otherwise."""
        try:
            resp = self._client.post(
                self._url,
                data={
                    "chat_id": msg.chat_id,
                    "photo": msg.photo_url,
                    "caption": msg.caption,
                    "disable_web_page_preview": "true",
                },
            )
        except httpx.HTTPError as e:
            logger.warning("Telegram send_photo network error: %s", e)
            return False

        if resp.status_code == 200:
            return True

        # Surface Telegram's error description when possible — these are
        # usually informative (e.g. "wrong file identifier", "Forbidden:
        # bot was blocked by the user").
        try:
            payload = resp.json()
            desc = payload.get("description", resp.text[:200])
        except Exception:
            desc = resp.text[:200]
        logger.warning(
            "Telegram send_photo HTTP %d: %s", resp.status_code, desc
        )
        return False

    def close(self) -> None:
        self._client.close()
