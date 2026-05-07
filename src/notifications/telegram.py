"""Telegram Bot API client (sendMediaGroup via URL).

Talks directly to https://api.telegram.org over HTTPS so we are not coupled
to a wrapper library that may lag the Bot API. The Bot API method we use
(sendMediaGroup) is stable and has been part of the API since 2.3.

Security: the bot token sits inside the request URL path. We must NEVER let
httpx's default INFO request log or its exception strings leak the token to
disk — those would write the token into pipeline.log on every send.
"""

from __future__ import annotations

import json
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
    """A message ready to be sent. `photo_urls` is a 2–10 element list; the
    images are sent as a single swipeable album. Caption attaches to the
    first photo (the album's caption)."""

    chat_id: str
    photo_urls: list[str]
    caption: str


class TelegramNotifier:
    """Synchronous Telegram client for sending a photo album.

    Each URL in `photo_urls` must be publicly reachable and < 5 MB; Telegram
    fetches them itself.
    """

    def __init__(self, bot_token: str, timeout: float = 10.0):
        self._token = bot_token
        self._url = f"{_API_BASE}/bot{bot_token}/sendMediaGroup"
        self._client = httpx.Client(timeout=timeout)

    def _scrub(self, s: str) -> str:
        # httpx exception strings often include the request URL, which has
        # the bot token in the path. Strip it before any logging.
        return s.replace(self._token, "<bot_token>") if self._token else s

    def send_album(self, msg: TelegramMessage) -> bool:
        """Send an album of photos. Returns True on success, False otherwise."""
        if not 2 <= len(msg.photo_urls) <= 10:
            logger.warning(
                "Telegram album needs 2-10 photos, got %d", len(msg.photo_urls)
            )
            return False
        media = [
            {"type": "photo", "media": url, **({"caption": msg.caption} if i == 0 else {})}
            for i, url in enumerate(msg.photo_urls)
        ]
        try:
            resp = self._client.post(
                self._url,
                data={
                    "chat_id": msg.chat_id,
                    # The Bot API expects `media` as a JSON-encoded string in
                    # form-encoded requests, not a nested object.
                    "media": json.dumps(media),
                },
            )
        except httpx.HTTPError as e:
            logger.warning(
                "Telegram send_album %s: %s",
                e.__class__.__name__,
                self._scrub(str(e)),
            )
            return False

        if resp.status_code == 200:
            logger.debug(
                "Telegram sendMediaGroup OK (chat=%s, photos=%d, caption=%r)",
                msg.chat_id, len(msg.photo_urls),
                msg.caption.splitlines()[0][:80],
            )
            return True

        # Surface Telegram's error description (e.g. "Forbidden: bot was
        # blocked by the user"). Scrub defensively in case any response body
        # echoes the URL back.
        try:
            payload = resp.json()
            desc = payload.get("description", resp.text[:200])
        except Exception:
            desc = resp.text[:200]
        logger.warning(
            "Telegram send_album HTTP %d: %s",
            resp.status_code, self._scrub(desc),
        )
        return False

    def close(self) -> None:
        self._client.close()
