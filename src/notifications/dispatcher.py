"""Background dispatcher that owns the Telegram worker thread.

The pipeline calls `maybe_enqueue` on the camera-loop thread; the worker
thread does the HTTP I/O, retries with backoff, and never blocks the
camera loop.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime

from src.notifications.telegram import TelegramMessage, TelegramNotifier
from src.notifications.watchlist import WatchlistGate

logger = logging.getLogger(__name__)


class NotificationDispatcher:
    def __init__(
        self,
        notifier: TelegramNotifier,
        gate: WatchlistGate,
        max_queue: int = 100,
    ):
        self._notifier = notifier
        self._gate = gate
        self._queue: queue.Queue[TelegramMessage | None] = queue.Queue(
            maxsize=max_queue
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="telegram-dispatcher", daemon=True,
        )
        self._thread.start()

    def maybe_enqueue(
        self,
        nabirds_id: int | None,
        confidence: float,
        photo_url: str,
        caption: str,
        now: datetime,
    ) -> bool:
        """Gate the detection; if pass, record + enqueue. Returns True if enqueued."""
        if not self._gate.should_notify(nabirds_id, confidence, now):
            return False
        # Record before enqueue so a flurry of detections within the cooldown
        # window can't queue multiple sends ahead of the worker.
        assert nabirds_id is not None
        self._gate.record_sent(nabirds_id, now)
        try:
            self._queue.put_nowait(TelegramMessage(
                chat_id=self._gate.cfg.chat_id,
                photo_url=photo_url,
                caption=caption,
            ))
        except queue.Full:
            logger.warning("Telegram queue full; dropping notification")
            return False
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg is None:
                break
            self._send_with_retry(msg)

    def _send_with_retry(self, msg: TelegramMessage) -> None:
        for attempt in range(3):
            if self._notifier.send_photo(msg):
                return
            time.sleep(min(2 ** attempt, 8))
        logger.warning("Telegram send failed after retries (chat=%s)", msg.chat_id)

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._notifier.close()
