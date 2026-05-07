"""Background dispatcher that owns the Telegram worker thread.

The pipeline calls `maybe_enqueue` on the camera-loop thread; the worker
thread does the HTTP I/O, retries with backoff, and never blocks the
camera loop.
"""

from __future__ import annotations

import logging
import queue
import threading
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
        """Atomically gate + reserve + enqueue. Returns True iff enqueued.
        On enqueue failure, releases the reservation so the slot/cooldown
        isn't burned for a notification that never went out."""
        if nabirds_id is None:
            return False
        if not self._gate.check_and_record(nabirds_id, confidence, now):
            return False
        try:
            self._queue.put_nowait(TelegramMessage(
                chat_id=self._gate.cfg.chat_id,
                photo_url=photo_url,
                caption=caption,
            ))
        except queue.Full:
            logger.warning(
                "Telegram queue full (size=%d); dropping notification for "
                "nabirds_id=%d", self._queue.maxsize, nabirds_id,
            )
            self._gate.revert_record(nabirds_id, now)
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

    def _send_with_retry(self, msg: TelegramMessage, max_attempts: int = 3) -> None:
        for attempt in range(max_attempts):
            if self._stop.is_set():
                return
            if self._notifier.send_photo(msg):
                return
            if attempt < max_attempts - 1:
                # Cancellable backoff: wakes immediately if shutdown is set.
                if self._stop.wait(timeout=min(2 ** attempt, 4)):
                    return
        logger.warning(
            "Telegram send failed after %d attempts (chat=%s)",
            max_attempts, msg.chat_id,
        )

    def shutdown(self, timeout: float = 12.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        # Only close the client after the worker has actually exited; closing
        # while a request is in flight can corrupt internal httpx state.
        if not self._thread.is_alive():
            self._notifier.close()
