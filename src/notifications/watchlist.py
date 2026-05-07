"""Watchlist gating: decide whether a detection should fire a notification.

Holds per-species cooldown timers and a daily cap, persisting to a JSON
file so a pipeline restart doesn't lose cooldown state. Quiet hours and
daily rollover are evaluated in the user's local timezone.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateConfig:
    enabled: bool
    chat_id: str
    site_base_url: str
    photo_kind: str
    cooldown_seconds: int
    daily_cap: int
    quiet_hours_start: str | None
    quiet_hours_end: str | None
    watchlist: dict[int, float] = field(default_factory=dict)
    timezone: str = "America/Chicago"


def _parse_hhmm(s: str | None) -> dtime | None:
    if not s:
        return None
    try:
        h, m = s.split(":")
        return dtime(int(h), int(m))
    except (ValueError, AttributeError):
        return None


def _in_quiet_hours(now: dtime, start: dtime, end: dtime) -> bool:
    """Inclusive of start, exclusive of end. Handles overnight ranges."""
    if start <= end:
        return start <= now < end
    # Overnight range, e.g. 22:00 → 07:00
    return now >= start or now < end


def _as_utc(now: datetime) -> datetime:
    """Pipeline timestamps are naive but contain UTC numbers (the pipeline
    explicitly does `replace(tzinfo=None)` after `fromtimestamp(tz=utc)`).
    Treating naive as system-local would make quiet hours and the daily
    rollover off by the UTC offset, so attach UTC explicitly."""
    return now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)


class WatchlistGate:
    """Decides whether a detection should fire a notification."""

    def __init__(self, cfg: GateConfig, state_path: Path):
        self.cfg = cfg
        self._state_path = state_path
        self._tz = ZoneInfo(cfg.timezone)
        self._quiet_start = _parse_hhmm(cfg.quiet_hours_start)
        self._quiet_end = _parse_hhmm(cfg.quiet_hours_end)
        self._lock = threading.Lock()
        self._last_sent: dict[int, float] = {}
        self._daily_date: str | None = None
        self._daily_count: int = 0
        self._load()

    def should_notify(
        self, nabirds_id: int | None, confidence: float, now: datetime
    ) -> bool:
        if not self.cfg.enabled or not self.cfg.chat_id:
            return False
        if nabirds_id is None:
            return False
        threshold = self.cfg.watchlist.get(nabirds_id)
        if threshold is None or confidence < threshold:
            return False
        utc = _as_utc(now)
        local = utc.astimezone(self._tz)
        if self._quiet_start and self._quiet_end:
            if _in_quiet_hours(local.time(), self._quiet_start, self._quiet_end):
                logger.debug(
                    "Notification suppressed (quiet hours): %s",
                    local.strftime("%H:%M"),
                )
                return False
        with self._lock:
            self._roll_daily(local.date().isoformat())
            if self._daily_count >= self.cfg.daily_cap:
                logger.debug(
                    "Notification suppressed (daily cap %d reached)",
                    self.cfg.daily_cap,
                )
                return False
            last = self._last_sent.get(nabirds_id)
            if last is not None:
                remaining = self.cfg.cooldown_seconds - (utc.timestamp() - last)
                if remaining > 0:
                    logger.debug(
                        "Notification suppressed (cooldown for %d, %.0fs left)",
                        nabirds_id, remaining,
                    )
                    return False
        return True

    def record_sent(self, nabirds_id: int, now: datetime) -> None:
        utc = _as_utc(now)
        local_date = utc.astimezone(self._tz).date().isoformat()
        with self._lock:
            self._roll_daily(local_date)
            self._last_sent[nabirds_id] = utc.timestamp()
            self._daily_count += 1
            snapshot = self._snapshot()
        # Persist outside the lock so disk I/O can't backpressure the
        # camera thread.
        self._save_snapshot(*snapshot)

    def check_and_record(
        self, nabirds_id: int | None, confidence: float, now: datetime
    ) -> bool:
        """Atomic gate + reservation. Returns True iff the caller should
        proceed to send (and the slot has been reserved). On a False return,
        nothing is mutated. Use revert_record() to release the reservation
        if the downstream send/enqueue fails."""
        if not self.cfg.enabled or not self.cfg.chat_id:
            return False
        if nabirds_id is None:
            return False
        threshold = self.cfg.watchlist.get(nabirds_id)
        if threshold is None:
            return False
        if confidence < threshold:
            logger.debug(
                "Notification suppressed (below threshold): nabirds_id=%d "
                "conf=%.2f < %.2f",
                nabirds_id, confidence, threshold,
            )
            return False
        utc = _as_utc(now)
        local = utc.astimezone(self._tz)
        if self._quiet_start and self._quiet_end:
            if _in_quiet_hours(local.time(), self._quiet_start, self._quiet_end):
                logger.debug(
                    "Notification suppressed (quiet hours): nabirds_id=%d "
                    "conf=%.2f at local %s",
                    nabirds_id, confidence, local.strftime("%H:%M"),
                )
                return False
        ts = utc.timestamp()
        with self._lock:
            self._roll_daily(local.date().isoformat())
            if self._daily_count >= self.cfg.daily_cap:
                logger.debug(
                    "Notification suppressed (daily cap %d reached)",
                    self.cfg.daily_cap,
                )
                return False
            last = self._last_sent.get(nabirds_id)
            if last is not None:
                remaining = self.cfg.cooldown_seconds - (ts - last)
                if remaining > 0:
                    logger.debug(
                        "Notification suppressed (cooldown): nabirds_id=%d "
                        "conf=%.2f, %.0fs left",
                        nabirds_id, confidence, remaining,
                    )
                    return False
            # All checks pass — reserve the slot atomically.
            self._last_sent[nabirds_id] = ts
            self._daily_count += 1
            snapshot = self._snapshot()
        self._save_snapshot(*snapshot)
        return True

    def revert_record(self, nabirds_id: int, now: datetime) -> None:
        """Undo a check_and_record reservation if the caller couldn't enqueue."""
        utc = _as_utc(now)
        ts = utc.timestamp()
        with self._lock:
            # Only remove if the timestamp still matches — guards against a
            # later record_sent for the same species silently being undone.
            if self._last_sent.get(nabirds_id) == ts:
                del self._last_sent[nabirds_id]
            self._daily_count = max(0, self._daily_count - 1)
            snapshot = self._snapshot()
        self._save_snapshot(*snapshot)

    def _snapshot(self) -> tuple[dict[int, float], str | None, int]:
        """Capture state under the caller's lock for off-lock persistence."""
        return (dict(self._last_sent), self._daily_date, self._daily_count)

    def _roll_daily(self, today: str) -> None:
        if self._daily_date != today:
            self._daily_date = today
            self._daily_count = 0

    def _load(self) -> None:
        try:
            data = json.loads(self._state_path.read_text())
        except FileNotFoundError:
            return
        except json.JSONDecodeError:
            logger.warning(
                "Notification state at %s is corrupt; starting fresh",
                self._state_path,
            )
            return
        try:
            self._last_sent = {
                int(k): float(v) for k, v in data.get("last_sent", {}).items()
            }
            self._daily_date = data.get("daily_date")
            self._daily_count = int(data.get("daily_count", 0))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid notification state at %s; ignoring",
                self._state_path,
            )

    def _save_snapshot(
        self,
        last_sent: dict[int, float],
        daily_date: str | None,
        daily_count: int,
    ) -> None:
        """Persist a snapshot. Caller must NOT hold self._lock — disk I/O
        can stall on a slow SD card and would otherwise backpressure the
        camera thread."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "last_sent": {str(k): v for k, v in last_sent.items()},
                "daily_date": daily_date,
                "daily_count": daily_count,
            }))
            tmp.replace(self._state_path)
        except OSError as e:
            logger.warning("Failed to persist notification state: %s", e)
