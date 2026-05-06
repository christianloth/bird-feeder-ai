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
from datetime import datetime, time as dtime
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
        local = now.astimezone(self._tz)
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
                logger.info(
                    "Notification suppressed (daily cap %d reached)",
                    self.cfg.daily_cap,
                )
                return False
            last = self._last_sent.get(nabirds_id)
            if last is not None:
                remaining = self.cfg.cooldown_seconds - (now.timestamp() - last)
                if remaining > 0:
                    logger.debug(
                        "Notification suppressed (cooldown for %d, %.0fs left)",
                        nabirds_id, remaining,
                    )
                    return False
        return True

    def record_sent(self, nabirds_id: int, now: datetime) -> None:
        local_date = now.astimezone(self._tz).date().isoformat()
        with self._lock:
            self._roll_daily(local_date)
            self._last_sent[nabirds_id] = now.timestamp()
            self._daily_count += 1
            self._save()

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

    def _save(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "last_sent": {str(k): v for k, v in self._last_sent.items()},
                "daily_date": self._daily_date,
                "daily_count": self._daily_count,
            }))
            tmp.replace(self._state_path)
        except OSError as e:
            logger.warning("Failed to persist notification state: %s", e)
