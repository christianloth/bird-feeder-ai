"""
Day/night mode manager for the detection pipeline.

Uses sunrise/sunset times from Open-Meteo to switch between daytime
(bird species) and nighttime (wildlife) inference. Night mode activates
30 minutes after sunset and deactivates 30 minutes before sunrise.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

from config.settings import settings
from src.backend.weather import WeatherService

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Which inference path the pipeline should use."""
    DAY = "day"      # Bird detection (COCO YOLO) + species classification
    NIGHT = "night"  # Wildlife detection (custom YOLO, 11 classes)


class DayNightManager:
    """
    Manages transitions between daytime and nighttime inference modes.

    Fetches sunrise/sunset times once per day from Open-Meteo. Checks
    the current time at a configurable interval and switches modes:
      - Night mode ON:  sunset + 30 minutes
      - Night mode OFF: sunrise - 30 minutes

    Usage:
        weather = WeatherService()
        manager = DayNightManager(weather)
        manager.update()  # call periodically in the pipeline loop
        if manager.mode == PipelineMode.NIGHT:
            # use wildlife detector
    """

    def __init__(
        self,
        weather_service: WeatherService,
        check_interval: int | None = None,
        night_offset: int | None = None,
        day_offset: int | None = None,
    ):
        self._weather = weather_service
        self._check_interval = check_interval or settings.mode_check_interval
        self._night_offset = timedelta(minutes=night_offset or settings.night_offset_minutes)
        self._day_offset = timedelta(minutes=day_offset or settings.day_offset_minutes)
        self._tz = ZoneInfo(settings.timezone)

        self._mode = PipelineMode.DAY
        self._last_check: float = 0.0

        # Cached sun times for today
        self._day_start: datetime | None = None   # sunrise - offset
        self._night_start: datetime | None = None  # sunset + offset
        self._sun_date = None  # date we fetched sun times for

    @property
    def mode(self) -> PipelineMode:
        """Current pipeline mode (DAY or NIGHT)."""
        return self._mode

    @property
    def is_night(self) -> bool:
        return self._mode == PipelineMode.NIGHT

    @property
    def is_day(self) -> bool:
        return self._mode == PipelineMode.DAY

    def _refresh_sun_times(self) -> bool:
        """
        Fetch today's sunrise/sunset if we haven't already.

        Returns True if sun times are available.
        """
        today = datetime.now(self._tz).date()
        if self._sun_date == today:
            return True

        sun_times = self._weather.get_daily_sun_times(today)
        if sun_times is None:
            logger.error("Failed to fetch sunrise/sunset times from Open-Meteo")
            return self._day_start is not None  # use stale data if available

        sunrise = sun_times["sunrise"].replace(tzinfo=self._tz)
        sunset = sun_times["sunset"].replace(tzinfo=self._tz)

        self._day_start = sunrise - self._day_offset
        self._night_start = sunset + self._night_offset
        self._sun_date = today

        logger.info(
            f"Sun times for {today}: sunrise {sunrise.strftime('%H:%M')}, "
            f"sunset {sunset.strftime('%H:%M')} | "
            f"day mode from {self._day_start.strftime('%H:%M')}, "
            f"night mode from {self._night_start.strftime('%H:%M')}"
        )
        return True

    def _compute_mode(self) -> PipelineMode:
        """Determine the correct mode based on current time and sun times."""
        now = datetime.now(self._tz)

        # day_start <= now < night_start  =>  DAY
        # otherwise                       =>  NIGHT
        if self._day_start <= now < self._night_start:
            return PipelineMode.DAY
        return PipelineMode.NIGHT

    def update(self) -> bool:
        """
        Check if a mode transition is needed.

        Call this periodically from the pipeline loop. It rate-limits
        checks to once per check_interval.

        Returns:
            True if the mode changed since the last call.
        """
        now = time.time()
        if now - self._last_check < self._check_interval:
            return False

        self._last_check = now

        if not self._refresh_sun_times():
            return False

        previous_mode = self._mode
        self._mode = self._compute_mode()

        if self._mode != previous_mode:
            logger.info(
                f"Mode transition: {previous_mode.value} -> {self._mode.value}"
            )
            return True

        return False

    def force_mode(self, mode: PipelineMode) -> None:
        """Manually override the pipeline mode."""
        if mode != self._mode:
            logger.info(f"Mode forced: {self._mode.value} -> {mode.value}")
            self._mode = mode

    def check_now(self) -> bool:
        """
        Force an immediate mode check, ignoring the rate limit.

        Returns True if the mode changed.
        """
        self._last_check = 0.0
        return self.update()
