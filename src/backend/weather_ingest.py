"""Background weather ingest that keeps weather_observations in sync."""

import asyncio
import logging
from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from src.backend.database import Detection, WeatherObservation
from src.backend.weather import WeatherService

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_WINDOW_DAYS = 7
DEFAULT_INTERVAL_S = 3600
MAX_BACKFILL_DAYS = 90


def _existing_hours(session, start: date, end: date) -> set[datetime]:
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt = datetime.combine(end, datetime.max.time())
    rows = session.execute(
        select(WeatherObservation.timestamp).where(
            WeatherObservation.timestamp >= start_dt,
            WeatherObservation.timestamp <= end_dt,
        )
    ).scalars().all()
    return {ts.replace(minute=0, second=0, microsecond=0) for ts in rows}


def backfill_range(
    session_factory: sessionmaker,
    weather_service: WeatherService,
    start_date: date,
    end_date: date,
) -> int:
    if end_date < start_date:
        return 0

    with session_factory() as session:
        have = _existing_hours(session, start_date, end_date)

    observations = weather_service.get_hourly_weather(
        start_date=start_date, end_date=end_date
    )

    with session_factory() as session:
        inserted = 0
        for obs in observations:
            ts = obs["timestamp"].replace(minute=0, second=0, microsecond=0)
            if ts in have:
                continue
            session.add(WeatherObservation(
                timestamp=ts,
                temperature_c=obs.get("temperature_c"),
                humidity_pct=obs.get("humidity_pct"),
                wind_speed_kmh=obs.get("wind_speed_kmh"),
                precipitation_mm=obs.get("precipitation_mm"),
                cloud_cover_pct=obs.get("cloud_cover_pct"),
                weather_code=obs.get("weather_code"),
            ))
            have.add(ts)
            inserted += 1
        session.commit()

    if inserted:
        logger.info(
            "Weather ingest: inserted %d hourly observations for %s..%s",
            inserted, start_date, end_date,
        )
    return inserted


def sync_weather(
    session_factory: sessionmaker,
    weather_service: WeatherService,
    refresh_window_days: int = DEFAULT_REFRESH_WINDOW_DAYS,
) -> int:
    today = date.today()
    earliest_allowed = today - timedelta(days=MAX_BACKFILL_DAYS)

    with session_factory() as session:
        min_detection = session.execute(
            select(func.min(Detection.timestamp))
        ).scalar_one_or_none()
        max_weather = session.execute(
            select(func.max(WeatherObservation.timestamp))
        ).scalar_one_or_none()

    candidates = [today - timedelta(days=refresh_window_days)]
    if min_detection is not None:
        candidates.append(min_detection.date())
    if max_weather is not None:
        candidates.append(max_weather.date())

    start = max(min(candidates), earliest_allowed)
    return backfill_range(session_factory, weather_service, start, today)


async def run_periodic_ingest(
    session_factory: sessionmaker,
    weather_service: WeatherService,
    interval_s: int = DEFAULT_INTERVAL_S,
    refresh_window_days: int = DEFAULT_REFRESH_WINDOW_DAYS,
) -> None:
    while True:
        try:
            await asyncio.to_thread(
                sync_weather, session_factory, weather_service, refresh_window_days
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Weather ingest iteration failed")
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
