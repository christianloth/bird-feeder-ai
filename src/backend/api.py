"""
FastAPI REST API for the bird feeder monitoring system.

Provides endpoints for querying detections, species stats, weather data,
and system status. Will be consumed by the future dashboard.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from src.backend.database import (
    Detection,
    Species,
    WeatherObservation,
    DailySummary,
    get_engine,
    create_tables,
    get_session_factory,
)
from src.backend.schemas import (
    DetectionResponse,
    DetectionStats,
    SpeciesResponse,
    WeatherResponse,
    DailySummaryResponse,
    SystemStatus,
)
from src.backend.weather import WeatherService, describe_weather_code
from src.backend.storage import ImageStorage

logger = logging.getLogger(__name__)

# Module-level state initialized at startup
_engine = None
_session_factory = None
_weather_service = None
_image_storage = None
_start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    global _engine, _session_factory, _weather_service, _image_storage, _start_time

    _engine = get_engine()
    create_tables(_engine)
    _session_factory = get_session_factory(_engine)
    _weather_service = WeatherService()
    _image_storage = ImageStorage()
    _start_time = datetime.now()

    logger.info("Bird Feeder AI API started.")
    yield

    if _weather_service:
        _weather_service.close()
    logger.info("Bird Feeder AI API stopped.")


app = FastAPI(
    title="Bird Feeder AI",
    description="API for querying bird detections, species stats, and weather data.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_db() -> Session:
    """Dependency that provides a database session."""
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


SessionDep = Annotated[Session, Depends(get_db)]


# --- Health ---

@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- Detections ---

@app.get("/api/detections", response_model=list[DetectionResponse])
def list_detections(
    session: SessionDep,
    skip: int = 0,
    limit: Annotated[int, Query(le=500)] = 50,
    species_id: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    min_confidence: float | None = None,
):
    """List detections with optional filtering."""
    query = session.query(Detection).join(Detection.species, isouter=True)

    if species_id is not None:
        query = query.filter(Detection.species_id == species_id)
    if since is not None:
        query = query.filter(Detection.timestamp >= since)
    if until is not None:
        query = query.filter(Detection.timestamp <= until)
    if min_confidence is not None:
        query = query.filter(Detection.confidence >= min_confidence)

    detections = query.order_by(desc(Detection.timestamp)).offset(skip).limit(limit).all()

    results = []
    for d in detections:
        resp = DetectionResponse.model_validate(d)
        resp.species_name = d.species.common_name if d.species else None
        results.append(resp)

    return results


@app.get("/api/detections/{detection_id}", response_model=DetectionResponse)
def get_detection(detection_id: int, session: SessionDep):
    """Get a single detection by ID."""
    detection = session.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    resp = DetectionResponse.model_validate(detection)
    resp.species_name = detection.species.common_name if detection.species else None
    return resp


@app.patch("/api/detections/{detection_id}/review")
def review_detection(
    detection_id: int,
    is_false_positive: bool,
    session: SessionDep,
):
    """Mark a detection as reviewed (correct or false positive)."""
    detection = session.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    detection.reviewed = True
    detection.is_false_positive = is_false_positive
    session.commit()
    return {"ok": True, "detection_id": detection_id, "is_false_positive": is_false_positive}


# --- Stats ---

@app.get("/api/stats", response_model=DetectionStats)
def get_stats(session: SessionDep):
    """Get overall detection statistics."""
    total = session.query(func.count(Detection.id)).scalar() or 0
    unique_species = (
        session.query(func.count(func.distinct(Detection.species_id)))
        .filter(Detection.species_id.isnot(None))
        .scalar()
        or 0
    )

    # Most common species
    most_common = (
        session.query(Species.common_name, func.count(Detection.id).label("cnt"))
        .join(Detection.species)
        .group_by(Species.id)
        .order_by(desc("cnt"))
        .first()
    )

    # Today's count
    today_start = datetime.combine(date.today(), datetime.min.time())
    today_count = (
        session.query(func.count(Detection.id))
        .filter(Detection.timestamp >= today_start)
        .scalar()
        or 0
    )

    # Average confidence
    avg_conf = session.query(func.avg(Detection.confidence)).scalar() or 0.0

    return DetectionStats(
        total_detections=total,
        unique_species=unique_species,
        most_common_species=most_common[0] if most_common else None,
        most_common_count=most_common[1] if most_common else 0,
        detections_today=today_count,
        avg_confidence=round(avg_conf, 3),
    )


@app.get("/api/stats/species", response_model=list[dict])
def get_species_stats(
    session: SessionDep,
    since: datetime | None = None,
):
    """Get detection count per species, ordered by frequency."""
    query = (
        session.query(
            Species.common_name,
            Species.scientific_name,
            func.count(Detection.id).label("count"),
            func.avg(Detection.confidence).label("avg_confidence"),
        )
        .join(Detection.species)
        .group_by(Species.id)
    )

    if since:
        query = query.filter(Detection.timestamp >= since)

    rows = query.order_by(desc("count")).all()

    return [
        {
            "species": row.common_name,
            "scientific_name": row.scientific_name,
            "count": row.count,
            "avg_confidence": round(row.avg_confidence, 3),
        }
        for row in rows
    ]


@app.get("/api/stats/hourly", response_model=list[dict])
def get_hourly_stats(
    session: SessionDep,
    target_date: date | None = None,
):
    """Get detection counts by hour for a given day."""
    target_date = target_date or date.today()
    start = datetime.combine(target_date, datetime.min.time())
    end = start + timedelta(days=1)

    detections = (
        session.query(Detection.timestamp)
        .filter(Detection.timestamp >= start, Detection.timestamp < end)
        .all()
    )

    # Count by hour
    hourly = {h: 0 for h in range(24)}
    for (ts,) in detections:
        hourly[ts.hour] += 1

    return [{"hour": h, "count": c} for h, c in hourly.items()]


# --- Species ---

@app.get("/api/species", response_model=list[SpeciesResponse])
def list_species(
    session: SessionDep,
    skip: int = 0,
    limit: Annotated[int, Query(le=1000)] = 100,
):
    """List all known species."""
    species = session.query(Species).offset(skip).limit(limit).all()
    return species


@app.get("/api/species/{species_id}", response_model=SpeciesResponse)
def get_species(species_id: int, session: SessionDep):
    """Get a single species by ID."""
    species = session.get(Species, species_id)
    if not species:
        raise HTTPException(status_code=404, detail="Species not found")
    return species


# --- Weather ---

@app.get("/api/weather/current", response_model=WeatherResponse)
def get_current_weather():
    """Get current weather conditions at the feeder location."""
    weather = _weather_service.get_current_weather()
    if not weather:
        raise HTTPException(status_code=503, detail="Weather service unavailable")
    weather["weather_description"] = describe_weather_code(weather.get("weather_code"))
    return weather


# --- Daily Summaries ---

@app.get("/api/daily", response_model=list[DailySummaryResponse])
def get_daily_summaries(
    session: SessionDep,
    days: Annotated[int, Query(le=365)] = 30,
):
    """Get daily summary data for the last N days."""
    cutoff = date.today() - timedelta(days=days)
    summaries = (
        session.query(DailySummary)
        .filter(DailySummary.date >= cutoff)
        .order_by(desc(DailySummary.date))
        .all()
    )
    return summaries


# --- System ---

@app.get("/api/system/status", response_model=SystemStatus)
def get_system_status(session: SessionDep):
    """Get system health and status information."""
    total = session.query(func.count(Detection.id)).scalar() or 0
    disk_mb = _image_storage.get_disk_usage_mb()
    uptime = (datetime.now() - _start_time).total_seconds() if _start_time else 0

    return SystemStatus(
        camera_connected=False,  # Updated when pipeline is integrated
        pipeline_running=False,  # Updated when pipeline is integrated
        total_detections=total,
        disk_usage_mb=round(disk_mb, 2),
        uptime_seconds=round(uptime, 1),
    )


@app.post("/api/system/cleanup")
def run_cleanup():
    """Manually trigger old image cleanup."""
    deleted = _image_storage.cleanup_old_images()
    return {"deleted_files": deleted}
