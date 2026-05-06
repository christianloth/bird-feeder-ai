"""
FastAPI REST API for the bird feeder monitoring system.

Provides endpoints for querying detections, species stats, weather data,
and system status. Will be consumed by the future dashboard.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from io import BytesIO
from typing import Annotated, Literal

from pathlib import Path as _Path

from PIL import Image
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from src.backend.database import (
    Detection,
    Species,
    DailySummary,
    get_engine,
    create_tables,
    get_session_factory,
    load_wildlife_species,
    migrate_species_category,
)
from src.backend.schemas import (
    DetectionResponse,
    DetectionReview,
    DetectionStats,
    SpeciesResponse,
    WeatherResponse,
    DailySummaryResponse,
    SystemStatus,
)
from src.backend.weather import WeatherService, describe_weather_code
from src.backend.weather_ingest import run_periodic_ingest, sync_weather
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
    migrate_species_category(_engine)
    _session_factory = get_session_factory(_engine)

    with _session_factory() as session:
        load_wildlife_species(session)

    _weather_service = WeatherService()
    _image_storage = ImageStorage()
    _start_time = datetime.now()

    try:
        await asyncio.to_thread(sync_weather, _session_factory, _weather_service)
    except Exception:
        logger.exception("Initial weather sync failed; will retry on schedule")

    ingest_task = asyncio.create_task(
        run_periodic_ingest(_session_factory, _weather_service)
    )

    logger.info("Bird Feeder AI API started.")
    yield

    ingest_task.cancel()
    try:
        await ingest_task
    except asyncio.CancelledError:
        pass

    if _weather_service:
        _weather_service.close()
    logger.info("Bird Feeder AI API stopped.")


app = FastAPI(
    title="Bird Feeder AI",
    description="API for querying bird detections, species stats, and weather data.",
    version="0.1.0",
    lifespan=lifespan,
)

_WEB_OUT = _Path(__file__).resolve().parents[2] / "web" / "out"


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


ReviewStatus = Literal["pending", "reviewed", "false_positive"]


def _apply_detection_filters(
    query,
    *,
    species_id: int | None,
    since: datetime | None,
    until: datetime | None,
    min_confidence: float | None,
    reviewed: ReviewStatus | None,
):
    if species_id is not None:
        query = query.filter(Detection.species_id == species_id)
    if since is not None:
        query = query.filter(Detection.timestamp >= since)
    if until is not None:
        query = query.filter(Detection.timestamp <= until)
    if min_confidence is not None:
        query = query.filter(Detection.confidence >= min_confidence)
    if reviewed == "pending":
        query = query.filter(Detection.reviewed.is_(False))
    elif reviewed == "reviewed":
        query = query.filter(
            Detection.reviewed.is_(True), Detection.is_false_positive.is_(False)
        )
    elif reviewed == "false_positive":
        query = query.filter(Detection.is_false_positive.is_(True))
    return query


@app.get("/api/detections", response_model=list[DetectionResponse])
def list_detections(
    session: SessionDep,
    skip: int = 0,
    limit: Annotated[int, Query(le=500)] = 50,
    species_id: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    min_confidence: float | None = None,
    reviewed: ReviewStatus | None = None,
):
    """List detections with optional filtering."""
    query = session.query(Detection).join(Detection.species, isouter=True)
    query = _apply_detection_filters(
        query,
        species_id=species_id,
        since=since,
        until=until,
        min_confidence=min_confidence,
        reviewed=reviewed,
    )

    detections = query.order_by(desc(Detection.timestamp)).offset(skip).limit(limit).all()

    results = []
    for d in detections:
        resp = DetectionResponse.model_validate(d)
        resp.species_name = d.species.common_name if d.species else None
        results.append(resp)

    return results


@app.get("/api/detections/count")
def count_detections(
    session: SessionDep,
    species_id: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    min_confidence: float | None = None,
    reviewed: ReviewStatus | None = None,
):
    """Return the total count of detections matching the same filters as /api/detections."""
    query = _apply_detection_filters(
        session.query(func.count(Detection.id)),
        species_id=species_id,
        since=since,
        until=until,
        min_confidence=min_confidence,
        reviewed=reviewed,
    )
    return {"count": query.scalar() or 0}


@app.get("/api/detections/{detection_id}", response_model=DetectionResponse)
def get_detection(detection_id: int, session: SessionDep):
    """Get a single detection by ID."""
    detection = session.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    resp = DetectionResponse.model_validate(detection)
    resp.species_name = detection.species.common_name if detection.species else None
    if detection.corrected_species:
        resp.corrected_species_name = detection.corrected_species.common_name
    return resp


@app.patch("/api/detections/{detection_id}/review")
def review_detection(
    detection_id: int,
    body: DetectionReview,
    session: SessionDep,
):
    """
    Mark a detection as reviewed.

    Set is_false_positive=true if the detection is wrong entirely.
    Set corrected_species_id if the species was misidentified (the detection
    is real but the classifier got the species wrong). The corrected label
    will be used when exporting training data.
    """
    detection = session.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    if body.corrected_species_id is not None:
        corrected = session.get(Species, body.corrected_species_id)
        if not corrected:
            raise HTTPException(
                status_code=404,
                detail=f"Corrected species ID {body.corrected_species_id} not found",
            )
        detection.corrected_species_id = body.corrected_species_id

    detection.reviewed = True
    detection.is_false_positive = body.is_false_positive
    session.commit()

    return {
        "ok": True,
        "detection_id": detection_id,
        "is_false_positive": body.is_false_positive,
        "corrected_species_id": detection.corrected_species_id,
    }


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
        session.query(func.count(Detection.id)).filter(Detection.timestamp >= today_start).scalar()
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


# --- Detection Images ---


@app.get("/api/detections/{detection_id}/crop")
def get_detection_crop(detection_id: int, session: SessionDep):
    """Serve the cropped detection image on-the-fly from the saved frame + bbox."""
    detection = session.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    if not detection.frame_path:
        raise HTTPException(status_code=404, detail="No frame image for this detection")

    frame_path = _image_storage.get_absolute_path(detection.frame_path)
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame image file not found on disk")

    if not all(
        v is not None
        for v in [
            detection.bbox_x1,
            detection.bbox_y1,
            detection.bbox_x2,
            detection.bbox_y2,
        ]
    ):
        raise HTTPException(status_code=400, detail="Detection missing bbox coordinates")

    crop = ImageStorage.crop_from_frame(
        frame_path,
        (detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2),
    )

    buf = BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/jpeg")


@app.get("/api/detections/{detection_id}/frame")
def get_detection_frame(detection_id: int, session: SessionDep):
    """Serve the full frame image for a detection."""
    detection = session.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    if not detection.frame_path:
        raise HTTPException(status_code=404, detail="No frame image for this detection")

    frame_path = _image_storage.get_absolute_path(detection.frame_path)
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame image file not found on disk")

    return Response(
        content=frame_path.read_bytes(),
        media_type="image/jpeg",
    )


# --- Annotated Frame ---


@app.get("/api/detections/{detection_id}/annotated")
def get_annotated_frame(detection_id: int, session: SessionDep):
    """Serve the full frame with bounding box drawn on it."""
    from PIL import ImageDraw, ImageFont

    detection = session.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    if not detection.frame_path:
        raise HTTPException(status_code=404, detail="No frame image for this detection")

    frame_path = _image_storage.get_absolute_path(detection.frame_path)
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame image file not found on disk")

    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)

    if all(
        v is not None
        for v in [detection.bbox_x1, detection.bbox_y1, detection.bbox_x2, detection.bbox_y2]
    ):
        x1, y1 = int(detection.bbox_x1), int(detection.bbox_y1)
        x2, y2 = int(detection.bbox_x2), int(detection.bbox_y2)

        # Draw bounding box
        bbox_color = "#ff0033"
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=bbox_color,
            )

        # Draw label background + text
        species_name = detection.species.common_name if detection.species else "Unknown"
        conf_pct = f"{detection.confidence * 100:.1f}%"
        label = f"{species_name} {conf_pct}"

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle(
            [x1, y1 - text_h - 8, x1 + text_w + 8, y1],
            fill=bbox_color,
        )
        draw.text((x1 + 4, y1 - text_h - 6), label, fill="white", font=font)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/jpeg")


# --- Delete Detection ---


@app.delete("/api/detections/{detection_id}")
def delete_detection(detection_id: int, session: SessionDep):
    """Delete a detection, removing both the database row and the image file from disk."""
    detection = session.get(Detection, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    # Delete the image file if it exists
    if detection.frame_path:
        frame_path = _image_storage.get_absolute_path(detection.frame_path)
        if frame_path.exists():
            frame_path.unlink()
            logger.info(f"Deleted image file: {detection.frame_path}")

            # Clean up empty parent directories
            parent = frame_path.parent
            while parent != _image_storage.base_dir:
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break

    session.delete(detection)
    session.commit()
    logger.info(f"Deleted detection #{detection_id}")

    return {"ok": True, "detection_id": detection_id}


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


# --- Web frontend (Next.js static export) ---
# Mounted last so all /api/* and /health routes registered above take precedence.
# Starlette's StaticFiles(html=True) handles directory-index resolution and
# trailing-slash redirects for /dashboard, /review, etc.

if _WEB_OUT.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(_WEB_OUT), html=True),
        name="web",
    )
else:
    logger.warning(
        "Next.js build output not found at %s. Run `cd web && npm run build`.",
        _WEB_OUT,
    )


if __name__ == "__main__":
    import uvicorn

    # Override uvicorn's default handlers to use stdout instead of stderr
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["handlers"]["default"]["stream"] = "ext://sys.stdout"
    log_config["handlers"]["access"]["stream"] = "ext://sys.stdout"

    uvicorn.run(
        "src.backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=log_config,
    )
