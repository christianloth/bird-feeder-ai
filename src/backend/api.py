"""
FastAPI REST API for the bird feeder monitoring system.

Provides endpoints for querying detections, species stats, weather data,
and system status. Will be consumed by the future dashboard.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from io import BytesIO
from typing import Annotated
from urllib.parse import urlencode

from pathlib import Path as _Path

from PIL import Image
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
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

_TEMPLATE_DIR = _Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


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


# --- Review UI ---


@app.get("/review", response_class=HTMLResponse)
def review_page(request: Request, session: SessionDep):
    """Serve the detection review UI."""
    # Count pending reviews
    pending = (
        session.query(func.count(Detection.id)).filter(Detection.reviewed.is_(False)).scalar() or 0
    )
    total = session.query(func.count(Detection.id)).scalar() or 0
    reviewed = total - pending

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={"pending": pending, "reviewed": reviewed, "total": total},
    )


@app.get("/api/review/next", response_class=HTMLResponse)
def review_next_card(request: Request, session: SessionDep):
    """Return the next unreviewed detection as an HTMX card fragment."""
    detection = (
        session.query(Detection)
        .filter(Detection.reviewed.is_(False))
        .order_by(desc(Detection.confidence))
        .first()
    )

    if not detection:
        return HTMLResponse(
            '<div id="review-card" class="card empty">'
            "<h2>All done!</h2><p>No more detections to review.</p>"
            "</div>"
        )

    species_name = detection.species.common_name if detection.species else "Unknown"

    # Get species list for correction dropdown (limited to likely species)
    species_list = session.query(Species).order_by(Species.common_name).all()

    # Count remaining
    pending = (
        session.query(func.count(Detection.id)).filter(Detection.reviewed.is_(False)).scalar() or 0
    )
    total = session.query(func.count(Detection.id)).scalar() or 0

    return templates.TemplateResponse(
        request=request,
        name="review_card.html",
        context={
            "detection": detection,
            "species_name": species_name,
            "species_list": species_list,
            "pending": pending,
            "total": total,
        },
    )


@app.post("/api/review/{detection_id}/confirm", response_class=HTMLResponse)
def review_confirm(detection_id: int, request: Request, session: SessionDep):
    """Confirm a detection as correct and return the next card."""
    detection = session.get(Detection, detection_id)
    if detection:
        detection.reviewed = True
        detection.is_false_positive = False
        session.commit()
    return review_next_card(request, session)


@app.post("/api/review/{detection_id}/reject", response_class=HTMLResponse)
def review_reject(detection_id: int, request: Request, session: SessionDep):
    """Mark a detection as false positive and return the next card."""
    detection = session.get(Detection, detection_id)
    if detection:
        detection.reviewed = True
        detection.is_false_positive = True
        session.commit()
    return review_next_card(request, session)


@app.post("/api/review/{detection_id}/correct/{species_id}", response_class=HTMLResponse)
def review_correct(
    detection_id: int,
    species_id: int,
    request: Request,
    session: SessionDep,
):
    """Correct the species and return the next card."""
    detection = session.get(Detection, detection_id)
    if detection:
        corrected = session.get(Species, species_id)
        if not corrected:
            raise HTTPException(status_code=404, detail="Species not found")
        detection.reviewed = True
        detection.is_false_positive = False
        detection.corrected_species_id = species_id
        session.commit()
    return review_next_card(request, session)


# --- Dashboard ---


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request, session: SessionDep):
    """Serve the main dashboard with gallery, stats, and filters."""
    total = session.query(func.count(Detection.id)).scalar() or 0
    unique_species = (
        session.query(func.count(func.distinct(Detection.species_id)))
        .filter(Detection.species_id.isnot(None))
        .scalar()
        or 0
    )
    today_start = datetime.combine(date.today(), datetime.min.time())
    today_count = (
        session.query(func.count(Detection.id)).filter(Detection.timestamp >= today_start).scalar()
        or 0
    )
    avg_conf = session.query(func.avg(Detection.confidence)).scalar() or 0.0
    disk_mb = _image_storage.get_disk_usage_mb()

    species_list = (
        session.query(Species)
        .filter(
            Species.id.in_(
                session.query(func.distinct(Detection.species_id)).filter(
                    Detection.species_id.isnot(None)
                )
            )
        )
        .order_by(Species.common_name)
        .all()
    )

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "total_detections": total,
            "unique_species": unique_species,
            "detections_today": today_count,
            "avg_confidence": round(avg_conf * 100, 1),
            "disk_usage_mb": round(disk_mb, 1),
            "species_list": species_list,
        },
    )


@app.get("/api/dashboard/gallery", response_class=HTMLResponse)
def dashboard_gallery(
    request: Request,
    session: SessionDep,
    species_id: str = "",
    since: str = "",
    until: str = "",
    min_confidence: str = "",
    reviewed: str = "",
    page: int = 1,
    per_page: int = 24,
):
    """Return gallery grid items as an HTMX fragment."""
    query = session.query(Detection).join(Detection.species, isouter=True)

    if species_id:
        query = query.filter(Detection.species_id == int(species_id))
    if since:
        query = query.filter(Detection.timestamp >= datetime.fromisoformat(since))
    if until:
        until_dt = datetime.fromisoformat(until) + timedelta(days=1)
        query = query.filter(Detection.timestamp < until_dt)
    if min_confidence:
        query = query.filter(Detection.confidence >= float(min_confidence))
    if reviewed == "pending":
        query = query.filter(Detection.reviewed.is_(False))
    elif reviewed == "reviewed":
        query = query.filter(Detection.reviewed.is_(True), Detection.is_false_positive.is_(False))
    elif reviewed == "false_positive":
        query = query.filter(Detection.is_false_positive.is_(True))

    total = query.count()
    offset = (page - 1) * per_page
    detections = query.order_by(desc(Detection.timestamp)).offset(offset).limit(per_page).all()
    has_more = offset + per_page < total

    # Build filter query string for pagination
    filter_params = {}
    if species_id:
        filter_params["species_id"] = species_id
    if since:
        filter_params["since"] = since
    if until:
        filter_params["until"] = until
    if min_confidence:
        filter_params["min_confidence"] = min_confidence
    if reviewed:
        filter_params["reviewed"] = reviewed
    filters_qs = urlencode(filter_params) if filter_params else ""

    return templates.TemplateResponse(
        request=request,
        name="gallery_items.html",
        context={
            "detections": detections,
            "page": page,
            "total": total,
            "has_more": has_more,
            "filters": filters_qs,
        },
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
        bbox_color = "#3fb950"
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
