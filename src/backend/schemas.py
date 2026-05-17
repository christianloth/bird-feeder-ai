"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime, date, time
from pydantic import BaseModel, Field


# --- Species ---

class SpeciesResponse(BaseModel):
    id: int
    common_name: str
    scientific_name: str
    family: str | None = None
    class_index: int

    model_config = {"from_attributes": True}


# --- Detections ---

class DetectionCreate(BaseModel):
    species_id: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    detection_model: str | None = None
    classifier_model: str | None = None
    bbox_x1: float | None = None
    bbox_y1: float | None = None
    bbox_x2: float | None = None
    bbox_y2: float | None = None
    frame_path: str | None = None


class DetectionResponse(BaseModel):
    id: int
    timestamp: datetime
    species_id: int | None = None
    species_name: str | None = None
    confidence: float                        # Stage-2 ViT classifier probability
    detector_confidence: float | None = None # Stage-1 detector (YOLO) bbox confidence (NULL for older rows)
    detection_model: str | None = None
    classifier_model: str | None = None
    bbox_x1: float | None = None
    bbox_y1: float | None = None
    bbox_x2: float | None = None
    bbox_y2: float | None = None
    frame_path: str | None = None
    reviewed: bool = False
    is_false_positive: bool = False
    corrected_species_id: int | None = None
    corrected_species_name: str | None = None
    source: str | None = None

    model_config = {"from_attributes": True}


class DetectionReview(BaseModel):
    is_false_positive: bool = False
    corrected_species_id: int | None = None  # Set if the species was misidentified


class BulkDeleteRequest(BaseModel):
    ids: list[int] = Field(default_factory=list)


class BulkDeleteResponse(BaseModel):
    deleted: int
    not_found: list[int] = Field(default_factory=list)


class IgnoreRegionResponse(BaseModel):
    id: int
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    frame_width: int | None = None
    frame_height: int | None = None
    overlap_threshold: float | None = None
    enabled: bool = True
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class IgnoreRegionCreate(BaseModel):
    label: str = ""
    x1: float
    y1: float
    x2: float
    y2: float
    frame_width: int | None = None
    frame_height: int | None = None
    overlap_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    enabled: bool = True


class IgnoreRegionUpdate(BaseModel):
    label: str | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    overlap_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    enabled: bool | None = None


class IgnoreSettings(BaseModel):
    overlap_threshold: float = Field(ge=0.0, le=1.0)


class IgnoreSettingsUpdate(BaseModel):
    overlap_threshold: float = Field(ge=0.0, le=1.0)


class DetectionStats(BaseModel):
    total_detections: int
    unique_species: int
    most_common_species: str | None = None
    most_common_count: int = 0
    detections_today: int = 0
    avg_confidence: float = 0.0


# --- Weather ---

class WeatherResponse(BaseModel):
    temperature_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_kmh: float | None = None
    precipitation_mm: float | None = None
    cloud_cover_pct: float | None = None
    weather_code: int | None = None
    weather_description: str | None = None
    timestamp: datetime | None = None


# --- Daily Summary ---

class DailySummaryResponse(BaseModel):
    date: date
    total_detections: int
    unique_species: int
    most_common_species: str | None = None
    avg_temperature: float | None = None
    sunrise: time | None = None
    sunset: time | None = None

    model_config = {"from_attributes": True}


# --- System ---

class SystemStatus(BaseModel):
    camera_connected: bool = False
    pipeline_running: bool = False
    total_detections: int = 0
    disk_usage_mb: float = 0.0
    uptime_seconds: float = 0.0
