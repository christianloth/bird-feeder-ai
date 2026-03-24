"""
SQLAlchemy 2.0 database models and session management.

Tables:
- detections: Every bird sighting with species, confidence, bounding box, image path
- species: Lookup table mapping class indices to species names
- weather_observations: Hourly weather data for correlating bird activity
- daily_summary: Pre-computed daily aggregates for dashboard queries
"""

from datetime import datetime, date, time
from pathlib import Path

from sqlalchemy import (
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Date,
    Time,
    ForeignKey,
    Index,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session,
    sessionmaker,
)

from config.settings import settings


class Base(DeclarativeBase):
    pass


class Species(Base):
    __tablename__ = "species"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    common_name: Mapped[str] = mapped_column(String(200), nullable=False)
    scientific_name: Mapped[str] = mapped_column(String(200), nullable=False)
    family: Mapped[str | None] = mapped_column(String(200))
    class_index: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)

    detections: Mapped[list["Detection"]] = relationship(back_populates="species")

    def __repr__(self) -> str:
        return f"<Species {self.common_name} ({self.scientific_name})>"


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    species_id: Mapped[int | None] = mapped_column(ForeignKey("species.id"), index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detection_model: Mapped[str | None] = mapped_column(String(100))
    classifier_model: Mapped[str | None] = mapped_column(String(100))
    bbox_x1: Mapped[float | None] = mapped_column(Float)
    bbox_y1: Mapped[float | None] = mapped_column(Float)
    bbox_x2: Mapped[float | None] = mapped_column(Float)
    bbox_y2: Mapped[float | None] = mapped_column(Float)
    image_path: Mapped[str | None] = mapped_column(String(500))
    thumbnail_path: Mapped[str | None] = mapped_column(String(500))
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    is_false_positive: Mapped[bool] = mapped_column(Boolean, default=False)

    species: Mapped[Species | None] = relationship(back_populates="detections")

    __table_args__ = (
        Index("idx_detections_confidence", "confidence"),
        Index("idx_detections_timestamp_species", "timestamp", "species_id"),
    )

    def __repr__(self) -> str:
        species_name = self.species.common_name if self.species else "Unknown"
        return f"<Detection {species_name} @ {self.timestamp} ({self.confidence:.2f})>"


class WeatherObservation(Base):
    __tablename__ = "weather_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    temperature_c: Mapped[float | None] = mapped_column(Float)
    humidity_pct: Mapped[float | None] = mapped_column(Float)
    wind_speed_kmh: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float)
    weather_code: Mapped[int | None] = mapped_column(Integer)


class DailySummary(Base):
    __tablename__ = "daily_summary"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    total_detections: Mapped[int] = mapped_column(Integer, default=0)
    unique_species: Mapped[int] = mapped_column(Integer, default=0)
    most_common_species_id: Mapped[int | None] = mapped_column(ForeignKey("species.id"))
    avg_temperature: Mapped[float | None] = mapped_column(Float)
    sunrise: Mapped[time | None] = mapped_column(Time)
    sunset: Mapped[time | None] = mapped_column(Time)


def _set_sqlite_pragmas(dbapi_conn, connection_record):
    """Enable WAL mode and other performance pragmas for SQLite."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    cursor.close()


def get_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine for the bird database."""
    url = database_url or settings.database_url
    engine = create_engine(url, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _set_sqlite_pragmas)
    return engine


def create_tables(engine=None):
    """Create all database tables if they don't exist."""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine=None) -> sessionmaker[Session]:
    """Create a session factory bound to the engine."""
    engine = engine or get_engine()
    return sessionmaker(bind=engine)


def get_session(engine=None) -> Session:
    """Create a new database session. Remember to close it when done."""
    factory = get_session_factory(engine)
    return factory()


def load_species_from_dataset(session: Session, classes_file: Path) -> None:
    """
    Load species from the NABirds classes.txt file into the database.

    Args:
        session: Active database session
        classes_file: Path to NABirds classes.txt (format: "class_id species_name")
    """
    existing = session.query(Species).count()
    if existing > 0:
        return

    with open(classes_file) as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                original_class_id, name = parts
                species = Species(
                    common_name=name,
                    scientific_name=name,
                    class_index=idx,
                )
                session.add(species)

    session.commit()
