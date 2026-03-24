"""Tests for the FastAPI REST API endpoints."""

from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.backend.database import Base, Species, Detection
from src.backend.api import app, get_db


test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(test_engine)
TestSession = sessionmaker(bind=test_engine)


def override_get_db():
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def _seed_data():
    with TestSession() as session:
        session.query(Detection).delete()
        session.query(Species).delete()

        cardinal = Species(
            id=1, common_name="Northern Cardinal",
            scientific_name="Cardinalis cardinalis",
            family="Cardinalidae", class_index=0,
        )
        blue_jay = Species(
            id=2, common_name="Blue Jay",
            scientific_name="Cyanocitta cristata",
            family="Corvidae", class_index=1,
        )
        session.add_all([cardinal, blue_jay])
        session.flush()

        session.add_all([
            Detection(timestamp=datetime(2025, 3, 15, 10, 0), species_id=1, confidence=0.95),
            Detection(timestamp=datetime(2025, 3, 15, 11, 0), species_id=1, confidence=0.88),
            Detection(timestamp=datetime(2025, 3, 15, 12, 0), species_id=2, confidence=0.92),
        ])
        session.commit()


def test_health():
    assert client.get("/health").status_code == 200


def test_list_detections_and_filter():
    _seed_data()
    # All detections
    data = client.get("/api/detections").json()
    assert len(data) == 3
    assert data[0]["species_name"] == "Blue Jay"  # Most recent first

    # Filter by species
    data = client.get("/api/detections", params={"species_id": 1}).json()
    assert len(data) == 2

    # Filter by confidence
    data = client.get("/api/detections", params={"min_confidence": 0.90}).json()
    assert len(data) == 2


def test_review_detection():
    _seed_data()
    resp = client.patch("/api/detections/1/review", params={"is_false_positive": True})
    assert resp.status_code == 200

    det = client.get("/api/detections/1").json()
    assert det["reviewed"] is True
    assert det["is_false_positive"] is True


def test_stats():
    _seed_data()
    data = client.get("/api/stats").json()
    assert data["total_detections"] == 3
    assert data["unique_species"] == 2
    assert data["most_common_species"] == "Northern Cardinal"

    species_data = client.get("/api/stats/species").json()
    assert species_data[0]["species"] == "Northern Cardinal"
    assert species_data[0]["count"] == 2


def test_species_endpoints():
    _seed_data()
    data = client.get("/api/species").json()
    assert len(data) == 2

    assert client.get("/api/species/9999").status_code == 404
