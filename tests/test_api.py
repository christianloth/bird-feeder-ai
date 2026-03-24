"""Tests for the FastAPI REST API."""

from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.backend.database import Base, Species, Detection
from src.backend.api import app, get_db


# Use StaticPool so all connections share the same in-memory database
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
    """Insert test species and detections."""
    with TestSession() as session:
        # Clear existing data
        session.query(Detection).delete()
        session.query(Species).delete()

        cardinal = Species(
            id=1,
            common_name="Northern Cardinal",
            scientific_name="Cardinalis cardinalis",
            family="Cardinalidae",
            class_index=0,
        )
        blue_jay = Species(
            id=2,
            common_name="Blue Jay",
            scientific_name="Cyanocitta cristata",
            family="Corvidae",
            class_index=1,
        )
        session.add_all([cardinal, blue_jay])
        session.flush()

        detections = [
            Detection(timestamp=datetime(2025, 3, 15, 10, 0), species_id=1, confidence=0.95),
            Detection(timestamp=datetime(2025, 3, 15, 11, 0), species_id=1, confidence=0.88),
            Detection(timestamp=datetime(2025, 3, 15, 12, 0), species_id=2, confidence=0.92),
        ]
        session.add_all(detections)
        session.commit()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_species():
    _seed_data()
    response = client.get("/api/species")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    names = {s["common_name"] for s in data}
    assert "Northern Cardinal" in names
    assert "Blue Jay" in names


def test_get_species_by_id():
    _seed_data()
    response = client.get("/api/species/1")
    assert response.status_code == 200
    assert response.json()["common_name"] == "Northern Cardinal"


def test_get_species_not_found():
    response = client.get("/api/species/9999")
    assert response.status_code == 404


def test_list_detections():
    _seed_data()
    response = client.get("/api/detections")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    # Should be ordered by timestamp descending
    assert data[0]["species_name"] == "Blue Jay"


def test_list_detections_filter_by_species():
    _seed_data()
    response = client.get("/api/detections", params={"species_id": 1})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(d["species_id"] == 1 for d in data)


def test_list_detections_filter_by_confidence():
    _seed_data()
    response = client.get("/api/detections", params={"min_confidence": 0.90})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(d["confidence"] >= 0.90 for d in data)


def test_get_detection_by_id():
    _seed_data()
    response = client.get("/api/detections/1")
    assert response.status_code == 200
    assert response.json()["confidence"] == 0.95


def test_review_detection():
    _seed_data()
    response = client.patch(
        "/api/detections/1/review",
        params={"is_false_positive": True},
    )
    assert response.status_code == 200
    assert response.json()["is_false_positive"] is True

    # Verify it persisted
    response = client.get("/api/detections/1")
    assert response.json()["reviewed"] is True
    assert response.json()["is_false_positive"] is True


def test_get_stats():
    _seed_data()
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_detections"] == 3
    assert data["unique_species"] == 2
    assert data["most_common_species"] == "Northern Cardinal"
    assert data["most_common_count"] == 2


def test_get_species_stats():
    _seed_data()
    response = client.get("/api/stats/species")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    # Northern Cardinal should be first (most frequent)
    assert data[0]["species"] == "Northern Cardinal"
    assert data[0]["count"] == 2
