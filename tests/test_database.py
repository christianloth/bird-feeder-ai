"""Tests for database models and operations."""

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.backend.database import Base, Species, Detection, create_tables


def get_test_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_create_tables():
    engine = get_test_engine()
    table_names = Base.metadata.tables.keys()
    assert "species" in table_names
    assert "detections" in table_names
    assert "weather_observations" in table_names
    assert "daily_summary" in table_names


def test_add_species():
    engine = get_test_engine()
    with Session(engine) as session:
        species = Species(
            common_name="Northern Cardinal",
            scientific_name="Cardinalis cardinalis",
            family="Cardinalidae",
            class_index=0,
        )
        session.add(species)
        session.commit()

        result = session.query(Species).filter_by(common_name="Northern Cardinal").first()
        assert result is not None
        assert result.scientific_name == "Cardinalis cardinalis"
        assert result.class_index == 0


def test_add_detection_with_species():
    engine = get_test_engine()
    with Session(engine) as session:
        species = Species(
            common_name="Blue Jay",
            scientific_name="Cyanocitta cristata",
            class_index=1,
        )
        session.add(species)
        session.flush()

        detection = Detection(
            timestamp=datetime.now(),
            species_id=species.id,
            confidence=0.95,
            detection_model="yolov8n",
            classifier_model="mobilenetv2",
            bbox_x1=100.0,
            bbox_y1=200.0,
            bbox_x2=300.0,
            bbox_y2=400.0,
        )
        session.add(detection)
        session.commit()

        result = session.query(Detection).first()
        assert result is not None
        assert result.confidence == 0.95
        assert result.species.common_name == "Blue Jay"


def test_detection_without_species():
    engine = get_test_engine()
    with Session(engine) as session:
        detection = Detection(
            timestamp=datetime.now(),
            confidence=0.42,
            detection_model="yolov8n",
        )
        session.add(detection)
        session.commit()

        result = session.query(Detection).first()
        assert result is not None
        assert result.species is None
        assert result.species_id is None
