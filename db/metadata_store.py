"""
Metadata Store Module

WHAT: This module defines the relational database models and helper functions
      for storing and querying video metadata (titles, file paths, durations)
      and frame timestamps.
HOW:  It uses SQLAlchemy ORM to define `Video` and `Keyframe` models. It implements
      a robust connection manager that attempts to connect to PostgreSQL (configured
      for Docker or localhost) and falls back to a local SQLite database if Postgres is down.
WHY:  Relational metadata (like frame timestamps mapped to video files and processing state)
      is structured and requires ACID transactions, which is best handled by standard SQL
      databases rather than vector databases like Qdrant.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Generator
from contextlib import contextmanager

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class Video(Base):
    """
    Represents a video file uploaded to the search engine.
    """
    __tablename__ = "videos"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    title: str = Column(String, nullable=False)
    filepath: str = Column(String, nullable=False)
    duration: Optional[float] = Column(Float, nullable=True)  # in seconds
    status: str = Column(String, default="pending", nullable=False)  # pending, processing, completed, failed
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: datetime = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    keyframes = relationship("Keyframe", back_populates="video", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Video(id={self.id}, title='{self.title}', status='{self.status}')>"


class Keyframe(Base):
    """
    Represents a single extracted frame of a video, associated with a timestamp.
    """
    __tablename__ = "keyframes"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    video_id: int = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    timestamp: float = Column(Float, nullable=False)  # Seconds into the video
    image_path: str = Column(String, nullable=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    video = relationship("Video", back_populates="keyframes")

    def __repr__(self) -> str:
        return f"<Keyframe(id={self.id}, video_id={self.video_id}, timestamp={self.timestamp}s)>"


# Enable foreign key support for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Ensures that foreign key constraints are enforced in SQLite databases.
    """
    if dbapi_connection.__class__.__module__ == "sqlite3":
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception:
            pass


# Engine setup with fallbacks
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@postgres:5432/video_search")

def get_engine() -> Engine:
    """
    Tries to connect to Postgres using the configured environment.
    If it fails (e.g. running locally outside of the Docker network),
    it tries localhost. If that fails, it falls back to a local SQLite file.
    """
    # 1. Try configured DATABASE_URL (Docker network or custom)
    try:
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as conn:
            pass
        logger.info(f"Connected to database using primary URL: {DATABASE_URL.split('@')[-1]}")
        return engine
    except OperationalError:
        # 2. Try swapping 'postgres' hostname with 'localhost' (Local host development)
        if "postgres" in DATABASE_URL:
            local_url = DATABASE_URL.replace("@postgres", "@localhost")
            try:
                engine = create_engine(local_url)
                with engine.connect() as conn:
                    pass
                logger.info(f"Connected to local database using: {local_url.split('@')[-1]}")
                return engine
            except OperationalError:
                pass

        # 3. Fallback to local SQLite database
        sqlite_path = Path("data")
        sqlite_path.mkdir(exist_ok=True)
        sqlite_url = "sqlite:///data/metadata.db"
        logger.warning(f"Could not connect to PostgreSQL. Falling back to SQLite at {sqlite_url}")
        return create_engine(sqlite_url)

# Initialize Session
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db() -> None:
    """
    Creates the relational database tables.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager dependency for database sessions.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# CRUD Helpers

def create_video(db: Session, title: str, filepath: str) -> Video:
    """
    Creates a new video entry in the database.
    """
    video = Video(title=title, filepath=filepath, status="pending")
    db.add(video)
    db.commit()
    db.refresh(video)
    return video

def update_video_status(db: Session, video_id: int, status: str, duration: Optional[float] = None) -> Optional[Video]:
    """
    Updates the processing status and optional duration of a video.
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if video:
        video.status = status
        if duration is not None:
            video.duration = duration
        db.commit()
        db.refresh(video)
    return video

def get_video(db: Session, video_id: int) -> Optional[Video]:
    """
    Retrieves a single video by its ID.
    """
    return db.query(Video).filter(Video.id == video_id).first()

def get_all_videos(db: Session) -> List[Video]:
    """
    Retrieves all video records from the database.
    """
    return db.query(Video).order_by(Video.created_at.desc()).all()

def add_keyframe(db: Session, video_id: int, timestamp: float, image_path: str) -> Keyframe:
    """
    Adds a single keyframe entry associated with a video.
    """
    keyframe = Keyframe(video_id=video_id, timestamp=timestamp, image_path=image_path)
    db.add(keyframe)
    db.commit()
    db.refresh(keyframe)
    return keyframe

def get_keyframes_for_video(db: Session, video_id: int) -> List[Keyframe]:
    """
    Retrieves all keyframes associated with a given video.
    """
    return db.query(Keyframe).filter(Keyframe.video_id == video_id).order_by(Keyframe.timestamp.asc()).all()
