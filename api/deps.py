"""
API Dependencies Module

WHAT: This module provides reusable dependencies for FastAPI routes, such as database sessions 
      and search service instances.
HOW:  It wraps SQLAlchemy context managers and search class constructors in dependency injection
      functions using FastAPI's `Depends`.
WHY:  Dependency injection keeps code modular, promotes reuse, makes it easy to write unit tests,
      and simplifies database session lifetime management.
"""

from typing import Generator
from sqlalchemy.orm import Session
from db.metadata_store import SessionLocal
from search.retriever import Retriever
from search.reranker import TemporalReranker


def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency to yield a database session and close it when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_retriever() -> Retriever:
    """
    Dependency to instantiate the search retriever.
    """
    return Retriever()


def get_reranker() -> TemporalReranker:
    """
    Dependency to instantiate the temporal reranker.
    """
    # By default, use a 10.0-second deduplication window
    return TemporalReranker(time_window_seconds=10.0)
