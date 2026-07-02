"""
API Schemas Module

WHAT: This module defines validation and serialization schemas (Pydantic models) for the endpoints.
HOW:  It structures Pydantic schemas representing request payloads and formatted JSON API responses.
WHY:  FastAPI automatically uses Pydantic schemas to validate incoming requests, serialize 
      endpoint returns, and generate OpenAPI (Swagger) schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """
    Schema for initiating video ingestion.
    """
    filepath: str = Field(..., description="Absolute or relative path to the video file to ingest.")


class IngestResponse(BaseModel):
    """
    Schema representing the ingestion initialization status response.
    """
    video_id: int = Field(..., description="ID of the created video record.")
    title: str = Field(..., description="Stemmed title of the video.")
    filepath: str = Field(..., description="Path to the video file.")
    status: str = Field(..., description="Processing status (e.g. pending, processing, completed, failed).")
    message: str = Field(..., description="Descriptive status message.")


class SearchResultResponse(BaseModel):
    """
    Schema representing a single search result matching a query.
    """
    video_id: int = Field(..., description="ID of the matching video.")
    video_title: str = Field(..., description="Title of the video.")
    video_path: str = Field(..., description="Absolute path to the video file.")
    keyframe_id: int = Field(..., description="ID of the matching keyframe.")
    frame_path: str = Field(..., description="Absolute path to the keyframe image.")
    timestamp: float = Field(..., description="Timestamp of the keyframe in seconds.")
    score: float = Field(..., description="Cosine similarity score.")
    clip_start: float = Field(..., description="Start of ±2s clip window in seconds.")
    clip_end: float = Field(..., description="End of ±2s clip window in seconds.")


class SearchResponse(BaseModel):
    """
    Schema representing the complete search query response.
    """
    query: str = Field(..., description="The search query text.")
    results: List[SearchResultResponse] = Field(..., description="List of matching search results.")
    count: int = Field(..., description="Number of results returned.")
