"""
API Main Module

WHAT: This module defines the FastAPI web application, routing `/ingest` and `/search` requests.
HOW:  - `/ingest`: Creates a 'pending' SQL video record, offloads keyframe extraction and CLIP embedding 
        to a background thread (using FastAPI's `BackgroundTasks`), and returns the video metadata.
      - `/search`: Takes a text query, generates a CLIP text embedding, searches Qdrant for semantic 
        frame matches, hydrates the results using PostgreSQL/SQLite records, and calculates a ±2s clip window.
WHY:  In Python, web servers are single-process, asynchronous (event loop) systems. 
      Heavy computation like FFmpeg video decoding and PyTorch inference (CLIP) would block the 
      entire event loop, preventing the API from answering other requests. Just like Sidekiq 
      in Rails, we offload this work asynchronously to background workers or threads.
"""

from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Query, BackgroundTasks, HTTPException, File, UploadFile
from sqlalchemy.orm import Session

from api.deps import get_db_session, get_retriever, get_reranker
from api.schemas import IngestRequest, IngestResponse, SearchResponse, SearchResultResponse
from db.metadata_store import create_video, get_video, init_db
from ingestion.tasks import process_video_ingestion
from search.query import SearchQuery
from search.retriever import Retriever
from search.reranker import TemporalReranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application lifespan events, such as initializing database tables on startup.
    """
    init_db()
    yield


app = FastAPI(title="Video Search Engine API", lifespan=lifespan)


@app.get("/health")
def health():
    """
    Checks the health of the API application.
    """
    return {"status": "ok"}


def background_ingest_wrapper(video_path: str, video_id: int):
    """
    Wrapper function to execute process_video_ingestion in a background thread.
    """
    try:
        process_video_ingestion(video_path=video_path, video_id=video_id)
    except Exception as e:
        # Relational database states are already updated to 'failed' inside process_video_ingestion
        pass


@app.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The video file to upload and ingest."),
    db: Session = Depends(get_db_session)
):
    """
    Asynchronously ingests a video file into the search engine.
    
    1. Saves the uploaded file to data/videos/ folder.
    2. Creates a pending video entry in the relational database.
    3. Spawns a background task to extract keyframes, generate CLIP embeddings,
       and store them in Qdrant and SQL.
    4. Returns the immediate state of the ingestion request.
    """
    videos_dir = Path("data/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = videos_dir / file.filename
    try:
        with open(filepath, "wb") as f:
            f.write(file.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
        
    title = filepath.stem
    
    # Create the DB entry synchronously so we have a valid video_id to return
    video = create_video(db, title=title, filepath=str(filepath))
    
    # Enqueue background task (similar to Sidekiq's perform_async)
    background_tasks.add_task(background_ingest_wrapper, str(filepath), video.id)
    
    return IngestResponse(
        video_id=video.id,
        title=video.title,
        filepath=video.filepath,
        status=video.status,
        message="Video uploaded and ingestion queued successfully."
    )


@app.get("/search", response_model=SearchResponse)
def search_videos(
    q: str = Query(..., description="The semantic text query to search for."),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of candidate results to retrieve."),
    threshold: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold."),
    video_ids: Optional[List[int]] = Query(default=None, description="Optional list of video IDs to restrict the search to."),
    rerank: bool = Query(default=True, description="Whether to apply temporal deduplication reranking."),
    rewrite: bool = Query(default=True, description="Whether to apply NLP query rewriting."),
    db: Session = Depends(get_db_session),
    retriever: Retriever = Depends(get_retriever),
    reranker: TemporalReranker = Depends(get_reranker)
):
    """
    Executes a semantic search query against the video collection.
    
    1. Optionally rewrites conversational queries into visual descriptions (Stage 1).
    2. Embeds the search string using the CLIP text encoder (Stage 2).
    3. Performs vector similarity search in Qdrant (with optional video filter).
    4. Hydrates the points with video title/path metadata from the relational DB.
    5. Calculates a ±2s playback window around the matched frame.
    6. Optionally applies Temporal Non-Maximum Suppression to deduplicate adjacent frames.
    """
    query_params = SearchQuery(
        text=q,
        limit=limit,
        score_threshold=threshold,
        video_ids=video_ids
    )
    
    # Retrieve candidates (Stage 1 query rewriting + Stage 2 vector search)
    results = retriever.retrieve(query_params, db=db, rewrite=rewrite)
    
    # Apply temporal deduplication NMS
    if rerank and results:
        results = reranker.rerank(results)
        
    formatted_results = [
        SearchResultResponse(
            video_id=res.video_id,
            video_title=res.video_title,
            video_path=res.video_path,
            keyframe_id=res.keyframe_id,
            frame_path=res.frame_path,
            timestamp=res.timestamp,
            score=res.score,
            clip_start=res.clip_start,
            clip_end=res.clip_end
        )
        for res in results
    ]
    
    return SearchResponse(
        query=q,
        rewritten_query=retriever.last_rewritten_query,
        results=formatted_results,
        count=len(formatted_results)
    )
