"""
Search Retriever Module

WHAT: This module retrieves matching keyframes from Qdrant and hydrates them with video metadata
      from PostgreSQL/SQLite. It also computes a ±2s window around each matching frame.
HOW:  It encodes the search text, queries Qdrant with optional filters, fetches corresponding
      video records from the SQL database using SQLAlchemy, and calculates the segment start/end bounds.
WHY:  In Rails, ActiveRecord makes single queries easy but can lead to N+1 queries if we load associations
      individually. Here, we fetch all matching videos in a single batch query (using `.in_()`), 
      optimizing database performance. We calculate the clip window relative to the matching timestamp 
      to allow frontends to play a segment around the event.
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from db.vector_store import VectorStore
from db.metadata_store import Video
from search.query import QueryEncoder, SearchQuery, QueryRewriter
from qdrant_client.http import models
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """
    Unified search result mapping vector similarities to SQL metadata
    along with calculated ±2s clip window bounds.
    """
    video_id: int
    video_title: str
    video_path: str
    keyframe_id: int
    frame_path: str
    timestamp: float
    score: float
    clip_start: float
    clip_end: float


class Retriever:
    """
    Retrieves and hydrates candidate keyframes matching a query.
    """
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        query_encoder: Optional[QueryEncoder] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        collection_name: str = "video_frames"
    ):
        """
        Initializes the retriever.
        
        Args:
            vector_store: Vector store client wrapper.
            query_encoder: Encoder to convert query strings.
            query_rewriter: Query rewriter to optimize query strings.
            collection_name: Name of Qdrant collection to search.
        """
        self.vector_store = vector_store or VectorStore()
        self.query_encoder = query_encoder or QueryEncoder()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.collection_name = collection_name
        self.last_rewritten_query: Optional[str] = None

    def retrieve(self, query: SearchQuery, db: Session, rewrite: bool = True) -> List[SearchResult]:
        """
        Retrieves matching keyframes from vector store and hydrates with SQL metadata and ±2s clip bounds.
        
        Args:
            query: Validate SearchQuery parameters.
            db: The SQLAlchemy Session.
            rewrite: Whether to apply NLP query rewriting.
            
        Returns:
            A list of SearchResult objects containing matched frames and clip bounds.
        """
        logger.info(f"Retrieving keyframes for query: '{query.text}'")
        
        # 1. Stage 1: Query Rewriting
        search_text = query.text
        if rewrite:
            search_text = self.query_rewriter.rewrite(query.text)
            
        self.last_rewritten_query = search_text
        
        # 2. Stage 2: Vector Search
        query_vector = self.query_encoder.encode(search_text)
         
        print (f"Encoded query vector: {query_vector[:5]}... (length: {len(query_vector)})")
        
        # 2. Build filter conditions if video_ids are supplied
        query_filter = None
        if query.video_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id",
                        match=models.MatchAny(any=query.video_ids)
                    )
                ]
            )
            
        # 3. Query Qdrant Vector Store
        scored_points = self.vector_store.search_vectors(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=query.limit,
            score_threshold=query.score_threshold,
            query_filter=query_filter
        )
        
        if not scored_points:
            logger.info("No matching points found in Qdrant.")
            return []
            
        # 4. Extract unique video IDs to batch query SQL database (Avoid N+1 queries)
        video_ids = list(set(
            point.payload["video_id"] 
            for point in scored_points 
            if point.payload and "video_id" in point.payload
        ))
        videos = db.query(Video).filter(Video.id.in_(video_ids)).all()
        video_map = {v.id: v for v in videos}
        
        # 5. Build hydrated results and calculate ±2s clip window
        results = []
        for point in scored_points:
            if not point.payload or "video_id" not in point.payload:
                continue
                
            vid_id = point.payload["video_id"]
            video = video_map.get(vid_id)
            
            # If the video does not exist in relational DB, skip it (data consistency)
            if not video:
                continue
                
            timestamp = point.payload.get("timestamp", 0.0)
            keyframe_id = point.payload.get("keyframe_id", 0)
            frame_path = point.payload.get("frame_path", "")
            
            # Calculate ±2s clip window bounds
            clip_start = max(0.0, timestamp - 2.0)
            clip_end = timestamp + 2.0
            if video.duration is not None:
                clip_end = min(video.duration, clip_end)
                
            results.append(
                SearchResult(
                    video_id=vid_id,
                    video_title=video.title,
                    video_path=video.filepath,
                    keyframe_id=keyframe_id,
                    frame_path=frame_path,
                    timestamp=timestamp,
                    score=point.score,
                    clip_start=clip_start,
                    clip_end=clip_end
                )
            )
            
        logger.info(f"Retrieved and hydrated {len(results)} search results.")
        return results
