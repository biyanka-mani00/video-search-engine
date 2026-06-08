"""
Ingestion Tasks Module

WHAT: Orchestrates the ingestion pipeline by connecting the Extractor, Embedder, 
      and (eventually) the Vector Store.
HOW:  Defines a flow where a video is first broken into frames, then those frames 
      are embedded, and finally stored.
WHY:  Using a centralized task logic makes it easy to offload these heavy 
      computations to Celery workers later.
"""

import logging
from pathlib import Path
from typing import Optional
from ingestion.extractor import VideoExtractor
from ingestion.embedder import FrameEmbedder

logger = logging.getLogger(__name__)

def process_video_ingestion(video_path: str, interval: float = 2.0):
    """
    Complete ingestion pipeline for a single video.
    
    1. Extract frames (FFmpeg)
    2. Generate embeddings (CLIP)
    3. (Future) Store in Qdrant & PostgreSQL
    """
    logger.info(f"Starting ingestion pipeline for: {video_path}")
    
    # 1. Extraction
    extractor = VideoExtractor()
    frame_paths = extractor.extract_frames(video_path, interval=interval)
    
    if not frame_paths:
        logger.error("No frames extracted. Aborting.")
        return
    
    # 2. Embedding
    embedder = FrameEmbedder()
    embeddings = embedder.embed_frames(frame_paths)
    
    logger.info(f"Ingestion step 1 & 2 complete for {video_path}.")
    logger.info(f"Extracted {len(frame_paths)} frames and generated embeddings shape {embeddings.shape}.")
    
    # 3. Storage (To be implemented in Day 4 & 6)
    logger.info("Ready for vector storage integration.")
    
    return frame_paths, embeddings

if __name__ == "__main__":
    # This will be used for testing the core logic
    pass
