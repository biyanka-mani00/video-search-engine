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
from db.vector_store import VectorStore

logger = logging.getLogger(__name__)

def process_video_ingestion(video_path: str, interval: float = 2.0, batch_size: int = 32):
    """
    Complete ingestion pipeline for a single video.
    
    1. Extract frames (FFmpeg)
    2. Generate embeddings (CLIP)
    3. Store in Qdrant
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
    embeddings = embedder.embed_frames(frame_paths, batch_size=batch_size)
    
    # 3. Storage
    logger.info("Connecting to Qdrant for storage...")
    store = VectorStore() # Let it use defaults/env variables
    store.create_collection("video_frames", vector_size=512)
    
    # Prepare metadata for each frame
    metadata = []
    for i, path in enumerate(frame_paths):
        # Calculate timestamp: frame 1 is 0s, frame 2 is interval*1s, etc.
        timestamp = i * interval
        metadata.append({
            "video_path": video_path,
            "frame_path": str(path),
            "timestamp": timestamp
        })
    
    # Save to Qdrant
    store.upsert_embeddings("video_frames", embeddings, metadata)
    
    logger.info(f"Ingestion complete for {video_path}.")
    logger.info(f"Extracted {len(frame_paths)} frames and stored in Qdrant.")
    
    return frame_paths, embeddings

if __name__ == "__main__":
    # Test the full pipeline
    test_video = "data/videos/test_video.mp4"
    import os
    
    if os.path.exists(test_video):
        # We'll use a 5-second interval for the test to keep it fast
        process_video_ingestion(test_video, interval=5.0)
    else:
        print(f"Test video not found at {test_video}")
