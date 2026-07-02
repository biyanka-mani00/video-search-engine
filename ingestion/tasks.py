"""
Ingestion Tasks Module

WHAT: Orchestrates the ingestion pipeline by connecting the Extractor, Embedder, 
      Vector Store, and PostgreSQL/SQLite Metadata Store.
HOW:  Defines a flow where video records are created, keyframes are extracted,
      embedded using CLIP, saved to Qdrant, and mapped in the SQL database with timestamps.
WHY:  Using a centralized task logic ensures database consistency and makes it easy 
      to offload these heavy computations to Celery workers later.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import torch

from ingestion.extractor import VideoExtractor
from ingestion.embedder import FrameEmbedder
from db.vector_store import VectorStore
from db.metadata_store import get_db, create_video, update_video_status, add_keyframe, init_db

logger = logging.getLogger(__name__)

def process_video_ingestion(
    video_path: str, 
    interval: float = 2.0, 
    batch_size: int = 32,
    video_id: Optional[int] = None
) -> Optional[Tuple[List[Path], torch.Tensor]]:
    """
    Complete ingestion pipeline for a single video:
    
    1. Initialize the SQL database schema.
    2. Create a 'pending' video record in the database if not already created.
    3. Update status to 'processing' and extract keyframes using FFmpeg.
    4. Generate frame vector embeddings using CLIP.
    5. Save keyframe timestamps and paths to the database.
    6. Upsert vector embeddings and linked metadata to Qdrant.
    7. Update video status to 'completed' with duration.
    """
    logger.info(f"Starting ingestion pipeline for: {video_path}")
    
    # Ensure database schemas are initialized
    init_db()
    
    # Step 0: Insert Video entry into relational DB if not provided
    title = Path(video_path).stem
    if video_id is None:
        with get_db() as db:
            video = create_video(db, title=title, filepath=video_path)
            video_id = video.id
            
    with get_db() as db:
        update_video_status(db, video_id, status="processing")
        
    try:
        # Step 1: Keyframe Extraction
        extractor = VideoExtractor()
        frame_paths = extractor.extract_frames(video_path, interval=interval)
        
        if not frame_paths:
            logger.error("No frames extracted. Aborting.")
            with get_db() as db:
                update_video_status(db, video_id, status="failed")
            return None
        
        # Step 2: Generate Vector Embeddings
        embedder = FrameEmbedder()
        embeddings = embedder.embed_frames(frame_paths, batch_size=batch_size)
        
        # Step 3: Establish Vector Store Collection
        logger.info("Connecting to Qdrant for storage...")
        store = VectorStore()
        store.create_collection("video_frames", vector_size=512)
        
        # Step 4: Write metadata to DB & prepare Qdrant payload
        qdrant_payloads = []
        with get_db() as db:
            for i, path in enumerate(frame_paths):
                # Calculate timestamp: frame index * sampling interval
                timestamp = i * interval
                
                # Write frame reference to SQL database
                keyframe = add_keyframe(db, video_id=video_id, timestamp=timestamp, image_path=str(path))
                
                # Build Qdrant payload referencing the database IDs
                qdrant_payloads.append({
                    "video_id": video_id,
                    "keyframe_id": keyframe.id,
                    "video_path": video_path,
                    "frame_path": str(path),
                    "timestamp": timestamp
                })
            
            # Save vectors to Qdrant
            store.upsert_embeddings("video_frames", embeddings, qdrant_payloads)
            
            # Update video state to completed
            duration = len(frame_paths) * interval
            update_video_status(db, video_id, status="completed", duration=duration)
            
        logger.info(f"Ingestion complete for video '{title}' (DB ID: {video_id}).")
        logger.info(f"Extracted and stored {len(frame_paths)} keyframes.")
        
        return frame_paths, embeddings

    except Exception as e:
        logger.exception(f"Error occurred during video ingestion: {e}")
        with get_db() as db:
            update_video_status(db, video_id, status="failed")
        raise e

if __name__ == "__main__":
    test_video = "data/videos/test_video.mp4"
    import os
    
    if os.path.exists(test_video):
        # Run pipeline with a 5-second interval for testing
        process_video_ingestion(test_video, interval=5.0)
    else:
        print(f"Test video not found at {test_video}")
