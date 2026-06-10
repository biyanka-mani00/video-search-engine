"""
Vector Store Module

WHAT: This module manages the storage and retrieval of high-dimensional vectors 
      (embeddings) using Qdrant, a specialized vector database.
HOW:  It uses the `qdrant-client` library to interface with a Qdrant instance. 
      It ensures a "collection" (similar to a table) exists with the correct 
      configuration (512 dimensions for CLIP, Cosine similarity).
WHY:  Traditional databases (like PostgreSQL) are not optimized for "nearest neighbor" 
      searches. Qdrant allows us to find frames that are mathematically similar to 
      a search query in milliseconds, even with millions of vectors.
"""

import logging
from typing import List, Optional, Union
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Handles interactions with the Qdrant vector database.
    """

    def __init__(self, host: str = "localhost", port: int = 6333, location: Optional[str] = None):
        """
        Initializes the Qdrant client.
        
        Args:
            host: Qdrant server host.
            port: Qdrant server port.
            location: If provided (e.g., ':memory:' or a path), uses local mode.
        """
        if location:
            self.client = QdrantClient(location=location)
            logger.info(f"Connected to local Qdrant at {location}")
        else:
            # Default to server mode
            try:
                self.client = QdrantClient(host=host, port=port)
                logger.info(f"Connected to Qdrant server at {host}:{port}")
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant server: {e}. Falling back to local storage.")
                self.client = QdrantClient(location="data/qdrant_storage")

    def create_collection(self, collection_name: str, vector_size: int = 512):
        """
        Ensures a collection exists with the specified vector dimensions.

        Args:
            collection_name: Name of the collection (like a table name).
            vector_size: Dimensionality of the vectors (512 for CLIP ViT-B-32).
        """
        if not self.client.collection_exists(collection_name):
            logger.info(f"Creating collection: {collection_name} with size {vector_size}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE
                ),
            )
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    def upsert_embeddings(
        self, 
        collection_name: str, 
        embeddings: Union[torch.Tensor, np.ndarray, list], 
        metadata: List[dict]
    ):
        """
        Saves embeddings and metadata into the collection.

        Args:
            collection_name: Target collection.
            embeddings: The vector data (Tensor or Array).
            metadata: A list of dicts containing info like 'video_path', 'timestamp', etc.
        """
        # Convert embeddings to list of lists if they are Tensors/Arrays
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy().tolist()
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        points = []
        for i, (vector, meta) in enumerate(zip(embeddings, metadata)):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=meta
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Successfully upserted {len(points)} vectors to '{collection_name}'.")

if __name__ == "__main__":
    # Test local mode
    store = VectorStore(location=":memory:")
    store.create_collection("test_collection", vector_size=512)
    
    # Dummy data
    dummy_vector = torch.randn(1, 512)
    dummy_meta = [{"frame_id": 1, "video": "test.mp4"}]
    
    store.upsert_embeddings("test_collection", dummy_vector, dummy_meta)
