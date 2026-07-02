"""
Search Query Module

WHAT: This module defines data validation schemas and encoding logic for search queries.
HOW:  It leverages Pydantic for validation and utilizes the pre-trained CLIP model
      (via `FrameEmbedder.embed_text`) to transform raw text queries into 512-dimensional vector embeddings.
WHY:  In Rails, inputs are validated at the controller or model level (ActiveModel::Validations).
      In FastAPI, Pydantic handles this at the API boundary, parsing and casting incoming JSON into 
      strongly-typed Python objects. Decoupling the query representation from the retrieval engine 
      keeps the code modular and easily testable.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
import torch
from ingestion.embedder import FrameEmbedder


class SearchQuery(BaseModel):
    """
    Data validation schema for a search query.
    """
    text: str = Field(..., description="The semantic text query to search for.")
    limit: int = Field(default=5, ge=1, le=100, description="Maximum number of search results to return.")
    score_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold.")
    video_ids: Optional[List[int]] = Field(default=None, description="Optional list of video IDs to filter results by.")


class QueryEncoder:
    """
    Encodes text queries into vector embeddings using the CLIP model.
    """
    def __init__(self, embedder: Optional[FrameEmbedder] = None):
        """
        Initializes the query encoder with a FrameEmbedder instance.
        """
        self.embedder = embedder or FrameEmbedder()

    def encode(self, query_text: str) -> torch.Tensor:
        """
        Converts the text query to a normalized embedding tensor.
        
        Args:
            query_text: The search phrase to encode.
            
        Returns:
            A 1D normalized torch.Tensor of shape (512,).
        """
        return self.embedder.embed_text(query_text)
