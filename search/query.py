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
import os
import logging
import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import torch
from ingestion.embedder import FrameEmbedder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class QueryRewriter:
    """
    Transforms natural language questions into descriptive, visual-focused captions
    suitable for CLIP semantic vector search.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the query rewriter using the next-generation google-genai SDK.
        """
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.5-flash"
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def rewrite(self, query_text: str) -> str:
        """
        Rewrites conversational query_text into a descriptive, visual caption.
        If no API key is configured, it falls back to a rule-based query cleaner.
        
        Args:
            query_text: The original user question.
            
        Returns:
            An optimized visual search string.
        """
        if not self.api_key or not self.client:
            logger.warning("GEMINI_API_KEY not configured. Falling back to rule-based query cleaning.")
            return self._rule_based_cleanup(query_text)
            
        logger.info(f"Rewriting query '{query_text}' using modern Google GenAI SDK...")
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=query_text,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a translation assistant for a video search engine. "
                        "Your task is to rewrite user questions into simple, descriptive visual descriptions (captions) "
                        "that can be matched against video keyframe images. "
                        "Strip away question words ('when', 'where', 'how', 'why', 'who', 'what'), temporal references, "
                        "and conversational filler. Return only the core visual objects, entities, and actions. "
                        "Do not include quotes or surrounding text. Keep it brief."
                    ),
                    temperature=0.1,
                    max_output_tokens=30
                )
            )
            text = response.text.strip()
            logger.info(f"Successfully rewrote query to: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Failed to query Google GenAI SDK: {e}")
            return self._rule_based_cleanup(query_text)

    def _rule_based_cleanup(self, query_text: str) -> str:
        """
        A basic fallback parser to strip common question syntax.
        
        Args:
            query_text: The user question.
            
        Returns:
            An optimized visual search string.
        """
        text = query_text.lower().strip("?.! ")
        
        # Strip common question starts
        starts_to_strip = [
            "when did", "when does", "when do", "where is", "where are",
            "show me", "can you find", "find the", "search for"
        ]
        for start in starts_to_strip:
            if text.startswith(start):
                text = text[len(start):].strip()
                
        # Strip common question ends
        ends_to_strip = [
            "appear", "appears", "appeared"
        ]
        for end in ends_to_strip:
            if text.endswith(end):
                text = text[:-len(end)].strip()
                text = f"{text} appearing"
                
        return text if text else query_text
