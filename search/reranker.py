"""
Search Reranker Module

WHAT: This module applies temporal deduplication (Non-Maximum Suppression) to retrieved keyframes.
HOW:  It sorts matching frames by similarity score, then prunes any frames from the same video 
      that fall within a temporal proximity window (e.g., 10 seconds) of a higher-scoring frame.
WHY:  In video retrieval, multiple consecutive frames from the same scene will match the same text.
      Pruning redundant results ensures that the search results page represents diverse parts of 
      the video rather than 5 frames of the exact same 2-second action.
"""

import logging
from typing import List
from search.retriever import SearchResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalReranker:
    """
    Reranks search results by applying temporal deduplication.
    """
    def __init__(self, time_window_seconds: float = 10.0):
        """
        Initializes the reranker.
        
        Args:
            time_window_seconds: The temporal distance window. Frames from the same video
                                 within this window will be deduplicated, keeping the highest score.
        """
        self.time_window_seconds = time_window_seconds

    def rerank(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Applies Temporal Non-Maximum Suppression (NMS) to deduplicate near-identical
        keyframes within the same time window.
        
        Args:
            results: A list of SearchResult objects, typically ordered by score descending.
            
        Returns:
            A pruned/reordered list of SearchResult objects.
        """
        if not results:
            return []

        # Ensure results are sorted by score descending
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        filtered_results: List[SearchResult] = []
        
        # Track selected frame timestamps for each video_id: {video_id: [selected_timestamps]}
        selected_map = {}
        
        for res in sorted_results:
            vid_id = res.video_id
            timestamp = res.timestamp
            
            if vid_id not in selected_map:
                selected_map[vid_id] = []
                
            # Check if this frame is too close to any already selected frame for this video
            is_redundant = False
            for selected_ts in selected_map[vid_id]:
                if abs(selected_ts - timestamp) <= self.time_window_seconds:
                    is_redundant = True
                    break
                    
            if not is_redundant:
                filtered_results.append(res)
                selected_map[vid_id].append(timestamp)
                
        logger.info(f"Temporal Reranking complete: reduced results from {len(results)} to {len(filtered_results)}")
        return filtered_results
