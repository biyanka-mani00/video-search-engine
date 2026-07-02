import unittest
from unittest.mock import MagicMock, patch
import torch
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from api.deps import get_db_session, get_retriever, get_reranker
from db.metadata_store import Base, Video, Keyframe
from db.vector_store import VectorStore
from search.query import SearchQuery, QueryEncoder
from search.retriever import Retriever, SearchResult
from search.reranker import TemporalReranker


class TestSearchQueryAndEncoder(unittest.TestCase):
    @patch('search.query.FrameEmbedder')
    def test_encoder(self, mock_embedder_class):
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = torch.ones(512)
        mock_embedder_class.return_value = mock_embedder

        encoder = QueryEncoder(embedder=mock_embedder)
        vector = encoder.encode("a dog running")
        
        mock_embedder.embed_text.assert_called_once_with("a dog running")
        self.assertEqual(vector.shape, (512,))


class TestTemporalReranker(unittest.TestCase):
    def test_reranker_deduplication(self):
        results = [
            SearchResult(
                video_id=1, video_title="V1", video_path="v1.mp4", 
                keyframe_id=1, frame_path="v1_1.jpg", timestamp=2.0, 
                score=0.9, clip_start=0.0, clip_end=4.0
            ),
            SearchResult(
                video_id=2, video_title="V2", video_path="v2.mp4", 
                keyframe_id=3, frame_path="v2_3.jpg", timestamp=4.0, 
                score=0.85, clip_start=2.0, clip_end=6.0
            ),
            SearchResult(
                video_id=1, video_title="V1", video_path="v1.mp4", 
                keyframe_id=2, frame_path="v1_2.jpg", timestamp=5.0, 
                score=0.8, clip_start=3.0, clip_end=7.0
            ),
            SearchResult(
                video_id=1, video_title="V1", video_path="v1.mp4", 
                keyframe_id=4, frame_path="v1_4.jpg", timestamp=25.0, 
                score=0.7, clip_start=23.0, clip_end=27.0
            ),
        ]

        reranker = TemporalReranker(time_window_seconds=10.0)
        reranked = reranker.rerank(results)

        # Expected remaining:
        # 1. Video 1 at 2.0 (score 0.9)
        # 2. Video 2 at 4.0 (score 0.85)
        # 3. Video 1 at 25.0 (score 0.7)
        # Video 1 at 5.0 (score 0.8) is pruned because it's within 10s of the 2.0s frame.
        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0].video_id, 1)
        self.assertEqual(reranked[0].timestamp, 2.0)
        self.assertEqual(reranked[1].video_id, 2)
        self.assertEqual(reranked[1].timestamp, 4.0)
        self.assertEqual(reranked[2].video_id, 1)
        self.assertEqual(reranked[2].timestamp, 25.0)


class TestRetriever(unittest.TestCase):
    def test_retrieve(self):
        mock_vector_store = MagicMock()
        mock_query_encoder = MagicMock()
        
        # Mock query encoding
        mock_query_encoder.encode.return_value = torch.ones(512)
        
        # Mock vector search output from Qdrant
        from qdrant_client.http import models
        mock_scored_point = models.ScoredPoint(
            id="some-uuid",
            version=1,
            score=0.9,
            payload={
                "video_id": 1,
                "keyframe_id": 10,
                "video_path": "test.mp4",
                "frame_path": "frame_10.jpg",
                "timestamp": 20.0
            }
        )
        mock_vector_store.search_vectors.return_value = [mock_scored_point]
        
        # Mock SQLAlchemy Session and Video retrieval
        mock_db = MagicMock()
        mock_video = MagicMock()
        mock_video.id = 1
        mock_video.title = "Test Video"
        mock_video.filepath = "test.mp4"
        mock_video.duration = 60.0
        
        # Mock DB query
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_video]
        
        retriever = Retriever(
            vector_store=mock_vector_store,
            query_encoder=mock_query_encoder,
            collection_name="test_collection"
        )
        
        query = SearchQuery(text="a white cat", limit=5)
        results = retriever.retrieve(query, db=mock_db)
        
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res.video_id, 1)
        self.assertEqual(res.video_title, "Test Video")
        self.assertEqual(res.score, 0.9)
        self.assertEqual(res.timestamp, 20.0)
        # Check ±2s bounds: 20s - 2s = 18s; 20s + 2s = 22s
        self.assertEqual(res.clip_start, 18.0)
        self.assertEqual(res.clip_end, 22.0)
        
        # Ensure search_vectors was called with correct parameters
        mock_vector_store.search_vectors.assert_called_once()
        args, kwargs = mock_vector_store.search_vectors.call_args
        self.assertEqual(kwargs['collection_name'], "test_collection")
        self.assertEqual(kwargs['limit'], 5)
        self.assertIsNone(kwargs['query_filter'])


class TestAPISearch(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.app = app

    def tearDown(self):
        self.app.dependency_overrides.clear()

    def test_search_api_endpoint(self):
        mock_retriever = MagicMock()
        mock_reranker = MagicMock()
        
        # Override dependencies
        mock_db = MagicMock()
        self.app.dependency_overrides[get_db_session] = lambda: mock_db
        self.app.dependency_overrides[get_retriever] = lambda: mock_retriever
        self.app.dependency_overrides[get_reranker] = lambda: mock_reranker
        
        # Setup mock return values
        mock_retriever.retrieve.return_value = [
            SearchResult(
                video_id=1, video_title="V1", video_path="v1.mp4", 
                keyframe_id=10, frame_path="f10.jpg", timestamp=20.0, 
                score=0.95, clip_start=18.0, clip_end=22.0
            ),
            SearchResult(
                video_id=1, video_title="V1", video_path="v1.mp4", 
                keyframe_id=11, frame_path="f11.jpg", timestamp=22.0, 
                score=0.85, clip_start=20.0, clip_end=24.0
            ),
        ]
        
        mock_reranker.rerank.return_value = [
            SearchResult(
                video_id=1, video_title="V1", video_path="v1.mp4", 
                keyframe_id=10, frame_path="f10.jpg", timestamp=20.0, 
                score=0.95, clip_start=18.0, clip_end=22.0
            ),
        ]
        
        response = self.client.get("/search?q=running+dog&limit=5&threshold=0.5&rerank=true")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["query"], "running dog")
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["video_id"], 1)
        self.assertEqual(data["results"][0]["score"], 0.95)
        self.assertEqual(data["results"][0]["clip_start"], 18.0)
        self.assertEqual(data["results"][0]["clip_end"], 22.0)
        
        mock_retriever.retrieve.assert_called_once()
        mock_reranker.rerank.assert_called_once()


class TestAPIIntegration(unittest.TestCase):
    """
    Full Integration Test: Ingest Video -> Search -> Get Results.
    Uses an isolated SQLite database and in-memory Qdrant client.
    """
    def setUp(self):
        # 1. Setup isolated in-memory SQLAlchemy DB for tests
        from sqlalchemy.pool import StaticPool
        self.test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        self.TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.test_engine)
        Base.metadata.create_all(bind=self.test_engine)

        # 2. Patch dependencies in FastAPI app
        app.dependency_overrides[get_db_session] = self.override_db_session
        
        # 3. Create isolated in-memory VectorStore for tests
        self.test_vector_store = VectorStore(location=":memory:")
        self.test_vector_store.create_collection("video_frames", vector_size=512)
        
        # Override retriever's VectorStore
        def override_retriever():
            return Retriever(vector_store=self.test_vector_store)
        app.dependency_overrides[get_retriever] = override_retriever

        # 4. Patch ingestion module references to metadata DB engine and VectorStore
        self.patcher_db_engine = patch('db.metadata_store.engine', self.test_engine)
        self.patcher_db_sessionmaker = patch('db.metadata_store.SessionLocal', self.TestingSessionLocal)
        
        # Patch tasks.py to use our in-memory Qdrant VectorStore
        self.patcher_tasks_store = patch('ingestion.tasks.VectorStore', return_value=self.test_vector_store)

        self.patcher_db_engine.start()
        self.patcher_db_sessionmaker.start()
        self.patcher_tasks_store.start()

        self.client = TestClient(app)

    def tearDown(self):
        self.patcher_db_engine.stop()
        self.patcher_db_sessionmaker.stop()
        self.patcher_tasks_store.stop()
        app.dependency_overrides.clear()

    def override_db_session(self):
        db = self.TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    def test_full_ingestion_and_search_flow(self):
        # Path to actual test video in workspace
        video_path = "data/videos/test_video.mp4"
        self.assertTrue(Path(video_path).exists(), f"Test video not found at {video_path}")

        # 1. Trigger ingest endpoint (runs keyframe extraction and CLIP embedding in background thread)
        # Note: TestClient runs background tasks synchronously before returning, so when post returns, ingestion is fully complete!
        with open(video_path, "rb") as f:
            ingest_response = self.client.post(
                "/ingest",
                files={"file": (Path(video_path).name, f, "video/mp4")}
            )
        
        self.assertEqual(ingest_response.status_code, 202)
        ingest_data = ingest_response.json()
        self.assertEqual(ingest_data["status"], "pending")
        video_id = ingest_data["video_id"]
        
        # Verify video record exists in our test DB
        db = self.TestingSessionLocal()
        from db.metadata_store import Video, Keyframe
        video_record = db.query(Video).filter(Video.id == video_id).first()
        self.assertIsNotNone(video_record)
        self.assertEqual(video_record.status, "completed")
        self.assertGreater(video_record.duration, 0)

        # Verify keyframes were extracted and added to SQL metadata
        keyframes = db.query(Keyframe).filter(Keyframe.video_id == video_id).all()
        self.assertGreater(len(keyframes), 0)
        db.close()

        # 2. Query search endpoint
        # We search for semantic query "video". Since CLIP embeddings are mapped, it should find matches.
        search_response = self.client.get("/search?q=video&limit=3&rerank=true")
        self.assertEqual(search_response.status_code, 200)
        search_data = search_response.json()

        self.assertEqual(search_data["query"], "video")
        self.assertGreater(search_data["count"], 0)
        
        # Verify the hydrated fields and ±2s clip bounds exist and are logical
        first_result = search_data["results"][0]
        self.assertEqual(first_result["video_id"], video_id)
        self.assertIn("test_video", first_result["video_title"])
        self.assertGreater(first_result["score"], 0.0)
        self.assertEqual(first_result["clip_start"], max(0.0, first_result["timestamp"] - 2.0))
        self.assertEqual(first_result["clip_end"], min(video_record.duration, first_result["timestamp"] + 2.0))


if __name__ == '__main__':
    unittest.main()
