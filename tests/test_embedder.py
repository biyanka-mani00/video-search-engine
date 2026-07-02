import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import torch
from ingestion.embedder import FrameEmbedder

class TestFrameEmbedder(unittest.TestCase):
    @patch('open_clip.create_model_and_transforms')
    @patch('open_clip.get_tokenizer')
    def setUp(self, mock_get_tokenizer, mock_create_model):
        # Mocking the model, preprocess, and tokenizer
        self.mock_model = MagicMock()
        self.mock_preprocess = MagicMock()
        mock_create_model.return_value = (self.mock_model, None, self.mock_preprocess)
        
        self.embedder = FrameEmbedder(model_name="ViT-B-32", device="cpu")

    @patch('PIL.Image.open')
    def test_embed_frames_batching(self, mock_image_open):
        # Setup mock for preprocess to return a dummy tensor
        self.mock_preprocess.return_value = torch.randn(3, 224, 224)
        
        # Mock model.encode_image to return dummy embeddings
        def side_effect(image_input):
            return torch.randn(image_input.shape[0], 512)
        self.mock_model.encode_image.side_effect = side_effect
        
        # Test with 10 frames and batch_size 3
        frame_paths = [Path(f"frame_{i}.jpg") for i in range(10)]
        
        embeddings = self.embedder.embed_frames(frame_paths, batch_size=3)
        
        # Check if model.encode_image was called 4 times (3+3+3+1)
        self.assertEqual(self.mock_model.encode_image.call_count, 4)
        self.assertEqual(embeddings.shape, (10, 512))

if __name__ == '__main__':
    unittest.main()
