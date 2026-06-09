"""
Frame Embedder Module

WHAT: This module converts images (extracted video frames) into numerical vector 
      embeddings using the CLIP (Contrastive Language-Image Pre-training) model.
HOW:  It uses `open_clip` or `clip-by-openai` to load a pre-trained Vision Transformer 
      (e.g., ViT-B/32). Each image is preprocessed and passed through the image 
      encoder to produce a 512-dimensional (typical) vector.
WHY:  CLIP is the "secret sauce" for multimodal search. Because it was trained on 
      image-text pairs, the resulting vectors for "a cat" (text) and an image of 
      a cat are close together in the vector space, enabling semantic search.
"""

import torch
import open_clip
from PIL import Image
import logging
from pathlib import Path
from typing import List, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameEmbedder:
    """
    Handles the generation of embeddings for images using CLIP.
    """

    def __init__(self, model_name: str = "ViT-B-32", device: str = None):
        """
        Initializes the CLIP model.
        
        Args:
            model_name: The CLIP model variant to use.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model {model_name} on {self.device}...")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained='openai', 
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def embed_frames(self, frame_paths: List[Path]) -> torch.Tensor:
        """
        Generates embeddings for a list of image paths.

        Args:
            frame_paths: List of Paths to the image files.
            
        Returns:
            A torch.Tensor of shape (num_frames, embedding_dim).
        """
        if not frame_paths:
            return torch.empty(0)

        logger.info(f"Generating embeddings for {len(frame_paths)} frames...")
        
        # Load and preprocess all images
        images = []
        for path in frame_paths:
            try:
                img = self.preprocess(Image.open(path)).unsqueeze(0).to(self.device)
                images.append(img)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue

        if not images:
            return torch.empty(0)

        # Batch process for efficiency
        image_input = torch.cat(images)
        
        with torch.no_grad():
            # Generate image features
            image_features = self.model.encode_image(image_input)
            
            # Normalize embeddings for cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        logger.info(f"Embeddings generated. Shape: {image_features.shape}")
        return image_features

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Generates an embedding for a text query.
        
        Args:
            text: The search query string.
            
        Returns:
            A normalized 1D torch.Tensor embedding.
        """
        text_input = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.squeeze(0)

if __name__ == "__main__":
    # Path to the frames we just extracted
    frame_dir = Path("data/keyframes/test_video")
    
    if not frame_dir.exists():
        print(f"Frame directory not found at {frame_dir}. Please run extractor.py first.")
    else:
        # Get list of frames
        frame_paths = sorted(list(frame_dir.glob("*.jpg")))
        
        if not frame_paths:
            print(f"No .jpg files found in {frame_dir}")
        else:
            print(f"Found {len(frame_paths)} frames. Loading CLIP model...")
            embedder = FrameEmbedder()
            
            try:
                # Generate embeddings
                embeddings = embedder.embed_frames(frame_paths)
                
                print("\n--- Embedding Complete ---")
                print(f"Total frames processed: {len(frame_paths)}")
                print(f"Embeddings Tensor Shape: {embeddings.shape}")
                print("\nExample Embedding (first 10 values of first frame):")
                print(embeddings[0][:10])
                
            except Exception as e:
                print(f"An error occurred during embedding: {e}")
