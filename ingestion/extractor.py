"""
Video Keyframe Extractor Module

WHAT: This module extracts individual frames (keyframes) from a video file at a 
      specified temporal interval (e.g., one frame every 2 seconds).
HOW:  It leverages the `ffmpeg-python` wrapper to execute FFmpeg commands. 
      Specifically, it uses the `fps` video filter to sample frames at a constant rate.
WHY:  A fixed interval ensures consistent coverage across the video's timeline. 
      While scene detection is an alternative, fixed intervals are more predictable 
      for downstream CLIP embedding and vector search, especially in learning projects.
"""

import os
import ffmpeg
import logging
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoExtractor:
    """
    Handles the extraction of frames from video files using FFmpeg.
    """

    def __init__(self, output_dir: str = "data/keyframes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path: str, interval: float = 2.0) -> List[Path]:
        """
        Extracts frames from the video at the given interval.

        Args:
            video_path: Path to the input video file.
            interval: Seconds between each extracted frame (e.g., 0.5, 2.0).
            
        Returns:
            A list of Paths to the extracted image files.
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create a sub-directory for this specific video to avoid collisions
        video_id = video_path_obj.stem
        video_output_dir = self.output_dir / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting frames from {video_path} every {interval}s to {video_output_dir}")

        try:
            # FFmpeg command: ffmpeg -i input.mp4 -vf "fps=1/interval" output_%04d.jpg
            (
                ffmpeg
                .input(video_path)
                .filter('fps', fps=1/interval)
                .output(str(video_output_dir / "frame_%04d.jpg"), **{'q:v': 2})
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise RuntimeError(f"Failed to extract frames from {video_path}")

        # Get list of extracted files
        extracted_files = sorted(list(video_output_dir.glob("frame_*.jpg")))
        logger.info(f"Successfully extracted {len(extracted_files)} frames.")
        
        return extracted_files

if __name__ == "__main__":
    # Ensure the test video exists (we created it earlier in data/videos/test_video.mp4)
    test_video = "data/videos/test_video.mp4"

    if not os.path.exists(test_video):
        print(f"Test video not found at {test_video}. Please ensure it exists.")
    else:
        extractor = VideoExtractor()
        try:
            # Extract frames every 2 seconds
            frames = extractor.extract_frames(test_video, interval=2.0)

            print("\n--- Extraction Complete ---")
            print(f"Video: {test_video}")
            print(f"Frames extracted: {len(frames)}")
            for i, frame in enumerate(frames):
                print(f"  Frame {i+1}: {frame}")

        except Exception as e:
            print(f"An error occurred during extraction: {e}")

