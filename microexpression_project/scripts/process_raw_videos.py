#!/usr/bin/env python3
"""
Clean Web Batch Video Processor for Micro-Expression Recognition
WEB INFERENCE ONLY - No labels, no image saving, no CASME assumptions
"""

import torch
from pathlib import Path
from typing import List, Tuple
import tempfile
import shutil

# Add src to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in str(__import__('sys').path):
    __import__('sys').path.insert(0, str(src_path))

from preprocessing_pipeline import VideoPreprocessor


class WebBatchVideoProcessor:
    """
    Batch processor for WEB INFERENCE ONLY
    
    Characteristics:
    - No labels (web uploads have no ground truth)
    - No image saving (memory-efficient)
    - No CASME assumptions (works with any uploaded video)
    - Temporary processing only
    """

    def __init__(self, input_dir: str = None, use_temp: bool = True):
        """
        Initialize web batch processor
        
        Args:
            input_dir: Directory containing uploaded videos (optional)
            use_temp: Whether to use temporary directory for processing
        """
        self.input_dir = Path(input_dir) if input_dir else None
        self.preprocessor = VideoPreprocessor()
        
        # Use temp directory for processing
        if use_temp:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="microexp_web_"))
        else:
            self.temp_dir = Path.cwd() / "temp_web_processing"
            self.temp_dir.mkdir(exist_ok=True)
        
        print(f"🔧 Web processor initialized with temp dir: {self.temp_dir}")

    def find_all_videos(self, input_dir: str = None) -> List[Path]:
        """
        Find all video files in input directory
        
        Args:
            input_dir: Override input directory (optional)
            
        Returns:
            List of video file paths
        """
        search_dir = Path(input_dir) if input_dir else self.input_dir
        
        if not search_dir or not search_dir.exists():
            print(f"❌ Input directory not found: {search_dir}")
            return []
        
        video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(search_dir.rglob(f'*{ext}'))
        
        return sorted(video_files)

    def process_video(self, video_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process single video for web inference
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames_tensor, flows_tensor) ready for model
        """
        try:
            print(f"📹 Web inference: {video_path.name}")
            
            # Use preprocessor directly - no image saving, no labels needed
            frames_tensor, flows_tensor = self.preprocessor.preprocess_video(str(video_path))
            
            print(f"✅ Processed: frames {frames_tensor.shape}, flows {flows_tensor.shape}")
            return frames_tensor, flows_tensor
            
        except Exception as e:
            print(f"❌ Error processing {video_path.name}: {e}")
            return None, None

    def process_batch(self, input_dir: str = None, max_videos: int = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process multiple videos for batch inference
        
        Args:
            input_dir: Override input directory (optional)
            max_videos: Maximum number of videos to process (optional)
            
        Returns:
            List of (frames_tensor, flows_tensor) tuples
        """
        video_files = self.find_all_videos(input_dir)
        
        if not video_files:
            print("❌ No video files found!")
            return []
        
        # Limit number of videos if specified
        if max_videos:
            video_files = video_files[:max_videos]
        
        print(f"🎬 Processing {len(video_files)} uploaded videos")
        print("=" * 60)
        
        results = []
        successful = 0
        failed = 0
        
        for video_path in video_files:
            frames_tensor, flows_tensor = self.process_video(video_path)
            
            if frames_tensor is not None and flows_tensor is not None:
                results.append((frames_tensor, flows_tensor))
                successful += 1
            else:
                failed += 1
        
        print("=" * 60)
        print(f"📊 Batch Processing Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {successful/(successful+failed)*100:.1f}%")
        
        return results

    def process_uploaded_file(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process single uploaded file (Flask integration)
        
        Args:
            video_path: Path to uploaded video file
            
        Returns:
            Tuple of (frames_tensor, flows_tensor) ready for model
        """
        video_path = Path(video_path)
        return self.process_video(video_path)

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Could not cleanup temp directory: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


# Convenience functions for Flask integration
def process_single_uploaded_video(video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process single uploaded video - Flask integration helper
    
    Args:
        video_path: Path to uploaded video file
        
    Returns:
        Tuple of (frames_tensor, flows_tensor) ready for model
    """
    processor = WebBatchVideoProcessor()
    try:
        return processor.process_uploaded_file(video_path)
    finally:
        processor.cleanup()


def process_uploaded_batch(upload_dir: str, max_videos: int = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Process batch of uploaded videos - Flask integration helper
    
    Args:
        upload_dir: Directory containing uploaded videos
        max_videos: Maximum number of videos to process
        
    Returns:
        List of (frames_tensor, flows_tensor) tuples
    """
    processor = WebBatchVideoProcessor(upload_dir)
    try:
        return processor.process_batch(max_videos=max_videos)
    finally:
        processor.cleanup()


# Demo and testing
def main():
    """Demo function for testing"""
    print("🎬 Web Batch Video Processor - DEMO")
    print("=" * 60)
    
    # Test with data/predict directory (has sample videos)
    project_root = Path(__file__).parent.parent
    test_dir = project_root / 'data' / 'predict'
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        print("💡 This demo requires videos in data/predict/")
        return
    
    # Initialize processor
    processor = WebBatchVideoProcessor(str(test_dir))
    
    try:
        # Process first few videos as demo
        results = processor.process_batch(max_videos=3)
        
        if results:
            print(f"\n✅ Successfully processed {len(results)} videos")
            print("📊 Tensor shapes:")
            for i, (frames, flows) in enumerate(results):
                print(f"   Video {i+1}: frames {frames.shape}, flows {flows.shape}")
            
            print("\n🎉 Demo complete! Ready for model inference.")
        else:
            print("❌ No videos processed successfully")
            
    finally:
        processor.cleanup()


if __name__ == '__main__':
    main()
