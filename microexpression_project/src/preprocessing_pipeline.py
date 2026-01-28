#!/usr/bin/env python3
"""
Video Preprocessing Pipeline for Micro-Expression Recognition
Handles face detection, cropping, resizing, and normalization
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torchvision import transforms

class OnsetApexOffsetSelector:
    """Simple frame selector for onset, apex, and offset frames"""
    
    def __init__(self, labels_file: str = None):
        """
        Initialize frame selector
        
        Args:
            labels_file: Path to labels file (CSV with emotion labels)
        """
        self.labels_file = labels_file
        self.labels_df = None
        if labels_file and Path(labels_file).exists():
            self.labels_df = pd.read_csv(labels_file)
            print(f"✅ Loaded {len(self.labels_df)} emotion labels from {labels_file}")
        else:
            print(f"⚠️ No labels file found at {labels_file}")
    
    def select_frames(self, frames: List[np.ndarray], metadata: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select onset, apex, and offset frames"""
        num_frames = len(frames)
        if num_frames == 0:
            # If no frames provided, create safe dummy frames
            dummy_frame = np.zeros((64, 64, 3), dtype=np.float32)
            return dummy_frame, dummy_frame, dummy_frame
        elif num_frames < 3:
            # If less than 3 frames, duplicate the middle frame
            middle_frame = frames[num_frames // 2]
            return middle_frame, middle_frame, middle_frame
        
        onset_idx = 0
        apex_idx = num_frames // 2
        offset_idx = num_frames - 1
        
        return frames[onset_idx], frames[apex_idx], frames[offset_idx]
    
    def get_all_samples(self, data_root: str) -> List[dict]:
        """Get all samples from data directory"""
        samples = []
        data_path = Path(data_root)
        
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_root}")
            return samples
        
        print(f"🔍 Scanning data directory: {data_root}")
        
        # Scan for subject directories
        for subject_dir in data_path.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub'):
                print(f"📁 Found subject: {subject_dir.name}")
                
                # Scan for episode directories
                for episode_dir in subject_dir.iterdir():
                    if episode_dir.is_dir():
                        print(f"  📂 Found episode: {episode_dir.name}")
                        
                        # Check for image files
                        image_files = list(episode_dir.glob('*.jpg')) + list(episode_dir.glob('*.png'))
                        if image_files:
                            # Get actual emotion label from CSV
                            emotion = 'happiness'  # default
                            label = 0
                            
                            if self.labels_df is not None:
                                # Find matching episode in labels
                                subject_match = self.labels_df[self.labels_df['subject_id'] == subject_dir.name]
                                if not subject_match.empty:
                                    episode_match = subject_match[subject_match['episode_id'] == episode_dir.name]
                                    if not episode_match.empty:
                                        emotion = episode_match.iloc[0]['emotion_label']
                                        # Map emotion to label index
                                        from config import EMOTION_LABELS
                                        label = EMOTION_LABELS.get(emotion, 0)
                            
                            # Create a sample entry
                            sample = {
                                'subject': subject_dir.name,
                                'episode': episode_dir.name,
                                'video_path': str(episode_dir),
                                'image_files': [str(img) for img in image_files],
                                'num_frames': len(image_files),
                                'emotion': emotion,
                                'label': label,
                                'onset_frame': 0,
                                'apex_frame': len(image_files) // 2,
                                'offset_frame': len(image_files) - 1
                            }
                            samples.append(sample)
                            print(f"    📸 Found {len(image_files)} frames - emotion: {emotion}")
        
        print(f"✅ Total samples found: {len(samples)}")
        return samples

class VideoPreprocessor:
    """Complete video preprocessing pipeline for micro-expression recognition"""
    
    def __init__(self, face_cascade_path: str = None):
        """
        Initialize preprocessor with face detection cascade
        
        Args:
            face_cascade_path: Path to OpenCV face cascade file
        """
        # Load face detection cascade
        if face_cascade_path is None:
            # Use OpenCV's built-in face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        if self.face_cascade.empty():
            print("⚠️  Warning: Face cascade not loaded, using full frame")
            self.face_detection_enabled = False
        else:
            self.face_detection_enabled = True
            print("✅ Face detection enabled")
        
        # Define face crop coordinates (same as training data)
        self.face_x1, self.face_y1 = 128, 128
        self.face_x2, self.face_y2 = 384, 384
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame using OpenCV cascade
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Face bounding box (x, y, w, h) or None if no face detected
        """
        if not self.face_detection_enabled:
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
        
        # Return the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        return tuple(largest_face)
    
    def crop_face_region(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Crop face region from frame
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, w, h) from detect_face
            
        Returns:
            Cropped face region (256x256)
        """
        h, w = frame.shape[:2]
        
        if face_bbox is not None:
            # Use detected face
            x, y, face_w, face_h = face_bbox
            
            # Expand bounding box slightly
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + face_w + margin)
            y2 = min(h, y + face_h + margin)
            
            # Crop face region
            face_region = frame[y1:y2, x1:x2]
        else:
            # Use fixed center crop (same as training data)
            center_x, center_y = w // 2, h // 2
            crop_size = 256
            
            x1 = max(0, center_x - crop_size // 2)
            y1 = max(0, center_y - crop_size // 2)
            x2 = min(w, center_x + crop_size // 2)
            y2 = min(h, center_y + crop_size // 2)
            
            face_region = frame[y1:y2, x1:x2]
        
        # Resize to 256x256
        face_resized = cv2.resize(face_region, (256, 256))
        
        return face_resized
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame to [0, 1] range
        
        Args:
            frame: Input frame (0-255 range)
            
        Returns:
            Normalized frame (0-1 range)
        """
        return frame.astype(np.float32) / 255.0
    
    def extract_onset_apex_offset(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract onset, apex, and offset frames from video
        
        Args:
            frames: List of video frames
            
        Returns:
            Tuple of (onset_frame, apex_frame, offset_frame)
        """
        if len(frames) < 3:
            raise ValueError("Video must have at least 3 frames")
        
        # Simple strategy: first, middle, last frames
        onset_idx = 0
        apex_idx = len(frames) // 2
        offset_idx = len(frames) - 1
        
        return frames[onset_idx], frames[apex_idx], frames[offset_idx]
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess frames for model input
        
        Args:
            frames: List of video frames (BGR format)
            
        Returns:
            Tuple of (frames_tensor, flows_tensor)
            frames_tensor: (3, 3, 64, 64) - TCHW format
            flows_tensor: (6, 64, 64) - CHW format
        """
        # Extract onset, apex, offset frames
        onset, apex, offset = self.extract_onset_apex_offset(frames)
        
        # Process each frame
        processed_frames = []
        
        for frame in [onset, apex, offset]:
            # Detect and crop face
            face_bbox = self.detect_face(frame)
            face_region = self.crop_face_region(frame, face_bbox)
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Normalize
            face_normalized = self.normalize_frame(face_rgb)
            
            # Apply transforms and convert to tensor
            face_tensor = self.transform(face_normalized)
            processed_frames.append(face_tensor)
        
        # Stack frames into tensor with correct shape (3, 3, 64, 64)
        frames_tensor = torch.stack(processed_frames)  # Shape: (3, 3, 64, 64)
        
        # For now, create dummy flows (in a real implementation, you'd compute optical flow)
        # This is a placeholder - you'd need to implement actual optical flow computation
        flows_tensor = torch.zeros(6, 64, 64)  # Shape: (6, 64, 64)
        
        return frames_tensor, flows_tensor
    
    def preprocess_video(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete video preprocessing pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames_tensor, flows_tensor)
        """
        # Read video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames found in video")
        
        print(f"📹 Loaded {len(frames)} frames from {video_path}")
        
        # Preprocess frames
        frames_tensor, flows_tensor = self.preprocess_frames(frames)
        
        print(f"✅ Preprocessed: frames {frames_tensor.shape}, flows {flows_tensor.shape}")
        
        return frames_tensor, flows_tensor

def test_preprocessor():
    """Test the video preprocessor"""
    
    print("=== TESTING VIDEO PREPROCESSOR ===")
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor()
    
    # Test with a sample video
    test_video = "data/predict/sub01/EP02_01f.avi"
    
    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        return
    
    try:
        # Preprocess video
        frames_tensor, flows_tensor = preprocessor.preprocess_video(test_video)
        
        print(f"✅ Preprocessing successful!")
        print(f"   Frames shape: {frames_tensor.shape}")
        print(f"   Flows shape: {flows_tensor.shape}")
        print(f"   Frames range: [{frames_tensor.min():.3f}, {frames_tensor.max():.3f}]")
        print(f"   Frames dtype: {frames_tensor.dtype}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_preprocessor()
