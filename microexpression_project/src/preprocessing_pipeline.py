#!/usr/bin/env python3
"""
Video Preprocessing Pipeline for Micro-Expression Recognition
Handles face detection, cropping, resizing, and normalization
"""

import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torchvision import transforms

from config import EMOTION_LABELS
from optical_flow_utils import triplet_to_six_channel_flow


def _plog(msg: str) -> None:
    """Print ``msg`` without failing on narrow Windows consoles (cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def _coarse_motion_peak_index(frames: List[np.ndarray]) -> int:
    """Frame index near strongest short-interval motion (helps long clips)."""
    n = len(frames)
    if n <= 2:
        return 0
    step = max(1, n // 200)
    best_i = 0
    best_m = -1.0
    for i in range(0, n - step, step):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        j = min(i + step, n - 1)
        g2 = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
        g1 = cv2.resize(g1, (96, 96), interpolation=cv2.INTER_AREA)
        g2 = cv2.resize(g2, (96, 96), interpolation=cv2.INTER_AREA)
        m = float(np.mean(np.abs(g1.astype(np.float32) - g2.astype(np.float32))))
        if m > best_m:
            best_m = m
            best_i = min(n - 1, i + step // 2)
    return int(best_i)


def _linspace_indices(lo: int, hi: int, k: int) -> List[int]:
    """k indices in [lo, hi] inclusive; handles short spans."""
    lo = int(max(0, lo))
    hi = int(max(lo, hi))
    if hi - lo + 1 <= k:
        return [int(x) for x in np.linspace(lo, hi, num=min(k, hi - lo + 1), dtype=int)]
    return [int(x) for x in np.linspace(lo, hi, num=k, dtype=int)]


def _motion_window_indices(n: int, peak: int, k: int) -> List[int]:
    """Sample k frame indices in a window around peak (micro-expression–friendly)."""
    half_span = max(k * 2, n // 8, 24)
    lo = max(0, peak - half_span)
    hi = min(n - 1, peak + half_span)
    while hi - lo < min(k - 1, n - 1):
        lo = max(0, lo - 1)
        hi = min(n - 1, hi + 1)
        if lo == 0 and hi == n - 1:
            break
    return _linspace_indices(lo, hi, k)


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
            _plog(f"[OK] Loaded {len(self.labels_df)} emotion labels from {labels_file}")
        else:
            _plog(f"[WARN] No labels file found at {labels_file}")
    
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
    
    @staticmethod
    def natural_image_paths(episode_dir: Path) -> List[Path]:
        """Sorted list of .jpg/.png paths under an episode (reg_img) folder."""
        paths = list(episode_dir.glob("*.jpg")) + list(episode_dir.glob("*.png"))
        return sorted(paths, key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)])

    @staticmethod
    def _path_closest_to_frame_number(image_paths: List[Path], target: int) -> Path:
        if not image_paths:
            raise ValueError("No image paths")
        best_p, best_d = image_paths[0], float("inf")
        for p in image_paths:
            m = re.search(r"(\d+)", p.name)
            n = int(m.group(1)) if m else 0
            d = abs(n - int(target))
            if d < best_d:
                best_d, best_p = d, p
        return best_p

    def load_onset_apex_offset_rgb(self, sample: dict, size: int = 64) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load onset / apex / offset RGB frames from disk for one episode.
        Uses label CSV frame indices matched to the closest image filename number.
        """
        ep = Path(sample["video_path"])
        if not ep.is_dir():
            return None
        paths = self.natural_image_paths(ep)
        if not paths:
            return None

        o, a, off = int(sample["onset_frame"]), int(sample["apex_frame"]), int(sample["offset_frame"])
        po = self._path_closest_to_frame_number(paths, o)
        pa = self._path_closest_to_frame_number(paths, a)
        poff = self._path_closest_to_frame_number(paths, off)

        def load_rgb(p: Path) -> np.ndarray:
            bgr = cv2.imread(str(p))
            if bgr is None:
                return np.zeros((size, size, 3), dtype=np.float32)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)

        return load_rgb(po), load_rgb(pa), load_rgb(poff)

    def get_all_samples(self, data_root: str) -> List[dict]:
        """Get all samples from data directory"""
        samples = []
        data_path = Path(data_root)
        
        if not data_path.exists():
            _plog(f"[ERROR] Data directory not found: {data_root}")
            return samples
        
        _plog(f"[scan] Scanning data directory: {data_root}")

        skipped_hidden = 0
        skipped_no_images = 0
        skipped_no_subject_label = 0
        skipped_no_episode_label = 0
        skipped_non_target_emotion = 0
        
        # Scan for subject directories
        for subject_dir in sorted(
            (p for p in data_path.iterdir() if p.is_dir()),
            key=lambda p: p.name.lower(),
        ):
            if subject_dir.name.startswith('.'):
                skipped_hidden += 1
                continue
            if subject_dir.is_dir() and subject_dir.name.startswith('sub'):
                _plog(f"[scan] Found subject: {subject_dir.name}")
                
                # Scan for episode directories
                for episode_dir in sorted(
                    (p for p in subject_dir.iterdir() if p.is_dir()),
                    key=lambda p: p.name.lower(),
                ):
                    if episode_dir.name.startswith('.'):
                        skipped_hidden += 1
                        continue
                    if episode_dir.is_dir():
                        _plog(f"  [scan] Found episode: {episode_dir.name}")
                        
                        # Check for image files
                        image_files = sorted(list(episode_dir.glob('*.jpg')) + list(episode_dir.glob('*.png')))
                        if image_files:
                            nimg = len(image_files)
                            onset_frame, apex_frame, offset_frame = 0, nimg // 2, max(0, nimg - 1)
                            emotion = 'happiness'
                            label = 0

                            if self.labels_df is not None:
                                subject_match = self.labels_df[self.labels_df['subject_id'] == subject_dir.name]
                                if subject_match.empty:
                                    skipped_no_subject_label += 1
                                    continue
                                episode_match = subject_match[subject_match['episode_id'] == episode_dir.name]
                                if episode_match.empty:
                                    skipped_no_episode_label += 1
                                    continue
                                row = episode_match.iloc[0]
                                emotion = str(row['emotion_label'])
                                if emotion not in EMOTION_LABELS:
                                    skipped_non_target_emotion += 1
                                    continue
                                label = EMOTION_LABELS[emotion]
                                onset_frame = int(row['onset_frame'])
                                apex_frame = int(row['apex_frame'])
                                offset_frame = int(row['offset_frame'])
                            else:
                                if emotion not in EMOTION_LABELS:
                                    skipped_non_target_emotion += 1
                                    continue

                            sample = {
                                'subject': subject_dir.name,
                                'episode': episode_dir.name,
                                'video_path': str(episode_dir),
                                'image_files': [str(img) for img in sorted(image_files)],
                                'num_frames': len(image_files),
                                'emotion': emotion,
                                'label': label,
                                'onset_frame': onset_frame,
                                'apex_frame': apex_frame,
                                'offset_frame': offset_frame,
                            }
                            samples.append(sample)
                            _plog(f"    [scan] Found {len(image_files)} frames - emotion: {emotion}")
                        else:
                            skipped_no_images += 1
        
        _plog(f"[OK] Total samples found: {len(samples)}")
        if self.labels_df is not None:
            _plog(
                "[scan] Skipped episodes: "
                f"hidden={skipped_hidden}, no_images={skipped_no_images}, "
                f"no_subject_label={skipped_no_subject_label}, no_episode_label={skipped_no_episode_label}, "
                f"non_target_emotion={skipped_non_target_emotion}"
            )
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
            _plog("[WARN] Face cascade not loaded, using full frame")
            self.face_detection_enabled = False
        else:
            self.face_detection_enabled = True
            _plog("[OK] Face detection enabled")
        
        # Define face crop coordinates (same as training data)
        self.face_x1, self.face_y1 = 128, 128
        self.face_x2, self.face_y2 = 384, 384
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        self._last_input_frame_count = 0
        self._last_faces_detected = 0
    
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
        Pick onset / apex / offset from a video clip.

        Uses motion-onset heuristic: apex is near the strongest frame-to-frame change
        (micro-expression dynamics), with onset at the start and offset at the end of
        the subsampled clip — closer to training than a blind middle frame on sparse samples.
        """
        n = len(frames)
        if n < 3:
            mid = frames[n // 2] if n else np.zeros((64, 64, 3), dtype=np.uint8)
            while len(frames) < 3:
                frames = list(frames) + [mid]
            n = len(frames)
        if n == 3:
            return frames[0], frames[1], frames[2]

        motions: List[float] = []
        for i in range(n - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            g1 = cv2.resize(g1, (96, 96), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(g2, (96, 96), interpolation=cv2.INTER_AREA)
            motions.append(float(np.mean(np.abs(g1.astype(np.float32) - g2.astype(np.float32)))))

        m = np.asarray(motions, dtype=np.float32)
        if len(m) >= 5:
            k = np.ones(3, dtype=np.float32) / 3.0
            smooth = np.convolve(m, k, mode="same")
            peak = int(np.argmax(smooth))
        else:
            peak = int(np.argmax(m))

        # Apex frame index: end of strongest motion interval, kept away from boundaries
        apex_idx = min(n - 2, max(1, peak + 1))
        onset_idx = 0
        offset_idx = n - 1
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

        self._last_faces_detected = 0
        
        # Process each frame
        processed_frames = []
        
        for frame in [onset, apex, offset]:
            # Detect and crop face
            face_bbox = self.detect_face(frame)
            if face_bbox is not None:
                self._last_faces_detected += 1
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

        o = processed_frames[0].numpy().transpose(1, 2, 0)
        a = processed_frames[1].numpy().transpose(1, 2, 0)
        off = processed_frames[2].numpy().transpose(1, 2, 0)
        flow_np = triplet_to_six_channel_flow(o, a, off)
        flows_tensor = torch.from_numpy(flow_np).float()

        return frames_tensor, flows_tensor
    
    def preprocess_video(
        self,
        video_path: str,
        max_input_frames: int = 64,
        verbose: bool = True,
        *,
        motion_focused_subsample: bool = True,
        max_buffer_frames: int = 900,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete video preprocessing pipeline
        
        Args:
            video_path: Path to video file
            max_input_frames: Subsample to at most this many frames before
                onset/apex/offset selection (speed + stable dynamics on long clips).
            motion_focused_subsample: If True and the clip is longer than max_input_frames,
                keep frames from a temporal window around peak motion instead of uniform
                sampling across the whole file (closer to a micro-expression in long videos).
            max_buffer_frames: If the decoder yields more frames than this, shrink with
                uniform sampling before motion analysis (memory + speed cap).
            
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

        n_raw = len(frames)
        if n_raw > max_buffer_frames:
            idx_buf = np.linspace(0, n_raw - 1, max_buffer_frames, dtype=int)
            frames = [frames[int(i)] for i in idx_buf]
            if verbose:
                _plog(
                    f"[video] Buffered {n_raw} -> {len(frames)} frames (max_buffer_frames={max_buffer_frames})"
                )

        n_in = len(frames)
        if n_in > max_input_frames:
            if motion_focused_subsample:
                peak = _coarse_motion_peak_index(frames)
                idx = _motion_window_indices(n_in, peak, max_input_frames)
                if verbose:
                    _plog(
                        f"[video] Motion-focused subsample {n_in} -> {len(idx)} frames (peak~{peak}) from {video_path}"
                    )
            else:
                idx = list(np.linspace(0, n_in - 1, max_input_frames, dtype=int))
                if verbose:
                    _plog(f"[video] Uniform subsample {n_in} -> {len(idx)} frames from {video_path}")
            frames = [frames[int(i)] for i in idx]
        else:
            if verbose:
                _plog(f"[video] Loaded {len(frames)} frames from {video_path}")
        self._last_input_frame_count = len(frames)
        
        # Preprocess frames
        frames_tensor, flows_tensor = self.preprocess_frames(frames)
        
        if verbose:
            _plog(f"[OK] Preprocessed: frames {frames_tensor.shape}, flows {flows_tensor.shape}")
        
        return frames_tensor, flows_tensor

def test_preprocessor():
    """Test the video preprocessor"""
    
    _plog("=== TESTING VIDEO PREPROCESSOR ===")
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor()
    
    # Test with a sample video
    test_video = "data/predict/sub01/EP02_01f.avi"
    
    if not Path(test_video).exists():
        _plog(f"[ERROR] Test video not found: {test_video}")
        return
    
    try:
        # Preprocess video
        frames_tensor, flows_tensor = preprocessor.preprocess_video(test_video)
        
        _plog("[OK] Preprocessing successful!")
        _plog(f"   Frames shape: {frames_tensor.shape}")
        _plog(f"   Flows shape: {flows_tensor.shape}")
        _plog(f"   Frames range: [{frames_tensor.min():.3f}, {frames_tensor.max():.3f}]")
        _plog(f"   Frames dtype: {frames_tensor.dtype}")
        
    except Exception as e:
        _plog(f"[ERROR] Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_preprocessor()
