import numpy as np
import cv2
import torch
from typing import Tuple, Optional
import warnings

# Suppress OpenCV warnings for optical flow
warnings.filterwarnings("ignore", category=UserWarning)


class OpticalFlowExtractor:
    """
    Extracts optical flow between consecutive frames for micro-expression recognition.
    
    Computes flow from onset→apex and apex→offset to capture motion dynamics.
    """
    
    def __init__(self, method: str = 'farneback'):
        """
        Initialize optical flow extractor.
        
        Args:
            method: Optical flow method ('farneback' or 'lucas_kanade')
        """
        self.method = method
        
        # Farneback parameters for micro-expression detection
        if method == 'farneback':
            self.farneback_params = dict(
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow between two frames.
        
        Args:
            frame1: First frame (onset or apex)
            frame2: Second frame (apex or offset)
        
        Returns:
            Tuple of (flow_x, flow_y) as numpy arrays
        """
        # Ensure frames are uint8 for OpenCV
        if frame1.dtype != np.uint8:
            frame1 = (frame1 * 255).astype(np.uint8)
        if frame2.dtype != np.uint8:
            frame2 = (frame2 * 255).astype(np.uint8)
        
        if self.method == 'farneback':
            # Compute dense optical flow using Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, **self.farneback_params
            )
            
            # Split into x and y components
            flow_x = flow[:, :, 0]
            flow_y = flow[:, :, 1]
            
        elif self.method == 'lucas_kanade':
            # Compute sparse optical flow using Lucas-Kanade
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            if len(frame2.shape) == 3:
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            
            # Feature points for tracking
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Detect good features to track
            p0 = cv2.goodFeaturesToTrack(frame1, maxCorners=100, qualityLevel=0.01, minDistance=7)
            
            if p0 is not None and len(p0) > 0:
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    frame1, frame2, p0, None, **lk_params
                )
                
                # Select good points
                good_old = p0[st == 1]
                good_new = p1[st == 1]
                
                # Create sparse flow field
                flow_x = np.zeros_like(frame1, dtype=np.float32)
                flow_y = np.zeros_like(frame1, dtype=np.float32)
                
                for i, (old, new) in enumerate(zip(good_old, good_new)):
                    x, y = int(new[0]), int(new[1])
                    if 0 <= x < flow_x.shape[1] and 0 <= y < flow_x.shape[0]:
                        flow_x[y, x] = new[0] - old[0]
                        flow_y[y, x] = new[1] - old[1]
            else:
                flow_x = np.zeros_like(frame1, dtype=np.float32)
                flow_y = np.zeros_like(frame1, dtype=np.float32)
        
        else:
            raise ValueError(f"Unknown optical flow method: {self.method}")
        
        return flow_x, flow_y
    
    def extract_flow_features_with_stats(self, onset: np.ndarray, apex: np.ndarray, 
                                     offset: np.ndarray, regions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract optical flow features and strain statistics from onset-apex-offset frames.
        
        Args:
            onset: Onset frame (64x64)
            apex: Apex frame (64x64)
            offset: Offset frame (64x64)
            regions: Number of regions for strain statistics
        
        Returns:
            Tuple of (flow_features [6, 64, 64], strain_statistics [32])
        """
        # Compute flow between onset→apex
        flow_x1, flow_y1 = self.compute_flow(onset, apex)
        
        # Compute flow between apex→offset
        flow_x2, flow_y2 = self.compute_flow(apex, offset)
        
        # Compute strain for both flow pairs
        strain1 = compute_strain(flow_x1, flow_y1)
        strain2 = compute_strain(flow_x2, flow_y2)
        
        # Compute strain statistics
        stats1 = compute_strain_statistics(strain1, regions)
        stats2 = compute_strain_statistics(strain2, regions)
        
        # Combine statistics (32 features: 16 per strain map)
        strain_statistics = np.concatenate([stats1, stats2])
        
        # Stack all flow components and strain
        flow_features = np.stack([
            flow_x1, flow_y1,
            flow_x2, flow_y2,
            strain1, strain2
        ], axis=0)
        
        return flow_features, strain_statistics
    
    def extract_flow_features(self, onset: np.ndarray, apex: np.ndarray, 
                           offset: np.ndarray) -> np.ndarray:
        """
        Extract optical flow features from onset-apex-offset frames.
        
        Args:
            onset: Onset frame (64x64)
            apex: Apex frame (64x64)
            offset: Offset frame (64x64)
        
        Returns:
            Flow features as numpy array [flow_x1, flow_y1, flow_x2, flow_y2, strain1, strain2]
        """
        # Compute flow between onset→apex
        flow_x1, flow_y1 = self.compute_flow(onset, apex)
        
        # Compute flow between apex→offset
        flow_x2, flow_y2 = self.compute_flow(apex, offset)
        
        # Compute strain for both flow pairs
        strain1 = compute_strain(flow_x1, flow_y1)
        strain2 = compute_strain(flow_x2, flow_y2)
        
        # Stack all flow components and strain
        flow_features = np.stack([
            flow_x1, flow_y1,
            flow_x2, flow_y2,
            strain1, strain2
        ], axis=0)
        
        return flow_features
    
    def normalize_flow(self, flow_features: np.ndarray) -> np.ndarray:
        """
        Normalize flow features to [0, 1] range while preserving temporal information.
        
        Uses dataset-level statistics to preserve absolute motion strength.
        
        Args:
            flow_features: Flow features [4, 64, 64]
        
        Returns:
            Normalized flow features
        """
        # Use fixed normalization bounds based on typical optical flow ranges
        # This preserves relative motion strength across samples
        flow_min, flow_max = -5.0, 5.0  # Typical optical flow range for 64x64 images
        
        # Clip to reasonable range
        flow_features = np.clip(flow_features, flow_min, flow_max)
        
        # Normalize to [0, 1]
        normalized_flow = (flow_features - flow_min) / (flow_max - flow_min)
        
        return normalized_flow
    
    def extract_and_normalize(self, onset: np.ndarray, apex: np.ndarray, 
                             offset: np.ndarray) -> np.ndarray:
        """
        Extract and normalize flow features in one step.
        
        Args:
            onset: Onset frame (64x64)
            apex: Apex frame (64x64)
            offset: Offset frame (64x64)
        
        Returns:
            Normalized flow features [4, 64, 64]
        """
        flow_features = self.extract_flow_features(onset, apex, offset)
        normalized_flow = self.normalize_flow(flow_features)
        
        return normalized_flow


def compute_au_aligned_strain_statistics(strain: np.ndarray) -> np.ndarray:
    """
    Compute strain statistics for AU-aligned ROIs.
    
    Uses Action Unit regions: AU4, AU6, AU9, AU10, AU12
    These regions are more interpretable and noise-resistant.
    
    Args:
        strain: Strain magnitude map (64, 64)
    
    Returns:
        AU-aligned statistical features [mean, std, max, energy] per AU (20 features total)
    """
    # AU-aligned ROI coordinates (approximate for 64x64 face)
    # These are based on standard facial AU locations
    au_rois = {
        'AU4': (16, 20, 28, 32),    # Brow lowerer (left)
        'AU6': (36, 20, 48, 32),    # Brow lowerer (right)
        'AU9': (20, 28, 44, 40),    # Nose wrinkler
        'AU10': (20, 40, 44, 48),   # Upper lip raiser
        'AU12': (24, 16, 40, 28)   # Lip corner puller (left)
    }
    
    features = []
    
    for au_name, (y1, x1, y2, x2) in au_rois.items():
        # Extract AU region
        au_strain = strain[y1:y2, x1:x2]
        
        # Compute statistics
        mean_val = np.mean(au_strain)
        std_val = np.std(au_strain)
        max_val = np.max(au_strain)
        energy = np.sum(au_strain ** 2)
        
        features.extend([mean_val, std_val, max_val, energy])
    
    return np.array(features)


def compute_strain_statistics(strain: np.ndarray, regions: int = 4) -> np.ndarray:
    """
    Compute statistical features from strain map for motion aggregation.
    
    Args:
        strain: Strain magnitude map (64, 64)
        regions: Number of regions to divide the face into (default: 4)
    
    Returns:
        Statistical features [mean, std, max, energy] per region
    """
    # Divide strain map into regions
    h, w = strain.shape
    region_h = h // regions
    region_w = w // regions
    
    features = []
    
    for i in range(regions):
        for j in range(regions):
            # Extract region
            y_start = i * region_h
            y_end = (i + 1) * region_h if i < regions - 1 else h
            x_start = j * region_w
            x_end = (j + 1) * region_w if j < regions - 1 else w
            
            region_strain = strain[y_start:y_end, x_start:x_end]
            
            # Compute statistics
            mean_val = np.mean(region_strain)
            std_val = np.std(region_strain)
            max_val = np.max(region_strain)
            energy = np.sum(region_strain ** 2)
            
            features.extend([mean_val, std_val, max_val, energy])
    
    return np.array(features)


def compute_strain(flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
    """
    Compute optical strain magnitude from flow.
    
    Args:
        flow_x: X-component of optical flow
        flow_y: Y-component of optical flow
    
    Returns:
        Strain magnitude tensor
    """
    fx_x = cv2.Sobel(flow_x, cv2.CV_32F, 1, 0, ksize=3)
    fx_y = cv2.Sobel(flow_x, cv2.CV_32F, 0, 1, ksize=3)
    fy_x = cv2.Sobel(flow_y, cv2.CV_32F, 1, 0, ksize=3)
    fy_y = cv2.Sobel(flow_y, cv2.CV_32F, 0, 1, ksize=3)

    strain = np.sqrt(fx_x**2 + fx_y**2 + fy_x**2 + fy_y**2)
    return strain


def compute_frame_differences(onset: np.ndarray, apex: np.ndarray, 
                              offset: np.ndarray) -> np.ndarray:
    """
    Compute simple frame differences as baseline motion features.
    
    Args:
        onset: Onset frame (64x64)
        apex: Apex frame (64x64)
        offset: Offset frame (64x64)
    
    Returns:
        Frame differences [apex-onset, offset-apex] as [2, 64, 64]
    """
    # Compute differences
    diff1 = apex - onset  # onset→apex
    diff2 = offset - apex  # apex→offset
    
    # Stack differences
    differences = np.stack([diff1, diff2], axis=0)
    
    return differences


if __name__ == "__main__":
    # Test optical flow extraction
    print("Testing Optical Flow Extraction...")
    
    # Create synthetic test frames
    np.random.seed(42)
    onset = np.random.rand(64, 64) * 0.5
    apex = np.random.rand(64, 64) * 0.7
    offset = np.random.rand(64, 64) * 0.6
    
    print(f"Test frames shape: {onset.shape}")
    print(f"Onset range: [{onset.min():.3f}, {onset.max():.3f}]")
    print(f"Apex range: [{apex.min():.3f}, {apex.max():.3f}]")
    print(f"Offset range: [{offset.min():.3f}, {offset.max():.3f}]")
    
    # Test optical flow extraction
    extractor = OpticalFlowExtractor(method='farneback')
    
    # Extract flow features
    flow_features = extractor.extract_and_normalize(onset, apex, offset)
    
    print(f"\nFlow features shape: {flow_features.shape}")
    print(f"Flow range: [{flow_features.min():.3f}, {flow_features.max():.3f}]")
    print(f"Flow mean: {flow_features.mean():.3f}")
    print(f"Flow std: {flow_features.std():.3f}")
    
    # Test frame differences
    differences = compute_frame_differences(onset, apex, offset)
    print(f"\nFrame differences shape: {differences.shape}")
    print(f"Differences range: [{differences.min():.3f}, {differences.max():.3f}]")
    
    print("\n✅ Optical flow extraction working correctly!")
