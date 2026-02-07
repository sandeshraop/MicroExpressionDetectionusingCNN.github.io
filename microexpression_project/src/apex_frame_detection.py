#!/usr/bin/env python3
"""
Adaptive Apex Frame Detection for Micro-Expression Recognition

Key Innovation: Identifies frame with maximum facial motion intensity
Micro-expressions peak at apex, providing most discriminative features.

Algorithm: Motion-based peak detection with temporal constraints
- Micro-expressions last 200-500ms
- Apex frame contains maximum motion intensity
- Temporal smoothing reduces noise
"""

import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
from typing import Tuple, List, Optional, Dict
import torch


class ApexFrameDetector:
    """
    Adaptive apex frame detection for micro-expression sequences.
    
    Identifies the frame with maximum facial motion intensity,
    considering micro-expression duration constraints (200-500ms).
    """
    
    def __init__(self, fps: float = 30.0, min_duration_ms: float = 200.0, 
                 max_duration_ms: float = 500.0, smoothing_sigma: float = 1.0):
        """
        Initialize apex frame detector.
        
        Args:
            fps: Frame rate of video sequence
            min_duration_ms: Minimum micro-expression duration in milliseconds
            max_duration_ms: Maximum micro-expression duration in milliseconds
            smoothing_sigma: Sigma for Gaussian smoothing of motion signal
        """
        self.fps = fps
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.smoothing_sigma = smoothing_sigma
        
        # Convert duration constraints to frame counts
        self.min_frames = max(1, int(min_duration_ms / 1000.0 * fps))
        self.max_frames = max(2, int(max_duration_ms / 1000.0 * fps))
        
        print(f"🎯 Apex Detection Config:")
        print(f"   FPS: {fps}")
        print(f"   Duration range: {min_duration_ms}ms - {max_duration_ms}ms")
        print(f"   Frame range: {self.min_frames} - {self.max_frames} frames")
    
    def compute_motion_magnitude(self, optical_flows: List[np.ndarray]) -> np.ndarray:
        """
        Compute motion magnitude for each optical flow.
        
        Args:
            optical_flows: List of optical flow arrays (height, width, 2)
            
        Returns:
            Array of motion magnitudes (num_flows,)
        """
        motion_magnitudes = []
        
        for flow in optical_flows:
            # Compute flow magnitude
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            magnitude = np.sqrt(flow_x**2 + flow_y**2)
            
            # Use mean magnitude as overall motion measure
            mean_magnitude = np.mean(magnitude)
            motion_magnitudes.append(mean_magnitude)
        
        return np.array(motion_magnitudes)
    
    def compute_advanced_motion_features(self, optical_flows: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute advanced motion features for robust apex detection.
        
        Args:
            optical_flows: List of optical flow arrays
            
        Returns:
            Dictionary of motion features
        """
        features = {}
        
        # Basic magnitude
        features['magnitude'] = self.compute_motion_magnitude(optical_flows)
        
        # Vertical motion emphasis (FaceSleuth integration)
        vertical_motions = []
        horizontal_motions = []
        
        for flow in optical_flows:
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            vertical_motions.append(np.mean(np.abs(flow_y)))
            horizontal_motions.append(np.mean(np.abs(flow_x)))
        
        features['vertical_motion'] = np.array(vertical_motions)
        features['horizontal_motion'] = np.array(horizontal_motions)
        features['vertical_dominance'] = features['vertical_motion'] / (features['horizontal_motion'] + 1e-8)
        
        # Motion consistency (how stable the motion is)
        features['motion_consistency'] = self._compute_motion_consistency(optical_flows)
        
        # Flow divergence (expansion/contraction patterns)
        features['flow_divergence'] = self._compute_flow_divergence(optical_flows)
        
        return features
    
    def _compute_motion_consistency(self, optical_flows: List[np.ndarray]) -> np.ndarray:
        """
        Compute motion consistency across frames.
        
        Args:
            optical_flows: List of optical flow arrays
            
        Returns:
            Array of consistency scores
        """
        if len(optical_flows) < 2:
            return np.array([1.0])
        
        consistencies = []
        
        for i in range(len(optical_flows)):
            # Compare with neighboring frames
            if i == 0:
                neighbor_flow = optical_flows[1]
            elif i == len(optical_flows) - 1:
                neighbor_flow = optical_flows[-2]
            else:
                # Average of previous and next
                neighbor_flow = (optical_flows[i-1] + optical_flows[i+1]) / 2
            
            # Compute correlation with neighbor
            current_flow = optical_flows[i]
            correlation = np.corrcoef(current_flow.flatten(), neighbor_flow.flatten())[0, 1]
            
            # Handle NaN values
            if np.isnan(correlation):
                correlation = 0.0
            
            consistencies.append(abs(correlation))
        
        return np.array(consistencies)
    
    def _compute_flow_divergence(self, optical_flows: List[np.ndarray]) -> np.ndarray:
        """
        Compute flow divergence (expansion/contraction patterns).
        
        Args:
            optical_flows: List of optical flow arrays
            
        Returns:
            Array of divergence values
        """
        divergences = []
        
        for flow in optical_flows:
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            # Compute gradients
            if flow_x.ndim >= 2:
                grad_x_x = np.gradient(flow_x, axis=1)
                grad_y_y = np.gradient(flow_y, axis=0)
            else:
                grad_x_x = np.zeros_like(flow_x)
                grad_y_y = np.zeros_like(flow_y)
            
            # Divergence = ∂u/∂x + ∂v/∂y
            divergence = grad_x_x + grad_y_y
            mean_divergence = np.mean(np.abs(divergence))
            
            divergences.append(mean_divergence)
        
        return np.array(divergences)
    
    def detect_apex_frame(self, optical_flows: List[np.ndarray], 
                         method: str = 'adaptive') -> Tuple[int, Dict]:
        """
        Detect apex frame from optical flow sequence.
        
        Args:
            optical_flows: List of optical flow arrays
            method: Detection method ('adaptive', 'magnitude', 'combined')
            
        Returns:
            Tuple of (apex_frame_index, detection_info)
        """
        if not optical_flows:
            raise ValueError("No optical flows provided")
        
        if len(optical_flows) == 1:
            # Single frame case
            return 0, {'method': 'single_frame', 'confidence': 1.0}
        
        # Compute motion features
        features = self.compute_advanced_motion_features(optical_flows)
        
        if method == 'adaptive':
            apex_idx, info = self._adaptive_apex_detection(features)
        elif method == 'magnitude':
            apex_idx, info = self._magnitude_based_detection(features)
        elif method == 'combined':
            apex_idx, info = self._combined_detection(features)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        # Add metadata
        info.update({
            'total_frames': len(optical_flows),
            'method': method,
            'fps': self.fps,
            'duration_range_ms': (self.min_duration_ms, self.max_duration_ms)
        })
        
        return apex_idx, info
    
    def _adaptive_apex_detection(self, features: Dict[str, np.ndarray]) -> Tuple[int, Dict]:
        """
        Adaptive apex detection using multiple motion features.
        
        Args:
            features: Dictionary of motion features
            
        Returns:
            Tuple of (apex_index, detection_info)
        """
        # Smooth the motion signals
        smoothed_magnitude = gaussian_filter1d(features['magnitude'], self.smoothing_sigma)
        smoothed_vertical = gaussian_filter1d(features['vertical_motion'], self.smoothing_sigma)
        
        # Combine features with weights
        combined_signal = (0.5 * smoothed_magnitude + 
                         0.3 * smoothed_vertical + 
                         0.2 * features['motion_consistency'])
        
        # Find peaks within duration constraints
        peaks, properties = find_peaks(
            combined_signal,
            distance=self.min_frames,
            prominence=np.std(combined_signal) * 0.5
        )
        
        if len(peaks) == 0:
            # No peaks found, use maximum
            apex_idx = np.argmax(combined_signal)
            confidence = 0.5
            method_used = 'no_peaks'
        else:
            # Select most prominent peak
            prominences = properties['prominences']
            apex_idx = peaks[np.argmax(prominences)]
            confidence = float(prominences[np.argmax(prominences)] / np.max(prominences))
            method_used = 'peak_detection'
        
        info = {
            'apex_frame': apex_idx,
            'confidence': confidence,
            'method_used': method_used,
            'peaks_found': len(peaks),
            'combined_signal': combined_signal.tolist(),
            'smoothed_magnitude': smoothed_magnitude.tolist()
        }
        
        return apex_idx, info
    
    def _magnitude_based_detection(self, features: Dict[str, np.ndarray]) -> Tuple[int, Dict]:
        """
        Simple magnitude-based apex detection.
        
        Args:
            features: Dictionary of motion features
            
        Returns:
            Tuple of (apex_index, detection_info)
        """
        # Smooth magnitude signal
        smoothed_magnitude = gaussian_filter1d(features['magnitude'], self.smoothing_sigma)
        
        # Find maximum
        apex_idx = np.argmax(smoothed_magnitude)
        max_magnitude = smoothed_magnitude[apex_idx]
        
        # Compute confidence based on how prominent the peak is
        confidence = max_magnitude / (np.mean(smoothed_magnitude) + 1e-8)
        confidence = min(1.0, confidence / 2.0)  # Normalize to [0, 1]
        
        info = {
            'apex_frame': apex_idx,
            'confidence': confidence,
            'method_used': 'magnitude_max',
            'max_magnitude': float(max_magnitude),
            'mean_magnitude': float(np.mean(smoothed_magnitude)),
            'smoothed_magnitude': smoothed_magnitude.tolist()
        }
        
        return apex_idx, info
    
    def _combined_detection(self, features: Dict[str, np.ndarray]) -> Tuple[int, Dict]:
        """
        Combined detection using multiple criteria.
        
        Args:
            features: Dictionary of motion features
            
        Returns:
            Tuple of (apex_index, detection_info)
        """
        # Weighted combination of features
        magnitude_score = gaussian_filter1d(features['magnitude'], self.smoothing_sigma)
        vertical_score = gaussian_filter1d(features['vertical_motion'], self.smoothing_sigma)
        consistency_score = features['motion_consistency']
        
        # Normalize scores to [0, 1]
        magnitude_norm = magnitude_score / (np.max(magnitude_score) + 1e-8)
        vertical_norm = vertical_score / (np.max(vertical_score) + 1e-8)
        
        # Combined score with emphasis on vertical motion (FaceSleuth)
        combined_score = 0.4 * magnitude_norm + 0.4 * vertical_norm + 0.2 * consistency_score
        
        # Find apex
        apex_idx = np.argmax(combined_score)
        max_score = combined_score[apex_idx]
        
        # Confidence based on score distribution
        confidence = (max_score - np.mean(combined_score)) / (np.std(combined_score) + 1e-8)
        confidence = min(1.0, max(0.0, confidence / 3.0))  # Normalize to [0, 1]
        
        info = {
            'apex_frame': apex_idx,
            'confidence': confidence,
            'method_used': 'combined_weighted',
            'max_score': float(max_score),
            'combined_score': combined_score.tolist()
        }
        
        return apex_idx, info
    
    def extract_apex_flow(self, frames: List[np.ndarray], optical_flows: List[np.ndarray]) -> Tuple[np.ndarray, int, Dict]:
        """
        Extract the optical flow at the apex frame.
        
        Args:
            frames: List of video frames
            optical_flows: List of optical flow arrays
            
        Returns:
            Tuple of (apex_flow, apex_frame_index, detection_info)
        """
        apex_idx, detection_info = self.detect_apex_frame(optical_flows)
        
        # Get the flow at apex (or preceding flow if apex is last frame)
        if apex_idx < len(optical_flows):
            apex_flow = optical_flows[apex_idx]
        else:
            apex_flow = optical_flows[-1]
            apex_idx = len(optical_flows) - 1
        
        return apex_flow, apex_idx, detection_info


# Utility functions for integration
def detect_apex_frame_simple(optical_flows: List[np.ndarray]) -> int:
    """
    Simple apex detection for quick integration.
    
    Args:
        optical_flows: List of optical flow arrays
        
    Returns:
        Index of apex frame
    """
    if not optical_flows:
        return 0
    
    motion_magnitudes = [np.mean(np.abs(flow)) for flow in optical_flows]
    return int(np.argmax(motion_magnitudes))


def create_apex_detector(fps: float = 30.0) -> ApexFrameDetector:
    """
    Factory function to create apex detector.
    
    Args:
        fps: Video frame rate
        
    Returns:
        Configured ApexFrameDetector instance
    """
    return ApexFrameDetector(fps=fps)


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing Apex Frame Detection...")
    
    # Create dummy optical flows
    num_flows = 10
    dummy_flows = []
    for i in range(num_flows):
        # Create flow with increasing then decreasing motion (simulating apex)
        flow = np.random.rand(64, 64, 2) * 0.1
        if 3 <= i <= 5:  # Apex region
            flow *= 3.0  # Higher motion
        dummy_flows.append(flow)
    
    # Test detector
    detector = ApexFrameDetector(fps=30.0)
    apex_idx, info = detector.detect_apex_frame(dummy_flows, method='adaptive')
    
    print(f"✅ Apex frame detected at index: {apex_idx}")
    print(f"✅ Detection confidence: {info['confidence']:.3f}")
    print(f"✅ Method used: {info['method_used']}")
    
    # Test apex extraction
    apex_flow, apex_idx, info = detector.extract_apex_flow([], dummy_flows)
    print(f"✅ Apex flow shape: {apex_flow.shape}")
    
    print("🎉 Apex Frame Detection implementation complete!")
