#!/usr/bin/env python3
"""
FaceSleuth Optical Flow with Vertical Motion Bias

Key Innovation: Micro-expressions show stronger vertical movements
- Eyebrow raises/furrows (surprise, fear, sadness)
- Mouth corner raises (happiness) or pulls down (disgust)
- Upper lip raises (disgust)

Implementation: Amplify vertical component by α=1.5
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch


class FaceSleuthOpticalFlow:
    """
    Enhanced optical flow computation with vertical motion bias
    for micro-expression recognition.
    """
    
    def __init__(self, vertical_emphasis_alpha: float = 1.5):
        """
        Initialize FaceSleuth optical flow processor.
        
        Args:
            vertical_emphasis_alpha: Amplification factor for vertical motion (default: 1.5)
        """
        self.alpha = vertical_emphasis_alpha
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        }
    
    def compute_vertical_biased_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow with vertical motion bias.
        
        Args:
            frame1: First frame (onset)
            frame2: Second frame (offset/apex)
            
        Returns:
            Vertical-biased optical flow (height, width, 2)
        """
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2
        
        # Compute standard optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.farneback_params)
        
        # Apply vertical bias (FaceSleuth innovation)
        flow[..., 1] *= self.alpha  # Amplify vertical component
        
        return flow
    
    def compute_flow_magnitude_angle(self, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude and angle from flow with vertical bias.
        
        Args:
            flow: Optical flow array (height, width, 2)
            
        Returns:
            Tuple of (magnitude, angle) arrays
        """
        flow_x = flow[..., 0]  # Horizontal component
        flow_y = flow[..., 1]  # Vertical component (already biased)
        
        # Compute magnitude and angle
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        angle = np.arctan2(flow_y, flow_x)
        
        return magnitude, angle
    
    def extract_vertical_dominance_features(self, flow: np.ndarray) -> dict:
        """
        Extract features that capture vertical motion dominance.
        
        Args:
            flow: Vertical-biased optical flow
            
        Returns:
            Dictionary of vertical dominance features
        """
        # Separate components
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        # Compute statistics
        vertical_mean = np.mean(np.abs(flow_y))
        horizontal_mean = np.mean(np.abs(flow_x))
        vertical_std = np.std(flow_y)
        horizontal_std = np.std(flow_x)
        
        # Vertical dominance ratio
        vertical_dominance = vertical_mean / (horizontal_mean + 1e-8)
        
        # Vertical motion energy
        vertical_energy = np.sum(flow_y**2)
        horizontal_energy = np.sum(flow_x**2)
        total_energy = vertical_energy + horizontal_energy
        vertical_ratio = vertical_energy / (total_energy + 1e-8)
        
        return {
            'vertical_dominance': vertical_dominance,
            'vertical_ratio': vertical_ratio,
            'vertical_mean': vertical_mean,
            'horizontal_mean': horizontal_mean,
            'vertical_std': vertical_std,
            'horizontal_std': horizontal_std,
            'total_energy': total_energy
        }
    
    def process_sequence_flows(self, frames: list) -> list:
        """
        Process a sequence of frames to compute vertical-biased flows.
        
        Args:
            frames: List of frames in temporal order
            
        Returns:
            List of vertical-biased optical flows
        """
        flows = []
        
        for i in range(len(frames) - 1):
            flow = self.compute_vertical_biased_flow(frames[i], frames[i + 1])
            flows.append(flow)
        
        return flows


def apply_vertical_bias_to_tensor(flow_tensor: torch.Tensor, alpha: float = 1.5) -> torch.Tensor:
    """
    Apply vertical bias to flow tensor (for PyTorch integration).
    
    Args:
        flow_tensor: Flow tensor (batch, channels, height, width) where channels=2
        alpha: Vertical emphasis factor
        
    Returns:
        Vertical-biased flow tensor
    """
    # Clone to avoid modifying original
    biased_flow = flow_tensor.clone()
    
    # Apply bias to vertical component (channel 1)
    biased_flow[:, 1, :, :] *= alpha
    
    return biased_flow


# Utility function for integration with existing pipeline
def enhance_optical_flow_with_facesleuth(flow: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Quick integration function to enhance existing flow with FaceSleuth bias.
    
    Args:
        flow: Existing optical flow (height, width, 2)
        alpha: Vertical emphasis factor
        
    Returns:
        Enhanced flow with vertical bias
    """
    enhanced_flow = flow.copy()
    enhanced_flow[..., 1] *= alpha
    return enhanced_flow


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing FaceSleuth Optical Flow...")
    
    # Create dummy frames
    frame1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Initialize processor
    processor = FaceSleuthOpticalFlow(vertical_emphasis_alpha=1.5)
    
    # Compute flow
    flow = processor.compute_vertical_biased_flow(frame1, frame2)
    print(f"✅ Flow shape: {flow.shape}")
    
    # Extract features
    features = processor.extract_vertical_dominance_features(flow)
    print(f"✅ Vertical dominance: {features['vertical_dominance']:.3f}")
    print(f"✅ Vertical ratio: {features['vertical_ratio']:.3f}")
    
    print("🎉 FaceSleuth Optical Flow implementation complete!")
