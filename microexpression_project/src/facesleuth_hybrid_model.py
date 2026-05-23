#!/usr/bin/env python3
"""
FaceSleuth Hybrid Model - Complete Implementation

Integrates all FaceSleuth innovations into a unified architecture:
- Vertical Bias Optical Flow (α=1.5)
- AU-aware Soft Boosting (λ=0.3)
- Apex Frame Detection
- Graph Convolutional Network for ROI interactions
- Temporal Transformer for sequence modeling

Architecture:
Input: (B, T, 3, 64, 64) frames + (B, T, 6, 64, 64) flows
↓
ROI Extraction (3 parallel encoders: eyebrows, eyes, mouth)
↓ CNN Feature Extraction (256-D per ROI)
↓ Temporal Modeling (8-head attention)
↓ Graph Convolution (ROI interactions)
↓ Feature Fusion (768-D combined)
↓ AU-aware Soft Boosting
↓ Classification (5 emotions; matches ``config.NUM_EMOTIONS``)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json

# Import FaceSleuth components
from facesleuth_optical_flow import FaceSleuthOpticalFlow
from config import NUM_EMOTIONS
from au_soft_boosting import AUSoftBoosting
from apex_frame_detection import ApexFrameDetector
from graph_convolutional_network import GraphConvolutionalNetwork, create_facial_gcn
from temporal_transformer import TemporalTransformer, create_temporal_transformer


@dataclass
class FaceSleuthConfig:
    """Configuration for FaceSleuth Hybrid Model."""
    
    # Optical Flow Configuration
    vertical_emphasis_alpha: float = 1.5
    
    # AU Soft Boosting Configuration
    lambda_boost: float = 0.3
    confidence_threshold: float = 0.3
    
    # Apex Detection Configuration
    fps: float = 30.0
    min_duration_ms: float = 200.0
    max_duration_ms: float = 500.0
    
    # CNN Architecture
    num_rois: int = 3
    roi_feature_dim: int = 256
    cnn_channels: List[int] = None
    
    # GCN Configuration
    gcn_hidden_dim: int = 256
    gcn_num_heads: int = 4
    gcn_num_layers: int = 2
    
    # Transformer Configuration
    transformer_embed_dim: int = 768
    transformer_num_heads: int = 8
    transformer_num_layers: int = 4
    
    # Classification (must match sklearn / CNN label space)
    num_classes: int = field(default_factory=lambda: int(NUM_EMOTIONS))
    dropout: float = 0.3
    
    def __post_init__(self):
        """Initialize default values."""
        if self.cnn_channels is None:
            self.cnn_channels = [3, 32, 64, 128, 256]


class ROICNNEncoder(nn.Module):
    """
    CNN encoder for individual facial ROIs.
    
    Processes each ROI (eyebrows, eyes, mouth) independently
    to extract spatial features.
    """
    
    def __init__(self, in_channels: int = 3, feature_dim: int = 256, 
                 channels: List[int] = None, dropout: float = 0.3):
        """
        Initialize ROI CNN encoder.
        
        Args:
            in_channels: Number of input channels (3 for RGB, 6 for flow)
            feature_dim: Output feature dimension
            channels: Channel progression
            dropout: Dropout rate
        """
        super().__init__()
        
        if channels is None:
            channels = [in_channels, 32, 64, 128, 256]
        
        self.channels = channels
        self.feature_dim = feature_dim
        
        # Build CNN layers
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout if i < len(channels) - 2 else 0)
            ])
        
        self.cnn = nn.Sequential(*layers)
        
        # Adaptive pooling and projection
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(channels[-1], feature_dim)
        self.final_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ROI encoder.
        
        Args:
            x: ROI input (batch_size, channels, height, width)
            
        Returns:
            ROI features (batch_size, feature_dim)
        """
        # CNN feature extraction
        features = self.cnn(x)
        
        # Global average pooling
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        
        # Projection to target dimension
        features = self.projection(features)
        features = self.final_dropout(features)
        
        return features


class FaceSleuthHybridModel(nn.Module):
    """
    Complete FaceSleuth Hybrid Model integrating all innovations.
    
    Combines vertical bias optical flow, AU-aware soft boosting,
    apex detection, GCN, and temporal transformer for state-of-the-art
    micro-expression recognition.
    """
    
    def __init__(self, config: Optional[FaceSleuthConfig] = None):
        """
        Initialize FaceSleuth Hybrid Model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Configuration
        self.config = config if config is not None else FaceSleuthConfig()
        
        # Initialize FaceSleuth components
        self.optical_flow_processor = FaceSleuthOpticalFlow(
            vertical_emphasis_alpha=self.config.vertical_emphasis_alpha
        )
        self.au_booster = AUSoftBoosting(
            lambda_boost=self.config.lambda_boost,
            confidence_threshold=self.config.confidence_threshold
        )
        self.apex_detector = ApexFrameDetector(
            fps=self.config.fps,
            min_duration_ms=self.config.min_duration_ms,
            max_duration_ms=self.config.max_duration_ms
        )
        
        frame_ch = [3] + list(self.config.cnn_channels[1:])
        flow_ch = [6] + list(self.config.cnn_channels[1:])

        self.roi_encoders = nn.ModuleList([
            ROICNNEncoder(
                in_channels=3,
                feature_dim=self.config.roi_feature_dim,
                channels=frame_ch,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_rois)
        ])

        self.flow_encoders = nn.ModuleList([
            ROICNNEncoder(
                in_channels=6,
                feature_dim=self.config.roi_feature_dim,
                channels=flow_ch,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.num_rois)
        ])

        pair_dim = self.config.roi_feature_dim * 2
        self.roi_temporal_projections = nn.ModuleList(
            [
                nn.Linear(pair_dim, self.config.transformer_embed_dim)
                for _ in range(self.config.num_rois)
            ]
        )
        
        # Temporal transformer
        self.temporal_transformer = create_temporal_transformer(
            embed_dim=self.config.transformer_embed_dim,
            num_heads=self.config.transformer_num_heads,
            num_layers=self.config.transformer_num_layers
        )
        
        # Graph Convolutional Network
        self.gcn = create_facial_gcn(
            num_rois=self.config.num_rois,
            input_dim=self.config.roi_feature_dim,
            hidden_dim=self.config.gcn_hidden_dim
        )
        
        # Feature fusion layers
        self.temporal_projection = nn.Linear(
            self.config.transformer_embed_dim, 
            self.config.roi_feature_dim * self.config.num_rois
        )
        self.gcn_projection = nn.Linear(
            self.config.gcn_hidden_dim, 
            self.config.roi_feature_dim * self.config.num_rois
        )
        
        # Final fusion and classification
        total_feature_dim = self.config.roi_feature_dim * self.config.num_rois * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_feature_dim, total_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(total_feature_dim // 2, self.config.num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def _roi_spatial_crop(self, x: torch.Tensor, roi_idx: int) -> torch.Tensor:
        """Vertical band crops: 0 upper (brows), 1 mid (eyes/nose), 2 lower (mouth)."""
        if x.dim() < 4:
            raise ValueError("expected (N, C, H, W)")
        _, _, H, _W = x.shape
        if roi_idx == 0:
            y0, y1 = 0, max(8, int(0.36 * H))
        elif roi_idx == 1:
            y0, y1 = int(0.22 * H), int(0.72 * H)
        else:
            y0, y1 = int(0.55 * H), H
        y1 = max(y1, y0 + 4)
        return x[..., y0:y1, :]

    def extract_roi_features(self, frames: torch.Tensor, flows: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract ROI features from frames and flows (one spatial band per ROI encoder).

        Args:
            frames: Frame tensor (batch_size, seq_len, 3, H, W)
            flows: Flow tensor (batch_size, seq_len, 6, H, W)

        Returns:
            Tuple of (frame_roi_feature_list, flow_roi_feature_list), length ``num_rois``.
        """
        batch_size, seq_len = frames.size(0), frames.size(1)
        bf, sf = flows.size(0), flows.size(1)

        frames_flat = frames.view(-1, *frames.shape[2:])
        flows_flat = flows.view(-1, *flows.shape[2:])

        frame_roi_features = []
        flow_roi_features = []

        for roi_idx in range(self.config.num_rois):
            crop_f = self._roi_spatial_crop(frames_flat, roi_idx)
            crop_fl = self._roi_spatial_crop(flows_flat, roi_idx)
            frame_roi_features.append(self.roi_encoders[roi_idx](crop_f))
            flow_roi_features.append(self.flow_encoders[roi_idx](crop_fl))

        frame_roi_features = [feat.view(batch_size, seq_len, -1) for feat in frame_roi_features]
        flow_roi_features = [feat.view(bf, sf, -1) for feat in flow_roi_features]

        return frame_roi_features, flow_roi_features
    
    def apply_vertical_bias_to_flows(self, flows: torch.Tensor) -> torch.Tensor:
        """
        Apply vertical bias to flow tensors (delegates to ``FaceSleuthOpticalFlow``).
        """
        return self.optical_flow_processor.apply_tensor_vertical_bias_6ch(flows)
    
    def detect_apex_frames(self, flows: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Detect apex frames from flow sequence.
        
        Args:
            flows: Flow tensor (batch_size, seq_len, 6, H, W)
            
        Returns:
            Tuple of (apex_flows, apex_indices)
        """
        batch_size, seq_len = flows.size(0), flows.size(1)
        apex_indices = []
        apex_flows = []
        
        for batch_idx in range(batch_size):
            # Convert flows to numpy for apex detection
            batch_flows = flows[batch_idx].cpu().numpy()
            
            # Convert to list of flow arrays
            flow_list = []
            for t in range(seq_len):
                # Convert 6-channel flow to 2-channel for apex detection
                # ApexFrameDetector expects (H, W, 2) with last-dim = (fx, fy)
                flow_2ch = np.stack([batch_flows[t][0], batch_flows[t][1]], axis=-1)
                flow_list.append(flow_2ch)
            
            # Detect apex
            apex_idx, detection_info = self.apex_detector.detect_apex_frame(flow_list)
            apex_indices.append(apex_idx)
            
            # Get apex flow
            apex_flow = flows[batch_idx, apex_idx]
            apex_flows.append(apex_flow)
        
        apex_flows = torch.stack(apex_flows)
        return apex_flows, apex_indices
    
    def forward(self, frames: torch.Tensor, flows: torch.Tensor, 
                au_activations: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FaceSleuth Hybrid Model.
        
        Args:
            frames: Frame tensor (batch_size, seq_len, 3, H, W)
            flows: Flow tensor (batch_size, seq_len, 6, H, W)
            au_activations: AU activation tensor (batch_size, 27)
            
        Returns:
            Dictionary with predictions and intermediate features
        """
        batch_size, seq_len = frames.size(0), frames.size(1)
        
        # Step 1: Apply vertical bias to flows
        biased_flows = self.apply_vertical_bias_to_flows(flows)
        
        # Step 2: Extract ROI features
        frame_roi_features, flow_roi_features = self.extract_roi_features(frames, biased_flows)
        
        # Step 3: Temporal modeling with transformer
        # Combine frame and flow features for each ROI
        temporal_features = []
        for i in range(self.config.num_rois):
            roi_temporal = torch.cat([frame_roi_features[i], flow_roi_features[i]], dim=-1)
            temporal_features.append(self.roi_temporal_projections[i](roi_temporal))
        
        # Transformer expects (batch, seq, embed_dim); average ROI projections
        combined_temporal = torch.stack(temporal_features, dim=0).mean(dim=0)
        temporal_context = self.temporal_transformer(combined_temporal)
        
        # Step 4: Graph Convolution — apex-aligned frames + flows (same T for reshape)
        apex_flows, apex_indices = self.detect_apex_frames(biased_flows)
        apex_frames = torch.stack(
            [frames[b, apex_indices[b]] for b in range(batch_size)], dim=0
        ).unsqueeze(1)
        _, apex_flow_roi = self.extract_roi_features(apex_frames, apex_flows.unsqueeze(1))

        roi_features_for_gcn = torch.stack(
            [x.squeeze(1) for x in apex_flow_roi], dim=1
        )
        gcn_features = self.gcn(roi_features_for_gcn)
        gcn_aggregated = self.gcn.aggregate_rois(gcn_features, method='mean')
        
        # Step 5: Feature fusion
        temporal_proj = self.temporal_projection(temporal_context)
        gcn_proj = self.gcn_projection(gcn_aggregated)
        
        fused_features = torch.cat([temporal_proj, gcn_proj], dim=-1)
        
        # Step 6: Classification
        logits = self.fusion_layer(fused_features)
        probabilities = F.softmax(logits, dim=-1)
        
        if au_activations is not None:
            au_np = au_activations.detach().cpu().numpy() if torch.is_tensor(au_activations) else au_activations
            boosted_probabilities = self.au_booster.apply_soft_boosting(probabilities, au_np)
        else:
            boosted_probabilities = probabilities
        
        # Compile results
        results = {
            'logits': logits,
            'probabilities': probabilities,
            'boosted_probabilities': boosted_probabilities,
            'predictions': torch.argmax(boosted_probabilities, dim=-1),
            'temporal_context': temporal_context,
            'gcn_features': gcn_features,
            'apex_indices': apex_indices,
            'detection_info': {
                'vertical_bias_applied': True,
                'alpha': self.config.vertical_emphasis_alpha,
                'apex_frames_detected': len(apex_indices),
                'au_boosting_applied': au_activations is not None
            }
        }
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model configuration and information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'FaceSleuthHybridModel',
            'config': self.config.__dict__,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'innovations': [
                'Vertical Bias Optical Flow (α=1.5)',
                'AU-aware Soft Boosting (λ=0.3)',
                'Adaptive Apex Frame Detection',
                'Graph Convolutional Network (ROI interactions)',
                'Temporal Transformer (8-head attention)'
            ],
            'architecture': {
                'roi_encoders': self.config.num_rois,
                'feature_dimension': self.config.roi_feature_dim,
                'transformer_heads': self.config.transformer_num_heads,
                'transformer_layers': self.config.transformer_num_layers,
                'gcn_layers': self.config.gcn_num_layers,
                'num_classes': self.config.num_classes
            }
        }


# Factory functions
def create_facesleuth_model(config: Optional[FaceSleuthConfig] = None) -> FaceSleuthHybridModel:
    """
    Factory function to create FaceSleuth Hybrid Model.
    
    Args:
        config: Model configuration
        
    Returns:
        Configured FaceSleuthHybridModel
    """
    return FaceSleuthHybridModel(config)


def create_default_facesleuth_model() -> FaceSleuthHybridModel:
    """
    Create FaceSleuth model with default configuration.
    
    Returns:
        FaceSleuthHybridModel with default settings
    """
    config = FaceSleuthConfig()
    return create_facesleuth_model(config)


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing FaceSleuth Hybrid Model...")
    
    # Create model
    model = create_default_facesleuth_model()
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data
    batch_size = 2
    seq_len = 4
    height, width = 64, 64
    
    frames = torch.randn(batch_size, seq_len, 3, height, width)
    flows = torch.randn(batch_size, seq_len, 6, height, width)
    au_activations = torch.randn(batch_size, 27)  # 27 Action Units
    
    print(f"✅ Input frames shape: {frames.shape}")
    print(f"✅ Input flows shape: {flows.shape}")
    print(f"✅ AU activations shape: {au_activations.shape}")
    
    # Forward pass
    with torch.no_grad():
        results = model(frames, flows, au_activations)
    
    print(f"✅ Output predictions shape: {results['predictions'].shape}")
    print(f"✅ Output probabilities shape: {results['probabilities'].shape}")
    print(f"✅ Boosted probabilities shape: {results['boosted_probabilities'].shape}")
    
    # Model info
    model_info = model.get_model_info()
    print(f"✅ Model innovations: {len(model_info['innovations'])}")
    for innovation in model_info['innovations']:
        print(f"   - {innovation}")
    
    print("🎉 FaceSleuth Hybrid Model implementation complete!")
