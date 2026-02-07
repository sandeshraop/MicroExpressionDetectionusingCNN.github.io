#!/usr/bin/env python3
"""
Graph Convolutional Network for Facial ROI Interactions

Key Innovation: Models interactions between facial regions using graph structure
- ROI nodes: Eyes, Eyebrows, Mouth (3 main regions)
- Edge connections: Natural facial muscle relationships
- Attention mechanism: Weights important interactions

Architecture:
- Graph Attention Layers (multi-head attention)
- ELU activation + Dropout
- Mean pooling over ROIs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for modeling ROI interactions.
    
    Implements multi-head attention over graph nodes (facial ROIs).
    """
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, 
                 alpha: float = 0.2, dropout: float = 0.3):
        """
        Initialize Graph Attention Layer.
        
        Args:
            in_features: Number of input features per ROI
            out_features: Number of output features per ROI
            num_heads: Number of attention heads
            alpha: LeakyReLU negative slope
            dropout: Dropout rate
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout
        
        # Multi-head attention parameters
        self.W = nn.Parameter(torch.empty(size=(num_heads, in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * out_features)))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # LeakyReLU activation
        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Graph Attention Layer.
        
        Args:
            h: Node features (batch_size, num_rois, in_features)
            adj: Adjacency matrix (num_rois, num_rois)
            
        Returns:
            Updated node features (batch_size, num_rois, out_features)
        """
        batch_size, num_rois, _ = h.size()
        
        # Linear transformation for each head
        Wh = torch.einsum('bij,hjk->bhik', h, self.W)  # (batch, heads, rois, out_features)
        
        # Compute attention coefficients
        Wh1 = torch.matmul(Wh, self.a[:, :self.out_features, :])  # (batch, heads, rois, 1)
        Wh2 = torch.matmul(Wh, self.a[:, self.out_features:, :])  # (batch, heads, rois, 1)
        
        # Broadcast and sum for attention coefficients
        e = Wh1 + Wh2.transpose(-2, -1)  # (batch, heads, rois, rois)
        e = self.leakyrelu(e)
        
        # Apply adjacency mask
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # (batch, heads, rois, out_features)
        
        # Aggregate heads (mean pooling)
        h_prime = h_prime.mean(dim=1)  # (batch, rois, out_features)
        
        return h_prime


class FacialAdjacencyMatrix:
    """
    Creates adjacency matrices for facial ROI interactions.
    
    Defines natural relationships between facial regions:
    - Eyebrows ↔ Eyes (coordinated movement)
    - Eyes ↔ Mouth (expression coordination)
    - Eyebrows ↔ Mouth (upper-lower face coordination)
    """
    
    def __init__(self, num_rois: int = 3):
        """
        Initialize facial adjacency matrix creator.
        
        Args:
            num_rois: Number of facial ROIs
        """
        self.num_rois = num_rois
        self.roi_names = ['eyebrows', 'eyes', 'mouth'] if num_rois == 3 else ['roi_{}'.format(i) for i in range(num_rois)]
    
    def create_facial_adjacency(self, connectivity: str = 'full') -> torch.Tensor:
        """
        Create adjacency matrix for facial ROIs.
        
        Args:
            connectivity: Type of connectivity ('full', 'facial', 'sparse')
            
        Returns:
            Adjacency matrix (num_rois, num_rois)
        """
        adj = torch.zeros(self.num_rois, self.num_rois)
        
        if connectivity == 'full':
            # All ROIs connected to each other
            adj.fill_(1.0)
            torch.fill_diagonal(adj, 1.0)  # Self-connections
            
        elif connectivity == 'facial':
            # Natural facial connectivity
            if self.num_rois == 3:
                # Eyebrows ↔ Eyes ↔ Mouth (natural chain)
                adj[0, 1] = adj[1, 0] = 1.0  # Eyebrows ↔ Eyes
                adj[1, 2] = adj[2, 1] = 1.0  # Eyes ↔ Mouth
                adj[0, 2] = adj[2, 0] = 0.5  # Eyebrows ↔ Mouth (weaker)
                torch.fill_diagonal(adj, 1.0)  # Self-connections
                
        elif connectivity == 'sparse':
            # Minimal connectivity
            adj[0, 1] = adj[1, 0] = 1.0  # Only adjacent connections
            adj[1, 2] = adj[2, 1] = 1.0
            torch.fill_diagonal(adj, 1.0)
        
        return adj
    
    def create_learnable_adjacency(self) -> nn.Parameter:
        """
        Create learnable adjacency matrix.
        
        Returns:
            Learnable adjacency matrix parameter
        """
        adj_init = self.create_facial_adjacency('facial')
        return nn.Parameter(adj_init.clone())


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph Convolutional Network for facial ROI interactions.
    
    Models how different facial regions interact during micro-expressions.
    """
    
    def __init__(self, num_rois: int = 3, input_dim: int = 256, hidden_dim: int = 256, 
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.3,
                 connectivity: str = 'facial'):
        """
        Initialize GCN for facial ROI interactions.
        
        Args:
            num_rois: Number of facial ROIs
            input_dim: Input feature dimension per ROI
            hidden_dim: Hidden feature dimension
            num_layers: Number of GCN layers
            num_heads: Number of attention heads per layer
            dropout: Dropout rate
            connectivity: Type of ROI connectivity
        """
        super().__init__()
        self.num_rois = num_rois
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Create adjacency matrix
        adjacency_creator = FacialAdjacencyMatrix(num_rois)
        self.register_buffer('adjacency', adjacency_creator.create_facial_adjacency(connectivity))
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.gcn_layers.append(
            GraphAttentionLayer(input_dim, hidden_dim, num_heads, dropout=dropout)
        )
        
        # Additional layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.gcn_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout=dropout)
            )
        
        # Activation and dropout
        self.activation = nn.ELU()
        self.dropout_layer = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN.
        
        Args:
            roi_features: ROI features (batch_size, num_rois, input_dim)
            
        Returns:
            ROI interaction-aware features (batch_size, num_rois, hidden_dim)
        """
        h = roi_features
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            # Graph convolution
            h = gcn_layer(h, self.adjacency)
            
            # Residual connection (except for first layer if dimensions don't match)
            if i > 0 or h.shape[-1] == roi_features.shape[-1]:
                h = h + roi_features if i == 0 else h + h_prev
            
            # Activation and normalization
            h = self.activation(h)
            h = self.layer_norms[i](h)
            h = self.dropout_layer(h)
            
            h_prev = h
        
        # Output projection
        h = self.output_projection(h)
        
        return h
    
    def aggregate_rois(self, roi_features: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """
        Aggregate ROI features into a single vector.
        
        Args:
            roi_features: ROI features (batch_size, num_rois, hidden_dim)
            method: Aggregation method ('mean', 'max', 'attention')
            
        Returns:
            Aggregated features (batch_size, hidden_dim)
        """
        if method == 'mean':
            return roi_features.mean(dim=1)
        elif method == 'max':
            return roi_features.max(dim=1)[0]
        elif method == 'attention':
            # Learnable attention pooling
            attention_weights = torch.softmax(
                torch.mean(roi_features, dim=-1), dim=-1
            ).unsqueeze(-1)
            return (roi_features * attention_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


class MultiScaleGCN(nn.Module):
    """
    Multi-scale Graph Convolutional Network.
    
    Processes ROI features at multiple scales for comprehensive analysis.
    """
    
    def __init__(self, num_rois: int = 3, input_dim: int = 256, hidden_dim: int = 256,
                 scales: List[int] = [1, 2, 4]):
        """
        Initialize multi-scale GCN.
        
        Args:
            num_rois: Number of facial ROIs
            input_dim: Input feature dimension
            hidden_dim: Hidden feature dimension
            scales: List of scales for multi-scale processing
        """
        super().__init__()
        self.num_rois = num_rois
        self.scales = scales
        
        # GCN for each scale
        self.gcn_modules = nn.ModuleDict()
        for scale in scales:
            self.gcn_modules[str(scale)] = GraphConvolutionalNetwork(
                num_rois=num_rois,
                input_dim=input_dim,
                hidden_dim=hidden_dim // len(scales),
                num_layers=2,
                num_heads=4
            )
        
        # Feature fusion
        self.fusion_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale GCN.
        
        Args:
            roi_features: ROI features (batch_size, num_rois, input_dim)
            
        Returns:
            Multi-scale aggregated features (batch_size, hidden_dim)
        """
        scale_features = []
        
        for scale in self.scales:
            # Process at current scale
            scale_gcn = self.gcn_modules[str(scale)]
            scale_feat = scale_gcn(roi_features)
            
            # Aggregate ROIs
            scale_feat = scale_gcn.aggregate_rois(scale_feat, method='mean')
            scale_features.append(scale_feat)
        
        # Concatenate multi-scale features
        multi_scale_feat = torch.cat(scale_features, dim=-1)
        
        # Fusion layer
        fused_features = self.fusion_layer(multi_scale_feat)
        
        return fused_features


# Utility functions
def create_facial_gcn(num_rois: int = 3, input_dim: int = 256, hidden_dim: int = 256) -> GraphConvolutionalNetwork:
    """
    Factory function to create facial GCN.
    
    Args:
        num_rois: Number of facial ROIs
        input_dim: Input feature dimension
        hidden_dim: Hidden feature dimension
        
    Returns:
        Configured GraphConvolutionalNetwork
    """
    return GraphConvolutionalNetwork(
        num_rois=num_rois,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
        connectivity='facial'
    )


def create_multi_scale_gcn(num_rois: int = 3, input_dim: int = 256, hidden_dim: int = 256) -> MultiScaleGCN:
    """
    Factory function to create multi-scale GCN.
    
    Args:
        num_rois: Number of facial ROIs
        input_dim: Input feature dimension
        hidden_dim: Hidden feature dimension
        
    Returns:
        Configured MultiScaleGCN
    """
    return MultiScaleGCN(
        num_rois=num_rois,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        scales=[1, 2, 4]
    )


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing Graph Convolutional Network...")
    
    # Create dummy ROI features
    batch_size = 4
    num_rois = 3
    input_dim = 256
    
    roi_features = torch.randn(batch_size, num_rois, input_dim)
    print(f"✅ ROI features shape: {roi_features.shape}")
    
    # Test basic GCN
    gcn = create_facial_gcn(num_rois=num_rois, input_dim=input_dim, hidden_dim=256)
    gcn_output = gcn(roi_features)
    print(f"✅ GCN output shape: {gcn_output.shape}")
    
    # Test ROI aggregation
    aggregated = gcn.aggregate_rois(gcn_output, method='mean')
    print(f"✅ Aggregated features shape: {aggregated.shape}")
    
    # Test multi-scale GCN
    multi_gcn = create_multi_scale_gcn(num_rois=num_rois, input_dim=input_dim, hidden_dim=256)
    multi_output = multi_gcn(roi_features)
    print(f"✅ Multi-scale GCN output shape: {multi_output.shape}")
    
    print("🎉 Graph Convolutional Network implementation complete!")
