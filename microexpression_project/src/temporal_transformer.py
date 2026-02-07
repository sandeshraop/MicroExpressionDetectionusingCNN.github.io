#!/usr/bin/env python3
"""
Temporal Transformer for Micro-Expression Sequence Modeling

Key Innovation: Multi-head attention for temporal dynamics
- Captures long-range dependencies in micro-expression sequences
- Models onset → apex → offset progression
- 8-head attention for comprehensive temporal analysis

Architecture:
- Multi-head attention (embed_dim=768, num_heads=8)
- Layer normalization + Residual connections
- Feed-forward network + GELU activation
- Mean pooling over temporal dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences.
    
    Adds positional information to temporal embeddings
    to maintain sequence order information.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 1000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (seq_len, batch_size, embed_dim)
            
        Returns:
            Input with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head attention specifically designed for temporal sequences.
    
    Captures temporal dependencies in micro-expression sequences
    with specialized attention mechanisms for time series.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 attention_type: str = 'standard'):
        """
        Initialize multi-head temporal attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            attention_type: Type of attention ('standard', 'causal', 'bidirectional')
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_type = attention_type
        
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, embed_dim)
            key: Key tensor (batch_size, seq_len, embed_dim)
            value: Value tensor (batch_size, seq_len, embed_dim)
            mask: Attention mask (batch_size, seq_len, seq_len)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections
        Q = self.q_proj(query)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(key)    # (batch_size, seq_len, embed_dim)
        V = self.v_proj(value)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask if needed
        if self.attention_type == 'causal':
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            if scores.is_cuda:
                causal_mask = causal_mask.cuda()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply provided mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # Softmax attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        output = self.out_proj(output)
        output = self.output_dropout(output)
        
        return output, attention_weights.mean(dim=1)  # Average attention over heads


class TemporalTransformerBlock(nn.Module):
    """
    Single transformer block for temporal processing.
    
    Combines multi-head attention with feed-forward network
    and residual connections.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, ff_dim: int = 2048,
                 dropout: float = 0.1, attention_type: str = 'bidirectional'):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            attention_type: Type of attention mechanism
        """
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadTemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Attention mask
            
        Returns:
            Output tensor (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for micro-expression sequence modeling.
    
    Processes temporal sequences of facial features to capture
    onset → apex → offset dynamics in micro-expressions.
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 4,
                 ff_dim: int = 2048, dropout: float = 0.1, max_seq_len: int = 100,
                 attention_type: str = 'bidirectional'):
        """
        Initialize temporal transformer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            attention_type: Type of attention ('bidirectional', 'causal')
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_type = attention_type
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TemporalTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                attention_type=attention_type
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # Layer normalization for output
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize output projection weights."""
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through temporal transformer.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Attention mask (batch_size, seq_len, seq_len)
            
        Returns:
            Temporal context vector (batch_size, embed_dim)
        """
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # Output projection and normalization
        x = self.output_projection(x)
        x = self.output_norm(x)
        
        # Mean pooling over temporal dimension
        temporal_context = x.mean(dim=1)
        
        return temporal_context
    
    def extract_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for interpretability.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            
        Returns:
            Attention weights from first layer (batch_size, num_heads, seq_len, seq_len)
        """
        # Add positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Get attention from first layer
        first_layer = self.transformer_layers[0]
        _, attention_weights = first_layer.attention(x, x, x)
        
        return attention_weights


class AdaptiveTemporalTransformer(nn.Module):
    """
    Adaptive Temporal Transformer with sequence-specific processing.
    
    Adapts to different sequence lengths and temporal patterns
    in micro-expression data.
    """
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, num_layers: int = 4,
                 ff_dim: int = 2048, dropout: float = 0.1):
        """
        Initialize adaptive temporal transformer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Base transformer
        self.transformer = TemporalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            attention_type='bidirectional'
        )
        
        # Sequence length embedding
        self.seq_length_embedding = nn.Embedding(100, embed_dim)  # Max seq length 100
        
        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with sequence length adaptation.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            seq_lengths: Actual sequence lengths (batch_size,)
            
        Returns:
            Temporal context vector (batch_size, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Add sequence length embedding if provided
        if seq_lengths is not None:
            seq_len_emb = self.seq_length_embedding(seq_lengths)
            x = x + seq_len_emb.unsqueeze(1)
        
        # Process through transformer
        temporal_context = self.transformer(x)
        
        return temporal_context


# Utility functions
def create_temporal_transformer(embed_dim: int = 768, num_heads: int = 8, 
                               num_layers: int = 4) -> TemporalTransformer:
    """
    Factory function to create temporal transformer.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        
    Returns:
        Configured TemporalTransformer
    """
    return TemporalTransformer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=2048,
        dropout=0.1,
        attention_type='bidirectional'
    )


def create_adaptive_temporal_transformer(embed_dim: int = 768, num_heads: int = 8,
                                       num_layers: int = 4) -> AdaptiveTemporalTransformer:
    """
    Factory function to create adaptive temporal transformer.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        
    Returns:
        Configured AdaptiveTemporalTransformer
    """
    return AdaptiveTemporalTransformer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=2048,
        dropout=0.1
    )


if __name__ == "__main__":
    # Test the implementation
    print("🧪 Testing Temporal Transformer...")
    
    # Create dummy temporal features
    batch_size = 4
    seq_len = 4  # 4 temporal windows (pre-onset, onset, apex, post-apex)
    embed_dim = 768
    
    temporal_features = torch.randn(batch_size, seq_len, embed_dim)
    print(f"✅ Temporal features shape: {temporal_features.shape}")
    
    # Test basic transformer
    transformer = create_temporal_transformer(embed_dim=embed_dim, num_heads=8, num_layers=4)
    output = transformer(temporal_features)
    print(f"✅ Transformer output shape: {output.shape}")
    
    # Test adaptive transformer
    adaptive_transformer = create_adaptive_temporal_transformer(embed_dim=embed_dim, num_heads=8)
    seq_lengths = torch.tensor([4, 3, 4, 2])  # Variable sequence lengths
    adaptive_output = adaptive_transformer(temporal_features, seq_lengths)
    print(f"✅ Adaptive transformer output shape: {adaptive_output.shape}")
    
    # Test attention extraction
    attention_weights = transformer.extract_attention_weights(temporal_features)
    print(f"✅ Attention weights shape: {attention_weights.shape}")
    
    print("🎉 Temporal Transformer implementation complete!")
