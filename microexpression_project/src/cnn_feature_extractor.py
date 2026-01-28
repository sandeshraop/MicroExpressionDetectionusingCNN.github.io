import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from config import NUM_EMOTIONS


class FlowCNN(nn.Module):
    """
    CNN model for optical flow-based micro-expression recognition.
    
    Input: 4-channel flow tensor [flow_x1, flow_y1, flow_x2, flow_y2] of shape (4, 64, 64)
    Architecture: Reduced capacity for small datasets to prevent memorization
    """
    
    def __init__(self, num_classes: int = NUM_EMOTIONS):
        """
        Initialize the Flow CNN model.
        
        Args:
            num_classes: Number of emotion classes (default: 4)
        """
        super(FlowCNN, self).__init__()
        
        # Reduced convolutional layers for flow processing
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # (6, 64, 64) -> (16, 64, 64)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (16, 64, 64) -> (32, 64, 64)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (32, 64, 64) -> (64, 64, 64)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Halves spatial dimensions
        
        # Fully connected layers (reduced capacity)
        # After 3 pooling operations: 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4, 64, 64)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch_size, 16, 32, 32)
        
        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch_size, 32, 16, 16)
        
        # Third conv block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch_size, 64, 8, 8)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size, 64 * 8 * 8)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)  # No activation here, CrossEntropyLoss handles it
        
        return x
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (returns class indices).
        
        Args:
            x: Input tensor
        
        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make probability predictions.
        
        Args:
            x: Input tensor
        
        Returns:
            Class probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        
        return probabilities
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'FlowCNN',
            'input_shape': '(batch_size, 6, 64, 64)',
            'input_description': '[flow_x1, flow_y1, flow_x2, flow_y2, strain1, strain2] from onset→apex, apex→offset',
            'num_classes': NUM_EMOTIONS,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Conv(6->16)->ReLU->Pool->Conv(16->32)->ReLU->Pool->Conv(32->64)->ReLU->Pool->FC(4096->128)->ReLU->FC(128->64)->ReLU->FC(64->4) (Reduced Capacity, Deterministic, No Dropout)'
        }


class HybridFlowCNN(nn.Module):
    """
    Hybrid CNN that processes both original frames and optical flow.
    
    Combines spatial information from frames with motion information from flow
    for improved micro-expression recognition.
    """
    
    def __init__(self, num_classes: int = NUM_EMOTIONS):
        """
        Initialize the Hybrid Flow CNN model.
        
        Args:
            num_classes: Number of emotion classes (default: 4)
        """
        super(HybridFlowCNN, self).__init__()
        
        # Frame encoder (spatial information)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )
        
        # Flow encoder (motion information)
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(128 * 8 * 8 * 2, 256),  # Combined features
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, frames: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.
        
        Args:
            frames: Frame tensor of shape (batch_size, 3, 64, 64)
            flows: Flow tensor of shape (batch_size, 4, 64, 64)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Handle single sample case
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        if flows.dim() == 3:
            flows = flows.unsqueeze(0)
        
        batch_size = frames.size(0)
        
        # Encode frames
        frame_features = self.frame_encoder(frames)  # (batch_size, 128, 8, 8)
        frame_features = frame_features.view(batch_size, -1)  # (batch_size, 128*8*8)
        
        # Encode flows
        flow_features = self.flow_encoder(flows)  # (batch_size, 128, 8, 8)
        flow_features = flow_features.view(batch_size, -1)  # (batch_size, 128*8*8)
        
        # Concatenate features
        combined_features = torch.cat([frame_features, flow_features], dim=1)  # (batch_size, 128*8*8*2)
        
        # Fusion and classification
        output = self.fusion(combined_features)
        
        return output
    
    def predict(self, frames: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (returns class indices).
        
        Args:
            frames: Frame tensor
            flows: Flow tensor
        
        Returns:
            Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(frames, flows)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'HybridFlowCNN',
            'input_shape': '(batch_size, 3, 64, 64) + (batch_size, 6, 64, 64)',
            'input_description': 'Frames + Optical Flow with Strain',
            'num_classes': NUM_EMOTIONS,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'FrameEncoder(Conv(3->32)->ReLU->Pool->Conv(32->64)->ReLU->Pool->Conv(64->128)->ReLU->Pool) + FlowEncoder(Conv(6->32)->ReLU->Pool->Conv(32->64)->ReLU->Pool->Conv(64->128)->ReLU->Pool) + Fusion(FC(8192*2->512)->ReLU->FC(512->256)->ReLU->FC(256->4)) (Deterministic, No Dropout)'
        }


def create_flow_model(model_type: str = 'flow', num_classes: int = NUM_EMOTIONS) -> nn.Module:
    """
    Create a flow-based CNN model.
    
    Args:
        model_type: 'flow' for FlowCNN, 'hybrid' for HybridFlowCNN
        num_classes: Number of emotion classes
    
    Returns:
        Flow-based CNN model instance
    """
    if model_type == 'flow':
        return FlowCNN(num_classes)
    elif model_type == 'hybrid':
        return HybridFlowCNN(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test flow-based CNN models
    print("Testing Flow-based CNN models...")
    
    # Test FlowCNN
    print("\n=== FlowCNN ===")
    model1 = FlowCNN()
    info1 = model1.get_model_info()
    print(f"Model info: {info1}")
    
    # Test forward pass
    x = torch.randn(2, 6, 64, 64)  # Updated to 6 channels
    output1 = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output1.shape}")
    
    # Test HybridFlowCNN
    print("\n=== HybridFlowCNN ===")
    model2 = HybridFlowCNN()
    info2 = model2.get_model_info()
    print(f"Model info: {info2}")
    
    # Test forward pass
    frames = torch.randn(2, 3, 64, 64)
    flows = torch.randn(2, 6, 64, 64)  # Updated to 6 channels
    output2 = model2(frames, flows)
    print(f"Frames shape: {frames.shape}")
    print(f"Flows shape: {flows.shape}")
    print(f"Output shape: {output2.shape}")
    
    # Test predictions
    pred1 = model1.predict(x)
    pred2 = model2.predict(frames, flows)
    print(f"FlowCNN predictions: {pred1}")
    print(f"HybridFlowCNN predictions: {pred2}")
    
    print("\n✅ Flow-based CNN models working correctly!")
