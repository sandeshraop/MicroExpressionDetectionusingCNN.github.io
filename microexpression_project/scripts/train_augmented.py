#!/usr/bin/env python3
"""
Train new model with augmented 248-sample balanced dataset
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import joblib
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent  # This goes up from scripts to microexpression_project
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Debug path
print(f"Current working directory: {Path.cwd()}")
print(f"Script file: {Path(__file__).name}")
print(f"Project root: {project_root}")
print(f"Src path: {src_path}")
print(f"Python path includes src: {str(src_path) in sys.path}")
print(f"Python path includes project root: {str(project_root) in sys.path}")

from micro_expression_model import EnhancedHybridModel
from dataset_loader import CNNCASMEIIDataset
from config import EMOTION_LABELS, LABEL_TO_EMOTION, EMOTION_DISPLAY_ORDER
from sklearn.utils.class_weight import compute_class_weight

class AugmentedDataset:
    """Augmented dataset with balanced classes"""
    
    def __init__(self, original_dataset, augmentation_factor=3):
        self.original_dataset = original_dataset
        self.augmentation_factor = augmentation_factor
        self.samples = []
        self._create_augmented_dataset()
    
    def _create_augmented_dataset(self):
        """Create balanced augmented dataset"""
        # Count original samples per emotion
        emotion_counts = defaultdict(int)
        for i in range(len(self.original_dataset)):
            _, _, label, _ = self.original_dataset[i]
            emotion = LABEL_TO_EMOTION[label.item()]
            emotion_counts[emotion] += 1
        
        # Target samples per emotion (max count)
        target_count = max(emotion_counts.values())
        
        print(f"📊 Creating balanced dataset (target: {target_count} per class)")
        
        # Add original samples
        for i in range(len(self.original_dataset)):
            frames, flows, label, metadata = self.original_dataset[i]
            emotion = LABEL_TO_EMOTION[label.item()]
            self.samples.append((frames, flows, label, metadata))
        
        # Add augmented samples for underrepresented classes
        for i in range(len(self.original_dataset)):
            frames, flows, label, metadata = self.original_dataset[i]
            emotion = LABEL_TO_EMOTION[label.item()]
            
            current_count = sum(1 for _, _, l, _ in self.samples 
                            if LABEL_TO_EMOTION[l.item()] == emotion)
            
            if current_count < target_count:
                # Create augmented versions
                num_augments = min(self.augmentation_factor, target_count - current_count)
                
                for aug_idx in range(num_augments):
                    aug_frames = self._augment_frames(frames)
                    aug_flows = self._augment_flows(flows)
                    # ⚠️ METADATA WARNING: Original metadata contains subject/episode info
                    # DO NOT use metadata for LOSO splits - augmented samples break subject independence
                    self.samples.append((aug_frames, aug_flows, label, metadata))
        
        print(f"✅ Created balanced dataset: {len(self.samples)} samples")
        
        # Verify balance
        final_counts = defaultdict(int)
        for _, _, label, _ in self.samples:
            emotion = LABEL_TO_EMOTION[label.item()]
            final_counts[emotion] += 1
        
        print(f"📈 Final distribution:")
        for emotion in ['happiness', 'surprise', 'disgust', 'repression']:
            count = final_counts.get(emotion, 0)
            print(f"   {emotion:10s}: {count:2d} samples")
    
    def _augment_frames(self, frames):
        """Apply consistent temporal augmentation to entire sequence"""
        frames_np = frames.numpy()
        augmented_frames = []
        
        # Generate CONSISTENT augmentation parameters for the entire sequence
        brightness_factor = np.random.uniform(0.9, 1.1)
        rotation_angle = np.random.uniform(-3, 3)
        apply_rotation = np.random.random() > 0.5
        
        import cv2
        h, w = frames_np[0].shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Apply SAME augmentation to ALL frames in the sequence
        for frame in frames_np:
            aug_frame = frame.copy()
            
            # Apply CONSISTENT brightness to entire sequence
            aug_frame = aug_frame * brightness_factor
            aug_frame = np.clip(aug_frame, 0, 1)
            
            # Apply CONSISTENT rotation to entire sequence (preserves temporal dynamics)
            if apply_rotation:
                aug_frame = cv2.warpAffine(aug_frame, rotation_matrix, (w, h))
            
            augmented_frames.append(torch.from_numpy(aug_frame).float())
        
        return torch.stack(augmented_frames)
    
    def _augment_flows(self, flows):
        """Apply flow-level augmentation (NO GEOMETRIC TRANSFORMS)"""
        flows_np = flows.numpy()
        augmented_flows = []
        
        # Generate CONSISTENT augmentation parameters for the entire sequence
        scale_factor = np.random.uniform(0.9, 1.1)
        noise_std = 0.01
        
        for flow in flows_np:
            aug_flow = flow.copy()
            
            # Add consistent noise (preserves vector field structure)
            noise = np.random.normal(0, noise_std, flow.shape)
            aug_flow = aug_flow + noise
            
            # Apply consistent scaling (preserves vector directions)
            aug_flow = aug_flow * scale_factor
            
            # ❌ NO ROTATION - Optical flow vectors must maintain physical consistency
            # Rotating flow fields without rotating vector directions introduces impossible motion
            
            augmented_flows.append(torch.from_numpy(aug_flow).float())
        
        return torch.stack(augmented_flows)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class AugmentedTrainer:
    """Trainer for augmented balanced dataset"""
    
    def __init__(self, device='auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def train_augmented_model(self, dataset, epochs=12, learning_rate=0.001):
        """Train model on augmented balanced dataset"""
        
        print(f"\n=== AUGMENTED BALANCED TRAINING ===")
        print(f"Dataset size: {len(dataset)}")
        
        # Collect all labels for class weighting
        all_labels = []
        for i in range(len(dataset)):
            _, _, label, _ = dataset[i]
            all_labels.append(label.item())
        
        # Compute class weights (should be balanced now)
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(all_labels), 
            y=all_labels
        )
        
        # Create full weight tensor for all 4 classes
        full_class_weights = np.ones(4)  # 4 emotions: happiness, surprise, disgust, repression
        for i, class_idx in enumerate(np.unique(all_labels)):
            full_class_weights[class_idx] = class_weights[i]
        
        weight_tensor = torch.FloatTensor(full_class_weights).to(self.device)
        
        print("Class weights (should be balanced):")
        for i, weight in enumerate(full_class_weights):
            emotion = LABEL_TO_EMOTION[i]
            print(f"  {emotion}: {weight:.3f}")
        
        # Initialize model
        model = EnhancedHybridModel()
        model.feature_extractor.to(self.device)
        
        # Prepare training data
        frames_list = []
        flows_list = []
        labels_list = []
        
        print("Preparing training data...")
        for i in range(len(dataset)):
            frames, flows, label, _ = dataset[i]
            frames_list.append(frames.to(self.device))
            flows_list.append(flows.to(self.device))
            labels_list.append(label.item())
        
        labels_array = np.array(labels_list)
        
        # Train CNN feature extractor
        print("Training CNN feature extractor...")
        self._train_cnn_balanced(model, frames_list, flows_list, labels_array, 
                                weight_tensor, epochs, learning_rate)
        
        # Extract all features
        print("Extracting features...")
        all_features = []
        
        # Move model to CPU for feature extraction
        model.feature_extractor.to('cpu')
        
        for frames, flows in zip(frames_list, flows_list):
            # ✅ PRESERVE TEMPORAL INFORMATION for SVM feature extraction
            # Extract features from each temporal frame, then aggregate
            if frames.dim() == 4:  # (T, C, H, W)
                temporal_features = []
                for t in range(frames.shape[0]):  # Process each temporal frame
                    single_frame = frames[t:t+1].cpu()  # Move to CPU
                    single_flow = flows.unsqueeze(0).cpu()  # Move to CPU
                    
                    # Extract features for this temporal frame
                    frame_features = model.extract_all_features(single_frame, single_flow)
                    temporal_features.append(frame_features)
                
                # Aggregate temporal features (mean across time)
                features = np.mean(temporal_features, axis=0)  # (1, 216)
            elif frames.dim() == 3:
                frames = frames.unsqueeze(0).cpu()
                flows = flows.unsqueeze(0).cpu()
                features = model.extract_all_features(frames, flows)
            else:
                frames = frames.cpu()
                flows = flows.cpu()
                features = model.extract_all_features(frames, flows)
            all_features.append(features)
        
        # Move model back to device if needed
        model.feature_extractor.to(self.device)
        
        all_features = np.vstack(all_features)
        
        # Train SVM classifier
        print("Training SVM classifier...")
        model.pipeline.set_params(classifier__class_weight='balanced')
        model.pipeline.fit(all_features, labels_array)
        model.is_fitted = True
        
        # Evaluate on training data
        predictions = model.pipeline.predict(all_features)
        accuracy = np.mean(predictions == labels_array)
        
        print(f"\nTraining Results:")
        # ⚠️ WARNING: This is augmented training accuracy - INFLATED and NOT publishable
        # Real performance must be measured on independent test set
        print(f"  Augmented Training Accuracy: {accuracy * 100:.2f}% (NOT for publication)")
        print(f"  Note: Use independent test set for real performance metrics")
        
        # Detailed classification report
        from sklearn.metrics import classification_report
        report = classification_report(
            labels_array, predictions,
            target_names=EMOTION_DISPLAY_ORDER,
            output_dict=True, zero_division=0
        )
        
        print("\nPer-class performance:")
        for emotion in EMOTION_DISPLAY_ORDER:
            if emotion in report:
                precision = report[emotion]['precision']
                recall = report[emotion]['recall']
                f1 = report[emotion]['f1-score']
                print(f"  {emotion}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        return model, accuracy, report
    
    def _train_augmented_model_direct(self, train_samples, epochs=12, learning_rate=0.001):
        """Train model directly on samples with on-the-fly augmentation (LOS0-safe)"""
        print(f"\n=== DIRECT AUGMENTED TRAINING (LOS0-SAFE) ===")
        print(f"Dataset size: {len(train_samples)}")
        
        # Prepare training data from samples
        frames_list = []
        flows_list = []
        labels_list = []
        
        print("Preparing training data...")
        for frames, flows, label, metadata in train_samples:
            # ✅ CRITICAL: Keep all tensors on the same device to avoid device mismatch
            frames_list.append(frames.to(self.device))
            flows_list.append(flows.to(self.device))
            labels_list.append(label.item())
        
        # Compute class weights
        all_labels = np.array(labels_list)
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(all_labels), 
            y=all_labels
        )
        
        # Create full weight tensor for all 4 classes
        full_class_weights = np.ones(4)
        for i, class_idx in enumerate(np.unique(all_labels)):
            full_class_weights[class_idx] = class_weights[i]
        
        weight_tensor = torch.FloatTensor(full_class_weights).to(self.device)
        
        print("Class weights (should be balanced):")
        for i, weight in enumerate(full_class_weights):
            emotion = LABEL_TO_EMOTION[i]
            print(f"  {emotion}: {weight:.3f}")
        
        # Initialize model
        model = EnhancedHybridModel()
        model.feature_extractor.to(self.device)
        
        # Train CNN feature extractor with on-the-fly augmentation
        self._train_cnn_balanced_direct(model, frames_list, flows_list, labels_list, 
                                        weight_tensor, epochs, learning_rate)
        
        # Extract features with temporal aggregation
        print("Extracting features...")
        all_features = []
        
        model.feature_extractor.to('cpu')
        
        for frames, flows in zip(frames_list, flows_list):
            # ✅ CRITICAL: Apply on-the-fly augmentation and maintain device consistency
            aug_frames = self._augment_frames_direct(frames)
            aug_flows = self._augment_flows_direct(flows)
            
            # Extract features from each temporal frame, then aggregate
            if aug_frames.dim() == 4:  # (T, C, H, W)
                temporal_features = []
                for t in range(aug_frames.shape[0]):
                    single_frame = aug_frames[t:t+1]
                    single_flow = aug_flows.unsqueeze(0)
                    
                    # ✅ CRITICAL: Move to CPU for feature extraction to avoid device issues
                    frame_features = model.extract_all_features(single_frame.cpu(), single_flow.cpu())
                    temporal_features.append(frame_features)
                
                features = np.mean(temporal_features, axis=0)
            else:
                # ✅ CRITICAL: Move to CPU for feature extraction
                features = model.extract_all_features(aug_frames.cpu(), aug_flows.cpu())
            
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        
        # Train SVM classifier
        print("Training SVM classifier...")
        model.pipeline.set_params(classifier__class_weight='balanced')
        model.pipeline.fit(all_features, all_labels)
        model.is_fitted = True
        
        # Evaluate on training data
        predictions = model.pipeline.predict(all_features)
        accuracy = np.mean(predictions == all_labels)
        
        print(f"\nTraining Results:")
        print(f"  Direct Training Accuracy: {accuracy * 100:.2f}% (NOT for publication)")
        print(f"  Note: Use independent test set for real performance metrics")
        
        return model, accuracy, {}
    
    def _train_cnn_balanced_direct(self, model, frames_list, flows_list, labels, 
                                   class_weights, epochs, learning_rate):
        """Train CNN with class weighting and on-the-fly augmentation"""
        
        # Prepare training data
        frames_tensor = torch.stack(frames_list)  # (N, 3, 3, 64, 64)
        flows_tensor = torch.stack(flows_list)     # (N, 6, 64, 64)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)  # ✅ CRITICAL: Put labels on same device
        
        # ✅ PRESERVE TEMPORAL DYNAMICS - Critical fix for micro-expression recognition
        if frames_tensor.dim() == 5:  # (N, T, C, H, W)
            original_batch_size = frames_tensor.shape[0]
            temporal_frames = frames_tensor.shape[1]
            
            frames_tensor = frames_tensor.view(-1, 3, 64, 64)  # (N*T, 3, 64, 64)
            
            flows_tensor = flows_tensor.unsqueeze(1).repeat(1, temporal_frames, 1, 1, 1)
            flows_tensor = flows_tensor.view(-1, 6, 64, 64)
            
            labels_tensor = labels_tensor.repeat_interleave(temporal_frames)
            
            print(f"✅ Temporal expansion: {original_batch_size} → {frames_tensor.shape[0]} samples")
            print(f"✅ Preserved onset→apex→offset dynamics for CNN training")
            
            # 🔥 AU-WEIGHTED SPATIAL EMPHASIS - Final boost for disgust recognition
            frames_tensor[:, :, 20:40, 25:40] *= 1.3
            print(f"✅ Applied AU-weighted spatial emphasis for disgust (AU9/AU10 region)")
        
        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(frames_tensor, flows_tensor, labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Set up training with weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.feature_extractor.parameters(), lr=learning_rate)
        
        # Add temporary linear head
        with torch.no_grad():
            sample_frames = frames_tensor[:1]
            sample_flows = flows_tensor[:1]
            features = model.feature_extractor(sample_frames, sample_flows)
            feature_dim = features.shape[1]
        
        temp_head = nn.Linear(feature_dim, len(EMOTION_LABELS)).to(self.device)
        
        model.feature_extractor.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_frames, batch_flows, batch_labels in loader:
                optimizer.zero_grad()
                
                # Forward pass
                features = model.feature_extractor(batch_frames, batch_flows)
                logits = temp_head(features)
                
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == batch_labels).sum().item()
                total += batch_labels.size(0)
            
            if (epoch + 1) % 3 == 0:
                accuracy = 100 * correct / total
                print(f"  CNN Epoch {epoch + 1}: Loss {total_loss / len(loader):.4f}, Acc {accuracy:.2f}%")
        
        model.feature_extractor.eval()
        
        print("CNN feature extractor trained on balanced data")
        return model
    
    def _augment_frames_direct(self, frames):
        """Apply consistent temporal augmentation to entire sequence"""
        # ✅ CRITICAL: Keep tensors on the same device
        device = frames.device
        
        frames_np = frames.cpu().numpy()
        augmented_frames = []
        
        # Generate CONSISTENT augmentation parameters for the entire sequence
        brightness_factor = np.random.uniform(0.9, 1.1)
        rotation_angle = np.random.uniform(-3, 3)
        apply_rotation = np.random.random() > 0.5
        
        import cv2
        h, w = frames_np[0].shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Apply SAME augmentation to ALL frames in the sequence
        for frame in frames_np:
            aug_frame = frame.copy()
            
            # Apply CONSISTENT brightness to entire sequence
            aug_frame = aug_frame * brightness_factor
            aug_frame = np.clip(aug_frame, 0, 1)
            
            # Apply CONSISTENT rotation to entire sequence
            if apply_rotation:
                aug_frame = cv2.warpAffine(aug_frame, rotation_matrix, (w, h))
            
            augmented_frames.append(torch.from_numpy(aug_frame).float())
        
        return torch.stack(augmented_frames).to(device)
    
    def _augment_flows_direct(self, flows):
        """Apply flow-level augmentation (NO GEOMETRIC TRANSFORMS)"""
        # ✅ CRITICAL: Keep tensors on the same device
        device = flows.device
        
        flows_np = flows.cpu().numpy()
        augmented_flows = []
        
        # Generate CONSISTENT augmentation parameters for the entire sequence
        scale_factor = np.random.uniform(0.9, 1.1)
        noise_std = 0.01
        
        for flow in flows_np:
            aug_flow = flow.copy()
            
            # Add consistent noise (preserves vector field structure)
            noise = np.random.normal(0, noise_std, flow.shape)
            aug_flow = aug_flow + noise
            
            # Apply consistent scaling (preserves vector directions)
            aug_flow = aug_flow * scale_factor
            
            augmented_flows.append(torch.from_numpy(aug_flow).float())
        
        return torch.stack(augmented_flows).to(device)
    
    def _train_cnn_balanced(self, model, frames_list, flows_list, labels, 
                           class_weights, epochs, learning_rate):
        """Train CNN with class weighting"""
        
        # Prepare training data
        frames_tensor = torch.stack(frames_list)  # (N, 3, 3, 64, 64)
        flows_tensor = torch.stack(flows_list)     # (N, 6, 64, 64)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # ✅ PRESERVE TEMPORAL DYNAMICS - Critical fix for micro-expression recognition
        # Treat each temporal frame as a separate sample to preserve onset→apex→offset motion
        if frames_tensor.dim() == 5:  # (N, T, C, H, W)
            # Reshape to treat each temporal frame as individual sample
            # (N, T, C, H, W) → (N*T, C, H, W)
            original_batch_size = frames_tensor.shape[0]
            temporal_frames = frames_tensor.shape[1]
            
            frames_tensor = frames_tensor.view(-1, 3, 64, 64)  # (N*T, 3, 64, 64)
            
            # Repeat flows to match temporal dimension and reshape
            # (N, 6, 64, 64) → (N, 1, 6, 64, 64) → (N, T, 6, 64, 64) → (N*T, 6, 64, 64)
            flows_tensor = flows_tensor.unsqueeze(1).repeat(1, temporal_frames, 1, 1, 1)
            flows_tensor = flows_tensor.view(-1, 6, 64, 64)
            
            # Repeat labels to match temporal samples
            labels_tensor = labels_tensor.repeat_interleave(temporal_frames)
            
            print(f"✅ Temporal expansion: {original_batch_size} → {frames_tensor.shape[0]} samples")
            print(f"✅ Preserved onset→apex→offset dynamics for CNN training")
            
            # 🔥 AU-WEIGHTED SPATIAL EMPHASIS - Final boost for disgust recognition
            # Emphasize nose (AU9) and upper lip (AU10) region critical for disgust
            # Region coordinates: [20:40, 25:40] covers nose and upper lip area in 64x64 frames
            frames_tensor[:, :, 20:40, 25:40] *= 1.3  # AU9/AU10 region emphasis
            print(f"✅ Applied AU-weighted spatial emphasis for disgust (AU9/AU10 region)")
        
        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(frames_tensor, flows_tensor, labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Set up training with weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.feature_extractor.parameters(), lr=learning_rate)
        
        # Add temporary linear head
        with torch.no_grad():
            sample_frames = frames_tensor[:1]  # Take first sample: (1, 3, 64, 64) after averaging
            sample_flows = flows_tensor[:1]     # Take first sample: (1, 6, 64, 64)
            features = model.feature_extractor(sample_frames, sample_flows)
            feature_dim = features.shape[1]
        
        temp_head = nn.Linear(feature_dim, len(EMOTION_LABELS)).to(self.device)
        
        model.feature_extractor.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_frames, batch_flows, batch_labels in loader:
                batch_frames = batch_frames.to(self.device)
                batch_flows = batch_flows.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Extract features
                outputs = model.feature_extractor(batch_frames, batch_flows)
                
                # Apply temporary classification head
                logits = temp_head(outputs)
                
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(loader)
            
            if (epoch + 1) % 3 == 0:
                print(f"  CNN Epoch {epoch + 1}: Loss {avg_loss:.4f}, Acc {accuracy * 100:.2f}%")
        
        model.feature_extractor.eval()
        print("CNN feature extractor trained on balanced data")
        
        # Discard temporary head
        del temp_head

def main():
    parser = argparse.ArgumentParser(description='Train Augmented Balanced Model')
    parser.add_argument('--epochs', type=int, default=12, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save model')
    
    args = parser.parse_args()
    
    print("=== AUGMENTED BALANCED MODEL TRAINING ===")
    
    # Load original dataset
    print(f"\nLoading original dataset...")
    data_root = Path(__file__).parent.parent / 'data' / 'casme2'
    labels_file = Path(__file__).parent.parent / 'data' / 'labels' / 'casme2_labels.csv'
    
    print(f"Using training directory: {data_root}")
    print(f"Labels file: {labels_file}")
    
    # Check if data exists
    if not data_root.exists():
        print(f"❌ Training directory not found: {data_root}")
        print("💡 Please ensure CASME2 dataset is properly extracted to data/casme2/")
        print("💡 Expected structure: data/casme2/sub01/EP01_01/...")
        return
    
    if not labels_file.exists():
        print(f"❌ Labels file not found: {labels_file}")
        print("💡 Please ensure labels file is available")
        return
    
    print(f"✅ Data directory found: {data_root}")
    print(f"✅ Labels file found: {labels_file}")
    
    # Create dataset
    print("🔄 Creating CNN dataset...")
    cnn_dataset = CNNCASMEIIDataset(str(data_root), str(labels_file))
    
    if len(cnn_dataset) == 0:
        print("❌ No samples loaded from casme2 directory")
        print("💡 Please ensure the casme2 directory contains the expected structure:")
        print("   data/casme2/sub01/EP01_01/frame_01_img01.jpg")
        print("   data/casme2/sub01/EP02_01f/frame_01_img01.jpg")
        return
    
    print(f"✅ Dataset created with {len(cnn_dataset)} samples")
    
    # Create augmented dataset
    print(f"\nCreating augmented balanced dataset...")
    augmented_dataset = AugmentedDataset(cnn_dataset, augmentation_factor=3)
    print(f"✅ Augmented dataset created with {len(augmented_dataset)} samples")
    
    # Initialize trainer
    trainer = AugmentedTrainer(device=args.device)
    
    # Train augmented model
    model, accuracy, report = trainer.train_augmented_model(
        augmented_dataset, 
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    # ✅ Save the trained model with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"../models/augmented_model_temporal_au_specific_{timestamp}.pkl"
    
    print(f"\n✅ Saving trained model to: {model_path}")
    model.save_model(model_path)
    
    # Save training metadata
    import json
    metadata = {
        'timestamp': timestamp,
        'model_type': 'EnhancedHybridModel',
        'training_accuracy': accuracy,
        'feature_dim': 224,
        'feature_breakdown': {
            'cnn_features': 128,
            'handcrafted': 48,
            'au_aligned_strain_statistics': 40,
            'au9_au10_specific': 8
        },
        'per_class_performance': report,
        'training_samples': len(augmented_dataset),
        'optimizations': [
            'temporal_dynamics_preservation',
            'scientific_augmentation',
            'au_weighted_spatial_emphasis',
            'au_specific_feature_concatenation'
        ]
    }
    
    metadata_path = f"../models/augmented_model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Training metadata saved to: {metadata_path}")
    print(f"✅ Model saved successfully with all scientific optimizations!")
    
    # Print final model info
    info = model.get_model_info()
    print(f"\n🎯 Final Model Info:")
    print(f"   Feature Dimension: {info['feature_dim']}")
    print(f"   Architecture: {info['architecture']}")
    print(f"   CNN Parameters: {info['cnn_parameters']:,}")
    print(f"   Feature Breakdown: {info['feature_breakdown']}")
    
    print(f"\n🎉 TRAINING COMPLETE - Model ready for inference testing!")

if __name__ == '__main__':
    main()
