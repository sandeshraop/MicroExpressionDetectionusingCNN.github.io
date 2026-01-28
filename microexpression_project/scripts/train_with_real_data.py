#!/usr/bin/env python3
"""
Retrain Model with Real CASME-II Data
Uses processed cropped images for proper training
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from micro_expression_model import EnhancedHybridModel
from dataset_loader import CNNCASMEIIDataset
from config import EMOTION_LABELS, LABEL_TO_EMOTION, EMOTION_DISPLAY_ORDER
from sklearn.utils.class_weight import compute_class_weight

def check_processed_data(data_root: str) -> bool:
    """Check if processed data exists and is valid"""
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        return False
    
    # Check for subject directories
    subject_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('sub')]
    if not subject_dirs:
        print(f"❌ No subject directories found in: {data_path}")
        return False
    
    # Check for episode directories with images
    valid_episodes = 0
    for subject_dir in subject_dirs:
        episode_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]
        for episode_dir in episode_dirs:
            # Check for frame images
            frame_files = list(episode_dir.glob("frame_*.jpg"))
            if frame_files:
                valid_episodes += 1
    
    if valid_episodes == 0:
        print(f"❌ No valid episodes with frame images found")
        return False
    
    print(f"✅ Found {len(subject_dirs)} subjects with {valid_episodes} valid episodes")
    return True

def train_with_real_data(data_root: str, model_save_dir: str = "models", epochs: int = 12, learning_rate: float = 0.001):
    """
    Train model with real CASME-II cropped images
    
    Args:
        data_root: Directory containing processed images (data/casme2)
        model_save_dir: Directory to save trained model
        epochs: Number of training epochs
        learning_rate: Learning rate for training
    """
    print("🧠 TRAINING WITH REAL CASME-II DATA")
    print("=" * 60)
    
    # Check data availability
    if not check_processed_data(data_root):
        print("💡 Please run process_raw_videos.py first to convert raw videos to cropped images")
        return
    
    # Load labels file
    labels_file = project_root / 'data' / 'labels' / 'casme2_labels.csv'
    if not labels_file.exists():
        print(f"❌ Labels file not found: {labels_file}")
        return
    
    print(f"✅ Data directory: {data_root}")
    print(f"✅ Labels file: {labels_file}")
    
    # Create dataset
    print("🔄 Loading real CASME-II dataset...")
    try:
        dataset = CNNCASMEIIDataset(str(data_root), str(labels_file))
        
        if len(dataset) == 0:
            print("❌ No samples loaded from dataset")
            return
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Print emotion distribution
        emotion_counts = {}
        for i in range(len(dataset)):
            _, _, label, _ = dataset[i]
            emotion = LABEL_TO_EMOTION[label.item()]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("📊 Emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} samples")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Prepare training data
    print("📊 Preparing training data...")
    frames_list = []
    flows_list = []
    labels_list = []
    
    try:
        for i in range(len(dataset)):
            frames, flows, label, metadata = dataset[i]
            frames_list.append(frames)
            flows_list.append(flows)
            labels_list.append(label.item())
        
        print(f"✅ Prepared {len(frames_list)} training samples")
        
    except Exception as e:
        print(f"❌ Error preparing training data: {e}")
        return
    
    # Initialize model
    print("🧠 Initializing Enhanced Hybrid CNN-SVM model...")
    try:
        model = EnhancedHybridModel()
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    # Compute class weights
    print("⚖️ Computing class weights...")
    try:
        classes = np.unique(labels_list)
        class_weights = compute_class_weight('balanced', classes=classes, y=labels_list)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"✅ Class weights computed: {dict(zip(classes, class_weights.numpy()))}")
    except Exception as e:
        print(f"❌ Error computing class weights: {e}")
        class_weights = None
    
    # Train CNN feature extractor
    print("🏋️ Training CNN feature extractor...")
    try:
        # Convert to tensors
        frames_tensor = torch.stack(frames_list)  # (N, 3, 3, 64, 64)
        flows_tensor = torch.stack(flows_list)     # (N, 6, 64, 64)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        print(f"📊 Training tensors: frames {frames_tensor.shape}, flows {flows_tensor.shape}")
        
        # Simple CNN training (can be enhanced)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        
        model.feature_extractor.to(device)
        frames_tensor = frames_tensor.to(device)
        flows_tensor = flows_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        
        # Training setup
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.feature_extractor.parameters(), lr=learning_rate)
        
        # Training loop
        model.feature_extractor.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Simple batch processing (can be enhanced with proper DataLoader)
            batch_size = 8
            num_batches = len(frames_tensor) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_frames = frames_tensor[start_idx:end_idx]
                batch_flows = flows_tensor[start_idx:end_idx]
                batch_labels = labels_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model.feature_extractor(batch_frames, batch_flows)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            accuracy = 100 * correct / total
            avg_loss = total_loss / num_batches
            
            if (epoch + 1) % 3 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: Loss {avg_loss:.4f}, Acc {accuracy:.2f}%")
        
        model.feature_extractor.eval()
        print("✅ CNN feature extractor trained successfully")
        
    except Exception as e:
        print(f"❌ Error training CNN: {e}")
        return
    
    # Train SVM classifier
    print("📈 Training SVM classifier...")
    try:
        # Extract features from all samples
        model.feature_extractor.eval()
        with torch.no_grad():
            all_features = model.feature_extractor(frames_tensor, flows_tensor)
        
        # Convert to numpy for SVM
        features_np = all_features.cpu().numpy()
        labels_np = np.array(labels_list)
        
        # Train SVM
        model.pipeline.fit(features_np, labels_np)
        model.is_fitted = True
        
        # Evaluate
        predictions = model.pipeline.predict(features_np)
        accuracy = np.mean(predictions == labels_np)
        
        print(f"✅ SVM trained successfully")
        print(f"📊 Training accuracy: {accuracy * 100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error training SVM: {e}")
        return
    
    # Save model
    print("💾 Saving trained model...")
    try:
        # Create save directory
        save_dir = Path(model_save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_dir / f"real_data_model_{timestamp}.pkl"
        
        # Save model
        model.save_model(str(model_path))
        
        # Save metadata
        import json
        metadata = {
            'timestamp': timestamp,
            'model_type': 'EnhancedHybridModel',
            'training_data': 'Real CASME-II cropped images',
            'training_accuracy': accuracy,
            'feature_dim': 224,
            'feature_breakdown': {
                'cnn_features': 128,
                'handcrafted': 48,
                'au_aligned_strain_statistics': 40,
                'au9_au10_specific': 8
            },
            'training_samples': len(dataset),
            'emotion_distribution': emotion_counts,
            'data_root': str(data_root),
            'epochs': epochs,
            'learning_rate': learning_rate
        }
        
        metadata_path = save_dir / f"real_data_model_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved: {model_path}")
        print(f"✅ Metadata saved: {metadata_path}")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    print("\n🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Model trained on {len(dataset)} real CASME-II samples")
    print(f"🎯 Training accuracy: {accuracy * 100:.2f}%")
    print(f"💾 Model saved: {model_path}")
    print(f"📄 Metadata: {metadata_path}")
    
    # Update web app to use new model
    print(f"\n💡 To use this model in the web app:")
    print(f"   1. Copy {model_path} to models/augmented_model_temporal_au_specific_20260127_182653.pkl")
    print(f"   2. Restart the Flask server")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train model with real CASME-II data')
    parser.add_argument('--data_root', type=str, default='data/casme2', 
                       help='Directory containing processed CASME-II images')
    parser.add_argument('--model_save_dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=12,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    data_root = project_root / args.data_root
    model_save_dir = project_root / args.model_save_dir
    
    # Train model
    train_with_real_data(str(data_root), str(model_save_dir), args.epochs, args.learning_rate)

if __name__ == '__main__':
    main()
