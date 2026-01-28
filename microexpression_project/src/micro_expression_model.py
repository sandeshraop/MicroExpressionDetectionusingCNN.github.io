import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List, Tuple
import joblib

from cnn_feature_extractor import HybridFlowCNN
from optical_flow_utils import compute_au_aligned_strain_statistics
from config import NUM_EMOTIONS, EMOTION_DISPLAY_ORDER


class EnhancedHybridModel:
    """
    Enhanced hybrid model combining CNN features with AU-aligned strain statistics.
    
    Architecture:
    - 48-D handcrafted motion features (from Phase 1)
    - 40-D AU-aligned strain statistics (5 AUs x 4 stats x 2 strain maps)
    - 128-D CNN flow+strain features
    - Total: ~216-D feature vector
    
    AU-aligned regions: AU4, AU6, AU9, AU10, AU12
    More interpretable and noise-resistant than uniform grid.
    """
    
    def __init__(self, cnn_model: str = 'hybrid', classifier_type: str = 'svm'):
        """
        Initialize enhanced hybrid model.
        
        Args:
            cnn_model: 'hybrid' for HybridFlowCNN
            classifier_type: 'svm', 'rf', or 'xgb'
        """
        self.cnn_model = cnn_model
        self.classifier_type = classifier_type
        
        # Initialize CNN feature extractor
        if cnn_model == 'hybrid':
            self.feature_extractor = HybridFlowCNN(NUM_EMOTIONS)
        else:
            raise ValueError(f"Unsupported CNN model: {cnn_model}")
        
        # Remove final classification layer for feature extraction
        self.feature_extractor.fusion = self.feature_extractor.fusion[:-1]
        
        # Initialize classifier
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'xgb':
            try:
                from xgboost import XGBClassifier
                self.classifier = XGBClassifier(random_state=42, eval_metric='mlogloss')
            except ImportError:
                raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        else:
            raise ValueError(f"Unsupported classifier: {classifier_type}")
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.classifier)
        ])
        
        # Store training data for fitting
        self.is_fitted = False
    
    def extract_cnn_features(self, frames: torch.Tensor, flows: torch.Tensor) -> np.ndarray:
        """
        Extract CNN features from frames and flows.
        
        Args:
            frames: Frame tensor (batch_size, 3, 64, 64)
            flows: Flow tensor (batch_size, 6, 64, 64)
        
        Returns:
            Extracted CNN features as numpy array
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(frames, flows)
        
        return features.cpu().numpy()
    
    def extract_au_aligned_strain_statistics(self, flows: torch.Tensor) -> np.ndarray:
        """
        Extract AU-aligned strain statistics from flow tensor.
        
        Args:
            flows: Flow tensor (batch_size, 6, 64, 64) or (6, 64, 64)
        
        Returns:
            AU-aligned strain statistics as numpy array (40-D)
        """
        # Handle both 3D and 4D tensors
        if flows.dim() == 3:
            flows = flows.unsqueeze(0)  # Add batch dimension
        
        batch_size = flows.shape[0]
        all_stats = []
        
        for i in range(batch_size):
            # Extract strain components (last 2 channels)
            strain1 = flows[i, 4, :, :]  # onset→apex strain
            strain2 = flows[i, 5, :, :]  # apex→offset strain
            
            # Compute AU-aligned statistics for each strain map
            stats1 = compute_au_aligned_strain_statistics(strain1.numpy())
            stats2 = compute_au_aligned_strain_statistics(strain2.numpy())
            
            # Combine statistics (40 features total: 20 per strain map)
            combined_stats = np.concatenate([stats1, stats2])
            all_stats.append(combined_stats)
        
        return np.array(all_stats)
    
    def extract_handcrafted_features(self, frames: torch.Tensor) -> np.ndarray:
        """
        Extract enhanced handcrafted motion features.
        
        This replaces the simple placeholder with more meaningful statistics
        that better represent the 48-D Phase 1 motion descriptors.
        
        Args:
            frames: Frame tensor (batch_size, 3, 64, 64) or (3, 64, 64)
        
        Returns:
            Enhanced handcrafted features (48-D)
        """
        # Handle both 3D and 4D tensors
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)  # Add batch dimension
        
        batch_size = frames.shape[0]
        
        handcrafted_features = []
        
        for i in range(batch_size):
            frame_seq = frames[i]  # (3, 64, 64)
            
            # Extract individual frames
            onset = frame_seq[0]
            apex = frame_seq[1]
            offset = frame_seq[2]
            
            # Enhanced motion statistics (more representative of Phase 1 features)
            features = []
            
            # 1. Frame-level statistics (9 features)
            for frame in [onset, apex, offset]:
                features.extend([
                    torch.mean(frame).item(),
                    torch.std(frame).item(),
                    torch.max(frame).item()
                ])
            
            # 2. Motion difference statistics (12 features)
            diff_onset_apex = apex - onset
            diff_apex_offset = offset - apex
            diff_onset_offset = offset - onset
            
            for diff in [diff_onset_apex, diff_apex_offset, diff_onset_offset]:
                features.extend([
                    torch.mean(diff).item(),
                    torch.std(diff).item(),
                    torch.max(diff).item(),
                    torch.min(diff).item()
                ])
            
            # 3. Regional motion statistics (27 features)
            # Divide face into 9 regions (3x3 grid) for spatial analysis
            h, w = 64, 64
            region_h, region_w = h // 3, w // 3
            
            for diff in [diff_onset_apex, diff_apex_offset, diff_onset_offset]:
                for i in range(3):
                    for j in range(3):
                        y_start = i * region_h
                        y_end = (i + 1) * region_h if i < 2 else h
                        x_start = j * region_w
                        x_end = (j + 1) * region_w if j < 2 else w
                        
                        region_diff = diff[y_start:y_end, x_start:x_end]
                        features.append(torch.mean(torch.abs(region_diff)).item())
            
            # Ensure exactly 48 features
            if len(features) > 48:
                features = features[:48]
            elif len(features) < 48:
                features.extend([0.0] * (48 - len(features)))
            
            handcrafted_features.append(features)
        
        return np.array(handcrafted_features)
    
    def extract_au9_au10_stats(self, flows: torch.Tensor) -> np.ndarray:
        """
        Extract AU9 (nose wrinkler) and AU10 (upper lip raiser) specific strain statistics.
        
        Args:
            flows: Flow tensor (batch_size, 6, 64, 64) or (6, 64, 64)
        
        Returns:
            AU9/AU10 specific statistics as numpy array (8-D)
        """
        # Handle both 3D and 4D tensors
        if flows.dim() == 3:
            flows = flows.unsqueeze(0)
        
        batch_size = flows.shape[0]
        au9_au10_stats = []
        
        for i in range(batch_size):
            # Extract strain components (last 2 channels)
            strain1 = flows[i, 4, :, :].numpy()  # onset→apex strain
            strain2 = flows[i, 5, :, :].numpy()  # apex→offset strain
            
            # AU9 region: nose area (rows 20:35, cols 25:40)
            au9_region1 = strain1[20:35, 25:40]
            au9_region2 = strain2[20:35, 25:40]
            
            # AU10 region: upper lip area (rows 30:40, cols 20:45)
            au10_region1 = strain1[30:40, 20:45]
            au10_region2 = strain2[30:40, 20:45]
            
            # Compute statistics for each region and strain map
            au9_stats1 = [np.mean(au9_region1), np.std(au9_region1), np.max(au9_region1)]
            au9_stats2 = [np.mean(au9_region2), np.std(au9_region2), np.max(au9_region2)]
            au10_stats1 = [np.mean(au10_region1), np.std(au10_region1), np.max(au10_region1)]
            au10_stats2 = [np.mean(au10_region2), np.std(au10_region2), np.max(au10_region2)]
            
            # Combine all AU9/AU10 statistics (8 features)
            combined_stats = au9_stats1 + au9_stats2 + au10_stats1 + au10_stats2
            au9_au10_stats.append(combined_stats)
        
        return np.array(au9_au10_stats)
    
    def extract_all_features(self, frames: torch.Tensor, flows: torch.Tensor) -> np.ndarray:
        """
        Extract all features: handcrafted + AU-aligned strain statistics + CNN features.
        
        Args:
            frames: Frame tensor (batch_size, 3, 64, 64)
            flows: Flow tensor (batch_size, 6, 64, 64)
        
        Returns:
            Combined feature vector (~216-D)
        """
        # Extract different feature types
        handcrafted = self.extract_handcrafted_features(frames)  # (batch_size, 48)
        au_strain_stats = self.extract_au_aligned_strain_statistics(flows.cpu())  # (batch_size, 40)
        au9_au10_stats = self.extract_au9_au10_stats(flows)  # (batch_size, 8)
        cnn_features = self.extract_cnn_features(frames, flows)  # (batch_size, 128)
        
        # ✅ AU-SPECIFIC FEATURE CONCATENATION - Disgust boost
        # Final SVM feature vector: [CNN + handcrafted + AU_aligned + AU9_AU10]
        combined_features = np.concatenate([cnn_features, handcrafted, au_strain_stats, au9_au10_stats], axis=1)
        
        return combined_features  # (batch_size, 224) - 128+48+40+8
    
    def fit(self, frames_list: List[torch.Tensor], flows_list: List[torch.Tensor], 
            labels: np.ndarray, train_cnn: bool = True) -> None:
        """
        Fit the enhanced hybrid model.
        
        Args:
            frames_list: List of frame tensors
            flows_list: List of flow tensors
            labels: Training labels
            train_cnn: Whether to train CNN feature extractor
        """
        print(f"Training Enhanced Hybrid Model + {self.classifier_type.upper()}...")
        
        if train_cnn:
            print("Training CNN feature extractor...")
            self._train_cnn(frames_list, flows_list, labels)
        
        # Extract all features
        all_features = []
        
        for frames, flows in zip(frames_list, flows_list):
            # Ensure batch dimension
            if frames.dim() == 3:
                frames = frames.unsqueeze(0)
            if flows.dim() == 3:
                flows = flows.unsqueeze(0)
            
            features = self.extract_all_features(frames, flows)
            all_features.append(features)
        
        # Concatenate all features
        all_features = np.vstack(all_features)
        
        print(f"Extracted features shape: {all_features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Feature breakdown: 48 (handcrafted) + 40 (AU-aligned strain stats) + 128 (CNN) = {all_features.shape[1]} total")
        
        # Fit the pipeline
        self.pipeline.fit(all_features, labels)
        self.is_fitted = True
        
        print("Enhanced hybrid pipeline fitted successfully!")
    
    def _train_cnn(self, frames_list: List[torch.Tensor], flows_list: List[torch.Tensor], 
                   labels: np.ndarray, epochs: int = 10) -> None:
        """Train CNN feature extractor on training data."""
        import torch.nn as nn
        import torch.optim as optim
        
        # Prepare training data
        frames_tensor = torch.stack(frames_list)
        flows_tensor = torch.stack(flows_list)
        labels_tensor = torch.tensor(labels)
        
        # Verify flow tensor has 6 channels
        if flows_tensor.shape[1] != 6:
            raise ValueError(f"Expected 6 channels in flow tensor, got {flows_tensor.shape[1]}")
        
        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(frames_tensor, flows_tensor, labels_tensor.long())
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Set up training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.001)
        
        # Add temporary linear head for CNN training
        # Get feature dimension by forward pass
        with torch.no_grad():
            sample_frames = frames_tensor[:1]
            sample_flows = flows_tensor[:1]
            features = self.feature_extractor(sample_frames, sample_flows)
            feature_dim = features.shape[1]
        
        temp_head = nn.Linear(feature_dim, NUM_EMOTIONS)
        
        self.feature_extractor.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_frames, batch_flows, batch_labels in loader:
                optimizer.zero_grad()
                
                # Extract features
                outputs = self.feature_extractor(batch_frames, batch_flows)
                
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
        
        self.feature_extractor.eval()
        print("CNN feature extractor trained (with temporary classification head)")
        
        # Discard temporary head (it will be garbage collected)
        del temp_head
    
    def predict(self, frames: torch.Tensor, flows: torch.Tensor) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            frames: Frame tensor
            flows: Flow tensor
        
        Returns:
            Predicted class indices
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Extract all features
        features = self.extract_all_features(frames, flows)
        
        # Make predictions
        predictions = self.pipeline.predict(features)
        
        return predictions
    
    def predict_proba(self, frames: torch.Tensor, flows: torch.Tensor) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            frames: Frame tensor
            flows: Flow tensor
        
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        # Extract all features
        features = self.extract_all_features(frames, flows)
        
        # Make probability predictions
        probabilities = self.pipeline.predict_proba(features)
        
        return probabilities
    
    def evaluate(self, frames_list: List[torch.Tensor], flows_list: List[torch.Tensor], 
                labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the enhanced hybrid model with comprehensive metrics.
        
        Args:
            frames_list: List of frame tensors
            flows_list: List of flow tensors
            labels: True labels
        
        Returns:
            Evaluation results with UAR and per-class metrics
        """
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        # Extract all features and make predictions
        all_features = []
        for frames, flows in zip(frames_list, flows_list):
            if frames.dim() == 3:
                frames = frames.unsqueeze(0)
            if flows.dim() == 3:
                flows = flows.unsqueeze(0)
            
            features = self.extract_all_features(frames, flows)
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        
        # Make predictions
        predictions = self.pipeline.predict(all_features)
        probabilities = self.pipeline.predict_proba(all_features)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions, labels=[0,1,2,3])
        
        # Get unique labels present in this fold
        unique_labels = sorted(list(set(labels)))
        target_names = [EMOTION_DISPLAY_ORDER[i] for i in unique_labels]
        
        report = classification_report(labels, predictions,
                                     labels=unique_labels,
                                     target_names=target_names,
                                     output_dict=True, zero_division=0)
        
        # Calculate UAR (Unweighted Average Recall)
        def calculate_uar(report_dict):
            uar = np.mean([
                report_dict[emotion]['recall']
                for emotion in EMOTION_DISPLAY_ORDER
                if emotion in report_dict
            ])
            return uar
        
        uar = calculate_uar(report)
        
        # Extract per-class recall for detailed analysis
        per_class_recall = {}
        for emotion in EMOTION_DISPLAY_ORDER:
            if emotion in report:
                per_class_recall[emotion] = report[emotion]['recall']
        
        return {
            'accuracy': accuracy,
            'uar': uar,
            'per_class_recall': per_class_recall,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': labels,
            'probabilities': probabilities,
            'features': all_features
        }
    
    def save_model(self, model_path: str):
        """Save trained model to file."""
        import joblib
        import os
        
        model_data = {
            'cnn_model': self.cnn_model,
            'classifier_type': self.classifier_type,
            'feature_extractor_state': self.feature_extractor.state_dict(),
            'pipeline': self.pipeline,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model from file."""
        import joblib
        
        model_data = joblib.load(model_path)
        
        # Restore model state
        self.cnn_model = model_data['cnn_model']
        self.classifier_type = model_data['classifier_type']
        self.feature_extractor.load_state_dict(model_data['feature_extractor_state'])
        self.pipeline = model_data['pipeline']
        self.is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {model_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model details
        """
        cnn_params = sum(p.numel() for p in self.feature_extractor.parameters())
        
        return {
            'model_type': 'EnhancedHybridModel',
            'cnn_model': self.cnn_model,
            'classifier': self.classifier_type,
            'cnn_parameters': cnn_params,
            'feature_dim': 224,  # 128 + 48 + 40 + 8 (CNN + handcrafted + AU-aligned + AU9_AU10)
            'feature_breakdown': {
                'cnn_features': 128,
                'handcrafted': 48,
                'au_aligned_strain_statistics': 40,
                'au9_au10_specific': 8  # Disgust-specific features
            },
            'is_fitted': self.is_fitted,
            'architecture': 'CNN(128) + Handcrafted(48) + AU-Aligned(40) + AU9/AU10(8) -> StandardScaler -> SVM'
        }


if __name__ == "__main__":
    # Test enhanced hybrid model
    print("Testing Enhanced Hybrid Model...")
    
    # Create synthetic test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create test samples with 6-channel flow
    frames_list = [torch.randn(3, 64, 64) for _ in range(10)]
    flows_list = [torch.randn(6, 64, 64) for _ in range(10)]
    labels = np.random.randint(0, 4, 10)
    
    print(f"Test data: {len(frames_list)} samples")
    print(f"Frames shape: {frames_list[0].shape}")
    print(f"Flows shape: {flows_list[0].shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test with SVM
    print("\n=== Testing Enhanced Hybrid Model with SVM ===")
    enhanced_model = EnhancedHybridModel(cnn_model='hybrid', classifier_type='svm')
    
    # Fit the model
    enhanced_model.fit(frames_list, flows_list, labels)
    
    # Test predictions
    test_frames = torch.randn(3, 64, 64)
    test_flows = torch.randn(6, 64, 64)
    
    pred = enhanced_model.predict(test_frames, test_flows)
    proba = enhanced_model.predict_proba(test_frames, test_flows)
    
    print(f"Prediction: {pred}")
    print(f"Probabilities: {proba}")
    
    # Evaluate
    results = enhanced_model.evaluate(frames_list, flows_list, labels)
    print(f"Test accuracy: {results['accuracy']:.3f}")
    
    # Test model info
    info = enhanced_model.get_model_info()
    print(f"Model info: {info}")
    
    print("\nEnhanced Hybrid Model working correctly!")
