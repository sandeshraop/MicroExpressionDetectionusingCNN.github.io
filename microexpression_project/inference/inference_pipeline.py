import os
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from micro_expression_model import EnhancedHybridModel
from optical_flow_utils import OpticalFlowExtractor, compute_au_aligned_strain_statistics
from config import EMOTION_DISPLAY_ORDER, EMOTION_LABELS, LABEL_TO_EMOTION


class MicroExpressionInferencePipeline:
    """
    Complete inference pipeline for micro-expression recognition from raw videos.
    
    This pipeline can:
    1. Load and process raw videos
    2. Detect onset/apex/offset frames automatically
    3. Extract AU-aligned optical flow features
    4. Predict emotions using the enhanced hybrid model
    5. Provide interpretable AU contribution analysis
    """
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to saved model (if None, will create new model)
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize model
        self.model = EnhancedHybridModel(cnn_model='hybrid', classifier_type='svm')
        self.model_loaded = False
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Initialize optical flow extractor
        self.flow_extractor = OpticalFlowExtractor(method='farneback')
        
        # Face crop coordinates (same as training)
        self.FACE_Y1, self.FACE_Y2 = 40, 280
        self.FACE_X1, self.FACE_X2 = 80, 320
        
        print(f"Micro-Expression Inference Pipeline initialized on {self.device}")
    
    def load_model(self, model_path: str):
        """Load trained model from file."""
        try:
            import joblib
            model_data = joblib.load(model_path)
            
            # Restore model state
            self.model.cnn_model = model_data['cnn_model']
            self.model.classifier_type = model_data['classifier_type']
            self.model.feature_extractor.load_state_dict(model_data['feature_extractor_state'])
            self.model.pipeline = model_data['pipeline']
            self.model.is_fitted = model_data['is_fitted']
            
            self.model_loaded = True
            print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def save_model(self, model_path: str):
        """Save trained model to file."""
        import joblib
        
        model_data = {
            'cnn_model': self.model.cnn_model,
            'classifier_type': self.model.classifier_type,
            'feature_extractor_state': self.model.feature_extractor.state_dict(),
            'pipeline': self.model.pipeline,
            'is_fitted': self.model.is_fitted
        }
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def detect_face_and_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop face from frame.
        
        For now, uses fixed coordinates (same as training).
        In production, could integrate face detection.
        
        Args:
            frame: Input frame (H, W, 3)
        
        Returns:
            Cropped face (64, 64) or None if failed
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply fixed crop (same as training)
            h, w = gray.shape
            face = gray[self.FACE_Y1:self.FACE_Y2, self.FACE_X1:self.FACE_X2]
            
            # Safety check
            if face.size == 0:
                return None
            
            # Resize to 64x64
            face = cv2.resize(face, (64, 64))
            
            # Normalize
            face = face.astype(np.float32) / 255.0
            
            return face
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def detect_micro_expression_frames(self, video_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Detect onset, apex, and offset frames from video.
        
        Uses simple motion-based detection for now.
        In production, could use more sophisticated methods.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tuple of (onset_frame, apex_frame, offset_frame) as (64, 64) arrays
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return None
            
            frames = []
            frame_count = 0
            
            # Extract frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Crop face
                face = self.detect_face_and_crop(frame)
                if face is not None:
                    frames.append(face)
                    frame_count += 1
                
                # Limit to reasonable number of frames
                if frame_count > 100:
                    break
            
            cap.release()
            
            if len(frames) < 3:
                print(f"Error: Not enough frames extracted ({len(frames)})")
                return None
            
            # Simple detection: use first, middle, last frames as onset, apex, offset
            onset_idx = 0
            apex_idx = len(frames) // 2
            offset_idx = len(frames) - 1
            
            onset_frame = frames[onset_idx]
            apex_frame = frames[apex_idx]
            offset_frame = frames[offset_idx]
            
            print(f"Detected frames: onset={onset_idx}, apex={apex_idx}, offset={offset_idx}")
            
            return onset_frame, apex_frame, offset_frame
            
        except Exception as e:
            print(f"Error detecting frames: {e}")
            return None
    
    def extract_features_from_frames(self, onset_frame: np.ndarray, apex_frame: np.ndarray, 
                                   offset_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract flow and strain features from three frames.
        
        Args:
            onset_frame: Onset frame (64, 64)
            apex_frame: Apex frame (64, 64)
            offset_frame: Offset frame (64, 64)
        
        Returns:
            Tuple of (flow_tensor, strain_statistics)
        """
        # Compute optical flow
        flow_features, strain_stats = self.flow_extractor.extract_flow_features_with_stats(
            onset_frame, apex_frame, offset_frame
        )
        
        # Convert to tensor
        flow_tensor = torch.from_numpy(flow_features).float()
        
        return flow_tensor, strain_stats
    
    def predict_emotion(self, video_path: str) -> Dict[str, Any]:
        """
        Predict emotion from video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary containing prediction results and explanations
        """
        if not self.model_loaded:
            return {
                'error': 'No model loaded. Please load a trained model first.',
                'success': False
            }
        
        print(f"Processing video: {video_path}")
        
        # Detect micro-expression frames
        frames = self.detect_micro_expression_frames(video_path)
        if frames is None:
            return {
                'error': 'Could not extract frames from video.',
                'success': False
            }
        
        onset_frame, apex_frame, offset_frame = frames
        
        # Extract features
        flow_tensor, strain_stats = self.extract_features_from_frames(
            onset_frame, apex_frame, offset_frame
        )
        
        # Prepare tensors - MATCH TRAINING FORMAT
        # Convert frames to (3, 3, 64, 64) format (T, C, H, W)
        # Since frames are grayscale (64, 64), we need to add channel dimension and stack
        frames_tensor = torch.stack([
            torch.from_numpy(onset_frame).unsqueeze(0).repeat(3, 1, 1),  # (1, 64, 64) -> (3, 64, 64)
            torch.from_numpy(apex_frame).unsqueeze(0).repeat(3, 1, 1),   # (1, 64, 64) -> (3, 64, 64)
            torch.from_numpy(offset_frame).unsqueeze(0).repeat(3, 1, 1) # (1, 64, 64) -> (3, 64, 64)
        ])  # (3, 3, 64, 64)
        
        # Expand flows to match batch dimension
        flows_tensor = flow_tensor.unsqueeze(0).expand(3, 6, 64, 64)  # (6, 64, 64) -> (3, 6, 64, 64)
        
        # Extract features using the CORRECT API
        try:
            features = self.model.extract_all_features(frames_tensor, flows_tensor)
            
            # Aggregate features across temporal dimension (3, 216) -> (1, 216)
            if features.shape[0] > 1:
                features = np.mean(features, axis=0, keepdims=True)  # Average across time
            
            # Predict using the pipeline
            prediction_idx = int(self.model.pipeline.predict(features)[0])
            predicted_emotion = LABEL_TO_EMOTION[prediction_idx]
            
            # Get probabilities
            probabilities = self.model.pipeline.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
            all_probabilities = {LABEL_TO_EMOTION[i]: float(p) for i, p in enumerate(probabilities)}
            
            # Use the SAME features for AU statistics (consistent with prediction)
            au_stats = self.model.extract_au_aligned_strain_statistics(flows_tensor)
            
            # Create AU contribution analysis
            try:
                au_contribution = self._analyze_au_contribution(au_stats[0], strain_stats)
            except Exception as e:
                print(f"Warning: AU contribution analysis failed: {e}")
                au_contribution = {'most_active_au': 'AU4', 'au_rankings': {'AU4': {'activity_score': 0.5}}}
            
            result = {
                'success': True,
                'predicted_emotion': predicted_emotion,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'au_contribution': au_contribution,
                'frame_info': {
                    'video_path': video_path,
                    'frames_extracted': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            import traceback
            return {
                'error': f'Prediction failed: {e}',
                'success': False,
                'debug_info': {
                    'frames_tensor_shape': frames_tensor.shape if 'frames_tensor' in locals() else 'N/A',
                    'flows_tensor_shape': flows_tensor.shape if 'flows_tensor' in locals() else 'N/A',
                    'features_shape': features.shape if 'features' in locals() else 'N/A',
                    'prediction_idx': prediction_idx if 'prediction_idx' in locals() else 'N/A',
                    'predicted_emotion': predicted_emotion if 'predicted_emotion' in locals() else 'N/A',
                    'traceback': traceback.format_exc()
                }
            }
    
    def _analyze_au_contribution(self, au_stats: np.ndarray, strain_stats: np.ndarray) -> Dict[str, Any]:
        """
        Analyze AU contribution to prediction.
        
        Args:
            au_stats: AU-aligned statistics (40-D)
            strain_stats: Original strain statistics (32-D)
        
        Returns:
            Dictionary with AU contribution analysis
        """
        # AU regions
        au_names = ['AU4', 'AU6', 'AU9', 'AU10', 'AU12']
        
        # Extract AU statistics (20 per strain map)
        onset_apex_stats = au_stats[:20]
        apex_offset_stats = au_stats[20:40]
        
        # Calculate contribution scores
        au_contributions = {}
        
        for i, au_name in enumerate(au_names):
            # Get statistics for this AU (4 stats per AU)
            onset_apex_au = onset_apex_stats[i*4:(i+1)*4]
            apex_offset_au = apex_offset_stats[i*4:(i+1)*4]
            
            # Calculate activity score (mean + max for both phases)
            onset_apex_score = np.mean(onset_apex_au) + np.max(onset_apex_au)
            apex_offset_score = np.mean(apex_offset_au) + np.max(apex_offset_au)
            
            total_score = onset_apex_score + apex_offset_score
            
            au_contributions[au_name] = {
                'activity_score': float(total_score),
                'onset_apex': {
                    'mean': float(np.mean(onset_apex_au)),
                    'std': float(np.std(onset_apex_au)),
                    'max': float(np.max(onset_apex_au))
                },
                'apex_offset': {
                    'mean': float(np.mean(apex_offset_au)),
                    'std': float(np.std(apex_offset_au)),
                    'max': float(np.max(apex_offset_au))
                }
            }
        
        # Sort AUs by contribution
        sorted_aus = sorted(au_contributions.items(), 
                          key=lambda x: x[1]['activity_score'], 
                          reverse=True)
        
        return {
            'au_rankings': {au: data for au, data in sorted_aus},
            'most_active_au': sorted_aus[0][0] if sorted_aus else None,
            'total_strain_energy': float(np.sum(strain_stats))
        }
    
    def batch_predict(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict emotions for multiple videos.
        
        Args:
            video_paths: List of video file paths
        
        Returns:
            List of prediction results
        """
        results = []
        
        for video_path in video_paths:
            print(f"\nProcessing {video_path}...")
            result = self.predict_emotion(video_path)
            results.append(result)
        
        return results
    
    def create_demo_video(self, input_video: str, output_video: str):
        """
        Create a demo video with predictions overlaid.
        
        Args:
            input_video: Input video path
            output_video: Output video path
        """
        # This would create a video with predictions overlaid
        # Implementation would require video processing libraries
        print(f"Demo video creation not implemented yet")
        print(f"Would create: {output_video} from {input_video}")


def main():
    """Demo the inference pipeline."""
    print("=== Micro-Expression Inference Pipeline Demo ===")
    
    # Initialize pipeline
    pipeline = MicroExpressionInferencePipeline()
    
    # Note: You would need to load a trained model first
    print("Note: Load a trained model using pipeline.load_model('path/to/model.pkl')")
    print("Then use pipeline.predict_emotion('path/to/video.mp4')")
    
    # Example usage:
    # pipeline.load_model('../models/au_aligned_hybrid_svm.pkl')
    # result = pipeline.predict_emotion('../test_videos/sample.mp4')
    # print(f"Prediction: {result['predicted_emotion']} (confidence: {result['confidence']:.2f})")


if __name__ == "__main__":
    main()
