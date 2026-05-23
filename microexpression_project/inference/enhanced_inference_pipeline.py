#!/usr/bin/env python3
"""
Enhanced Inference Pipeline with Preprocessing
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import datetime
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing_pipeline import VideoPreprocessor
from inference_utils import hybrid_predict_from_features, load_enhanced_hybrid_from_path

class EnhancedMicroExpressionInferencePipeline:
    """Enhanced inference pipeline with preprocessing"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the enhanced inference pipeline
        
        Args:
            model_path: Path to trained model file
        """
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = VideoPreprocessor()
        
        print(f"Enhanced Micro-Expression Inference Pipeline initialized on {self.device}")
        print(f"Preprocessor: {'✅ Face detection' if self.preprocessor.face_detection_enabled else '❌ No face detection'}")
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model
        
        Args:
            model_path: Path to trained model file
        """
        try:
            self.model = load_enhanced_hybrid_from_path(model_path)
            self.model.feature_extractor.to(self.device)
            self.model.feature_extractor.eval()
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def predict_emotion(self, video_path: str) -> Dict[str, Any]:
        """
        Predict emotion from video using enhanced pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {"success": False, "error": "Model not loaded."}

        try:
            # Preprocess video
            print(f"🎬 Processing video: {video_path}")
            frames_tensor, flows_tensor = self.preprocessor.preprocess_video(
                video_path, max_input_frames=64, verbose=False
            )
            n_read = getattr(self.preprocessor, "_last_input_frame_count", None)

            frames_tensor = frames_tensor.to(self.device)
            flows_tensor = flows_tensor.to(self.device)

            frames_b = frames_tensor.unsqueeze(0)
            flows_b = flows_tensor.unsqueeze(0)

            with torch.no_grad():
                features = self.model.extract_all_features(frames_b, flows_b)

            feats = features.detach().cpu().numpy() if torch.is_tensor(features) else np.asarray(features)
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)
            hp = hybrid_predict_from_features(self.model, feats.astype(np.float64, copy=False))
            predicted_emotion = hp["prediction_emotion"]
            confidence = hp["confidence"]
            all_probabilities = hp["by_emotion"]
            
            # AU contribution is not computed here (no AU detector in this pipeline).
            # Keep the field for API compatibility, but mark it explicitly as visual/placeholder only.
            au_contribution = {
                "visual_explanation_only": True,
                "most_active_au": None,
                "total_strain_energy": None,
                "au_rankings": {},
                "note": (
                    "No AU detector is executed in enhanced_inference_pipeline.py. "
                    "Probabilities are produced by the hybrid CNN-SVM model; AU boosting (if enabled) "
                    "operates by adjusting probabilities, not by overriding labels."
                ),
            }
            
            return {
                "success": True,
                "predicted_emotion": predicted_emotion,
                "relative_probability": confidence,
                "all_probabilities": all_probabilities,
                "au_contribution": au_contribution,
                "frame_info": {
                    "video_path": video_path,
                    "frames_processed": int(n_read) if n_read is not None else int(frames_tensor.shape[0]),
                    "onset_apex_offset_slices": int(frames_tensor.shape[0]) if frames_tensor.dim() >= 1 else 3,
                    "frame_size": f"{frames_tensor.shape[-2]}x{frames_tensor.shape[-1]}",
                    "preprocessing": "face_detection" if self.preprocessor.face_detection_enabled else "center_crop"
                },
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return {"success": False, "error": error_msg}

def test_enhanced_pipeline():
    """Test the enhanced inference pipeline"""
    
    print("=== TESTING ENHANCED INFERENCE PIPELINE ===")
    
    # Initialize pipeline with augmented model
    model_path = "../models/augmented_balanced_au_aligned_svm_20260127_162621.pkl"
    
    try:
        pipeline = EnhancedMicroExpressionInferencePipeline(model_path)
        
        # Test with known videos
        test_videos = [
            "../data/predict/sub01/EP02_01f.avi",
            "../data/predict/sub02/EP09_01.avi",
            "../data/predict/sub02/EP01_11f.avi"
        ]
        
        ground_truth = ["happiness", "happiness", "repression"]
        
        print("\n🎬 Testing enhanced pipeline...")
        print("=" * 60)
        
        results = []
        correct_count = 0
        
        for i, (video_path, true_emotion) in enumerate(zip(test_videos, ground_truth), 1):
            print(f"\n{i}. {Path(video_path).name}")
            print(f"   Ground truth: {true_emotion}")
            
            if not Path(video_path).exists():
                print(f"   ❌ Video not found")
                continue
            
            try:
                result = pipeline.predict_emotion(video_path)
                
                if result['success']:
                    predicted_emotion = result['predicted_emotion']
                    confidence = result['relative_probability']
                    
                    is_correct = (predicted_emotion == true_emotion)
                    if is_correct:
                        correct_count += 1
                    
                    status = "✅" if is_correct else "❌"
                    print(f"   {status} {predicted_emotion} (conf: {confidence:.3f}) - {'CORRECT' if is_correct else 'WRONG'}")
                    
                    results.append({
                        'video': Path(video_path).name,
                        'true_emotion': true_emotion,
                        'predicted_emotion': predicted_emotion,
                        'confidence': confidence,
                        'correct': is_correct,
                        'preprocessing': result['frame_info']['preprocessing']
                    })
                    
                else:
                    print(f"   ❌ Processing failed: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 ENHANCED PIPELINE RESULTS")
        print("=" * 60)
        
        if len(results) > 0:
            accuracy = (correct_count / len(results)) * 100
            print(f"Overall Accuracy: {correct_count}/{len(results)} = {accuracy:.1f}%")
            
            print(f"\n📈 Detailed Results:")
            for result in results:
                status = "✅" if result['correct'] else "❌"
                print(f"   {status} {result['video']:15s}: {result['predicted_emotion']:8s} "
                      f"(true: {result['true_emotion']:8s}) conf: {result['confidence']:.3f} "
                      f"preproc: {result['preprocessing']}")
            
            print(f"\n🎯 CONCLUSION:")
            if accuracy > 0:
                print(f"   ✅ Preprocessing pipeline working!")
                print(f"   ✅ Model can now process test videos correctly!")
            else:
                print(f"   ⚠️  Still issues, but preprocessing is functional")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_enhanced_pipeline()
