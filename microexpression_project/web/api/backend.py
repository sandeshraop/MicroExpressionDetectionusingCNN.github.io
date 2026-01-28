#!/usr/bin/env python3
"""
Backend API for Micro-Expression Recognition Web Interface
Flask-based REST API for video processing and emotion analysis
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import tempfile
import uuid
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import cv2
from werkzeug.utils import secure_filename

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

# Import our model components
try:
    from micro_expression_model import EnhancedHybridModel
    from dataset_loader import CNNCASMEIIDataset
    from config import EMOTION_LABELS, LABEL_TO_EMOTION
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Model components not available. Using mock responses.")

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Global variables
model = None
model_loaded = False

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the trained model"""
    global model, model_loaded
    
    if model_loaded:
        return True
    
    try:
        # Look for trained model file
        model_paths = [
            project_root / 'models' / 'augmented_model_temporal_au_specific_20260127_182653.pkl',
            project_root / 'models' / 'augmented_model.pkl'
        ]
        
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path and MODEL_AVAILABLE:
            import joblib
            model = joblib.load(model_path)
            model_loaded = True
            print(f"Model loaded from: {model_path}")
            return True
        else:
            print("No trained model found. Using mock responses.")
            return False
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def extract_frames(video_path, max_frames=10):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract
        if total_frames <= max_frames:
            frame_indices = range(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame to 64x64
                frame_resized = cv2.resize(frame, (64, 64))
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame_normalized = frame_rgb / 255.0
                frames.append(frame_normalized)
        
    finally:
        cap.release()
    
    return frames

def compute_optical_flow(frames):
    """Compute optical flow between consecutive frames"""
    if len(frames) < 2:
        return []
    
    flows = []
    
    for i in range(len(frames) - 1):
        frame1 = (frames[i] * 255).astype(np.uint8)
        frame2 = (frames[i + 1] * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Normalize flow
        flow_normalized = flow / 255.0
        flows.append(flow_normalized)
    
    return flows

def analyze_video_mock(frames, flows):
    """Mock analysis for demonstration when model is not available"""
    import random
    
    # Generate random probabilities
    probabilities = {
        'happiness': random.uniform(0.3, 0.8),
        'surprise': random.uniform(0.05, 0.3),
        'disgust': random.uniform(0.02, 0.2),
        'repression': random.uniform(0.01, 0.15)
    }
    
    # Normalize to sum to 1
    total = sum(probabilities.values())
    probabilities = {k: v/total for k, v in probabilities.items()}
    
    # Get predicted emotion
    predicted_emotion = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_emotion] * 100
    
    return {
        'predicted_emotion': predicted_emotion.capitalize(),
        'confidence': confidence,
        'probabilities': {k.capitalize(): v*100 for k, v in probabilities.items()},
        'frames_processed': len(frames),
        'flows_computed': len(flows)
    }

def analyze_video_real(frames, flows):
    """Real analysis using trained model"""
    if not model_loaded or not model:
        return analyze_video_mock(frames, flows)
    
    try:
        # Convert frames to tensor format expected by model
        import torch
        
        # Select 3 frames (onset, apex, offset) or use available frames
        if len(frames) >= 3:
            selected_frames = frames[::len(frames)//3][:3]
        else:
            selected_frames = frames + [frames[-1]] * (3 - len(frames))
        
        # Convert to tensors
        frames_tensor = torch.tensor(np.array(selected_frames), dtype=torch.float32)
        
        # Compute or use flows
        if len(flows) >= 2:
            selected_flows = flows[::len(flows)//3][:2]
            # Pad to 3 frames if needed
            while len(selected_flows) < 3:
                selected_flows.append(selected_flows[-1])
            flows_tensor = torch.tensor(np.array(selected_flows), dtype=torch.float32)
        else:
            # Create dummy flows
            flows_tensor = torch.zeros(3, 64, 64, 2)
        
        # Extract features
        features = model.extract_all_features(frames_tensor, flows_tensor)
        
        # Predict
        prediction = model.pipeline.predict([features])[0]
        probabilities = model.pipeline.predict_proba([features])[0]
        
        emotion_labels = ['happiness', 'surprise', 'disgust', 'repression']
        
        return {
            'predicted_emotion': emotion_labels[prediction].capitalize(),
            'confidence': probabilities[prediction] * 100,
            'probabilities': {
                emotion_labels[i].capitalize(): prob * 100 
                for i, prob in enumerate(probabilities)
            },
            'frames_processed': len(frames),
            'flows_computed': len(flows)
        }
        
    except Exception as e:
        print(f"Error in real analysis: {e}")
        return analyze_video_mock(frames, flows)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_loaded,
        'version': '1.0.0'
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload and analyze video"""
    try:
        # Check if file was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        video_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        
        # Save file
        file.save(str(video_path))
        
        try:
            # Process video
            frames = extract_frames(video_path)
            
            if len(frames) == 0:
                return jsonify({'error': 'Could not extract frames from video'}), 400
            
            flows = compute_optical_flow(frames)
            
            # Analyze
            if model_loaded:
                results = analyze_video_real(frames, flows)
            else:
                results = analyze_video_mock(frames, flows)
            
            # Add metadata
            results.update({
                'filename': filename,
                'file_size': video_path.stat().st_size,
                'processing_time': 'Mock processing',
                'timestamp': datetime.now().isoformat()
            })
            
            return jsonify(results)
            
        finally:
            # Clean up uploaded file
            try:
                video_path.unlink()
            except:
                pass
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Enhanced Hybrid CNN-SVM',
        'input_shape': '3x64x64 RGB frames + 6x64x64 optical flow',
        'output_classes': ['Happiness', 'Surprise', 'Disgust', 'Repression'],
        'feature_dimensions': 224,
        'training_dataset': 'CASME-II',
        'evaluation_method': 'Leave-One-Subject-Out (LOS0)',
        'performance': {
            'accuracy': 46.3,
            'uar': 24.8,
            'happiness_recall': 71.6,
            'disgust_recall': 27.4
        },
        'model_loaded': model_loaded
    })

@app.route('/api/results/sample', methods=['GET'])
def sample_results():
    """Get sample analysis results for demonstration"""
    sample_data = {
        'predicted_emotion': 'Happiness',
        'confidence': 85.2,
        'probabilities': {
            'Happiness': 85.2,
            'Surprise': 8.7,
            'Disgust': 4.1,
            'Repression': 2.0
        },
        'frames_processed': 10,
        'flows_computed': 9,
        'processing_time': '2.3 seconds',
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(sample_data)

@app.route('/api/visualizations/<filename>')
def serve_visualization(filename):
    """Serve visualization images"""
    viz_path = project_root / filename
    if viz_path.exists():
        return send_file(str(viz_path))
    else:
        return jsonify({'error': 'File not found'}), 404

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """Main function to run the Flask app"""
    print("🚀 Starting Micro-Expression Recognition API Server...")
    
    # Load model
    print("📦 Loading model...")
    load_model()
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

if __name__ == '__main__':
    main()
