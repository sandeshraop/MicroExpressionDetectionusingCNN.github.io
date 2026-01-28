#!/usr/bin/env python3
"""
Enhanced Flask Web Application with Preprocessing
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'inference'))

from enhanced_inference_pipeline import EnhancedMicroExpressionInferencePipeline
from preprocessing_pipeline import VideoPreprocessor

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global variables
pipeline = None
model_loaded = False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_pipeline():
    """Initialize the enhanced inference pipeline"""
    global pipeline, model_loaded
    
    try:
        # Try to load the augmented model first
        model_path = Path(__file__).parent.parent / 'models' / 'augmented_balanced_au_aligned_svm_20260127_162621.pkl'
        
        if model_path.exists():
            pipeline = EnhancedMicroExpressionInferencePipeline(str(model_path))
            model_loaded = True
            print("✅ Enhanced pipeline with augmented model loaded!")
        else:
            print("⚠️  Augmented model not found, pipeline not initialized")
            
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        model_loaded = False

@app.route('/')
def index():
    """Main page"""
    return render_template('enhanced_index.html', modelLoaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video upload and prediction"""
    global pipeline, model_loaded
    
    if not model_loaded or pipeline is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server configuration.'
        }), 500
    
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Save uploaded file
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"📹 Video uploaded: {filename}")
        
        # Process video with enhanced pipeline
        result = pipeline.predict_emotion(filepath)
        
        if result['success']:
            # Save inference result
            result_data = {
                'filename': filename,
                'original_filename': file.filename,
                'timestamp': result['timestamp'],
                'prediction': result['predicted_emotion'],
                'confidence': result['relative_probability'],
                'all_probabilities': result['all_probabilities'],
                'au_contribution': result['au_contribution'],
                'preprocessing': result['frame_info']['preprocessing']
            }
            
            # Save to results file
            results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'results.json')
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                results = []
            
            results.append(result_data)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return jsonify({
                'success': True,
                'prediction': result['predicted_emotion'],
                'confidence': result['relative_probability'],
                'all_probabilities': result['all_probabilities'],
                'au_contribution': result['au_contribution'],
                'preprocessing': result['frame_info']['preprocessing'],
                'frame_info': result['frame_info'],
                'timestamp': result['timestamp']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500
            
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing video: {str(e)}'
        }), 500

@app.route('/results')
def get_results():
    """Get all inference results"""
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], 'results.json')
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return jsonify({'success': True, 'results': results})
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({'success': True, 'results': []})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'preprocessing': 'face_detection' if pipeline and pipeline.preprocessor.face_detection_enabled else 'center_crop',
        'device': str(pipeline.device) if pipeline else 'unknown'
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize pipeline
    initialize_pipeline()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5005)
