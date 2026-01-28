#!/usr/bin/env python3
"""
Production-ready API server for micro-expression recognition
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import redis
import uuid

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'inference'))

from enhanced_inference_pipeline import EnhancedMicroExpressionInferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = '/app/uploads'
app.config['RESULTS_FOLDER'] = '/app/results'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variables
pipeline = None
redis_client = None

def initialize_services():
    """Initialize pipeline and Redis connection"""
    global pipeline, redis_client
    
    try:
        # Initialize Redis
        redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed: {e}")
        redis_client = None
    
    try:
        # Initialize pipeline
        model_path = '/app/models/augmented_balanced_au_aligned_svm_20260127_162621.pkl'
        if Path(model_path).exists():
            pipeline = EnhancedMicroExpressionInferencePipeline(model_path)
            logger.info("✅ Enhanced pipeline initialized")
        else:
            logger.error(f"❌ Model not found: {model_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Pipeline initialization failed: {e}")
        return False
    
    return True

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'pipeline': pipeline is not None,
            'redis': redis_client is not None,
            'model_loaded': pipeline is not None
        }
    }
    
    if pipeline:
        status['preprocessing'] = 'face_detection' if pipeline.preprocessor.face_detection_enabled else 'center_crop'
        status['device'] = str(pipeline.device)
    
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion from uploaded video"""
    if not pipeline:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{request_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        logger.info(f"📹 Video uploaded: {saved_filename}")
        
        # Process video
        start_time = datetime.now()
        result = pipeline.predict_emotion(filepath)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result['success']:
            response_data = {
                'request_id': request_id,
                'prediction': result['predicted_emotion'],
                'confidence': result['relative_probability'],
                'all_probabilities': result['all_probabilities'],
                'au_contribution': result['au_contribution'],
                'preprocessing': result['frame_info']['preprocessing'],
                'processing_time': processing_time,
                'timestamp': result['timestamp']
            }
            
            # Cache result in Redis
            if redis_client:
                redis_client.setex(
                    f"prediction:{request_id}",
                    3600,  # 1 hour expiry
                    json.dumps(response_data)
                )
            
            # Save detailed result
            result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{request_id}.json")
            with open(result_file, 'w') as f:
                json.dump({
                    'request_id': request_id,
                    'filename': saved_filename,
                    'original_filename': filename,
                    'result': response_data,
                    'processing_time': processing_time
                }, f, indent=2)
            
            logger.info(f"✅ Prediction completed: {result['predicted_emotion']} ({result['relative_probability']:.3f})")
            return jsonify(response_data)
        else:
            logger.error(f"❌ Prediction failed: {result['error']}")
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"❌ Error processing video: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<request_id>', methods=['GET'])
def get_prediction(request_id):
    """Get prediction result by request ID"""
    if redis_client:
        cached_result = redis_client.get(f"prediction:{request_id}")
        if cached_result:
            return jsonify(json.loads(cached_result))
    
    # Check file-based storage
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{request_id}.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
        return jsonify(data['result'])
    
    return jsonify({'error': 'Prediction not found'}), 404

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    stats = {
        'service': 'micro-expression-api',
        'version': '1.0.0',
        'uptime': datetime.now().isoformat(),
        'model': 'augmented_balanced_au_aligned_svm_20260127_162621',
        'preprocessing': 'face_detection' if pipeline and pipeline.preprocessor.face_detection_enabled else 'center_crop',
        'device': str(pipeline.device) if pipeline else 'unknown'
    }
    
    # Redis stats
    if redis_client:
        try:
            redis_info = redis_client.info()
            stats['redis'] = {
                'connected_clients': redis_info.get('connected_clients', 0),
                'used_memory': redis_info.get('used_memory_human', 'unknown'),
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0)
            }
        except Exception as e:
            stats['redis'] = {'error': str(e)}
    
    return jsonify(stats)

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    models_dir = Path('/app/models')
    models = []
    
    if models_dir.exists():
        for model_file in models_dir.glob('*.pkl'):
            models.append({
                'name': model_file.name,
                'size': model_file.stat().st_size,
                'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            })
    
    return jsonify({'models': models})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 100MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Micro-Expression Recognition API Server")
    
    if initialize_services():
        logger.info("✅ Services initialized successfully")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("❌ Failed to initialize services")
        sys.exit(1)
