#!/usr/bin/env python3
"""
Improved Flask Web Application for Micro-Expression Recognition
Enhanced with security, proper exception handling, and performance optimizations
"""

import os
import sys
import json
import uuid
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import threading

# Flask imports
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# Scientific computing imports
import cv2
import numpy as np
import torch

# Add project paths
_web_dir = Path(__file__).resolve().parent
project_root = _web_dir.parent
for _p in (project_root / "scripts", project_root, project_root / "src"):
    _s = str(_p.resolve())
    if _s not in sys.path:
        sys.path.insert(0, _s)

# Import improved modules
try:
    from security_utils import SecurityValidator, FileUploadManager, temporary_video_file
    from logging_config import get_logger, setup_logging, SecurityLogger, PerformanceLogger
    from web_config import get_config_manager, get_config
    from model_cache import get_model_manager, get_model
    from config import EMOTION_LABELS, LABEL_TO_EMOTION
    from inference_utils import hybrid_predict_from_features
    from casme_predict_bridge import get_clip_tensors, default_regimg_search_roots, find_labels_row_fuzzy
    from preprocessing_pipeline import VideoPreprocessor, OnsetApexOffsetSelector
    import pandas as pd
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    security_logger = SecurityLogger()
    perf_logger = PerformanceLogger()
    
    # Get configuration
    config_manager = get_config_manager()
    config = get_config()
    
    MODEL_AVAILABLE = True
    logger.info("All modules imported successfully")
    
except ImportError as e:
    MODEL_AVAILABLE = False
    # Fallback logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import modules: {e}")
    
    # Fallback configuration
    class SimpleConfig:
        def __init__(self):
            self.security = type('Security', (), {
                'max_content_length': 100 * 1024 * 1024,
                'allowed_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
                'upload_folder': tempfile.gettempdir(),
                'cors_origins': "*"
            })()
            self.model = type('Model', (), {
                'max_video_frames': 64,
                'inference_timeout': 30
            })()
    
    config = SimpleConfig()


class ImprovedFlaskApp:
    """Improved Flask application with security and performance optimizations"""
    
    def __init__(self):
        """Initialize the improved Flask application"""
        self.app = Flask(__name__, static_folder='.', static_url_path='')
        self.model_manager = get_model_manager() if MODEL_AVAILABLE else None
        self.file_manager = None
        self.video_preprocessor = None
        self.labels_df = None
        self.selector = None
        self.model = None
        self.model_loaded = False
        
        # Setup Flask configuration
        self._setup_flask_config()
        
        # Setup CORS
        self._setup_cors()
        
        # Setup routes
        self._setup_routes()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Improved Flask application initialized")
    
    def _setup_flask_config(self):
        """Setup Flask configuration"""
        self.app.config['MAX_CONTENT_LENGTH'] = config.security.max_content_length
        self.app.config['UPLOAD_FOLDER'] = config.security.upload_folder
        self.app.config['SECRET_KEY'] = config.security.secret_key
        
        logger.info("Flask configuration setup complete", extra={
            'max_content_length': config.security.max_content_length,
            'upload_folder': config.security.upload_folder
        })
    
    def _setup_cors(self):
        """Setup CORS configuration"""
        cors_origins = config.security.cors_origins
        if cors_origins == "*":
            CORS(self.app)
        else:
            origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
            if origins:
                CORS(self.app, origins=origins)
            else:
                CORS(self.app)
        
        logger.info("CORS configuration setup complete", extra={'cors_origins': cors_origins})
    
    def _setup_routes(self):
        """Setup application routes"""
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint with detailed status"""
            try:
                health_data = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'model_loaded': self.model_loaded,
                    'model_available': MODEL_AVAILABLE,
                    'version': '2.0.0',
                    'environment': config.environment.value if hasattr(config, 'environment') else 'unknown'
                }
                
                # Add cache info if available
                if self.model_manager:
                    cache_info = self.model_manager.get_cache_info()
                    health_data['cache_info'] = cache_info
                
                return jsonify(health_data)
                
            except Exception as e:
                logger.error(f"Health check failed: {e}", exc_info=True)
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/api/upload', methods=['POST'])
        def upload_video():
            """Secure video upload endpoint"""
            start_time = time.time()
            
            try:
                # Validate file presence
                if 'video' not in request.files:
                    security_logger.log_validation_failure('missing_file', 'none')
                    return jsonify({
                        'success': False,
                        'error': 'No video file provided'
                    }), 400
                
                file = request.files['video']
                if file.filename == '':
                    security_logger.log_validation_failure('empty_filename', 'none')
                    return jsonify({
                        'success': False,
                        'error': 'No file selected'
                    }), 400
                
                # Security validation
                try:
                    SecurityValidator.validate_file_upload(file, config.security.max_content_length)
                except ValueError as e:
                    security_logger.log_validation_failure('file_upload', str(e))
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 400
                
                # Save file securely
                try:
                    file_path = self.file_manager.save_upload(file)
                    security_logger.log_file_upload(
                        file.filename, 
                        file.content_length if hasattr(file, 'content_length') else 0,
                        'video/unknown'
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to save uploaded file: {e}", exc_info=True)
                    return jsonify({
                        'success': False,
                        'error': 'Failed to save file'
                    }), 500
                
                # Process video
                try:
                    result = self._process_video_file(file_path)
                    
                    # Log performance
                    processing_time = time.time() - start_time
                    perf_logger.log_execution_time('upload_video', processing_time, 
                                                 filename=file.filename, 
                                                 file_size=file.content_length if hasattr(file, 'content_length') else 0)
                    
                    return jsonify({
                        'success': True,
                        'result': result,
                        'processing_time': f"{processing_time:.2f}s"
                    })
                    
                except Exception as e:
                    logger.error(f"Video processing failed: {e}", exc_info=True)
                    return jsonify({
                        'success': False,
                        'error': f'Video processing failed: {str(e)}'
                    }), 500
                
                finally:
                    # Cleanup
                    self.file_manager.cleanup_file(file_path)
                
            except Exception as e:
                logger.error(f"Upload endpoint error: {e}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': f'Upload failed: {str(e)}'
                }), 500
        
        @self.app.route('/api/analyze-casme-episode', methods=['POST'])
        def analyze_casme_episode():
            """Analyze CASME-II episode with proper validation"""
            start_time = time.time()
            
            try:
                if not self.model_loaded or not self.model:
                    return jsonify({
                        'success': False,
                        'error': 'Model not loaded'
                    }), 503
                
                # Get and validate input
                data = request.get_json(silent=True) or {}
                subject = data.get("subject_id", "").strip()
                episode = data.get("episode_id", "").strip()
                
                # Input validation
                try:
                    SecurityValidator.validate_casme_subject(subject)
                    SecurityValidator.validate_casme_episode(episode)
                except ValueError as e:
                    security_logger.log_validation_failure('casme_format', f"{subject}/{episode}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 400
                
                # Analyze episode
                try:
                    result = self._analyze_casme_episode_secure(subject, episode)
                    
                    processing_time = time.time() - start_time
                    perf_logger.log_execution_time('analyze_casme_episode', processing_time,
                                                 subject=subject, episode=episode)
                    
                    result['processing_time'] = f"{processing_time:.2f}s"
                    return jsonify(result)
                    
                except FileNotFoundError as e:
                    logger.warning(f"CASME episode not found: {e}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 404
                    
                except ValueError as e:
                    logger.warning(f"Invalid CASME episode data: {e}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 400
                    
                except Exception as e:
                    logger.error(f"CASME episode analysis failed: {e}", exc_info=True)
                    return jsonify({
                        'success': False,
                        'error': f'Analysis failed: {str(e)}'
                    }), 500
                
            except Exception as e:
                logger.error(f"CASME analysis endpoint error: {e}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': f'Request failed: {str(e)}'
                }), 500
        
        @self.app.errorhandler(413)
        def too_large(e):
            """Handle file too large error"""
            security_logger.log_validation_failure('file_too_large', 'unknown')
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size: {config.security.max_content_length // (1024*1024)}MB'
            }), 413
        
        @self.app.errorhandler(404)
        def not_found(e):
            """Handle 404 errors"""
            return jsonify({
                'success': False,
                'error': 'Endpoint not found'
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(e):
            """Handle 500 errors"""
            logger.error(f"Internal server error: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Internal server error'
            }), 500
    
    def _initialize_components(self):
        """Initialize application components"""
        try:
            # Initialize file manager
            upload_dir = Path(config.security.upload_folder)
            self.file_manager = FileUploadManager(upload_dir)
            
            # Initialize video preprocessor
            if MODEL_AVAILABLE:
                self.video_preprocessor = VideoPreprocessor()
                
                # Load CASME-II labels
                labels_file = project_root / "data" / "labels" / "casme2_labels.csv"
                if labels_file.exists():
                    self.labels_df = pd.read_csv(labels_file)
                    self.selector = OnsetApexOffsetSelector(str(labels_file))
                    logger.info(f"CASME-II labels loaded: {len(self.labels_df)} samples")
                
                # Load model
                self._load_model()
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}", exc_info=True)
    
    def _load_model(self):
        """Load model with caching"""
        if not MODEL_AVAILABLE:
            return
        
        try:
            model_path = project_root / "models" / "real_data_model_20260414_085233.pkl"
            
            if model_path.exists():
                self.model = get_model(str(model_path))
                self.model_loaded = True
                logger.info(f"Model loaded successfully: {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
    
    def _process_video_file(self, file_path: Path) -> Dict[str, Any]:
        """Process video file with proper error handling"""
        if not self.model_loaded or not self.video_preprocessor:
            raise ValueError("Model or preprocessor not available")
        
        # Read and process video
        frames_tensor, flows_tensor = self.video_preprocessor.preprocess_video(
            str(file_path), 
            max_input_frames=config.model.max_video_frames
        )
        
        # Extract features and predict
        features = self.model.extract_all_features(frames_tensor, flows_tensor)
        features = np.asarray(features, dtype=np.float64).reshape(1, -1)
        
        prediction_result = hybrid_predict_from_features(self.model, features)
        
        return {
            'prediction': prediction_result['prediction_emotion'],
            'confidence': prediction_result['confidence'],
            'all_probabilities': prediction_result['by_emotion'],
            'preprocessing': 'video_preprocessor',
            'frame_info': {
                'frames_processed': frames_tensor.shape[0] if frames_tensor.dim() > 0 else 0,
                'faces_detected': 1  # Assumed for video processing
            },
            'model_info': {
                'model_type': 'enhanced_hybrid',
                'inference_mode': 'video'
            }
        }
    
    def _analyze_casme_episode_secure(self, subject: str, episode: str) -> Dict[str, Any]:
        """Secure CASME-II episode analysis"""
        if not self.model_loaded or self.labels_df is None or not self.selector:
            raise ValueError("Required components not available")
        
        # Get clip tensors with proper error handling
        try:
            frames_tensor, flows_tensor, source_tag = get_clip_tensors(
                subject=subject,
                filename_stem=episode,
                video_path=None,
                casme2_root=project_root / "data" / "casme2",
                labels_df=self.labels_df,
                selector=self.selector,
                video_pre=self.video_preprocessor,
                max_video_frames=config.model.max_video_frames,
                extra_regimg_roots=default_regimg_search_roots(project_root)
            )
        except Exception as e:
            logger.error(f"Failed to get clip tensors: {e}", exc_info=True)
            raise FileNotFoundError(f"Could not load episode {subject}/{episode}: {str(e)}")
        
        # Extract features and predict
        try:
            features = self.model.extract_all_features(frames_tensor, flows_tensor)
            features = np.asarray(features, dtype=np.float64).reshape(1, -1)
            
            prediction_result = hybrid_predict_from_features(self.model, features)
            
            return {
                'success': True,
                'prediction': prediction_result['prediction_emotion'],
                'confidence': prediction_result['confidence'],
                'all_probabilities': prediction_result['by_emotion'],
                'source_tag': source_tag,
                'subject': subject,
                'episode': episode,
                'preprocessing': source_tag,
                'frame_info': {
                    'frames_processed': frames_tensor.shape[0] if frames_tensor.dim() > 0 else 0,
                    'faces_detected': 1
                },
                'model_info': {
                    'model_type': 'enhanced_hybrid',
                    'inference_mode': 'casme_episode'
                }
            }
            
        except Exception as e:
            logger.error(f"Feature extraction or prediction failed: {e}", exc_info=True)
            raise ValueError(f"Analysis failed: {str(e)}")
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the Flask application"""
        host = host or config.host
        port = port or config.port
        debug = debug if debug is not None else config.debug
        
        logger.info(f"Starting Flask application", extra={
            'host': host,
            'port': port,
            'debug': debug
        })
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Failed to start application: {e}", exc_info=True)
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Cleanup resources"""
        try:
            if self.file_manager:
                self.file_manager.cleanup_all()
            
            if self.model_manager:
                self.model_manager.shutdown()
            
            logger.info("Application shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


def create_app() -> Flask:
    """Create Flask application instance"""
    app_instance = ImprovedFlaskApp()
    return app_instance.app


def main():
    """Main entry point"""
    try:
        # Create and run application
        app_instance = ImprovedFlaskApp()
        app_instance.run()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
