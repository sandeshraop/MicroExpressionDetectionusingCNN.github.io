#!/usr/bin/env python3
"""
Configuration management for micro-expression recognition web application
Centralized configuration with environment variable support and validation
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from logging_config import get_logger

logger = get_logger(__name__)


class Environment(Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    max_content_length: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = field(default_factory=lambda: ['.mp4', '.avi', '.mov', '.mkv', '.webm'])
    allowed_mime_types: List[str] = field(default_factory=lambda: [
        'video/mp4', 'video/avi', 'video/quicktime', 
        'video/x-msvideo', 'video/webm', 'video/x-matroska'
    ])
    upload_folder: str = field(default_factory=lambda: os.path.join(tempfile.gettempdir(), 'uploads'))
    secret_key: str = field(default_factory=lambda: os.urandom(32).hex())
    session_timeout: int = 3600  # 1 hour
    cors_origins: str = "*"  # Comma-separated list or "*"
    
    def __post_init__(self):
        """Validate security configuration"""
        if self.max_content_length <= 0:
            raise ValueError("max_content_length must be positive")
        
        if not self.allowed_extensions:
            raise ValueError("allowed_extensions cannot be empty")
        
        if not self.secret_key:
            raise ValueError("secret_key cannot be empty")


@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_path: Optional[str] = None
    facesleuth_model_path: Optional[str] = None
    model_cache_size: int = 1
    inference_timeout: int = 30  # seconds
    max_video_frames: int = 64
    optical_flow_method: str = "farneback"
    facesleuth_vertical_alpha: float = 1.5
    facesleuth_boosting_lambda: float = 0.3
    uncertainty_threshold: float = 0.6
    
    def __post_init__(self):
        """Validate model configuration"""
        if self.model_cache_size <= 0:
            raise ValueError("model_cache_size must be positive")
        
        if self.inference_timeout <= 0:
            raise ValueError("inference_timeout must be positive")
        
        if self.max_video_frames < 1:
            raise ValueError("max_video_frames must be at least 1")
        
        if not 0.0 <= self.facesleuth_vertical_alpha <= 10.0:
            raise ValueError("facesleuth_vertical_alpha must be between 0.0 and 10.0")
        
        if not 0.0 <= self.facesleuth_boosting_lambda <= 1.0:
            raise ValueError("facesleuth_boosting_lambda must be between 0.0 and 1.0")
        
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be between 0.0 and 1.0")


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    console_output: bool = True
    structured_output: bool = False
    log_dir: str = "logs"
    
    def __post_init__(self):
        """Validate logging configuration"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")


@dataclass
class PerformanceConfig:
    """Performance configuration settings"""
    max_concurrent_requests: int = 10
    request_timeout: int = 60  # seconds
    cleanup_interval: int = 300  # seconds (5 minutes)
    memory_limit_mb: int = 1024  # 1GB
    
    def __post_init__(self):
        """Validate performance configuration"""
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        
        if self.cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be positive")
        
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")


@dataclass
class WebConfig:
    """Main web application configuration"""
    environment: Environment = Environment.DEVELOPMENT
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    # Sub-configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    def __post_init__(self):
        """Validate main configuration"""
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        
        if self.ssl_enabled and (not self.ssl_cert_path or not self.ssl_key_path):
            raise ValueError("ssl_cert_path and ssl_key_path required when ssl_enabled is True")


class ConfigManager:
    """Manages application configuration with environment variable support"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional configuration file path
        """
        self.config_file = config_file
        self.config = self._load_configuration()
        logger.info("Configuration loaded", extra={
            'environment': self.config.environment.value,
            'debug': self.config.debug,
            'config_file': str(config_file) if config_file else None
        })
    
    def _load_configuration(self) -> WebConfig:
        """Load configuration from environment variables and file"""
        # Start with default configuration
        config = WebConfig()
        
        # Load from environment variables
        config = self._load_from_environment(config)
        
        # Load from file if provided
        if self.config_file and self.config_file.exists():
            config = self._load_from_file(config)
        
        # Validate configuration
        self._validate_configuration(config)
        
        return config
    
    def _load_from_environment(self, config: WebConfig) -> WebConfig:
        """Load configuration from environment variables"""
        
        # Main configuration
        config.environment = Environment(os.environ.get('FLASK_ENV', 'development'))
        config.host = os.environ.get('FLASK_HOST', config.host)
        config.port = int(os.environ.get('FLASK_PORT', config.port))
        config.debug = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
        
        # Security configuration
        config.security.max_content_length = int(os.environ.get('MAX_CONTENT_LENGTH', config.security.max_content_length))
        config.security.secret_key = os.environ.get('SECRET_KEY', config.security.secret_key)
        config.security.cors_origins = os.environ.get('CORS_ORIGINS', config.security.cors_origins)
        config.security.upload_folder = os.environ.get('UPLOAD_FOLDER', config.security.upload_folder)
        
        # Model configuration
        config.model.model_path = os.environ.get('MODEL_PATH')
        config.model.facesleuth_model_path = os.environ.get('FACESELEUTH_MODEL_PATH')
        config.model.max_video_frames = int(os.environ.get('MAX_VIDEO_FRAMES', config.model.max_video_frames))
        config.model.optical_flow_method = os.environ.get('OPTICAL_FLOW_METHOD', config.model.optical_flow_method)
        
        # Logging configuration
        config.logging.log_level = os.environ.get('LOG_LEVEL', config.logging.log_level)
        config.logging.log_file = os.environ.get('LOG_FILE')
        config.logging.console_output = os.environ.get('CONSOLE_OUTPUT', 'true').lower() in ('true', '1', 'yes')
        
        # Performance configuration
        config.performance.max_concurrent_requests = int(os.environ.get('MAX_CONCURRENT_REQUESTS', config.performance.max_concurrent_requests))
        config.performance.request_timeout = int(os.environ.get('REQUEST_TIMEOUT', config.performance.request_timeout))
        
        return config
    
    def _load_from_file(self, config: WebConfig) -> WebConfig:
        """Load configuration from JSON file"""
        try:
            import json
            
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with file values
            if 'environment' in config_data:
                config.environment = Environment(config_data['environment'])
            
            if 'host' in config_data:
                config.host = config_data['host']
            
            if 'port' in config_data:
                config.port = config_data['port']
            
            if 'debug' in config_data:
                config.debug = config_data['debug']
            
            # Update security config
            if 'security' in config_data:
                for key, value in config_data['security'].items():
                    if hasattr(config.security, key):
                        setattr(config.security, key, value)
            
            # Update model config
            if 'model' in config_data:
                for key, value in config_data['model'].items():
                    if hasattr(config.model, key):
                        setattr(config.model, key, value)
            
            logger.info(f"Configuration loaded from file: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from file: {e}")
            raise
        
        return config
    
    def _validate_configuration(self, config: WebConfig) -> None:
        """Validate configuration"""
        try:
            # This will trigger __post_init__ validation
            config.__post_init__()
            config.security.__post_init__()
            config.model.__post_init__()
            config.logging.__post_init__()
            config.performance.__post_init__()
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_config(self) -> WebConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Re-validate after update
        self._validate_configuration(self.config)
    
    def save_config(self, file_path: Path) -> None:
        """Save current configuration to file"""
        try:
            import json
            
            config_dict = {
                'environment': self.config.environment.value,
                'host': self.config.host,
                'port': self.config.port,
                'debug': self.config.debug,
                'security': {
                    'max_content_length': self.config.security.max_content_length,
                    'allowed_extensions': self.config.security.allowed_extensions,
                    'upload_folder': self.config.security.upload_folder,
                    'cors_origins': self.config.security.cors_origins,
                },
                'model': {
                    'model_path': self.config.model.model_path,
                    'max_video_frames': self.config.model.max_video_frames,
                    'optical_flow_method': self.config.model.optical_flow_method,
                    'facesleuth_vertical_alpha': self.config.model.facesleuth_vertical_alpha,
                    'facesleuth_boosting_lambda': self.config.model.facesleuth_boosting_lambda,
                },
                'logging': {
                    'log_level': self.config.logging.log_level,
                    'log_file': self.config.logging.log_file,
                    'console_output': self.config.logging.console_output,
                },
                'performance': {
                    'max_concurrent_requests': self.config.performance.max_concurrent_requests,
                    'request_timeout': self.config.performance.request_timeout,
                }
            }
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        config_file = Path(__file__).parent.parent / "config" / "web_config.json"
        _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config() -> WebConfig:
    """Get current web configuration"""
    return get_config_manager().get_config()


def test_configuration():
    """Test configuration management"""
    print("🧪 Testing Configuration Management...")
    
    try:
        # Test configuration loading
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"✅ Environment: {config.environment.value}")
        print(f"✅ Host: {config.host}")
        print(f"✅ Port: {config.port}")
        print(f"✅ Debug: {config.debug}")
        print(f"✅ Max content length: {config.security.max_content_length}")
        print(f"✅ Model cache size: {config.model.model_cache_size}")
        
        # Test configuration validation
        try:
            invalid_config = WebConfig(port=70000)
            print("❌ Configuration validation failed")
        except ValueError:
            print("✅ Configuration validation working")
        
        print("🎉 Configuration test complete!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")


if __name__ == "__main__":
    test_configuration()
