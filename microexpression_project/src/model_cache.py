#!/usr/bin/env python3
"""
Model caching system for micro-expression recognition
Provides efficient model loading with memory management and thread safety
"""

import threading
import time
import weakref
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import gc

from logging_config import get_logger
from inference_utils import load_enhanced_hybrid_from_path, try_load_enhanced_hybrid_raw

logger = get_logger(__name__)


class ModelCache:
    """Thread-safe model cache with memory management"""
    
    def __init__(self, max_size: int = 1, cleanup_interval: int = 300):
        """
        Initialize model cache
        
        Args:
            max_size: Maximum number of models to cache
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[str, Any] = {}
        self._cache_times: Dict[str, float] = {}
        self._cache_lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"ModelCache initialized", extra={
            'max_size': max_size,
            'cleanup_interval': cleanup_interval
        })
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="ModelCache-Cleanup"
        )
        self._cleanup_thread.start()
    
    def _cleanup_worker(self) -> None:
        """Background cleanup worker"""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            self._cleanup_expired()
    
    def _cleanup_expired(self) -> None:
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        with self._cache_lock:
            for key, cache_time in self._cache_times.items():
                # Remove entries older than 1 hour
                if current_time - cache_time > 3600:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_model(key)
        
        if expired_keys:
            logger.info(f"Cleaned up expired models", extra={
                'expired_count': len(expired_keys),
                'expired_keys': expired_keys
            })
    
    def _remove_model(self, key: str) -> None:
        """Remove model from cache"""
        if key in self._cache:
            model = self._cache.pop(key)
            self._cache_times.pop(key, None)
            
            # Force garbage collection
            del model
            gc.collect()
            
            logger.debug(f"Model removed from cache", extra={'key': key})
    
    def get_model(self, model_path: str, loader_func: Optional[Callable] = None) -> Any:
        """
        Get model from cache or load it
        
        Args:
            model_path: Path to model file
            loader_func: Optional custom loader function
            
        Returns:
            Loaded model
        """
        # Normalize path
        model_path = str(Path(model_path).resolve())
        
        with self._cache_lock:
            # Check if model is in cache
            if model_path in self._cache:
                self._cache_times[model_path] = time.time()
                logger.debug(f"Model retrieved from cache", extra={'model_path': model_path})
                return self._cache[model_path]
            
            # Check if cache is full
            if len(self._cache) >= self.max_size:
                # Remove oldest model
                oldest_key = min(self._cache_times.keys(), key=self._cache_times.get)
                self._remove_model(oldest_key)
                logger.info(f"Removed oldest model to make space", extra={'removed_key': oldest_key})
        
        # Load model outside of lock
        try:
            if loader_func:
                model = loader_func(model_path)
            else:
                model = load_enhanced_hybrid_from_path(model_path)
            
            # Add to cache
            with self._cache_lock:
                self._cache[model_path] = model
                self._cache_times[model_path] = time.time()
            
            logger.info(f"Model loaded and cached", extra={'model_path': model_path})
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model", extra={
                'model_path': model_path,
                'error': str(e)
            })
            raise
    
    def clear_cache(self) -> None:
        """Clear all cached models"""
        with self._cache_lock:
            keys = list(self._cache.keys())
            for key in keys:
                self._remove_model(key)
        
        logger.info("Model cache cleared", extra={'cleared_count': len(keys)})
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        with self._cache_lock:
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'cached_models': list(self._cache.keys()),
                'cache_times': self._cache_times.copy()
            }
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup thread"""
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        self.clear_cache()
        logger.info("ModelCache shutdown complete")


class ModelManager:
    """Singleton model manager with caching and thread safety"""
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ModelManager':
        """Singleton pattern with thread safety"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model manager"""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.cache = ModelCache()
            self.loaders: Dict[str, Callable] = {}
            self.loading_locks: Dict[str, threading.Lock] = {}
            self._lock = threading.Lock()
            
            logger.info("ModelManager initialized")
    
    def register_loader(self, model_type: str, loader_func: Callable) -> None:
        """
        Register a custom model loader
        
        Args:
            model_type: Type identifier for the model
            loader_func: Loader function
        """
        with self._lock:
            self.loaders[model_type] = loader_func
            logger.info(f"Model loader registered", extra={'model_type': model_type})
    
    def get_model(self, model_path: str, model_type: str = "default") -> Any:
        """
        Get model with caching and thread safety
        
        Args:
            model_path: Path to model file
            model_type: Type of model for custom loader
            
        Returns:
            Loaded model
        """
        # Get or create loading lock for this model
        with self._lock:
            if model_path not in self.loading_locks:
                self.loading_locks[model_path] = threading.Lock()
            loading_lock = self.loading_locks[model_path]
        
        # Use loading lock to prevent concurrent loading of same model
        with loading_lock:
            try:
                # Get loader function
                loader_func = self.loaders.get(model_type)
                
                # Get from cache
                model = self.cache.get_model(model_path, loader_func)
                
                return model
                
            except Exception as e:
                logger.error(f"Failed to get model", extra={
                    'model_path': model_path,
                    'model_type': model_type,
                    'error': str(e)
                })
                raise
    
    def preload_model(self, model_path: str, model_type: str = "default") -> bool:
        """
        Preload model into cache
        
        Args:
            model_path: Path to model file
            model_type: Type of model
            
        Returns:
            True if successful
        """
        try:
            self.get_model(model_path, model_type)
            logger.info(f"Model preloaded", extra={'model_path': model_path})
            return True
        except Exception as e:
            logger.error(f"Failed to preload model", extra={
                'model_path': model_path,
                'error': str(e)
            })
            return False
    
    def clear_cache(self) -> None:
        """Clear model cache"""
        self.cache.clear_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return self.cache.get_cache_info()
    
    def shutdown(self) -> None:
        """Shutdown model manager"""
        self.cache.shutdown()
        
        # Clean up loading locks
        with self._lock:
            self.loading_locks.clear()
        
        logger.info("ModelManager shutdown complete")


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def get_model(model_path: str, model_type: str = "default") -> Any:
    """
    Get model with caching
    
    Args:
        model_path: Path to model file
        model_type: Type of model
        
    Returns:
        Loaded model
    """
    return get_model_manager().get_model(model_path, model_type)


def preload_model(model_path: str, model_type: str = "default") -> bool:
    """
    Preload model into cache
    
    Args:
        model_path: Path to model file
        model_type: Type of model
        
    Returns:
        True if successful
    """
    return get_model_manager().preload_model(model_path, model_type)


def test_model_cache():
    """Test model caching system"""
    print("🧪 Testing Model Cache...")
    
    try:
        # Create model manager
        manager = ModelManager()
        
        # Test cache info
        info = manager.get_cache_info()
        print(f"✅ Cache info: {info}")
        
        # Test with dummy model path (will fail but test the system)
        try:
            model = manager.get_model("nonexistent_model.pkl")
        except Exception as e:
            print(f"✅ Error handling working: {e}")
        
        # Clear cache
        manager.clear_cache()
        print("✅ Cache cleared")
        
        # Shutdown
        manager.shutdown()
        print("✅ Model cache test complete!")
        
    except Exception as e:
        print(f"❌ Model cache test failed: {e}")


if __name__ == "__main__":
    test_model_cache()
