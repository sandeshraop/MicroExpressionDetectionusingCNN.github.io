#!/usr/bin/env python3
"""
Performance monitoring system for micro-expression recognition
Tracks system resources, request metrics, and model performance
"""

import time
import threading
import psutil
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json

from logging_config import get_logger, PerformanceLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    timestamp: datetime
    endpoint: str
    method: str
    duration: float
    status_code: int
    memory_usage_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    active_threads: int
    open_files: int


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    timestamp: datetime
    model_name: str
    inference_time: float
    feature_extraction_time: float
    preprocessing_time: float
    total_time: float
    cache_hit: bool
    input_shape: tuple
    prediction_confidence: float


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, 
                 max_requests: int = 1000,
                 max_system_metrics: int = 100,
                 max_model_metrics: int = 100,
                 monitoring_interval: int = 30):
        """
        Initialize performance monitor
        
        Args:
            max_requests: Maximum number of request metrics to keep
            max_system_metrics: Maximum number of system metrics to keep
            max_model_metrics: Maximum number of model metrics to keep
            monitoring_interval: System monitoring interval in seconds
        """
        self.max_requests = max_requests
        self.max_system_metrics = max_system_metrics
        self.max_model_metrics = max_model_metrics
        self.monitoring_interval = monitoring_interval
        
        # Metrics storage
        self.request_metrics: deque = deque(maxlen=max_requests)
        self.system_metrics: deque = deque(maxlen=max_system_metrics)
        self.model_metrics: deque = deque(maxlen=max_model_metrics)
        
        # Aggregated statistics
        self.endpoint_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_duration': 0.0,
            'avg_duration': 0.0,
            'min_duration': float('inf'),
            'max_duration': 0.0,
            'last_request': None
        })
        
        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("Performance monitor initialized", extra={
            'max_requests': max_requests,
            'monitoring_interval': monitoring_interval
        })
    
    def _start_monitoring(self) -> None:
        """Start background monitoring thread"""
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitoring_thread.start()
    
    def _monitoring_worker(self) -> None:
        """Background monitoring worker"""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
    
    def _collect_system_metrics(self) -> None:
        """Collect system resource metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Thread and file counts
            active_threads = threading.active_count()
            try:
                open_files = len(psutil.Process().open_files())
            except (psutil.AccessDenied, AttributeError):
                open_files = 0
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                active_threads=active_threads,
                open_files=open_files
            )
            
            with self._lock:
                self.system_metrics.append(metrics)
            
            # Log warnings for high resource usage
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
            
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
            if disk.percent > 90:
                logger.warning(f"High disk usage: {disk.percent:.1f}%")
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def log_request(self, 
                   endpoint: str,
                   method: str,
                   duration: float,
                   status_code: int,
                   success: bool,
                   error_message: Optional[str] = None) -> None:
        """
        Log request metrics
        
        Args:
            endpoint: Request endpoint
            method: HTTP method
            duration: Request duration in seconds
            status_code: HTTP status code
            success: Whether request was successful
            error_message: Error message if failed
        """
        try:
            # Get current resource usage
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            metrics = RequestMetrics(
                timestamp=datetime.now(),
                endpoint=endpoint,
                method=method,
                duration=duration,
                status_code=status_code,
                memory_usage_mb=memory.used / (1024 * 1024),
                cpu_percent=cpu_percent,
                success=success,
                error_message=error_message
            )
            
            with self._lock:
                self.request_metrics.append(metrics)
                
                # Update endpoint statistics
                stats = self.endpoint_stats[endpoint]
                stats['total_requests'] += 1
                stats['last_request'] = metrics.timestamp.isoformat()
                
                if success:
                    stats['successful_requests'] += 1
                else:
                    stats['failed_requests'] += 1
                
                stats['total_duration'] += duration
                stats['avg_duration'] = stats['total_duration'] / stats['total_requests']
                stats['min_duration'] = min(stats['min_duration'], duration)
                stats['max_duration'] = max(stats['max_duration'], duration)
            
            # Log performance
            perf_logger.log_execution_time(f"{method} {endpoint}", duration,
                                         status_code=status_code, success=success)
            
        except Exception as e:
            logger.error(f"Failed to log request metrics: {e}")
    
    def log_model_inference(self,
                           model_name: str,
                           inference_time: float,
                           feature_extraction_time: float,
                           preprocessing_time: float,
                           cache_hit: bool,
                           input_shape: tuple,
                           prediction_confidence: float) -> None:
        """
        Log model inference metrics
        
        Args:
            model_name: Name of the model
            inference_time: Time spent on inference
            feature_extraction_time: Time spent on feature extraction
            preprocessing_time: Time spent on preprocessing
            cache_hit: Whether model was loaded from cache
            input_shape: Shape of input tensor
            prediction_confidence: Prediction confidence score
        """
        try:
            metrics = ModelMetrics(
                timestamp=datetime.now(),
                model_name=model_name,
                inference_time=inference_time,
                feature_extraction_time=feature_extraction_time,
                preprocessing_time=preprocessing_time,
                total_time=inference_time + feature_extraction_time + preprocessing_time,
                cache_hit=cache_hit,
                input_shape=input_shape,
                prediction_confidence=prediction_confidence
            )
            
            with self._lock:
                self.model_metrics.append(metrics)
            
            # Log performance
            perf_logger.log_execution_time(f"model_inference_{model_name}", metrics.total_time,
                                         cache_hit=cache_hit, confidence=prediction_confidence)
            
        except Exception as e:
            logger.error(f"Failed to log model metrics: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            # Request statistics
            total_requests = len(self.request_metrics)
            successful_requests = sum(1 for r in self.request_metrics if r.success)
            failed_requests = total_requests - successful_requests
            
            # Recent requests (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_requests = [r for r in self.request_metrics if r.timestamp > one_hour_ago]
            
            # Average response times
            avg_response_time = 0.0
            if recent_requests:
                avg_response_time = sum(r.duration for r in recent_requests) / len(recent_requests)
            
            # System metrics (latest)
            latest_system = self.system_metrics[-1] if self.system_metrics else None
            
            # Model metrics (latest)
            latest_model = self.model_metrics[-1] if self.model_metrics else None
            
            return {
                'timestamp': datetime.now().isoformat(),
                'request_stats': {
                    'total_requests': total_requests,
                    'successful_requests': successful_requests,
                    'failed_requests': failed_requests,
                    'success_rate': successful_requests / total_requests if total_requests > 0 else 0.0,
                    'recent_requests_1h': len(recent_requests),
                    'avg_response_time_recent': avg_response_time,
                    'endpoint_stats': dict(self.endpoint_stats)
                },
                'system_stats': {
                    'cpu_percent': latest_system.cpu_percent if latest_system else 0.0,
                    'memory_percent': latest_system.memory_percent if latest_system else 0.0,
                    'memory_used_mb': latest_system.memory_used_mb if latest_system else 0.0,
                    'disk_usage_percent': latest_system.disk_usage_percent if latest_system else 0.0,
                    'active_threads': latest_system.active_threads if latest_system else 0,
                    'open_files': latest_system.open_files if latest_system else 0
                } if latest_system else {},
                'model_stats': {
                    'last_inference_time': latest_model.total_time if latest_model else 0.0,
                    'last_cache_hit': latest_model.cache_hit if latest_model else False,
                    'last_prediction_confidence': latest_model.prediction_confidence if latest_model else 0.0,
                    'total_inferences': len(self.model_metrics)
                } if latest_model else {}
            }
    
    def get_detailed_metrics(self, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> Dict[str, List[Dict]]:
        """
        Get detailed metrics within time range
        
        Args:
            start_time: Start time for metrics
            end_time: End time for metrics
            
        Returns:
            Dictionary with detailed metrics
        """
        with self._lock:
            def filter_metrics(metrics, start_time, end_time):
                filtered = list(metrics)
                if start_time:
                    filtered = [m for m in filtered if m.timestamp >= start_time]
                if end_time:
                    filtered = [m for m in filtered if m.timestamp <= end_time]
                return [asdict(m) for m in filtered]
            
            return {
                'request_metrics': filter_metrics(self.request_metrics, start_time, end_time),
                'system_metrics': filter_metrics(self.system_metrics, start_time, end_time),
                'model_metrics': filter_metrics(self.model_metrics, start_time, end_time)
            }
    
    def export_metrics(self, file_path: Path) -> None:
        """
        Export metrics to file
        
        Args:
            file_path: Path to export file
        """
        try:
            metrics_data = {
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_summary(),
                'detailed_metrics': self.get_detailed_metrics()
            }
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics"""
        with self._lock:
            self.request_metrics.clear()
            self.system_metrics.clear()
            self.model_metrics.clear()
            self.endpoint_stats.clear()
        
        logger.info("Performance metrics cleared")
    
    def shutdown(self) -> None:
        """Shutdown performance monitor"""
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Performance monitor shutdown complete")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def log_request_metrics(endpoint: str, method: str, duration: float, 
                       status_code: int, success: bool, 
                       error_message: Optional[str] = None) -> None:
    """Log request metrics using global monitor"""
    get_performance_monitor().log_request(
        endpoint, method, duration, status_code, success, error_message
    )


def log_model_metrics(model_name: str, inference_time: float,
                     feature_extraction_time: float, preprocessing_time: float,
                     cache_hit: bool, input_shape: tuple,
                     prediction_confidence: float) -> None:
    """Log model metrics using global monitor"""
    get_performance_monitor().log_model_inference(
        model_name, inference_time, feature_extraction_time,
        preprocessing_time, cache_hit, input_shape, prediction_confidence
    )


def test_performance_monitor():
    """Test performance monitoring system"""
    print("🧪 Testing Performance Monitor...")
    
    try:
        # Create monitor
        monitor = PerformanceMonitor(monitoring_interval=1)
        
        # Log some test requests
        monitor.log_request("/api/test", "POST", 0.123, 200, True)
        monitor.log_request("/api/test", "POST", 0.456, 500, False, "Test error")
        
        # Log model metrics
        monitor.log_model_inference(
            "test_model", 0.1, 0.05, 0.02, True, (1, 3, 64, 64), 0.85
        )
        
        # Wait for system metrics
        time.sleep(2)
        
        # Get summary
        summary = monitor.get_summary()
        print(f"✅ Performance summary generated")
        print(f"   Total requests: {summary['request_stats']['total_requests']}")
        print(f"   Success rate: {summary['request_stats']['success_rate']:.2%}")
        
        # Export metrics
        export_path = Path("test_metrics.json")
        monitor.export_metrics(export_path)
        print(f"✅ Metrics exported to: {export_path}")
        
        # Cleanup
        monitor.shutdown()
        export_path.unlink(missing_ok=True)
        
        print("🎉 Performance monitor test complete!")
        
    except Exception as e:
        print(f"❌ Performance monitor test failed: {e}")


if __name__ == "__main__":
    test_performance_monitor()
