#!/usr/bin/env python3
"""
Structured logging configuration for micro-expression recognition system
Provides consistent logging across all modules with proper error tracking
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum


class LogLevel(Enum):
    """Log levels for the application"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for consistent log output"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data['extra'] = log_data.get('extra', {})
                log_data['extra'][key] = value
        
        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record with colors"""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format message
        formatted = f"{color}[{timestamp}] {record.levelname:8} {record.name:20} {record.getMessage()}{reset}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class LoggerManager:
    """Manages logging configuration for the application"""
    
    _instance = None
    _configured = False
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logger manager"""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def configure_logging(cls, 
                         log_level: LogLevel = LogLevel.INFO,
                         log_file: Optional[Path] = None,
                         console_output: bool = True,
                         structured_output: bool = False) -> None:
        """
        Configure logging for the entire application
        
        Args:
            log_level: Minimum log level
            log_file: Optional log file path
            console_output: Whether to output to console
            structured_output: Whether to use structured JSON format
        """
        if cls._configured:
            return
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level.value)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level.value)
            
            if structured_output:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(ColoredFormatter())
            
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler (10MB max, 5 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024, 
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level.value)
            file_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(file_handler)
        
        cls._configured = True
        
        # Log configuration
        logger = logging.getLogger(__name__)
        logger.info("Logging configured", extra={
            'log_level': log_level.name,
            'log_file': str(log_file) if log_file else None,
            'console_output': console_output,
            'structured_output': structured_output
        })
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get logger with specified name
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)


class PerformanceLogger:
    """Logger for performance monitoring"""
    
    def __init__(self, logger_name: str = "performance"):
        """Initialize performance logger"""
        self.logger = LoggerManager.get_logger(logger_name)
    
    def log_execution_time(self, 
                          function_name: str, 
                          execution_time: float, 
                          **kwargs) -> None:
        """
        Log function execution time
        
        Args:
            function_name: Name of the function
            execution_time: Execution time in seconds
            **kwargs: Additional context
        """
        self.logger.info(f"Function {function_name} executed", extra={
            'function_name': function_name,
            'execution_time': execution_time,
            'performance_metric': True,
            **kwargs
        })
    
    def log_memory_usage(self, 
                         operation: str, 
                         memory_mb: float, 
                         **kwargs) -> None:
        """
        Log memory usage
        
        Args:
            operation: Operation being performed
            memory_mb: Memory usage in MB
            **kwargs: Additional context
        """
        self.logger.info(f"Memory usage for {operation}", extra={
            'operation': operation,
            'memory_mb': memory_mb,
            'memory_metric': True,
            **kwargs
        })


class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self, logger_name: str = "security"):
        """Initialize security logger"""
        self.logger = LoggerManager.get_logger(logger_name)
    
    def log_security_event(self, 
                          event_type: str, 
                          severity: str, 
                          **kwargs) -> None:
        """
        Log security event
        
        Args:
            event_type: Type of security event
            severity: Event severity
            **kwargs: Additional context
        """
        self.logger.warning(f"Security event: {event_type}", extra={
            'security_event': True,
            'event_type': event_type,
            'severity': severity,
            **kwargs
        })
    
    def log_file_upload(self, 
                       filename: str, 
                       file_size: int, 
                       mime_type: str,
                       **kwargs) -> None:
        """
        Log file upload event
        
        Args:
            filename: Uploaded filename
            file_size: File size in bytes
            mime_type: MIME type
            **kwargs: Additional context
        """
        self.logger.info(f"File uploaded: {filename}", extra={
            'security_event': True,
            'event_type': 'file_upload',
            'filename': filename,
            'file_size': file_size,
            'mime_type': mime_type,
            **kwargs
        })
    
    def log_validation_failure(self, 
                              validation_type: str, 
                              input_value: str,
                              **kwargs) -> None:
        """
        Log validation failure
        
        Args:
            validation_type: Type of validation that failed
            input_value: The invalid input value
            **kwargs: Additional context
        """
        self.logger.warning(f"Validation failed: {validation_type}", extra={
            'security_event': True,
            'event_type': 'validation_failure',
            'validation_type': validation_type,
            'input_value': input_value,
            **kwargs
        })


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with specified name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return LoggerManager.get_logger(name)


def setup_logging(log_dir: Path = Path("logs"), 
                 log_level: LogLevel = LogLevel.INFO,
                 console_output: bool = True) -> None:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Log level
        console_output: Whether to output to console
    """
    log_file = log_dir / "microexpression.log"
    LoggerManager.configure_logging(
        log_level=log_level,
        log_file=log_file,
        console_output=console_output,
        structured_output=False  # Use colored output for console
    )


def test_logging():
    """Test logging configuration"""
    print("🧪 Testing Logging Configuration...")
    
    # Setup logging
    setup_logging()
    
    # Get loggers
    logger = get_logger("test")
    perf_logger = PerformanceLogger()
    security_logger = SecurityLogger()
    
    # Test basic logging
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Test performance logging
    perf_logger.log_execution_time("test_function", 0.123, test_param="value")
    perf_logger.log_memory_usage("test_operation", 45.6, operation_id="123")
    
    # Test security logging
    security_logger.log_file_upload("test.mp4", 1024, "video/mp4", user_id="test")
    security_logger.log_validation_failure("casme_subject", "invalid", user_id="test")
    
    print("✅ Logging test complete!")


if __name__ == "__main__":
    test_logging()
