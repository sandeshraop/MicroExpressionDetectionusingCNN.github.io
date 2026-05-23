#!/usr/bin/env python3
"""
Security utilities for micro-expression recognition web application
Implements file upload security, input validation, and path traversal protection
"""

import re
import magic
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Set
from werkzeug.utils import secure_filename as werkzeug_secure_filename

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Security validation utilities for file uploads and inputs"""
    
    # Allowed MIME types for video files
    ALLOWED_MIME_TYPES = {
        'video/mp4',
        'video/avi',
        'video/quicktime',
        'video/x-msvideo',
        'video/webm',
        'video/x-matroska'
    }
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    # Maximum file size (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # CASME-II validation patterns
    SUBJECT_PATTERN = re.compile(r'^sub\d{2}$', re.IGNORECASE)
    EPISODE_PATTERN = re.compile(r'^EP\d{2}_\d{2}f?$', re.IGNORECASE)
    
    @staticmethod
    def validate_file_upload(file, max_size: Optional[int] = None) -> bool:
        """
        Validate uploaded file for security
        
        Args:
            file: File object from Flask request
            max_size: Maximum allowed file size (defaults to MAX_FILE_SIZE)
            
        Returns:
            True if file is valid, raises ValueError if invalid
        """
        if max_size is None:
            max_size = SecurityValidator.MAX_FILE_SIZE
        
        # Check file size
        if hasattr(file, 'content_length') and file.content_length > max_size:
            raise ValueError(f"File too large: {file.content_length} bytes (max: {max_size})")
        
        # Check filename
        if not file.filename:
            raise ValueError("No filename provided")
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SecurityValidator.ALLOWED_EXTENSIONS:
            raise ValueError(f"File extension not allowed: {file_ext}")
        
        # Check file content
        try:
            # Read first 1KB for content detection
            file_content = file.read(1024)
            file.seek(0)  # Reset file pointer
            
            mime_type = magic.from_buffer(file_content, mime=True)
            if mime_type not in SecurityValidator.ALLOWED_MIME_TYPES:
                raise ValueError(f"File type not allowed: {mime_type}")
                
        except Exception as e:
            logger.error(f"File content validation failed: {e}")
            raise ValueError(f"File validation failed: {str(e)}")
        
        return True
    
    @staticmethod
    def secure_filename(filename: str) -> str:
        """
        Generate secure filename
        
        Args:
            filename: Original filename
            
        Returns:
            Secure filename
        """
        # Use werkzeug's secure_filename as base
        secure_name = werkzeug_secure_filename(filename)
        
        # Add timestamp prefix to prevent collisions
        import time
        # Use ns precision to avoid collisions in tests / fast successive calls
        timestamp = str(int(time.time_ns()))
        
        # Keep original extension
        ext = Path(filename).suffix.lower()
        if ext in SecurityValidator.ALLOWED_EXTENSIONS:
            secure_name = f"{timestamp}_{secure_name}"
        
        return secure_name
    
    @staticmethod
    def validate_casme_subject(subject: str) -> bool:
        """
        Validate CASME-II subject format
        
        Args:
            subject: Subject ID (e.g., 'sub01')
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not subject or not isinstance(subject, str):
            raise ValueError("Subject ID is required and must be a string")
        
        if not SecurityValidator.SUBJECT_PATTERN.match(subject.strip()):
            raise ValueError(f"Invalid subject format: {subject}. Expected format: subXX")
        
        return True
    
    @staticmethod
    def validate_casme_episode(episode: str) -> bool:
        """
        Validate CASME-II episode format
        
        Args:
            episode: Episode ID (e.g., 'EP02_01f')
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        if not episode or not isinstance(episode, str):
            raise ValueError("Episode ID is required and must be a string")
        
        if not SecurityValidator.EPISODE_PATTERN.match(episode.strip()):
            raise ValueError(f"Invalid episode format: {episode}. Expected format: EPXX_XX[f]")
        
        return True
    
    @staticmethod
    def safe_path_join(base_dir: Path, *path_parts) -> Path:
        """
        Safely join paths to prevent directory traversal
        
        Args:
            base_dir: Base directory
            *path_parts: Path parts to join
            
        Returns:
            Safe path within base directory
            
        Raises:
            ValueError: If path traversal is detected
        """
        try:
            # Join paths
            full_path = base_dir
            for part in path_parts:
                full_path = full_path / part
            
            # Resolve to absolute path
            resolved_path = full_path.resolve()
            
            # Ensure resolved path is within base directory
            base_resolved = base_dir.resolve()
            
            if not str(resolved_path).startswith(str(base_resolved)):
                raise ValueError(f"Path traversal detected: {resolved_path}")
            
            return resolved_path
            
        except Exception as e:
            logger.error(f"Safe path join failed: {e}")
            raise ValueError(f"Invalid path: {str(e)}")
    
    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate file hash: {e}")
            raise ValueError(f"Hash calculation failed: {str(e)}")


class FileUploadManager:
    """Manages secure file uploads with proper cleanup"""
    
    def __init__(self, upload_dir: Path):
        """
        Initialize file upload manager
        
        Args:
            upload_dir: Directory for uploads
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.active_files: Set[Path] = set()
        
        logger.info(f"FileUploadManager initialized with directory: {self.upload_dir}")
    
    def save_upload(self, file, filename: Optional[str] = None) -> Path:
        """
        Securely save uploaded file
        
        Args:
            file: File object from Flask request
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            # Validate file
            SecurityValidator.validate_file_upload(file)
            
            # Generate secure filename
            if filename is None:
                filename = SecurityValidator.secure_filename(file.filename)
            else:
                filename = SecurityValidator.secure_filename(filename)
            
            # Create safe file path
            file_path = SecurityValidator.safe_path_join(self.upload_dir, filename)
            
            # Save file
            file.save(str(file_path))
            
            # Track active file
            self.active_files.add(file_path)
            
            logger.info(f"File saved successfully: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save upload: {e}")
            raise
    
    def cleanup_file(self, file_path: Path) -> bool:
        """
        Clean up uploaded file
        
        Args:
            file_path: Path to file to clean up
            
        Returns:
            True if cleanup successful
        """
        try:
            if file_path.exists():
                file_path.unlink()
                self.active_files.discard(file_path)
                logger.info(f"File cleaned up: {file_path}")
                return True
            else:
                logger.warning(f"File not found for cleanup: {file_path}")
                return False
                
        except FileNotFoundError:
            logger.warning(f"File already deleted: {file_path}")
            self.active_files.discard(file_path)
            return True
        except PermissionError as e:
            logger.error(f"Permission denied deleting file {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")
            return False
    
    def cleanup_all(self) -> int:
        """
        Clean up all active files
        
        Returns:
            Number of files cleaned up
        """
        cleaned_count = 0
        for file_path in list(self.active_files):
            if self.cleanup_file(file_path):
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} files")
        return cleaned_count


# Context manager for temporary files
import contextlib
import tempfile
import os

@contextlib.contextmanager
def temporary_video_file(file_data: bytes, suffix: str = '.mp4'):
    """
    Context manager for temporary video files
    
    Args:
        file_data: Binary file data
        suffix: File suffix
        
    Yields:
        Path to temporary file
    """
    temp_file = None
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(file_data)
        temp_file.close()
        
        yield Path(temp_file.name)
        
    finally:
        # Clean up
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass


def test_security_utils():
    """Test security utilities"""
    print("🧪 Testing Security Utilities...")
    
    # Test CASME-II validation
    try:
        SecurityValidator.validate_casme_subject("sub01")
        SecurityValidator.validate_casme_episode("EP02_01f")
        print("✅ CASME-II validation tests passed")
    except Exception as e:
        print(f"❌ CASME-II validation test failed: {e}")
    
    # Test filename security
    try:
        secure = SecurityValidator.secure_filename("test.mp4")
        print(f"✅ Secure filename: {secure}")
    except Exception as e:
        print(f"❌ Secure filename test failed: {e}")
    
    print("🎉 Security utilities test complete!")


if __name__ == "__main__":
    test_security_utils()
