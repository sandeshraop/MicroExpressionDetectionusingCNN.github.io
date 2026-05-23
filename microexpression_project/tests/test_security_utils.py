#!/usr/bin/env python3
"""
Test suite for security utilities
"""

import pytest
import tempfile
import magic
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from security_utils import SecurityValidator, FileUploadManager, temporary_video_file


class TestSecurityValidator:
    """Test SecurityValidator class"""
    
    def test_validate_casme_subject_valid(self):
        """Test valid CASME subject validation"""
        assert SecurityValidator.validate_casme_subject("sub01")
        assert SecurityValidator.validate_casme_subject("sub15")
        assert SecurityValidator.validate_casme_subject("SUB01")  # Case insensitive
    
    def test_validate_casme_subject_invalid(self):
        """Test invalid CASME subject validation"""
        with pytest.raises(ValueError):
            SecurityValidator.validate_casme_subject("")
        
        with pytest.raises(ValueError):
            SecurityValidator.validate_casme_subject("sub1")  # Too short
        
        with pytest.raises(ValueError):
            SecurityValidator.validate_casme_subject("sub001")  # Too long
        
        with pytest.raises(ValueError):
            SecurityValidator.validate_casme_subject("invalid")
    
    def test_validate_casme_episode_valid(self):
        """Test valid CASME episode validation"""
        assert SecurityValidator.validate_casme_episode("EP02_01")
        assert SecurityValidator.validate_casme_episode("EP02_01f")
        assert SecurityValidator.validate_casme_episode("ep02_01f")  # Case insensitive
    
    def test_validate_casme_episode_invalid(self):
        """Test invalid CASME episode validation"""
        with pytest.raises(ValueError):
            SecurityValidator.validate_casme_episode("")
        
        with pytest.raises(ValueError):
            SecurityValidator.validate_casme_episode("EP2_01")  # Too short
        
        with pytest.raises(ValueError):
            SecurityValidator.validate_casme_episode("invalid")
    
    def test_secure_filename(self):
        """Test secure filename generation"""
        secure = SecurityValidator.secure_filename("test.mp4")
        assert secure.endswith(".mp4")
        assert "test" in secure
        assert secure.split("_", 1)[0].isdigit()
        
        # Test with timestamp
        secure2 = SecurityValidator.secure_filename("test.mp4")
        assert secure2 != secure  # Should be different due to timestamp
    
    def test_safe_path_join(self):
        """Test safe path joining"""
        base_dir = Path("/tmp/test")
        
        # Normal path
        safe_path = SecurityValidator.safe_path_join(base_dir, "subdir", "file.mp4")
        assert safe_path.is_absolute()
        assert str(safe_path).startswith(str(base_dir.resolve()))
        
        # Path traversal attempt
        with pytest.raises(ValueError):
            SecurityValidator.safe_path_join(base_dir, "..", "etc", "passwd")


class TestFileUploadManager:
    """Test FileUploadManager class"""
    
    def test_file_upload_manager_init(self):
        """Test FileUploadManager initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = FileUploadManager(Path(temp_dir))
            assert manager.upload_dir == Path(temp_dir)
            assert manager.upload_dir.exists()
            assert len(manager.active_files) == 0
    
    def test_cleanup_all(self):
        """Test cleanup of all files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = FileUploadManager(Path(temp_dir))
            
            # Create some dummy files
            for i in range(3):
                file_path = manager.upload_dir / f"test_{i}.txt"
                file_path.write_text("test content")
                manager.active_files.add(file_path)
            
            # Cleanup
            cleaned_count = manager.cleanup_all()
            assert cleaned_count == 3
            assert len(manager.active_files) == 0


class TestTemporaryVideoFile:
    """Test temporary video file context manager"""
    
    def test_temporary_video_file(self):
        """Test temporary video file context manager"""
        test_data = b"fake video data"
        
        with temporary_video_file(test_data, ".mp4") as temp_path:
            assert temp_path.exists()
            assert temp_path.suffix == ".mp4"
            
            # Read back data
            with open(temp_path, "rb") as f:
                assert f.read() == test_data
        
        # File should be cleaned up
        assert not temp_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
