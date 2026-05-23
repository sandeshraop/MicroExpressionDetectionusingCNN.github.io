#!/usr/bin/env python3
"""
Startup script for Micro-Expression Recognition Web Application
"""

import sys
import os
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError, AttributeError):
        pass

# Paths relative to this file (works from any cwd)
_web_dir = Path(__file__).resolve().parent
project_root = _web_dir.parent
sys.path.append(str(project_root))


def main():
    """Main startup function"""
    print("🚀 Micro-Expression Recognition Web Application")
    print("=" * 50)

    # Flask/static paths expect cwd = web/
    os.chdir(_web_dir)

    if not (_web_dir / "templates" / "index.html").exists():
        print("❌ Error: templates/index.html not found!")
        print(f"Expected at: {_web_dir / 'templates' / 'index.html'}")
        sys.exit(1)
    
    # Check requirements
    print("📦 Checking requirements...")
    try:
        import flask
        import cv2
        import numpy
        print("✅ All required packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Start the application
    print("🌐 Starting web application...")
    print("📍 Application will be available at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        from app import main as run_app
        run_app()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
