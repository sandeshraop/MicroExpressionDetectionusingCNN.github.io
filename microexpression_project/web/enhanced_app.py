"""
Backward-compatible entrypoint: older deployment docs referenced ``web/enhanced_app.py``.
Delegates to the real Flask app in ``web/app.py``.
"""

import sys
from pathlib import Path

_web = Path(__file__).resolve().parent
if str(_web) not in sys.path:
    sys.path.insert(0, str(_web))

import app as web_app  # noqa: E402

if __name__ == "__main__":
    web_app.main()
