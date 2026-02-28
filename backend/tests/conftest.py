"""
pytest configuration for backend integration tests.

Adds both the project root (for `config` package) and backend/ (for `agents`,
`db`, `prompts`, `utils` packages) to sys.path so imports resolve correctly
regardless of which directory pytest is invoked from.
"""
import sys
from pathlib import Path

# /ceChillHackers/backend/tests/conftest.py
_TESTS_DIR = Path(__file__).resolve().parent        # backend/tests/
_BACKEND_DIR = _TESTS_DIR.parent                    # backend/
_ROOT_DIR = _BACKEND_DIR.parent                     # ceChillHackers/ (project root)

for p in (_ROOT_DIR, _BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
