"""
conftest.py
-----------
Shared pytest configuration. Ensures the project root is on sys.path
so that ``from src.xxx import ...`` works from the tests directory.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
