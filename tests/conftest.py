"""
Pytest configuration and shared fixtures for ADA pipeline tests.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import pytest
import pandas as pd
import numpy as np

# Import all fixtures from fixture modules
pytest_plugins = [
    "tests.fixtures.session_state",
    "tests.fixtures.test_data",
    "tests.fixtures.mock_llm",
]
