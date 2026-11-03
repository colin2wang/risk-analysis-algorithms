"""Shared test utilities for the risk_analysis test suite."""

import os
import sys

# Ensure the package root is importable when tests are run directly
# (e.g. `python -m unittest discover tests`) from any directory.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
