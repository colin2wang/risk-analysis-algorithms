"""Qualitative risk analysis methods.

Modules
-------
risk_matrix
    Configurable likelihood-impact risk matrix with classification
    bands, batch evaluation and heat-map aggregation.
fmea
    Failure Mode and Effects Analysis with Risk Priority Number (RPN)
    scoring and threshold-based criticality screening.
"""

from .risk_matrix import RiskMatrix, MatrixCell
from .fmea import FMEAItem, FMEA

__all__ = [
    "RiskMatrix",
    "MatrixCell",
    "FMEAItem",
    "FMEA",
]
