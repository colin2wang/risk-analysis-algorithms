"""risk_analysis - A comprehensive library of mainstream risk analysis algorithms.

This package implements a wide spectrum of risk analysis techniques, ranging
from simple qualitative methods to advanced quantitative and statistical
models.  It is organised into four sub-packages:

``qualitative``
    Methods that rely on expert judgement expressed on ordinal scales:
    risk matrices, FMEA (Failure Mode and Effects Analysis), bow-tie
    barrier models.

``quantitative``
    Classic quantitative techniques: Annualized Loss Expectancy (ALE),
    decision-tree Expected Monetary Value (EMV), one-way sensitivity
    analysis (tornado charts) and scenario / stress testing.

``structural``
    System-oriented, topology-based methods: Fault Tree Analysis (FTA),
    Event Tree Analysis (ETA), Bow-Tie analysis and the Analytic
    Hierarchy Process (AHP).

``statistical``
    Simulation- and statistics-driven methods: generic Monte Carlo
    simulation, the FAIR (Factor Analysis of Information Risk) model,
    financial risk metrics (VaR / CVaR / Sharpe / drawdown), Markov
    chain risk state models and Bayesian inference / Naive Bayes
    classification.

Dependencies
------------
The library depends only on ``numpy``.  All functions accept plain Python
containers as well as numpy arrays where appropriate.

Example
-------
>>> from risk_analysis.qualitative import RiskMatrix
>>> matrix = RiskMatrix.default_5x5()
>>> matrix.classify(likelihood=4, impact=3).level
<RiskLevel.HIGH: 'High'>
"""

__version__ = "1.0.0"
__author__ = "Risk Analysis Library Contributors"

from . import core  # noqa: F401
