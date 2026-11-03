"""Statistical and simulation-based risk analysis methods.

Modules
-------
monte_carlo
    Generic Monte Carlo simulation engine with a distribution library
    (uniform, triangular, PERT, normal, lognormal, discrete,
    empirical) and percentile / exceedance reporting.
fair
    FAIR (Factor Analysis of Information Risk): Monte Carlo
    quantification of cyber loss exposure from threat event
    frequency, vulnerability and loss magnitude estimates.
var
    Financial risk metrics: Value at Risk (historical / parametric /
    Monte Carlo), Conditional VaR (Expected Shortfall), Sharpe ratio,
    Sortino ratio and maximum drawdown.
markov
    Discrete-time Markov chain models of risk state evolution,
    including stationary distributions and absorbing-state analysis.
bayesian
    Bayesian inference for risk updating and a Naive Bayes classifier
    with Laplace smoothing for data-driven risk scoring.
"""

from .monte_carlo import (
    Distribution,
    Uniform,
    Triangular,
    PERT,
    Normal,
    Lognormal,
    Discrete,
    Empirical,
    MonteCarloEngine,
    SimulationResult,
)
from .fair import FAIRAnalysis
from .var import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    cvar,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
)
from .markov import MarkovRiskModel
from .bayesian import bayes_update, NaiveBayesRiskClassifier

__all__ = [
    "Distribution",
    "Uniform",
    "Triangular",
    "PERT",
    "Normal",
    "Lognormal",
    "Discrete",
    "Empirical",
    "MonteCarloEngine",
    "SimulationResult",
    "FAIRAnalysis",
    "historical_var",
    "parametric_var",
    "monte_carlo_var",
    "cvar",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "MarkovRiskModel",
    "bayes_update",
    "NaiveBayesRiskClassifier",
]
