"""Classic quantitative risk analysis techniques.

Modules
-------
ale
    Single Loss Expectancy (SLE), Annualized Loss Expectancy (ALE) and
    Return on Security Investment (ROSI) / cost-benefit analysis.
decision_tree
    Decision-tree Expected Monetary Value (EMV) analysis for decisions
    under uncertainty.
sensitivity
    One-way sensitivity analysis producing tornado-chart data.
scenario
    Scenario analysis and stress testing with multiplicative shocks.
"""

from .ale import (
    sle,
    ale,
    annualized_rate_of_occurrence,
    ros,
    cost_benefit_analysis,
    multi_year_ale,
)
from .decision_tree import DecisionNode, DecisionTree
from .sensitivity import one_way_sensitivity, tornado_data
from .scenario import Scenario, ScenarioAnalysis

__all__ = [
    "sle",
    "ale",
    "annualized_rate_of_occurrence",
    "ros",
    "cost_benefit_analysis",
    "multi_year_ale",
    "DecisionNode",
    "DecisionTree",
    "one_way_sensitivity",
    "tornado_data",
    "Scenario",
    "ScenarioAnalysis",
]
