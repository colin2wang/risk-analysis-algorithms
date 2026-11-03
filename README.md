# Risk Analysis Algorithms Library (Python)

A comprehensive, self-contained Python library implementing the
mainstream risk analysis algorithms.

## Algorithm Catalogue

| # | Technique | Module | Complexity | Typical Domain |
|---|-----------|--------|-----------|----------------|
| 1 | Risk Matrix (Likelihood x Impact) | `qualitative.risk_matrix` | Simple | Generic qualitative assessment (ISO 31000) |
| 2 | FMEA / Risk Priority Number (RPN) | `qualitative.fmea` | Simple | Design & process safety (IEC 60812, AIAG) |
| 3 | SLE / ALE / ROSI (cost-benefit) | `quantitative.ale` | Simple | Security & IT risk economics (ISO 27005) |
| 4 | Decision Tree / EMV | `quantitative.decision_tree` | Simple-Medium | Project risk (PMI PMBOK) |
| 5 | One-way Sensitivity / Tornado | `quantitative.sensitivity` | Simple-Medium | Cost & schedule risk models |
| 6 | Scenario Analysis & Stress Testing | `quantitative.scenario` | Simple-Medium | Strategic, financial, continuity planning |
| 7 | Fault Tree Analysis (FTA) | `structural.fault_tree` | Medium | Reliability / safety (IEC 61025) |
| 8 | Event Tree Analysis (ETA) | `structural.event_tree` | Medium | Probabilistic risk assessment (PRA) |
| 9 | Bow-Tie (barrier model) | `structural.bow_tie` | Medium | Process safety (OGP/IPIECA, CCPS) |
| 10 | Analytic Hierarchy Process (AHP) | `structural.ahp` | Medium | Multi-criteria risk prioritisation (Saaty) |
| 11 | Monte Carlo Simulation engine | `statistical.monte_carlo` | Complex | Universal quantitative engine |
| 12 | FAIR (cyber risk quantification) | `statistical.fair` | Complex | Information security (Open FAIR) |
| 13 | VaR / CVaR / Sharpe / Sortino / Drawdown | `statistical.var` | Complex | Market & financial risk (Basel, FRM) |
| 14 | Markov Chain risk state models | `statistical.markov` | Complex | Reliability, credit migration, posture drift |
| 15 | Bayesian Inference & Naive Bayes | `statistical.bayesian` | Complex | Risk-informed inspection, indicator-based scoring |

## Project Layout

```
risk_analysis/
├── README.md
├── requirements.txt
├── setup.py
├── risk_analysis/                 # the package
│   ├── __init__.py
│   ├── core.py                    # shared scales, risk items, levels
│   ├── qualitative/               # risk_matrix, fmea
│   ├── quantitative/              # ale, decision_tree, sensitivity, scenario
│   ├── structural/                # fault_tree, event_tree, bow_tie, ahp
│   └── statistical/               # monte_carlo, fair, var, markov, bayesian
└── tests/                         # unittest suite, one file per module
```

## Installation

Requires Python 3.9+ and numpy.

```bash
pip install numpy
# or install the library itself:
pip install .
```

## Quick Start

```python
from risk_analysis.qualitative import RiskMatrix
from risk_analysis.quantitative import ale, ros
from risk_analysis.structural import FaultTree, Gate, BasicEvent, AHP
from risk_analysis.statistical import PERT, MonteCarloEngine, FAIRAnalysis

# 1. Qualitative: standard 5x5 risk matrix
matrix = RiskMatrix.default_5x5()
print(matrix.classify(4, 3).level)          # RiskLevel.HIGH

# 2. Quantitative: annualized loss expectancy
print(ale(sle_value=1_000_000, aro=0.5))    # 500000.0 per year

# 3. Structural: fault tree with two redundant channels
tree = FaultTree(Gate("AND", [
    Gate("OR", [BasicEvent("A", 0.01), BasicEvent("B", 0.02)]),
    Gate("OR", [BasicEvent("C", 0.03), BasicEvent("D", 0.04)]),
]))
print(tree.top_probability())               # exact top-event probability
print(tree.minimal_cut_sets())              # minimal cut sets

# 4. Statistical: Monte Carlo cost risk with a PERT estimate
engine = MonteCarloEngine(
    inputs={"cost": PERT(1_000_000, 5_000_000, 1_800_000)},
    model=lambda x: x["cost"], n_samples=50_000, seed=42)
result = engine.run()
print(result.summary())                     # mean / p50 / p95 / p99 ...

# 5. Statistical: FAIR cyber risk quantification
fair = FAIRAnalysis(50, 100, 200, 0.01, 0.03, 0.10,
                    100_000, 200_000, 1_000_000,
                    n_samples=10_000, seed=7)
print(fair.run()["annual_loss_p95"])        # 95th percentile annual loss
```

## Running the Tests

```bash
python -m unittest discover -s tests -v
# or from the project root:
python -m unittest discover -s risk_analysis/tests -v
```

The test suite covers every module with deterministic, seeded cases:
probability identities, conservation laws (event-tree totals, matrix
row-stochasticity), known closed-form results (AHP weights, EMV, Bayes
posterior), reproducibility (seeded simulations) and input validation.

## Design Notes

* **Only dependency**: `numpy` (used by the statistical and structural
  sub-packages). All qualitative modules are pure standard library.
* **Reproducibility**: every stochastic component accepts a `seed`
  (backed by `numpy.random.default_rng`).
* **Validation**: all public constructors and functions validate their
  inputs and raise `ValueError` with descriptive messages.
* **Extensibility**: new distributions subclass
  `statistical.monte_carlo.Distribution`; new tree gates can be added
  in `structural.fault_tree.Gate`.

## References

* ISO 31000 / ISO 27005 -- risk management and information security
* IEC 60812 -- FMEA; IEC 61025 -- Fault Tree Analysis
* PMI PMBOK -- EMV decision analysis, Monte Carlo schedule/cost risk
* Saaty, T.L. -- The Analytic Hierarchy Process
* Jones & Freund / The FAIR Institute -- Open FAIR risk quantification
* Jorion, P. -- Value at Risk; Basel frameworks for VaR / ES
