"""Scenario analysis and stress testing.

Scenario analysis evaluates a model under a small set of coherent,
plausible narratives (base case, best case, worst case, regulatory
shock, ...) rather than sampling individual variables independently as
Monte Carlo does.  Stress testing is a special case focused on severe
adverse scenarios -- a regulatory staple for banks (Basel/CCAR) and
widely used in operational risk and business continuity planning.

This module supports two composition styles:

* **Multiplicative shocks** -- each scenario defines a percentage
  change (multiplier) per input, e.g. ``{"revenue": -0.3}`` meaning
  revenue falls by 30%.
* **Absolute overrides** -- a scenario pins inputs to explicit values.

Example
-------
>>> from risk_analysis.quantitative import Scenario, ScenarioAnalysis
>>> def model(inputs):
...     return inputs["revenue"] * inputs["margin"]
>>> analysis = ScenarioAnalysis(model, base_inputs={
...     "revenue": 1000.0, "margin": 0.2})
>>> analysis.add_scenario(Scenario("Recession", shocks={
...     "revenue": -0.4, "margin": -0.25}))
>>> results = analysis.run()
>>> round(results["Recession"]["output"], 2)
120.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Scenario:
    """A single scenario definition.

    Attributes
    ----------
    name:
        Scenario name (must be unique within an analysis).
    shocks:
        Mapping from input name to a *multiplicative* change, where
        ``-0.3`` means "reduce the input by 30%".  Multipliers are
        additive with the base value: ``new = base * (1 + shock)``.
    overrides:
        Mapping from input name to an *absolute* replacement value.
        Overrides are applied after shocks and win over them.
    description:
        Optional narrative describing the scenario logic.
    probability:
        Optional subjective probability of the scenario (0..1).  When
        all scenarios carry probabilities, probability-weighted
        aggregate statistics can be computed.

    Raises
    ------
    ValueError
        If ``probability`` is provided outside ``[0, 1]``, or if both
        ``shocks`` and ``overrides`` are empty.
    """

    name: str
    shocks: Dict[str, float] = field(default_factory=dict)
    overrides: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    probability: Optional[float] = None

    def __post_init__(self) -> None:
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not self.shocks and not self.overrides:
            raise ValueError(
                "scenario must define at least one shock or override"
            )


class ScenarioAnalysis:
    """Evaluate a scalar model across a set of scenarios.

    Parameters
    ----------
    model:
        Callable accepting the input dictionary and returning a scalar
        output (loss, profit, capital requirement, ...).
    base_inputs:
        Baseline input values shared by all scenarios.

    Examples
    --------
    See the module docstring.
    """

    def __init__(
        self,
        model: Callable[[Dict[str, float]], float],
        base_inputs: Dict[str, float],
    ) -> None:
        if not base_inputs:
            raise ValueError("base_inputs must not be empty")
        self.model = model
        self.base_inputs = dict(base_inputs)
        self.scenarios: List[Scenario] = []

    def add_scenario(self, scenario: Scenario) -> None:
        """Register a scenario.

        Parameters
        ----------
        scenario:
            The :class:`Scenario` to add.

        Raises
        ------
        ValueError
            If a scenario with the same name already exists.
        """
        if any(s.name == scenario.name for s in self.scenarios):
            raise ValueError(f"duplicate scenario name: {scenario.name!r}")
        self.scenarios.append(scenario)

    def _build_inputs(self, scenario: Scenario) -> Dict[str, float]:
        """Materialise the input vector for one scenario.

        Base values are shock-multiplied and then overridden by any
        absolute overrides.  Unknown input names raise ``ValueError``.
        """
        inputs = dict(self.base_inputs)
        for name, shock in scenario.shocks.items():
            if name not in inputs:
                raise ValueError(f"unknown input for shock: {name!r}")
            inputs[name] = inputs[name] * (1.0 + float(shock))
        for name, value in scenario.overrides.items():
            if name not in inputs:
                raise ValueError(f"unknown input for override: {name!r}")
            inputs[name] = float(value)
        return inputs

    def run(self) -> Dict[str, dict]:
        """Evaluate every scenario plus the baseline.

        Returns
        -------
        dict
            Mapping from scenario name (``"__baseline__"`` for the
            unshocked base case) to a result dictionary containing the
            scenario ``inputs`` and the scalar ``output``.
        """
        results: Dict[str, dict] = {
            "__baseline__": {
                "inputs": dict(self.base_inputs),
                "output": float(self.model(dict(self.base_inputs))),
            }
        }
        for scenario in self.scenarios:
            inputs = self._build_inputs(scenario)
            results[scenario.name] = {
                "inputs": inputs,
                "output": float(self.model(inputs)),
            }
        return results

    def worst_case(self) -> dict:
        """Return the scenario producing the most adverse output.

        "Most adverse" means the *minimum* output value; for models
        where larger values are worse, wrap the model so that larger
        is always better (e.g. return ``-loss``).

        Returns
        -------
        dict
            The result entry (see :meth:`run`) of the worst scenario,
            with the scenario name added under key ``"name"``.
        """
        results = self.run()
        worst_name = min(
            (k for k in results if k != "__baseline__"),
            key=lambda k: results[k]["output"],
            default="__baseline__",
        )
        return {"name": worst_name, **results[worst_name]}

    def weighted_expectation(self) -> Optional[float]:
        """Probability-weighted expected output across scenarios.

        Returns
        -------
        float or None
            The weighted mean over all scenarios that define a
            probability, or ``None`` if no scenario has one.

        Raises
        ------
        ValueError
            If the defined probabilities do not sum to 1 (within
            tolerance).
        """
        weighted = [
            (s.probability, self._build_inputs(s))
            for s in self.scenarios if s.probability is not None
        ]
        if not weighted:
            return None
        total_p = sum(p for p, _ in weighted)
        if abs(total_p - 1.0) > 1e-9:
            raise ValueError(
                f"scenario probabilities must sum to 1, got {total_p!r}"
            )
        return float(sum(
            p * self.model(inputs) for p, inputs in weighted
        ))
