"""Annualized Loss Expectancy (ALE) family of quantitative risk metrics.

The SLE/ALE model is the canonical quantitative risk formula taught in
security management frameworks (e.g. CISM, CISSP) and risk management
standards (ISO 27005).  It expresses risk in monetary terms per year:

* **SLE (Single Loss Expectancy)** -- the expected monetary loss from a
  single occurrence of a risk event::

      SLE = AV * EF

  where ``AV`` is the Asset Value and ``EF`` the Exposure Factor
  (fraction of the asset lost in one event, 0..1).

* **ARO (Annualized Rate of Occurrence)** -- the expected number of
  occurrences per year (e.g. 0.5 means once every two years).

* **ALE (Annualized Loss Expectancy)**::

      ALE = SLE * ARO

* **ROSI (Return on Security Investment)** -- the annualised net
  benefit of a mitigation control::

      ROSI = (ALE_before - ALE_after) - annual_cost_of_control

  often expressed as a ratio or percentage of the control cost.

Example
-------
>>> sle(av=1_000_000, ef=0.4)
400000.0
>>> round(ale(sle_value=400_000, aro=0.25), 2)
100000.0
"""

from __future__ import annotations

from typing import Sequence


def sle(av: float, ef: float) -> float:
    """Compute the Single Loss Expectancy.

    Parameters
    ----------
    av:
        Asset Value in monetary units (must be non-negative).
    ef:
        Exposure Factor -- the fraction of the asset value destroyed or
        compromised by a single event, in ``[0, 1]``.

    Returns
    -------
    float
        ``av * ef`` -- expected loss from one occurrence.

    Raises
    ------
    ValueError
        If ``av`` is negative or ``ef`` is outside ``[0, 1]``.

    Examples
    --------
    >>> sle(av=500_000, ef=0.25)
    125000.0
    """
    if av < 0:
        raise ValueError("av must be non-negative")
    if not 0.0 <= ef <= 1.0:
        raise ValueError(f"ef must be in [0, 1], got {ef!r}")
    return float(av * ef)


def annualized_rate_of_occurrence(events_per_year: float) -> float:
    """Validate and normalise an Annualized Rate of Occurrence.

    Parameters
    ----------
    events_per_year:
        Expected number of events per year.  Values below 1 represent
        less-than-annual events (e.g. 0.1 = once per decade).

    Returns
    -------
    float
        The validated ARO.

    Raises
    ------
    ValueError
        If the rate is negative or not finite.
    """
    if events_per_year < 0:
        raise ValueError("ARO must be non-negative")
    return float(events_per_year)


def ale(sle_value: float, aro: float) -> float:
    """Compute the Annualized Loss Expectancy.

    Parameters
    ----------
    sle_value:
        Single Loss Expectancy in monetary units.
    aro:
        Annualized Rate of Occurrence (events per year, may be < 1).

    Returns
    -------
    float
        ``sle_value * aro`` -- expected annual loss.

    Examples
    --------
    >>> ale(sle_value=100_000, aro=2.0)
    200000.0
    >>> ale(sle_value=1_000_000, aro=0.05)   # once per 20 years
    50000.0
    """
    aro = annualized_rate_of_occurrence(aro)
    if sle_value < 0:
        raise ValueError("sle_value must be non-negative")
    return float(sle_value * aro)


def ros(
    ale_before: float,
    ale_after: float,
    annual_control_cost: float,
) -> dict:
    """Compute the Return on Security Investment of a mitigation.

    ROSI measures whether a control is economically justified: the
    annual risk reduction (ALE delta) is compared with the annualised
    cost of operating the control.

    Parameters
    ----------
    ale_before:
        Annualized loss expectancy without the control.
    ale_after:
        Annualized loss expectancy with the control in place.
    annual_control_cost:
        Total annual cost of the control (capex amortisation + opex).

    Returns
    -------
    dict
        Keys:

        * ``risk_reduction`` -- ``ale_before - ale_after`` (may be
          negative if the control is ineffective).
        * ``net_benefit`` -- ``risk_reduction - annual_control_cost``.
        * ``rosi_ratio`` -- ``risk_reduction / annual_control_cost``
          (values > 1 mean the control pays for itself).
        * ``rosi_percent`` -- ratio expressed in percent.
        * ``justified`` -- ``True`` when ``net_benefit > 0``.

    Raises
    ------
    ValueError
        If ``annual_control_cost`` is not positive or any ALE is
        negative.

    Examples
    --------
    >>> result = ros(ale_before=100_000, ale_after=20_000,
    ...              annual_control_cost=30_000)
    >>> result["justified"]
    True
    >>> result["net_benefit"]
    50000.0
    """
    if ale_before < 0 or ale_after < 0:
        raise ValueError("ALE values must be non-negative")
    if annual_control_cost <= 0:
        raise ValueError("annual_control_cost must be positive")
    risk_reduction = float(ale_before - ale_after)
    net_benefit = risk_reduction - float(annual_control_cost)
    ratio = risk_reduction / float(annual_control_cost)
    return {
        "risk_reduction": risk_reduction,
        "net_benefit": net_benefit,
        "rosi_ratio": ratio,
        "rosi_percent": ratio * 100.0,
        "justified": net_benefit > 0,
    }


def cost_benefit_analysis(
    scenarios: Sequence[dict],
) -> dict:
    """Compare several mitigation options using the ALE/ROSI framework.

    Parameters
    ----------
    scenarios:
        Sequence of dictionaries, each describing one mitigation
        option with keys ``name``, ``ale_after`` and
        ``annual_control_cost``.  The baseline (``ale_before``) is
        taken from the first scenario's ``ale_before`` key or must be
        present in every scenario.

    Returns
    -------
    dict
        ``{"baseline_ale": float, "ranked": [dict, ...]}`` where each
        entry of ``ranked`` contains the scenario's ROSI metrics plus
        its name, sorted so that the highest net benefit comes first.

    Raises
    ------
    ValueError
        If scenarios are empty or missing required keys.

    Examples
    --------
    >>> cba = cost_benefit_analysis([
    ...     {"name": "Do nothing", "ale_before": 100_000,
    ...      "ale_after": 100_000, "annual_control_cost": 0.0001},
    ...     {"name": "Backup site", "ale_before": 100_000,
    ...      "ale_after": 10_000, "annual_control_cost": 25_000},
    ... ])
    >>> cba["ranked"][0]["name"]
    'Backup site'
    """
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    required = {"name", "ale_after", "annual_control_cost"}
    baseline_values = [s.get("ale_before") for s in scenarios
                       if s.get("ale_before") is not None]
    if not baseline_values:
        raise ValueError("at least one scenario must define 'ale_before'")
    ale_before = float(baseline_values[0])

    ranked = []
    for scenario in scenarios:
        missing = required - set(scenario)
        if missing:
            raise ValueError(f"scenario missing keys: {sorted(missing)}")
        metrics = ros(
            ale_before=ale_before,
            ale_after=float(scenario["ale_after"]),
            annual_control_cost=float(scenario["annual_control_cost"]),
        )
        ranked.append({"name": scenario["name"], **metrics})
    ranked.sort(key=lambda entry: entry["net_benefit"], reverse=True)
    return {"baseline_ale": ale_before, "ranked": ranked}


def multi_year_ale(
    sle_value: float,
    aro: float,
    years: int,
    discount_rate: float = 0.0,
) -> dict:
    """Project ALE over a multi-year horizon with optional discounting.

    Useful for comparing one-off vs recurring control costs across a
    planning horizon using present-value (PV) reasoning.

    Parameters
    ----------
    sle_value:
        Single Loss Expectancy (constant across years).
    aro:
        Annualized Rate of Occurrence (constant across years).
    years:
        Planning horizon in years (must be >= 1).
    discount_rate:
        Annual discount rate used to convert future losses to present
        value (e.g. ``0.05`` for 5%).  ``0`` disables discounting.

    Returns
    -------
    dict
        Keys: ``annual_ale`` (undiscounted yearly ALE),
        ``total_ale`` (sum over horizon, undiscounted),
        ``total_pv`` (discounted present value of the loss stream).

    Raises
    ------
    ValueError
        If ``years < 1`` or ``discount_rate <= -1``.

    Examples
    --------
    >>> result = multi_year_ale(200_000, 0.5, years=3)
    >>> result["total_ale"]
    300000.0
    """
    if years < 1:
        raise ValueError("years must be >= 1")
    if discount_rate <= -1:
        raise ValueError("discount_rate must be > -1")
    annual = ale(sle_value=sle_value, aro=aro)
    total = annual * years
    pv = sum(
        annual / ((1.0 + discount_rate) ** t)
        for t in range(1, years + 1)
    )
    return {
        "annual_ale": annual,
        "total_ale": total,
        "total_pv": float(pv),
    }
