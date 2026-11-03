"""FAIR - Factor Analysis of Information Risk (cyber risk quantification).

FAIR (Jones & Freund, The FAIR Institute; Open FAIR standard) is the
de-facto standard ontology for quantifying cyber risk in monetary
terms.  It decomposes risk into:

* **Loss Event Frequency (LEF)** = Threat Event Frequency (TEF) x
  Vulnerability (Vuln) -- how many times per year a threat event
  actually becomes a loss event.
* **Loss Magnitude (LM)** -- the monetary cost of each loss event,
  typically modelled with a three-point estimate.
* **Risk / Annualized Loss Exposure (ALE)** = LEF x LM, aggregated
  over the year.

Because the factors are estimated as ranges (min / most-likely / max),
FAIR quantification is performed with Monte Carlo simulation:
thousands of plausible years are generated and the resulting loss
distribution is reported via percentiles.

This module implements the widely used simplified Open FAIR Monte
Carlo procedure (as popularised by the FAIR Institute's sharing
community):

1. Sample TEF ~ PERT(min, ml, max).
2. Sample Vulnerability ~ Uniform(min, max) or PERT.
3. Number of loss events ~ Poisson(TEF x Vuln).
4. For each loss event, sample LM ~ PERT(min, ml, max) and sum.
5. Aggregate the annual loss distribution.

Example
-------
>>> from risk_analysis.statistical import FAIRAnalysis
>>> fair = FAIRAnalysis(
...     tef_min=10, tef_ml=50, tef_max=200,          # events / year
...     vuln_min=0.01, vuln_ml=0.03, vuln_max=0.10,  # 1%..10%
...     lm_min=50_000, lm_ml=150_000, lm_max=1_000_000,
...     n_simulations=20_000, seed=7,
... )
>>> result = fair.run()
>>> result["annual_loss_p95"] > result["annual_loss_p50"]
True
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .monte_carlo import PERT


class FAIRAnalysis:
    """Open FAIR Monte Carlo quantification of annual loss exposure.

    Parameters
    ----------
    tef_min, tef_ml, tef_max:
        Three-point estimate of annual Threat Event Frequency.
    vuln_min, vuln_ml, vuln_max:
        Three-point estimate of Vulnerability -- the probability that
        a threat event becomes a loss event, in ``[0, 1]``.  The
        most-likely value anchors a PERT distribution.
    lm_min, lm_ml, lm_max:
        Three-point estimate of per-event Loss Magnitude in monetary
        units.
    n_simulations:
        Number of simulated years (default 10_000).
    seed:
        Optional seed for reproducibility.

    Raises
    ------
    ValueError
        If three-point estimates are not ordered ``min <= ml <= max``
        or any value is negative / vulnerability outside ``[0, 1]``.

    Examples
    --------
    See the module docstring.
    """

    def __init__(
        self,
        tef_min: float,
        tef_ml: float,
        tef_max: float,
        vuln_min: float,
        vuln_ml: float,
        vuln_max: float,
        lm_min: float,
        lm_ml: float,
        lm_max: float,
        n_simulations: int = 10_000,
        seed: int = None,
    ) -> None:
        for name, triple, lo, hi in (
            ("tef", (tef_min, tef_ml, tef_max), 0.0, float("inf")),
            ("vuln", (vuln_min, vuln_ml, vuln_max), 0.0, 1.0),
            ("lm", (lm_min, lm_ml, lm_max), 0.0, float("inf")),
        ):
            if not (lo <= triple[0] <= triple[1] <= triple[2] <= hi):
                raise ValueError(
                    f"{name} estimates must satisfy "
                    f"{lo} <= min <= ml <= max <= {hi}, got {triple}"
                )
        if n_simulations < 1:
            raise ValueError("n_simulations must be >= 1")
        self.tef_min, self.tef_ml, self.tef_max = tef_min, tef_ml, tef_max
        self.vuln_min, self.vuln_ml, self.vuln_max = vuln_min, vuln_ml, vuln_max
        self.lm_min, self.lm_ml, self.lm_max = lm_min, lm_ml, lm_max
        self.n_simulations = int(n_simulations)
        self.seed = seed

    def run(self) -> Dict[str, float]:
        """Execute the FAIR Monte Carlo simulation.

        Returns
        -------
        dict
            Keys:

            * ``loss_events_mean`` -- average number of loss events/yr.
            * ``annual_loss_mean`` -- mean annual loss.
            * ``annual_loss_p10 / p50 / p90 / p95 / p99`` --
              percentile estimates of the annual loss distribution.
            * ``annual_loss_max`` -- largest simulated annual loss.
            * ``n_simulations`` -- number of simulated years.
        """
        rng = np.random.default_rng(self.seed)
        n = self.n_simulations

        tef = PERT(self.tef_min, self.tef_ml, self.tef_max).sample(rng, n)
        vuln = PERT(self.vuln_min, self.vuln_ml, self.vuln_max).sample(rng, n)

        # Number of loss events ~ Poisson(expected events)
        expected_events = tef * vuln
        loss_event_counts = rng.poisson(expected_events)

        # Sample per-event loss magnitudes for the total demand
        total_events = int(loss_event_counts.sum())
        lm = PERT(self.lm_min, self.lm_ml, self.lm_max).sample(
            rng, total_events
        )

        # Aggregate per simulated year
        annual_losses = np.zeros(n, dtype=float)
        offsets = np.cumsum(loss_event_counts)
        start = 0
        for year in range(n):
            end = offsets[year]
            if end > start:
                annual_losses[year] = lm[start:end].sum()
            start = end

        return {
            "loss_events_mean": float(loss_event_counts.mean()),
            "annual_loss_mean": float(annual_losses.mean()),
            "annual_loss_p10": float(np.percentile(annual_losses, 10)),
            "annual_loss_p50": float(np.percentile(annual_losses, 50)),
            "annual_loss_p90": float(np.percentile(annual_losses, 90)),
            "annual_loss_p95": float(np.percentile(annual_losses, 95)),
            "annual_loss_p99": float(np.percentile(annual_losses, 99)),
            "annual_loss_max": float(annual_losses.max()),
            "n_simulations": n,
        }

    def compare_mitigation(
        self,
        vuln_min: float,
        vuln_ml: float,
        vuln_max: float,
    ) -> Dict[str, float]:
        """Estimate the risk reduction from lowering Vulnerability.

        Runs a second simulation with the supplied (post-control)
        vulnerability estimates and reports the mean and tail
        reduction versus the baseline.

        Parameters
        ----------
        vuln_min, vuln_ml, vuln_max:
            Post-control vulnerability three-point estimates.

        Returns
        -------
        dict
            Keys: ``baseline_mean``, ``mitigated_mean``,
            ``mean_reduction``, ``baseline_p95``, ``mitigated_p95``,
            ``p95_reduction``.
        """
        baseline = self.run()
        original = (self.vuln_min, self.vuln_ml, self.vuln_max)
        self.vuln_min, self.vuln_ml, self.vuln_max = vuln_min, vuln_ml, vuln_max
        try:
            mitigated = self.run()
        finally:
            self.vuln_min, self.vuln_ml, self.vuln_max = original
        return {
            "baseline_mean": baseline["annual_loss_mean"],
            "mitigated_mean": mitigated["annual_loss_mean"],
            "mean_reduction": baseline["annual_loss_mean"]
            - mitigated["annual_loss_mean"],
            "baseline_p95": baseline["annual_loss_p95"],
            "mitigated_p95": mitigated["annual_loss_p95"],
            "p95_reduction": baseline["annual_loss_p95"]
            - mitigated["annual_loss_p95"],
        }
