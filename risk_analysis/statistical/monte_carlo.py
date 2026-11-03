"""Generic Monte Carlo simulation engine with a small distribution library.

Monte Carlo simulation approximates the distribution of a model output
by repeatedly sampling its random inputs and re-evaluating the model.
It is the workhorse of modern quantitative risk analysis (schedule and
cost risk per PMI/NASA guidance, cyber risk per FAIR, financial risk
per VaR engines) whenever closed-form solutions are unavailable.

This module provides:

* A :class:`Distribution` abstract base and common implementations:
  :class:`Uniform`, :class:`Triangular`, :class:`PERT`, :class:`Normal`,
  :class:`Lognormal`, :class:`Discrete` and :class:`Empirical`.
* :class:`MonteCarloEngine` -- a seeded, reproducible simulator that
  maps sampled inputs through a user model and reports percentiles,
  moments and exceedance probabilities.

Example
-------
>>> import numpy as np
>>> from risk_analysis.statistical import PERT, MonteCarloEngine
>>> engine = MonteCarloEngine(
...     inputs={"cost": PERT(1_000_000, 1_500_000, 4_000_000)},
...     model=lambda s: s["cost"] * 1.1,
...     n_samples=50_000, seed=42,
... )
>>> result = engine.run()
>>> round(result.percentile(50) / 100_000) in (17, 18)   # near median
True
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Tuple

import numpy as np


class Distribution(ABC):
    """Abstract base class for all sampling distributions.

    Subclasses must implement :meth:`sample` drawing values from a
    supplied ``numpy.random.Generator`` so simulations are reproducible
    under a fixed seed.
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw ``size`` independent samples.

        Parameters
        ----------
        rng:
            Numpy random generator providing the random stream.
        size:
            Number of samples to draw.

        Returns
        -------
        numpy.ndarray
            1-D array of sampled values.
        """
        raise NotImplementedError


class Uniform(Distribution):
    """Continuous uniform distribution on ``[low, high]``.

    Parameters
    ----------
    low:
        Lower bound (inclusive).
    high:
        Upper bound (inclusive).

    Raises
    ------
    ValueError
        If ``low >= high``.
    """

    def __init__(self, low: float, high: float) -> None:
        if low >= high:
            raise ValueError("low must be < high")
        self.low = float(low)
        self.high = float(high)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw uniform samples on ``[low, high)``."""
        return rng.uniform(self.low, self.high, size)


class Triangular(Distribution):
    """Triangular distribution defined by min / mode / max.

    Commonly used in PERT-style schedule and cost models when only
    three-point estimates are available.

    Parameters
    ----------
    low:
        Minimum value.
    mode:
        Most likely value (must satisfy ``low <= mode <= high``).
    high:
        Maximum value.

    Raises
    ------
    ValueError
        If the ordering constraint is violated.
    """

    def __init__(self, low: float, mode: float, high: float) -> None:
        if not low <= mode <= high:
            raise ValueError("require low <= mode <= high")
        self.low = float(low)
        self.mode = float(mode)
        self.high = float(high)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw triangular samples."""
        return rng.triangular(self.low, self.mode, self.high, size)


class PERT(Triangular):
    """PERT (Beta-PERT) distribution from three-point estimates.

    The modified PERT distribution used in modern cost-risk practice is
    a Beta distribution reparameterised on ``[low, high]`` with mean
    pulled toward the mode via the ``gamma`` shape parameter.  With
    ``gamma=4`` it reduces to the classic PERT mean
    ``(low + 4*mode + high) / 6``.

    Parameters
    ----------
    low:
        Minimum value.
    mode:
        Most likely value.
    high:
        Maximum value.
    gamma:
        Shape parameter controlling peakedness (default 4; larger
        values concentrate samples nearer the mode).

    Raises
    ------
    ValueError
        If ``gamma <= 0``.
    """

    def __init__(
        self, low: float, mode: float, high: float, gamma: float = 4.0
    ) -> None:
        super().__init__(low, mode, high)
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = float(gamma)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw Beta-PERT samples via the reparameterised Beta law."""
        # Beta shape parameters on [0, 1] derived from mode and gamma
        span = self.high - self.low
        if span == 0:
            return np.full(size, self.low)
        mode_unit = (self.mode - self.low) / span
        alpha = 1.0 + self.gamma * mode_unit
        beta = 1.0 + self.gamma * (1.0 - mode_unit)
        samples = rng.beta(alpha, beta, size)
        return self.low + samples * span


class Normal(Distribution):
    """Normal (Gaussian) distribution with mean and standard deviation.

    Parameters
    ----------
    mean:
        Distribution mean.
    std:
        Standard deviation (must be > 0).

    Raises
    ------
    ValueError
        If ``std <= 0``.
    """

    def __init__(self, mean: float, std: float) -> None:
        if std <= 0:
            raise ValueError("std must be positive")
        self.mean = float(mean)
        self.std = float(std)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw normal samples."""
        return rng.normal(self.mean, self.std, size)


class Lognormal(Distribution):
    """Lognormal distribution parameterised by the underlying normal.

    A sample ``X`` is lognormal when ``ln(X) ~ Normal(mean_log,
    sigma_log)``.  Widely used for loss severities, which are
    non-negative and right-skewed.

    Parameters
    ----------
    mean_log:
        Mean of the underlying normal (log-space mean).
    sigma_log:
        Standard deviation of the underlying normal (log-space sigma,
        must be > 0).

    Raises
    ------
    ValueError
        If ``sigma_log <= 0``.
    """

    def __init__(self, mean_log: float, sigma_log: float) -> None:
        if sigma_log <= 0:
            raise ValueError("sigma_log must be positive")
        self.mean_log = float(mean_log)
        self.sigma_log = float(sigma_log)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw lognormal samples (always positive)."""
        return rng.lognormal(self.mean_log, self.sigma_log, size)


class Discrete(Distribution):
    """Finite discrete distribution over ``(value, probability)`` pairs.

    Parameters
    ----------
    outcomes:
        Sequence of ``(value, probability)`` tuples.  Probabilities
        must sum to 1 (within tolerance) and be non-negative.

    Raises
    ------
    ValueError
        If outcomes are empty, probabilities are invalid or do not sum
        to 1.
    """

    def __init__(self, outcomes: Sequence[Tuple[float, float]]) -> None:
        if not outcomes:
            raise ValueError("outcomes must not be empty")
        values = np.array([v for v, _ in outcomes], dtype=float)
        probs = np.array([p for _, p in outcomes], dtype=float)
        if (probs < 0).any():
            raise ValueError("probabilities must be non-negative")
        if abs(probs.sum() - 1.0) > 1e-9:
            raise ValueError("probabilities must sum to 1")
        self.values = values
        self.probs = probs

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw discrete samples via inverse-CDF lookup."""
        indices = rng.choice(len(self.values), size=size, p=self.probs)
        return self.values[indices]


class Empirical(Distribution):
    """Empirical distribution resampling from observed data.

    Implements bootstrapping: each sample is drawn uniformly (with
    replacement) from the supplied observations.  Used when no
    parametric form is justified.

    Parameters
    ----------
    data:
        Observed values (at least one).

    Raises
    ------
    ValueError
        If ``data`` is empty.
    """

    def __init__(self, data: Sequence[float]) -> None:
        if not data:
            raise ValueError("data must not be empty")
        self.data = np.asarray(data, dtype=float)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Bootstrap-resample the observed data."""
        indices = rng.integers(0, len(self.data), size=size)
        return self.data[indices]


@dataclass
class SimulationResult:
    """Container for the output samples of a simulation run.

    Attributes
    ----------
    samples:
        1-D array of model outputs.
    n_samples:
        Number of samples collected.
    inputs_sampled:
        Optional record of sampled input arrays keyed by input name
        (only populated when ``keep_inputs`` was requested).
    """

    samples: np.ndarray
    n_samples: int
    inputs_sampled: Dict[str, np.ndarray] = field(default_factory=dict)

    def percentile(self, q: float) -> float:
        """Return the q-th percentile of the output distribution.

        Parameters
        ----------
        q:
            Percentile in ``[0, 100]``.

        Returns
        -------
        float
            The percentile value.
        """
        if not 0 <= q <= 100:
            raise ValueError("q must be in [0, 100]")
        return float(np.percentile(self.samples, q))

    def mean(self) -> float:
        """Arithmetic mean of the output samples."""
        return float(self.samples.mean())

    def std(self) -> float:
        """Standard deviation of the output samples."""
        return float(self.samples.std(ddof=1)) if self.n_samples > 1 else 0.0

    def probability_of_exceedance(self, threshold: float) -> float:
        """Probability that the output exceeds a threshold.

        Parameters
        ----------
        threshold:
            Value to test against.

        Returns
        -------
        float
            Empirical ``P(output > threshold)`` in ``[0, 1]``.
        """
        return float((self.samples > threshold).mean())

    def exceedance(self, threshold: float) -> float:
        """Alias of :meth:`probability_of_exceedance`.

        Parameters
        ----------
        threshold:
            Value to test against.

        Returns
        -------
        float
            Empirical ``P(output > threshold)``.
        """
        return self.probability_of_exceedance(threshold)

    def probability_exceeded(self, threshold: float) -> float:
        """Alias of :meth:`probability_of_exceedance`.

        Parameters
        ----------
        threshold:
            Value to test against.

        Returns
        -------
        float
            Empirical ``P(output > threshold)``.
        """
        return self.probability_of_exceedance(threshold)

    def summary(self) -> dict:
        """Produce a compact summary of the output distribution.

        Returns
        -------
        dict
            Keys: ``n``, ``mean``, ``std``, ``p5``, ``p50``, ``p95``
            and ``min``/``max``.
        """
        return {
            "n": self.n_samples,
            "mean": self.mean(),
            "std": self.std(),
            "p5": self.percentile(5),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "min": float(self.samples.min()),
            "max": float(self.samples.max()),
        }

    def histogram(self, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a histogram of the output samples.

        Parameters
        ----------
        bins:
            Number of equal-width bins.

        Returns
        -------
        tuple of numpy.ndarray
            ``(counts, bin_edges)`` as produced by ``numpy.histogram``.
        """
        return np.histogram(self.samples, bins=bins)


class MonteCarloEngine:
    """A seeded Monte Carlo simulator for arbitrary scalar models.

    Parameters
    ----------
    inputs:
        Mapping from input name to a :class:`Distribution`.
    model:
        Callable receiving a dictionary ``{name: sampled_value}`` and
        returning a scalar output.
    n_samples:
        Number of iterations (default 10_000).
    seed:
        Optional seed for reproducible runs.

    Raises
    ------
    ValueError
        If ``n_samples < 1`` or ``inputs`` is empty.

    Examples
    --------
    See the module docstring.
    """

    def __init__(
        self,
        inputs: Dict[str, Distribution],
        model: Callable[[Dict[str, float]], float],
        n_samples: int = 10_000,
        seed: int = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if not inputs:
            raise ValueError("inputs must not be empty")
        self.inputs = dict(inputs)
        self.model = model
        self.n_samples = int(n_samples)
        self.seed = seed

    def run(self, keep_inputs: bool = False) -> SimulationResult:
        """Execute the simulation.

        Parameters
        ----------
        keep_inputs:
            When ``True``, sampled input arrays are retained in the
            result for later correlation/diagnostic analysis.

        Returns
        -------
        SimulationResult
            The collected output samples plus summary helpers.
        """
        rng = np.random.default_rng(self.seed)
        sampled: Dict[str, np.ndarray] = {
            name: dist.sample(rng, self.n_samples)
            for name, dist in self.inputs.items()
        }
        outputs = np.empty(self.n_samples, dtype=float)
        for i in range(self.n_samples):
            scenario = {name: float(arr[i]) for name, arr in sampled.items()}
            outputs[i] = float(self.model(scenario))
        return SimulationResult(
            samples=outputs,
            n_samples=self.n_samples,
            inputs_sampled=sampled if keep_inputs else {},
        )
