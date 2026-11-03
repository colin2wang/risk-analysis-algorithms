"""Financial risk metrics: VaR, CVaR, Sharpe, Sortino, drawdown.

This module implements the standard quantitative finance risk toolkit
used in market risk management (Basel market-risk framework, FRM
curriculum):

* **Value at Risk (VaR)** -- the loss threshold that will not be
  exceeded with a given confidence level over a horizon::

      VaR(alpha) = -quantile_{1-alpha}(portfolio return)

  Three estimation methods are provided: *historical* (empirical
  quantile of past returns), *parametric* (variance-covariance /
  delta-normal assuming normality) and *Monte Carlo* (simulated
  returns).

* **Conditional VaR / Expected Shortfall (CVaR/ES)** -- the expected
  loss *given* that the VaR threshold is exceeded; coherent (always
  >= VaR) and the preferred tail-risk measure under Basel FRTB.

* **Sharpe ratio** -- excess return per unit of total volatility.
* **Sortino ratio** -- excess return per unit of downside volatility.
* **Maximum drawdown** -- largest peak-to-trough decline of a
  cumulative return path.

All return series are expressed in simple (arithmetic) returns per
period; losses are reported as positive numbers.

Example
-------
>>> import numpy as np
>>> rng = np.random.default_rng(0)
>>> returns = rng.normal(0.0005, 0.01, 1000)
>>> var95 = parametric_var(returns, confidence_level=0.95)
>>> es95 = cvar(returns, confidence_level=0.95)
>>> es95 >= var95
True
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np


def _as_array(returns: Sequence[float]) -> np.ndarray:
    """Validate and convert a return sequence to a numpy array.

    Raises
    ------
    ValueError
        If fewer than 2 observations are supplied.
    """
    arr = np.asarray(list(returns), dtype=float)
    if arr.size < 2:
        raise ValueError("at least 2 return observations are required")
    return arr


def historical_var(
    returns: Sequence[float],
    confidence_level: float = 0.95,
) -> float:
    """Value at Risk from the empirical return distribution.

    Parameters
    ----------
    returns:
        Historical periodic (e.g. daily) simple returns.
    confidence_level:
        Confidence level in ``(0, 1)``; 0.95 and 0.99 are standard.

    Returns
    -------
    float
        Positive VaR: the loss threshold exceeded with probability
        ``1 - confidence_level``.

    Raises
    ------
    ValueError
        If the confidence level is outside ``(0, 1)``.

    Examples
    --------
    >>> historical_var([0.01, -0.02, 0.005, -0.01, 0.03, -0.03],
    ...                confidence_level=0.75) >= 0
    True
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    arr = _as_array(returns)
    var_quantile = np.percentile(arr, (1.0 - confidence_level) * 100.0)
    return float(max(0.0, -var_quantile))


def parametric_var(
    returns: Sequence[float],
    confidence_level: float = 0.95,
    mean_return: Optional[float] = None,
) -> float:
    """Value at Risk under the delta-normal (variance-covariance) model.

    Assumes returns are normally distributed:
    ``VaR = -(mu - z * sigma)`` where ``z`` is the (1 - alpha)
    quantile of the standard normal and ``mu`` the mean return.

    Parameters
    ----------
    returns:
        Historical periodic simple returns (used for mu/sigma).
    confidence_level:
        Confidence level in ``(0, 1)``.
    mean_return:
        Optional override of the mean; passing ``0`` yields the
        "zero-mean" VaR popular in short-horizon risk reporting.

    Returns
    -------
    float
        Positive VaR estimate.

    Raises
    ------
    ValueError
        If the confidence level is outside ``(0, 1)``.

    Examples
    --------
    >>> round(parametric_var([0.0] * 10 + [-0.01, 0.01],
    ...                      confidence_level=0.95, mean_return=0.0), 3)
    0.007
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    arr = _as_array(returns)
    mu = float(arr.mean()) if mean_return is None else float(mean_return)
    sigma = float(arr.std(ddof=1))
    z = NormalDist().inv_cdf(1.0 - confidence_level)
    return float(max(0.0, -(mu + z * sigma)))


def monte_carlo_var(
    returns: Sequence[float],
    confidence_level: float = 0.95,
    n_simulations: int = 100_000,
    seed: Optional[int] = None,
) -> float:
    """Value at Risk from Monte Carlo simulated returns.

    Returns are assumed normal with mean and volatility estimated from
    history; ``n_simulations`` draws approximate the future return
    distribution and the empirical quantile yields VaR.

    Parameters
    ----------
    returns:
        Historical periodic simple returns.
    confidence_level:
        Confidence level in ``(0, 1)``.
    n_simulations:
        Number of simulated returns (>= 100).
    seed:
        Optional seed for reproducibility.

    Returns
    -------
    float
        Positive VaR estimate.

    Raises
    ------
    ValueError
        If ``n_simulations < 100`` or confidence level invalid.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if n_simulations < 100:
        raise ValueError("n_simulations must be >= 100")
    arr = _as_array(returns)
    rng = np.random.default_rng(seed)
    simulated = rng.normal(
        arr.mean(), arr.std(ddof=1), int(n_simulations)
    )
    quantile = np.percentile(simulated, (1.0 - confidence_level) * 100.0)
    return float(max(0.0, -quantile))


def cvar(
    returns: Sequence[float],
    confidence_level: float = 0.95,
) -> float:
    """Conditional VaR (Expected Shortfall) from historical returns.

    CVaR is the average of the worst ``(1 - alpha)`` fraction of
    returns -- the expected loss *beyond* the VaR threshold.

    Parameters
    ----------
    returns:
        Historical periodic simple returns.
    confidence_level:
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        Positive expected shortfall; guaranteed >= the corresponding
        historical VaR.

    Examples
    --------
    >>> es = cvar([0.01, -0.05, -0.07, 0.02, -0.01, 0.03],
    ...           confidence_level=0.9)
    >>> es > 0
    True
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    arr = _as_array(returns)
    tail_fraction = 1.0 - confidence_level
    cutoff = int(np.floor(arr.size * tail_fraction))
    cutoff = max(cutoff, 1)  # always take at least the worst observation
    worst = np.sort(arr)[:cutoff]
    return float(max(0.0, -worst.mean()))


def sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Annualisable Sharpe ratio: excess return over total volatility.

    Computed per period; annualise by multiplying by ``sqrt(periods
    per year)`` when using sub-annual data.

    Parameters
    ----------
    returns:
        Periodic simple returns of the asset or portfolio.
    risk_free_rate:
        Periodic risk-free rate.

    Returns
    -------
    float
        ``(mean - rf) / std`` (population-style, ddof=1).  Returns
        0.0 when volatility is zero.

    Examples
    --------
    >>> sharpe_ratio([0.01, 0.02, 0.015, 0.005]) > 0
    True
    """
    arr = _as_array(returns)
    excess = arr - float(risk_free_rate)
    sigma = float(excess.std(ddof=1))
    if sigma == 0.0:
        return 0.0
    return float(excess.mean() / sigma)


def sortino_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Sortino ratio: excess return over *downside* volatility.

    Downside deviation considers only returns below the risk-free
    rate (target), penalising harmful volatility but not upside
    movements.

    Parameters
    ----------
    returns:
        Periodic simple returns.
    risk_free_rate:
        Periodic target/risk-free rate.

    Returns
    -------
    float
        ``(mean - rf) / downside_deviation``; 0.0 when no downside
        observations exist.

    Examples
    --------
    >>> sortino_ratio([0.03, 0.04, 0.02]) > 0
    True
    """
    arr = _as_array(returns)
    excess = arr - float(risk_free_rate)
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0
    return float(excess.mean() / np.sqrt((downside ** 2).mean()))


def max_drawdown(returns: Sequence[float]) -> float:
    """Maximum peak-to-trough decline of a cumulative return path.

    Parameters
    ----------
    returns:
        Periodic simple returns in chronological order.

    Returns
    -------
    float
        The maximum drawdown as a positive fraction (e.g. ``0.25``
        for a 25% decline).  Zero when the path never declines.

    Examples
    --------
    >>> round(max_drawdown([0.10, -0.20, 0.05]), 4)
    0.2
    """
    arr = _as_array(returns)
    cumulative = np.cumprod(1.0 + arr)
    running_peak = np.maximum.accumulate(cumulative)
    drawdowns = 1.0 - cumulative / running_peak
    return float(drawdowns.max())
