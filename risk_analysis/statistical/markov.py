"""Markov chain models of risk state evolution.

A discrete-time Markov chain models a system that moves among a finite
set of states (e.g. ``Healthy -> At Risk -> Compromised``) with
transition probabilities that depend only on the current state.  In
risk analysis Markov models are used for:

* Availability / reliability modelling (up, degraded, down).
* Cybersecurity posture drift (secure, exposed, breached).
* Credit rating migration (investment grade -> default).
* Maintenance policy analysis (operational, degraded, failed).

This module provides:

* State-distribution evolution over time (:meth:`evolve`).
* The stationary (long-run) distribution (:meth:`stationary_distribution`).
* Absorbing-state analysis: expected number of steps until absorption
  (:meth:`expected_steps_to_absorption`) using the fundamental matrix.

Example
-------
>>> import numpy as np
>>> from risk_analysis.statistical import MarkovRiskModel
>>> model = MarkovRiskModel(
...     states=["Healthy", "At Risk", "Breached"],
...     transition_matrix=[
...         [0.90, 0.09, 0.01],
...         [0.20, 0.70, 0.10],
...         [0.00, 0.00, 1.00],   # 'Breached' is absorbing
...     ],
... )
>>> dist = model.evolve([1.0, 0.0, 0.0], steps=1)
>>> round(dist[2], 4)
0.01
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class MarkovRiskModel:
    """A discrete-time Markov chain over risk states.

    Parameters
    ----------
    states:
        Ordered list of state names.
    transition_matrix:
        Row-stochastic ``n x n`` matrix where entry ``[i, j]`` is the
        probability of moving from state ``i`` to state ``j`` in one
        step.

    Raises
    ------
    ValueError
        If the matrix is not square, contains negative entries, or
        rows do not sum to 1.

    Examples
    --------
    See the module docstring.
    """

    def __init__(
        self,
        states: Sequence[str],
        transition_matrix: Sequence[Sequence[float]],
    ) -> None:
        self.states = list(states)
        self.matrix = np.asarray(transition_matrix, dtype=float)
        n = len(self.states)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"transition matrix must be {n}x{n}, got {self.matrix.shape}"
            )
        if (self.matrix < 0).any():
            raise ValueError("transition probabilities must be non-negative")
        row_sums = self.matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise ValueError(
                f"rows must sum to 1, got {row_sums.tolist()}"
            )

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------
    def state_index(self, state: str) -> int:
        """Return the index of a state name.

        Raises
        ------
        KeyError
            If the state is unknown.
        """
        try:
            return self.states.index(state)
        except ValueError:
            raise KeyError(f"unknown state: {state!r}") from None

    def _as_distribution(self, initial) -> np.ndarray:
        """Normalise an initial state (name, index, or distribution)."""
        if isinstance(initial, str):
            dist = np.zeros(len(self.states))
            dist[self.state_index(initial)] = 1.0
            return dist
        arr = np.asarray(initial, dtype=float)
        if arr.shape != (len(self.states),):
            raise ValueError("initial distribution has wrong size")
        total = arr.sum()
        if total <= 0:
            raise ValueError("initial distribution must have positive mass")
        return arr / total

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------
    def evolve(self, initial, steps: int) -> np.ndarray:
        """Propagate a state distribution ``steps`` periods forward.

        Parameters
        ----------
        initial:
            Starting point: a state name, a state index, or a full
            probability distribution over states.
        steps:
            Number of transitions to apply (>= 0).

        Returns
        -------
        numpy.ndarray
            The state distribution after ``steps`` transitions.

        Raises
        ------
        ValueError
            If ``steps`` is negative.

        Examples
        --------
        >>> model = MarkovRiskModel(["A", "B"], [[0.5, 0.5], [0.5, 0.5]])
        >>> model.evolve("A", steps=2).round(6).tolist()
        [0.5, 0.5]
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        dist = self._as_distribution(initial)
        if steps == 0:
            return dist
        return dist @ np.linalg.matrix_power(self.matrix, steps)

    def most_likely_state(self, initial, steps: int) -> str:
        """Name of the most probable state after ``steps`` transitions.

        Parameters
        ----------
        initial:
            Starting point (see :meth:`evolve`).
        steps:
            Number of transitions.

        Returns
        -------
        str
            The state with the highest probability mass.
        """
        dist = self.evolve(initial, steps)
        return self.states[int(np.argmax(dist))]

    # ------------------------------------------------------------------
    # Stationary behaviour
    # ------------------------------------------------------------------
    def stationary_distribution(self) -> np.ndarray:
        """Compute the stationary (long-run) distribution.

        The stationary distribution ``pi`` satisfies
        ``pi @ P = pi`` and ``sum(pi) = 1``; it is the normalised
        left eigenvector of ``P`` for eigenvalue 1.

        Returns
        -------
        numpy.ndarray
            The stationary distribution over states.
        """
        n = len(self.states)
        # Solve (P^T - I) pi = 0 with the normalisation constraint
        a = self.matrix.T - np.eye(n)
        a = np.vstack([a, np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(a, b, rcond=None)
        pi = np.clip(pi, 0.0, None)
        return pi / pi.sum()

    # ------------------------------------------------------------------
    # Absorbing analysis
    # ------------------------------------------------------------------
    def absorbing_states(self) -> List[int]:
        """Identify absorbing states (states that can never be left).

        Returns
        -------
        list of int
            Indices of states ``i`` where ``P[i, i] == 1``.
        """
        return [i for i in range(len(self.states))
                if self.matrix[i, i] == 1.0]

    def expected_steps_to_absorption(self, initial) -> Optional[float]:
        """Expected number of steps until an absorbing state is reached.

        Uses the fundamental matrix ``N = (I - Q)^{-1}`` of standard
        absorbing-Markov-chain theory, where ``Q`` is the transient
        block of the transition matrix.  The expected steps from
        transient state ``i`` is row sum ``i`` of ``N``.

        Parameters
        ----------
        initial:
            Starting point (see :meth:`evolve`).

        Returns
        -------
        float or None
            The expected number of steps, or ``None`` when the chain
            has no absorbing states or the start is already absorbed.
        """
        absorbing = set(self.absorbing_states())
        if not absorbing:
            return None
        transient = [i for i in range(len(self.states)) if i not in absorbing]
        if not transient:
            return None
        q = self.matrix[np.ix_(transient, transient)]
        n_matrix = np.linalg.inv(np.eye(len(transient)) - q)
        t = n_matrix.sum(axis=1)  # expected steps per transient start state
        dist = self._as_distribution(initial)
        transient_mass = float(dist[transient].sum())
        if transient_mass == 0.0:
            return None
        return float(dist[transient] @ t / transient_mass)
