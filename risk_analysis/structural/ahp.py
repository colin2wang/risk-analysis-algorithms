"""Analytic Hierarchy Process (AHP).

The Analytic Hierarchy Process, developed by Thomas L. Saaty in the
1970s, is a structured technique for multi-criteria decision making
and risk prioritisation.  It decomposes a decision into a hierarchy
(goal -> criteria -> alternatives) and derives weights from pairwise
comparison matrices using the fundamental scale 1-9 (and reciprocals).

Core mathematics
----------------
1. A pairwise comparison matrix ``A`` (n x n) is built where
   ``a[i][j]`` expresses how much more important criterion ``i`` is
   than criterion ``j`` (``a[j][i] = 1 / a[i][j]``, ``a[i][i] = 1``).
2. The priority weights are the normalised principal eigenvector of
   ``A``; the principal eigenvalue ``lambda_max`` measures consistency.
3. Consistency Index ``CI = (lambda_max - n) / (n - 1)``; Consistency
   Ratio ``CR = CI / RI`` where RI is the Random Index for matrices of
   the same size.  ``CR <= 0.10`` is the conventional acceptance rule.

Example
-------
>>> from risk_analysis.structural import AHP
>>> criteria = AHP(["Cost", "Schedule", "Quality"])
>>> criteria.set_comparison("Cost", "Schedule", 3.0)
>>> criteria.set_comparison("Cost", "Quality", 5.0)
>>> criteria.set_comparison("Schedule", "Quality", 2.0)
>>> criteria.is_consistent()
True
>>> weights = criteria.weights
>>> abs(sum(weights) - 1.0) < 1e-9
True
>>> weights.argmax()   # 'Cost' is the most important criterion
0
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

# Saaty's Random Index table for n = 1..15 (values for n <= 2 are 0 by
# definition because such matrices are always consistent).
_RANDOM_INDEX = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
    8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56,
    14: 1.57, 15: 1.59,
}


class AHP:
    """A pairwise-comparison matrix with weights and consistency checks.

    Parameters
    ----------
    criteria:
        Ordered list of criteria (or alternative) names.

    Raises
    ------
    ValueError
        If fewer than 2 criteria are supplied or names are duplicated.

    Examples
    --------
    See the module docstring.
    """

    def __init__(self, criteria: Sequence[str]) -> None:
        names = list(criteria)
        if len(names) < 2:
            raise ValueError("AHP requires at least 2 criteria")
        if len(set(names)) != len(names):
            raise ValueError("criteria names must be unique")
        self.criteria: List[str] = names
        self._n = len(names)
        self._index: Dict[str, int] = {c: i for i, c in enumerate(names)}
        # Start from the identity (all comparisons equal importance)
        self._matrix = np.eye(self._n, dtype=float)

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------
    def set_comparison(self, a: str, b: str, value: float) -> None:
        """Set the pairwise comparison between two criteria.

        Parameters
        ----------
        a:
            First criterion name (row).
        b:
            Second criterion name (column).
        value:
            Saaty fundamental scale value in ``[1/9, 9]``: how much
            more important ``a`` is than ``b``.  Use values < 1 to say
            ``b`` dominates; the reciprocal is filled automatically.

        Raises
        ------
        KeyError
            If either criterion is unknown.
        ValueError
            If ``value`` is outside the Saaty range.
        """
        if a not in self._index or b not in self._index:
            raise KeyError(f"unknown criterion in ({a!r}, {b!r})")
        if not 1 / 9 <= value <= 9:
            raise ValueError(f"comparison value must be in [1/9, 9], got {value!r}")
        i, j = self._index[a], self._index[b]
        self._matrix[i, j] = float(value)
        self._matrix[j, i] = 1.0 / float(value)

    def set_matrix(self, matrix: Sequence[Sequence[float]]) -> None:
        """Bulk-load a full pairwise comparison matrix.

        Parameters
        ----------
        matrix:
            An ``n x n`` matrix of comparison values.  Diagonal
            entries must be 1 and the matrix must be reciprocal
            (``m[j][i] == 1 / m[i][j]`` within tolerance).

        Raises
        ------
        ValueError
            If dimensions, reciprocity or the diagonal are violated.
        """
        m = np.asarray(matrix, dtype=float)
        if m.shape != (self._n, self._n):
            raise ValueError(f"matrix must be {self._n}x{self._n}")
        if not np.allclose(np.diag(m), 1.0):
            raise ValueError("matrix diagonal must be all ones")
        with np.errstate(divide="ignore", invalid="ignore"):
            reciprocal = np.where(m != 0, 1.0 / m, np.inf)
        if not np.allclose(m, reciprocal, atol=1e-9):
            raise ValueError("matrix must be reciprocal (a[j][i] = 1/a[i][j])")
        self._matrix = m

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        """Priority weights (normalised principal eigenvector).

        Computed on access; returns a 1-D numpy array aligned with the
        ``criteria`` order and summing to 1.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self._matrix)
        principal = int(np.argmax(eigenvalues.real))
        vector = np.abs(eigenvectors[:, principal].real)
        return vector / vector.sum()

    @property
    def lambda_max(self) -> float:
        """Principal (largest real) eigenvalue of the comparison matrix."""
        eigenvalues = np.linalg.eigvals(self._matrix)
        return float(eigenvalues.real.max())

    @property
    def consistency_index(self) -> float:
        """Consistency Index ``CI = (lambda_max - n) / (n - 1)``.

        Returns 0.0 for ``n <= 2`` (matrices are always consistent).
        """
        if self._n <= 2:
            return 0.0
        return (self.lambda_max - self._n) / (self._n - 1)

    def consistency_ratio(self) -> float:
        """Consistency Ratio ``CR = CI / RI``.

        Returns
        -------
        float
            The CR value; ``0.0`` for ``n <= 2``.  Values above 0.10
            indicate inconsistent judgements that should be revised.
        """
        ci = self.consistency_index
        ri = _RANDOM_INDEX.get(self._n, 1.60)
        if ri == 0.0:
            return 0.0
        return ci / ri

    def is_consistent(self, threshold: float = 0.10) -> bool:
        """Check the conventional CR <= 0.10 acceptance rule.

        Parameters
        ----------
        threshold:
            Maximum acceptable CR (default 0.10 per Saaty).

        Returns
        -------
        bool
            ``True`` when the judgements are acceptably consistent.
        """
        return self.consistency_ratio() <= threshold

    # ------------------------------------------------------------------
    # Alternative scoring
    # ------------------------------------------------------------------
    def rank_alternatives(
        self,
        alternative_scores: Dict[str, Sequence[float]],
        criteria_weights: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float]]:
        """Rank alternatives by weighted aggregation of criterion scores.

        Implements the standard AHP synthesis: each alternative is
        scored per criterion, the per-criterion scores are normalised
        across alternatives, and the global score is the weighted sum
        using the criteria weights.

        Parameters
        ----------
        alternative_scores:
            Mapping from alternative name to a score vector aligned
            with ``criteria`` (higher is better).
        criteria_weights:
            Optional replacement weights (e.g. derived from another
            AHP level); defaults to :attr:`weights`.

        Returns
        -------
        list of tuple
            ``(alternative, global_score)`` pairs sorted descending.

        Raises
        ------
        ValueError
            If score vectors do not match the number of criteria.

        Examples
        --------
        >>> ahp = AHP(["Cost", "Speed"])
        >>> ahp.set_comparison("Cost", "Speed", 2.0)
        >>> ranked = ahp.rank_alternatives({
        ...     "Vendor A": [0.8, 0.4],
        ...     "Vendor B": [0.5, 0.9],
        ... })
        >>> ranked[0][0] in ("Vendor A", "Vendor B")
        True
        """
        weights = criteria_weights if criteria_weights is not None else self.weights
        names = list(alternative_scores.keys())
        raw = np.array(
            [alternative_scores[name] for name in names], dtype=float
        )
        if raw.shape[1] != self._n:
            raise ValueError(
                f"each alternative needs {self._n} scores, "
                f"got {raw.shape[1]}"
            )
        # Column-normalise each criterion across alternatives
        column_sums = raw.sum(axis=0)
        column_sums[column_sums == 0] = 1.0  # guard degenerate columns
        normalised = raw / column_sums
        global_scores = normalised @ weights
        ranked = sorted(
            zip(names, global_scores.tolist()), key=lambda t: t[1],
            reverse=True,
        )
        return ranked
