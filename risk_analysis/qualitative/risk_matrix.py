"""Likelihood-Impact risk matrix analysis.

The risk matrix (also known as a probability-consequence matrix or
P-I matrix) is the most widely used qualitative risk assessment tool.
Risks are rated on two ordinal scales -- likelihood and impact -- and
plotted into a grid whose cells are classified into risk bands
(Low / Medium / High / Critical).

This module provides a fully configurable matrix implementation:

* Arbitrary grid sizes (default: the ubiquitous 5x5 matrix).
* Custom scoring functions (default: ``likelihood * impact``).
* Configurable band thresholds or absolute score bands.
* Batch evaluation of risk portfolios and heat-map aggregation.

Example
-------
>>> from risk_analysis.qualitative import RiskMatrix
>>> matrix = RiskMatrix.default_5x5()
>>> cell = matrix.classify(likelihood=5, impact=4)
>>> cell.level.value
'Critical'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..core import RiskItem, RiskLevel


@dataclass(frozen=True)
class MatrixCell:
    """A single cell of the risk matrix.

    Attributes
    ----------
    likelihood:
        Likelihood level (row index, 1-based).
    impact:
        Impact level (column index, 1-based).
    score:
        Numeric score produced by the matrix scoring function.
    level:
        Risk classification band assigned to this cell.
    """

    likelihood: int
    impact: int
    score: float
    level: RiskLevel

    def to_dict(self) -> Dict[str, object]:
        """Serialise the cell into a plain dictionary."""
        return {
            "likelihood": self.likelihood,
            "impact": self.impact,
            "score": self.score,
            "level": self.level.value,
        }


class RiskMatrix:
    """A configurable likelihood-impact risk matrix.

    Parameters
    ----------
    size:
        Number of levels on each axis (default 5, i.e. a 5x5 matrix).
    score_fn:
        Scoring function ``f(likelihood, impact) -> float``.  Defaults
        to the multiplicative convention ``likelihood * impact``.
    band_fn:
        Banding function ``f(score) -> RiskLevel``.  Defaults to
        :meth:`RiskLevel.from_score` with ``max_score`` derived from the
        matrix size and the default scoring function.
    level_names:
        Optional human-readable names for each level (1-based order),
        e.g. ``["Rare", "Unlikely", "Possible", "Likely", "Certain"]``.
        When omitted, generic ``"L1".."Ln"`` / ``"I1".."In"`` labels are
        generated.

    Raises
    ------
    ValueError
        If ``size`` is smaller than 2.

    Examples
    --------
    >>> matrix = RiskMatrix.default_5x5()
    >>> matrix.classify(3, 3).level
    <RiskLevel.MEDIUM: 'Medium'>
    """

    def __init__(
        self,
        size: int = 5,
        score_fn: Optional[Callable[[int, int], float]] = None,
        band_fn: Optional[Callable[[float], RiskLevel]] = None,
        level_names: Optional[Sequence[str]] = None,
    ) -> None:
        if size < 2:
            raise ValueError("matrix size must be at least 2")
        self.size = int(size)
        self._score_fn = score_fn or self._default_score_fn
        if band_fn is not None:
            self._band_fn = band_fn
        else:
            max_score = self._score_fn(self.size, self.size)
            self._band_fn = lambda s: RiskLevel.from_score(s, max_score=max_score)
        if level_names is not None and len(level_names) != self.size:
            raise ValueError(
                f"level_names must contain exactly {self.size} entries"
            )
        self._level_names = list(level_names) if level_names else None

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_score_fn(likelihood: int, impact: int) -> float:
        """Default multiplicative scoring ``likelihood * impact``.

        This is the de-facto standard for qualitative risk matrices and
        yields a maximum score of ``size ** 2``.
        """
        return float(likelihood * impact)

    def validate_level(self, level: int, axis: str = "level") -> int:
        """Validate that a level index lies within the matrix bounds.

        Parameters
        ----------
        level:
            1-based level index to validate.
        axis:
            Axis name used in error messages (``"likelihood"`` or
            ``"impact"``).

        Returns
        -------
        int
            The validated level as an ``int``.

        Raises
        ------
        ValueError
            If the level is outside ``[1, size]``.
        """
        if not 1 <= int(level) <= self.size:
            raise ValueError(
                f"{axis} must be in [1, {self.size}], got {level!r}"
            )
        return int(level)

    def score(self, likelihood: int, impact: int) -> float:
        """Compute the numeric score for a matrix cell.

        Parameters
        ----------
        likelihood:
            Likelihood level (1-based).
        impact:
            Impact level (1-based).

        Returns
        -------
        float
            Score produced by the configured scoring function.
        """
        likelihood = self.validate_level(likelihood, "likelihood")
        impact = self.validate_level(impact, "impact")
        return float(self._score_fn(likelihood, impact))

    def classify(self, likelihood: int, impact: int) -> MatrixCell:
        """Classify a single likelihood/impact combination.

        Parameters
        ----------
        likelihood:
            Likelihood level (1-based).
        impact:
            Impact level (1-based).

        Returns
        -------
        MatrixCell
            The fully classified cell (score + risk band).

        Examples
        --------
        >>> matrix = RiskMatrix.default_5x5()
        >>> matrix.classify(5, 5).level
        <RiskLevel.CRITICAL: 'Critical'>
        """
        s = self.score(likelihood, impact)
        return MatrixCell(
            likelihood=int(likelihood),
            impact=int(impact),
            score=s,
            level=self._band_fn(s),
        )

    # ------------------------------------------------------------------
    # Portfolio evaluation
    # ------------------------------------------------------------------
    def evaluate(self, risks: Sequence[RiskItem]) -> List[MatrixCell]:
        """Evaluate a portfolio of risk items against the matrix.

        Parameters
        ----------
        risks:
            Sequence of :class:`~risk_analysis.core.RiskItem` objects
            whose ``likelihood`` and ``impact`` ratings are used.

        Returns
        -------
        list of MatrixCell
            One classified cell per risk, in input order.

        Raises
        ------
        ValueError
            If any risk has ratings outside the matrix bounds.
        """
        return [self.classify(r.likelihood, r.impact) for r in risks]

    def evaluate_with_items(
        self, risks: Sequence[RiskItem]
    ) -> List[tuple]:
        """Evaluate risks and keep them paired with their cells.

        Parameters
        ----------
        risks:
            Sequence of risk items.

        Returns
        -------
        list of tuple
            ``(risk_item, MatrixCell)`` pairs in input order.
        """
        return list(zip(risks, self.evaluate(risks)))

    def heat_map(self, risks: Sequence[RiskItem]) -> Dict[RiskLevel, int]:
        """Aggregate a risk portfolio into a heat-map band census.

        Parameters
        ----------
        risks:
            Sequence of risk items.

        Returns
        -------
        dict
            Mapping from each :class:`RiskLevel` band to the number of
            risks falling into that band.  All four bands are always
            present (zero counts included).
        """
        counts: Dict[RiskLevel, int] = {
            level: 0 for level in RiskLevel
        }
        for cell in self.evaluate(risks):
            counts[cell.level] += 1
        return counts

    def sort_by_severity(
        self, risks: Sequence[RiskItem]
    ) -> List[tuple]:
        """Sort a risk portfolio from most to least severe.

        Ties are broken by likelihood, then impact, then insertion
        order (stable sort).

        Parameters
        ----------
        risks:
            Sequence of risk items.

        Returns
        -------
        list of tuple
            ``(risk_item, MatrixCell)`` pairs sorted by descending
            score.
        """
        pairs = self.evaluate_with_items(risks)
        pairs.sort(
            key=lambda pair: (-pair[1].score, -pair[0].likelihood,
                              -pair[0].impact)
        )
        return pairs

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def level_name(self, level: int) -> str:
        """Return the human-readable name of a level index.

        Parameters
        ----------
        level:
            1-based level index.

        Returns
        -------
        str
            Configured level name or a generic ``"L<n>"`` label.
        """
        self.validate_level(level)
        if self._level_names is not None:
            return self._level_names[level - 1]
        return f"L{level}"

    def all_cells(self) -> List[MatrixCell]:
        """Return every cell of the matrix in row-major order.

        Returns
        -------
        list of MatrixCell
            All ``size * size`` classified cells, ordered from the
            highest likelihood row downwards and lowest impact column
            first.
        """
        return [
            self.classify(likelihood, impact)
            for likelihood in range(self.size, 0, -1)
            for impact in range(1, self.size + 1)
        ]

    @classmethod
    def default_5x5(cls) -> "RiskMatrix":
        """Create the standard 5x5 risk matrix with default scoring.

        Returns
        -------
        RiskMatrix
            A 5x5 matrix using multiplicative scoring and the default
            relative banding thresholds (25% / 50% / 75% of max score).
        """
        return cls(size=5)
