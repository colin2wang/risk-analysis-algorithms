"""Core shared building blocks for the risk analysis library.

This module defines the fundamental data structures and conventions used
across all risk analysis techniques:

* :class:`RiskItem` -- a lightweight representation of an individual risk
  (hazard, threat, failure mode, ...) with ordinal likelihood and impact
  ratings.
* :class:`LikelihoodScale` / :class:`ImpactScale` -- standard 5-point
  ordinal scales with human-readable descriptors.
* :class:`RiskLevel` -- the conventional four-band risk classification
  (Low / Medium / High / Critical).

All classes are deliberately dependency-free (pure Python + ``enum``) so
they can be reused by every sub-package without pulling in numpy.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RiskLevel(Enum):
    """Four-band risk classification used throughout the library.

    The bands follow common information-security / project-management
    conventions and are ordered from lowest to highest severity.
    """

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

    @staticmethod
    def from_score(score: float, max_score: float = 25.0) -> "RiskLevel":
        """Map a numeric risk score onto a :class:`RiskLevel` band.

        The mapping uses relative thresholds of the achievable maximum
        score so it works for any scale combination (e.g. 5x5 -> max 25,
        4x4 -> max 16).

        Parameters
        ----------
        score:
            The computed risk score (typically ``likelihood * impact``).
        max_score:
            The maximum achievable score on the scale in use.

        Returns
        -------
        RiskLevel
            The classification band:

            * ``score <= 25%`` of max  -> LOW
            * ``score <= 50%`` of max  -> MEDIUM
            * ``score <= 75%`` of max  -> HIGH
            * otherwise                -> CRITICAL
        """
        if max_score <= 0:
            raise ValueError("max_score must be positive")
        ratio = score / max_score
        if ratio <= 0.25:
            return RiskLevel.LOW
        if ratio <= 0.50:
            return RiskLevel.MEDIUM
        if ratio <= 0.75:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL


class LikelihoodScale(Enum):
    """Standard 5-point qualitative likelihood scale.

    Each level carries a short descriptor describing the expected
    frequency of occurrence within the assessment horizon.
    """

    RARE = (1, "Rare - may occur only in exceptional circumstances")
    UNLIKELY = (2, "Unlikely - could occur at some time")
    POSSIBLE = (3, "Possible - might occur at some time")
    LIKELY = (4, "Likely - will probably occur in most circumstances")
    ALMOST_CERTAIN = (5, "Almost certain - expected to occur")

    @property
    def level(self) -> int:
        """Numeric level of the likelihood (1 = rare ... 5 = certain)."""
        return self.value[0]

    @property
    def description(self) -> str:
        """Human-readable descriptor of the likelihood level."""
        return self.value[1]


class ImpactScale(Enum):
    """Standard 5-point qualitative impact (consequence) scale.

    Descriptors are written generically so the scale applies to safety,
    financial, operational and reputational impacts alike.
    """

    NEGLIGIBLE = (1, "Negligible - no material harm")
    MINOR = (2, "Minor - limited, easily absorbed harm")
    MODERATE = (3, "Moderate - noticeable harm, remediation required")
    MAJOR = (4, "Major - serious harm to objectives or operations")
    CATASTROPHIC = (5, "Catastrophic - threat to viability / mission")

    @property
    def level(self) -> int:
        """Numeric level of the impact (1 = negligible ... 5 = catastrophic)."""
        return self.value[0]

    @property
    def description(self) -> str:
        """Human-readable descriptor of the impact level."""
        return self.value[1]


@dataclass
class RiskItem:
    """A single risk entry shared by qualitative assessment methods.

    A :class:`RiskItem` captures the minimal information needed by the
    risk matrix and FMEA modules: an identifier, a human-readable name
    and ordinal likelihood / impact ratings on a 1-5 scale.

    Attributes
    ----------
    name:
        Short human-readable name of the risk.
    likelihood:
        Ordinal likelihood rating (1-5).  See :class:`LikelihoodScale`.
    impact:
        Ordinal impact rating (1-5).  See :class:`ImpactScale`.
    risk_id:
        Optional stable identifier; auto-generated when omitted.
    description:
        Optional free-text description of the risk.
    owner:
        Optional accountable party (person, team, department).
    attributes:
        Free-form dictionary for method-specific extra fields (e.g.
        severity / occurrence / detection for FMEA).

    Examples
    --------
    >>> risk = RiskItem(name="Data breach", likelihood=3, impact=5)
    >>> risk.risk_score
    15
    """

    name: str
    likelihood: int
    impact: int
    risk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    owner: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ordinal ratings immediately upon construction."""
        if not 1 <= int(self.likelihood) <= 5:
            raise ValueError(
                f"likelihood must be in [1, 5], got {self.likelihood!r}"
            )
        if not 1 <= int(self.impact) <= 5:
            raise ValueError(f"impact must be in [1, 5], got {self.impact!r}")

    @property
    def risk_score(self) -> int:
        """Default risk score: the product ``likelihood * impact``.

        This multiplicative convention is the most widely used scoring
        rule for 5x5 risk matrices.  Specialised modules (FMEA, AHP,
        ...) may compute alternative scores from ``attributes``.
        """
        return int(self.likelihood) * int(self.impact)

    @property
    def risk_level(self) -> RiskLevel:
        """Risk classification band derived from :meth:`risk_score`."""
        return RiskLevel.from_score(self.risk_score, max_score=25.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the risk item into a plain dictionary.

        Returns
        -------
        dict
            A JSON-serialisable representation including the derived
            score and level for reporting convenience.
        """
        return {
            "risk_id": self.risk_id,
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "likelihood": self.likelihood,
            "impact": self.impact,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "attributes": dict(self.attributes),
        }
