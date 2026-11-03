"""Bow-Tie analysis with barrier modelling.

The Bow-Tie diagram unifies fault tree thinking (causes on the left)
and event tree thinking (consequences on the right) around a central
**top event**.  It is the standard visual risk method in oil & gas
(OGP/IPIECA), aviation (ICAO) and process safety (CCPS).

Structure
---------
* **Threats** -- causes that can lead to the top event, each with an
  annual frequency.
* **Preventive barriers** -- controls placed between a threat and the
  top event; each has a Probability of Failure on Demand (PFD).
  The threat's frequency is reduced by the product of barrier
  effectiveness: ``freq_after = freq * prod(PFD_i)``.
* **Mitigative barriers** -- controls between the top event and a
  consequence, reducing the escalation frequency in the same way.
* **Consequences** -- outcomes of the top event, each with an
  unmitigated escalation fraction and mitigative barriers.

Example
-------
>>> from risk_analysis.structural import BowTie, Barrier
>>> bt = BowTie(top_event="Loss of containment")
>>> bt.add_threat("Corrosion", frequency=0.1,
...               barriers=[Barrier("Inspection", pfd=0.05)])
>>> bt.add_consequence("Fire", fraction=0.4,
...                    barriers=[Barrier("Deluge system", pfd=0.02)])
>>> round(bt.threat_frequency_after("Corrosion"), 5)
0.005
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class Barrier:
    """A control barrier with a Probability of Failure on Demand.

    Attributes
    ----------
    name:
        Barrier name (e.g. ``"Relief valve"``, ``"Operator training"``).
    pfd:
        Probability of Failure on Demand in ``[0, 1]`` -- the chance
        the barrier fails when called upon.  ``pfd=0.01`` means the
        barrier is 99% effective per demand.
    kind:
        ``"preventive"`` or ``"mitigative"`` (informational; the
        placement is determined by which side of the bow-tie the
        barrier is registered on).
    barrier_type:
        Free classification such as hardware, human, or organisational.

    Raises
    ------
    ValueError
        If ``pfd`` is outside ``[0, 1]``.
    """

    name: str
    pfd: float
    kind: str = "preventive"
    barrier_type: str = "hardware"

    def __post_init__(self) -> None:
        if not 0.0 <= self.pfd <= 1.0:
            raise ValueError(f"pfd must be in [0, 1], got {self.pfd!r}")

    @property
    def effectiveness(self) -> float:
        """Probability the barrier performs on demand (``1 - pfd``)."""
        return 1.0 - self.pfd


@dataclass
class Threat:
    """A threat (cause) leading to the top event.

    Attributes
    ----------
    name:
        Threat name.
    frequency:
        Unmitigated annual frequency of the threat.
    barriers:
        Preventive barriers in place for this threat.
    """

    name: str
    frequency: float
    barriers: List[Barrier] = field(default_factory=list)


@dataclass
class Consequence:
    """A consequence following the top event.

    Attributes
    ----------
    name:
        Consequence name (e.g. ``"Fatality"``, ``"Data breach"``).
    fraction:
        Conditional fraction of top events escalating into this
        consequence *before* mitigative barriers (fractions across all
        consequences should sum to <= 1; the remainder is benign).
    barriers:
        Mitigative barriers in place for this consequence.
    """

    name: str
    fraction: float
    barriers: List[Barrier] = field(default_factory=list)


class BowTie:
    """A complete bow-tie diagram with frequency propagation.

    Parameters
    ----------
    top_event:
        Name of the central top (critical) event.
    hazardous_event_frequency:
        Optional externally supplied frequency of the top event.  When
        omitted, it is derived as the sum of mitigated threat
        frequencies.

    Examples
    --------
    See the module docstring.
    """

    def __init__(
        self,
        top_event: str,
        hazardous_event_frequency: Optional[float] = None,
    ) -> None:
        if hazardous_event_frequency is not None and hazardous_event_frequency < 0:
            raise ValueError("hazardous_event_frequency must be non-negative")
        self.top_event = top_event
        self._explicit_frequency = hazardous_event_frequency
        self.threats: Dict[str, Threat] = {}
        self.consequences: Dict[str, Consequence] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def add_threat(
        self,
        name: str,
        frequency: float,
        barriers: Optional[Sequence[Barrier]] = None,
    ) -> None:
        """Register a threat with optional preventive barriers.

        Parameters
        ----------
        name:
            Threat name; must be unique.
        frequency:
            Unmitigated annual frequency (>= 0).
        barriers:
            Preventive :class:`Barrier` objects.
        """
        if frequency < 0:
            raise ValueError("frequency must be non-negative")
        if name in self.threats:
            raise ValueError(f"duplicate threat: {name!r}")
        for barrier in barriers or []:
            barrier.kind = "preventive"
        self.threats[name] = Threat(
            name=name, frequency=float(frequency),
            barriers=list(barriers or []),
        )

    def add_consequence(
        self,
        name: str,
        fraction: float,
        barriers: Optional[Sequence[Barrier]] = None,
    ) -> None:
        """Register a consequence with optional mitigative barriers.

        Parameters
        ----------
        name:
            Consequence name; must be unique.
        fraction:
            Conditional escalation fraction in ``[0, 1]``.
        barriers:
            Mitigative :class:`Barrier` objects.
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        if name in self.consequences:
            raise ValueError(f"duplicate consequence: {name!r}")
        for barrier in barriers or []:
            barrier.kind = "mitigative"
        self.consequences[name] = Consequence(
            name=name, fraction=float(fraction),
            barriers=list(barriers or []),
        )

    # ------------------------------------------------------------------
    # Frequency propagation
    # ------------------------------------------------------------------
    def threat_frequency_after(self, threat_name: str) -> float:
        """Mitigated annual frequency of one threat reaching the top event.

        Parameters
        ----------
        threat_name:
            Name of the threat.

        Returns
        -------
        float
            ``frequency * product(barrier PFDs)``.
        """
        threat = self.threats[threat_name]
        pfd_product = 1.0
        for barrier in threat.barriers:
            pfd_product *= barrier.pfd
        return threat.frequency * pfd_product

    def top_event_frequency(self) -> float:
        """Total frequency of the top event.

        Uses the explicitly supplied frequency when available;
        otherwise sums the mitigated frequencies of all threats.

        Returns
        -------
        float
            Top-event annual frequency.
        """
        if self._explicit_frequency is not None:
            return self._explicit_frequency
        return sum(
            self.threat_frequency_after(name) for name in self.threats
        )

    def consequence_frequency(self, consequence_name: str) -> float:
        """Mitigated annual frequency of one consequence.

        Parameters
        ----------
        consequence_name:
            Name of the consequence.

        Returns
        -------
        float
            ``top_freq * fraction * product(mitigative PFDs)``.
        """
        consequence = self.consequences[consequence_name]
        pfd_product = 1.0
        for barrier in consequence.barriers:
            pfd_product *= barrier.pfd
        return self.top_event_frequency() * consequence.fraction * pfd_product

    def risk_profile(self) -> dict:
        """Compute the full bow-tie risk profile.

        Returns
        -------
        dict
            Keys: ``top_event``, ``top_event_frequency``, ``threats``
            (name -> mitigated frequency) and ``consequences``
            (name -> mitigated frequency).
        """
        return {
            "top_event": self.top_event,
            "top_event_frequency": self.top_event_frequency(),
            "threats": {
                name: self.threat_frequency_after(name)
                for name in self.threats
            },
            "consequences": {
                name: self.consequence_frequency(name)
                for name in self.consequences
            },
        }
