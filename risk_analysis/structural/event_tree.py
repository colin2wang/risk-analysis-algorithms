"""Event Tree Analysis (ETA).

Event Tree Analysis is a forward, inductive technique (complementary
to the deductive fault tree) standardised alongside FTA in IEC 61025
and ubiquitous in probabilistic risk assessment (PRA) of nuclear
plants, chemical processes and safety systems.

Starting from an **initiating event**, the analysis asks, for each
safety function / system in a fixed order, whether it succeeds or
fails.  Every combination of successes and failures defines an
**accident sequence** whose probability is the product of the
branching probabilities along the path.  Each sequence is assigned a
consequence category, yielding a full frequency-consequence spectrum.

Example
-------
>>> from risk_analysis.structural import EventTree, Branch
>>> tree = EventTree(initiating_event="Pipe rupture",
...                  initiating_frequency=1e-4)
>>> tree.add_function("High pressure injection",
...                   Branch(success_prob=0.95, success="OK",
...                          failure="Core damage"))
>>> tree.add_function("Containment spray",
...                   Branch(success_prob=0.9, success="OK",
...                          failure="Offsite release"))
>>> sequences = tree.sequences()
>>> total = sum(s["frequency"] for s in sequences)
>>> abs(total - 1e-4) < 1e-12
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Branch:
    """Branching definition for one safety function (event tree heading).

    Attributes
    ----------
    success_prob:
        Probability that the function succeeds on demand (0..1).
    success:
        Consequence label when the function succeeds.
    failure:
        Consequence label when the function fails.  If omitted, the
        previous non-``"OK"`` consequence propagates (standard PRA
        behaviour: once containment is lost, later successes do not
        restore it).
    name:
        Optional short identifier for the function.

    Raises
    ------
    ValueError
        If ``success_prob`` is outside ``[0, 1]``.
    """

    success_prob: float
    success: str = "OK"
    failure: str = ""
    name: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.success_prob <= 1.0:
            raise ValueError(
                f"success_prob must be in [0, 1], got {self.success_prob!r}"
            )


class EventTree:
    """An event tree with ordered safety-function headings.

    Parameters
    ----------
    initiating_event:
        Name of the initiating event.
    initiating_frequency:
        Expected frequency of the initiating event per year (>= 0).

    Raises
    ------
    ValueError
        If the initiating frequency is negative.
    """

    def __init__(self, initiating_event: str, initiating_frequency: float) -> None:
        if initiating_frequency < 0:
            raise ValueError("initiating_frequency must be non-negative")
        self.initiating_event = initiating_event
        self.initiating_frequency = float(initiating_frequency)
        self.functions: List[Branch] = []

    def add_function(self, branch: Branch, name: Optional[str] = None) -> None:
        """Append a safety-function heading to the tree.

        Parameters
        ----------
        branch:
            The :class:`Branch` describing success/failure.
        name:
            Optional function name; falls back to ``branch.name`` or a
            generic ``"F<n>"`` label.
        """
        label = name or branch.name or f"F{len(self.functions) + 1}"
        branch.name = label
        self.functions.append(branch)

    def sequences(self) -> List[dict]:
        """Enumerate all accident sequences of the tree.

        Returns
        -------
        list of dict
            One entry per sequence with keys:

            * ``path`` -- list of ``"function: outcome"`` strings in
              heading order.
            * ``probability`` -- joint branching probability.
            * ``frequency`` -- ``initiating_frequency * probability``.
            * ``consequence`` -- the final consequence label (the last
              non-empty outcome, or ``"OK"`` if all functions
              succeeded).
        """
        results: List[dict] = []

        def walk(
            index: int,
            prob: float,
            path: List[str],
            consequence: str,
        ) -> None:
            if index == len(self.functions):
                results.append({
                    "path": list(path),
                    "probability": prob,
                    "frequency": self.initiating_frequency * prob,
                    "consequence": consequence or "OK",
                })
                return
            branch = self.functions[index]
            # Success branch
            walk(
                index + 1,
                prob * branch.success_prob,
                path + [f"{branch.name}: success"],
                branch.success if branch.success else consequence,
            )
            # Failure branch
            fail_consequence = branch.failure or consequence
            walk(
                index + 1,
                prob * (1.0 - branch.success_prob),
                path + [f"{branch.name}: failure"],
                fail_consequence,
            )

        walk(0, 1.0, [], "")
        return results

    def total_risk(self) -> dict:
        """Aggregate sequence frequencies by consequence category.

        Returns
        -------
        dict
            Mapping from consequence label to the summed annual
            frequency of all sequences ending in that consequence.
        """
        aggregated: dict = {}
        for seq in self.sequences():
            aggregated[seq["consequence"]] = (
                aggregated.get(seq["consequence"], 0.0) + seq["frequency"]
            )
        return aggregated

    def check_probabilities(self) -> bool:
        """Verify that every path probability structure is coherent.

        Because each heading contributes ``p + (1 - p) = 1``, the
        sequence probabilities always sum to 1.  This method confirms
        that invariant numerically (useful as a self-test after
        programmatic tree construction).

        Returns
        -------
        bool
            ``True`` when the summed sequence probability equals 1
            within floating-point tolerance.
        """
        return abs(sum(s["probability"] for s in self.sequences()) - 1.0) < 1e-9
