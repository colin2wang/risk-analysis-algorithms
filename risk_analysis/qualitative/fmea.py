"""Failure Mode and Effects Analysis (FMEA) with Risk Priority Numbers.

FMEA is a bottom-up, inductive qualitative technique originating from
the US military standard MIL-P-1629 and later adopted by the automotive
industry (AIAG) and IEC 60812.  Each potential failure mode of a system,
function or process is scored on three 1-10 ordinal scales:

* **Severity (S)** -- how serious the effect of the failure is.
* **Occurrence (O)** -- how frequently the cause of the failure is
  expected to happen.
* **Detection (D)** -- how unlikely the existing controls are to detect
  the cause or failure before it reaches the customer.

The **Risk Priority Number** ``RPN = S * O * D`` (range 1..1000) ranks
failure modes so that corrective actions can be prioritised.  A related
metric, the **criticality** ``S * O``, ignores detectability and is
closer to a classic likelihood-impact product.

Example
-------
>>> from risk_analysis.qualitative import FMEA, FMEAItem
>>> fmea = FMEA()
>>> fmea.add_item(FMEAItem(
...     function="Cooling pump",
...     failure_mode="Bearing seizure",
...     effect="Pump stops, reactor overheat",
...     cause="Lubrication starvation",
...     severity=9, occurrence=3, detection=4,
... ))
>>> fmea.items[0].rpn
108
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..core import RiskItem


@dataclass
class FMEAItem:
    """A single failure mode under FMEA analysis.

    Attributes
    ----------
    function:
        The function / component / process step being analysed.
    failure_mode:
        The manner in which the function fails to meet its intent.
    effect:
        The consequence of the failure mode for the customer or system.
    cause:
        The root cause that produces the failure mode.
    severity:
        Severity rating, 1 (no effect) to 10 (hazardous without
        warning).
    occurrence:
        Occurrence rating, 1 (extremely unlikely) to 10 (inevitable).
    detection:
        Detection rating, 1 (almost certain detection) to 10 (absolute
        uncertainty of detection by current controls).
    item_id:
        Optional stable identifier; auto-generated when omitted.
    current_controls:
        Optional description of existing prevention/detection controls.
    recommended_action:
        Optional description of recommended corrective action.

    Raises
    ------
    ValueError
        If any of S / O / D is outside the 1-10 range.

    Examples
    --------
    >>> item = FMEAItem("Valve", "Stuck closed", "Overpressure",
    ...                 "Corrosion", severity=8, occurrence=5, detection=6)
    >>> item.rpn
    240
    """

    function: str
    failure_mode: str
    effect: str
    cause: str
    severity: int
    occurrence: int
    detection: int
    item_id: str = field(default_factory=lambda: f"FMEA-{id(object()):x}")
    current_controls: str = ""
    recommended_action: str = ""

    def __post_init__(self) -> None:
        """Validate the S/O/D ratings."""
        for name, value in (
            ("severity", self.severity),
            ("occurrence", self.occurrence),
            ("detection", self.detection),
        ):
            if not 1 <= int(value) <= 10:
                raise ValueError(f"{name} must be in [1, 10], got {value!r}")

    @property
    def rpn(self) -> int:
        """Risk Priority Number: ``severity * occurrence * detection``.

        Returns
        -------
        int
            Value in the range 1..1000.
        """
        return int(self.severity * self.occurrence * self.detection)

    @property
    def criticality(self) -> int:
        """Criticality index ``severity * occurrence`` (detectability ignored).

        Returns
        -------
        int
            Value in the range 1..100.
        """
        return int(self.severity * self.occurrence)

    def to_risk_item(self) -> RiskItem:
        """Convert to a generic :class:`~risk_analysis.core.RiskItem`.

        The generic 1-5 likelihood rating is derived from ``occurrence``
        by scaling, and the impact rating from ``severity``.  This allows
        FMEA results to be plotted on a standard risk matrix.

        Returns
        -------
        RiskItem
            The mapped risk item with FMEA fields stored in
            ``attributes``.
        """
        likelihood_5 = max(1, min(5, (self.occurrence + 1) // 2))
        impact_5 = max(1, min(5, (self.severity + 1) // 2))
        return RiskItem(
            name=f"{self.function}: {self.failure_mode}",
            likelihood=likelihood_5,
            impact=impact_5,
            description=self.effect,
            attributes={
                "fmea_id": self.item_id,
                "rpn": self.rpn,
                "criticality": self.criticality,
            },
        )


class FMEA:
    """A worksheet-style container for an FMEA study.

    The class collects :class:`FMEAItem` entries and provides ranking,
    threshold screening and statistics used in FMEA reviews.

    Examples
    --------
    >>> fmea = FMEA(rpn_threshold=150)
    >>> fmea.add_item(FMEAItem("Brake", "Pad wear", "Longer stopping "
    ...     "distance", "Normal wear", severity=8, occurrence=6,
    ...     detection=3))
    FMEA-1
    >>> len(fmea.critical_items()) == 1
    True
    """

    def __init__(self, rpn_threshold: int = 125) -> None:
        """Initialise an empty FMEA worksheet.

        Parameters
        ----------
        rpn_threshold:
            RPN value at or above which an item is considered critical
            and requires corrective action.  Common practice uses
            thresholds between 100 and 200.
        """
        if not 1 <= rpn_threshold <= 1000:
            raise ValueError("rpn_threshold must be in [1, 1000]")
        self.rpn_threshold = int(rpn_threshold)
        self.items: List[FMEAItem] = []
        self._counter = 0

    def add_item(self, item: FMEAItem) -> str:
        """Add a failure mode to the worksheet.

        If the item has no meaningful identifier, a sequential one
        (``FMEA-1``, ``FMEA-2``, ...) is assigned.

        Parameters
        ----------
        item:
            The :class:`FMEAItem` to register.

        Returns
        -------
        str
            The identifier of the added item.
        """
        self._counter += 1
        if not item.item_id or item.item_id.startswith("FMEA-") is False:
            item.item_id = f"FMEA-{self._counter}"
        self.items.append(item)
        return item.item_id

    def ranked(self) -> List[FMEAItem]:
        """Return all items sorted by descending RPN.

        Returns
        -------
        list of FMEAItem
            Items ordered from highest to lowest RPN (stable for ties).
        """
        return sorted(self.items, key=lambda it: it.rpn, reverse=True)

    def critical_items(self) -> List[FMEAItem]:
        """Return items whose RPN meets or exceeds the threshold.

        Returns
        -------
        list of FMEAItem
            Critical items ordered by descending RPN.
        """
        return [it for it in self.ranked() if it.rpn >= self.rpn_threshold]

    def statistics(self) -> dict:
        """Compute summary statistics for the worksheet.

        Returns
        -------
        dict
            Keys:

            * ``total_items`` -- number of failure modes.
            * ``max_rpn`` / ``min_rpn`` / ``mean_rpn`` -- RPN spread.
            * ``critical_count`` -- items at/above the threshold.
            * ``max_criticality`` -- highest S*O value.
        """
        if not self.items:
            return {
                "total_items": 0,
                "max_rpn": 0,
                "min_rpn": 0,
                "mean_rpn": 0.0,
                "critical_count": 0,
                "max_criticality": 0,
            }
        rpns = [it.rpn for it in self.items]
        return {
            "total_items": len(self.items),
            "max_rpn": max(rpns),
            "min_rpn": min(rpns),
            "mean_rpn": sum(rpns) / len(rpns),
            "critical_count": len(self.critical_items()),
            "max_criticality": max(it.criticality for it in self.items),
        }
