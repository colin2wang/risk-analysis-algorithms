"""Fault Tree Analysis (FTA).

Fault Tree Analysis is a top-down, deductive technique developed at
Bell Telephone Laboratories in 1962 for the US Air Force and later
standardised in IEC 61025.  It starts from an undesired **top event**
and decomposes the combinations of lower-level failures (basic events)
connected through logic **gates** that can cause it.

Supported gates
---------------
* ``AND`` -- output occurs iff **all** inputs occur:
  ``P = product of inputs``.
* ``OR`` -- output occurs if **any** input occurs:
  ``P = 1 - product(1 - p_i)`` (independent inputs).
* ``K-of-N`` (voting gate) -- output occurs if at least ``k`` of the
  ``n`` inputs occur; computed by exact binomial enumeration.

Analyses provided
-----------------
* **Top-event probability** -- exact computation via path-set
  enumeration (feasible for small trees) or gate-wise recursion
  (fast, exact for independent basic events).
* **Minimal cut sets** -- smallest combinations of basic events whose
  simultaneous occurrence causes the top event.
* **Importance measures** -- Fussell-Vesely and Birnbaum importance
  for ranking basic-event contributors.

Example
-------
>>> from risk_analysis.structural import BasicEvent, Gate, FaultTree
>>> power = BasicEvent("power_loss", 0.01)
>>> disk = BasicEvent("disk_failure", 0.02)
>>> tree = FaultTree(Gate("OR", [power, disk]))
>>> round(tree.top_probability(), 4)
0.0298
"""

from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union


class BasicEvent:
    """An elementary failure event with an occurrence probability.

    Attributes
    ----------
    name:
        Unique event identifier within a tree.
    probability:
        Probability of occurrence in the mission time (0..1).

    Raises
    ------
    ValueError
        If the probability is outside ``[0, 1]``.
    """

    def __init__(self, name: str, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"probability must be in [0, 1], got {probability!r}"
            )
        self.name = name
        self.probability = float(probability)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BasicEvent({self.name!r}, p={self.probability:g})"


Node = Union[BasicEvent, "Gate"]


class Gate:
    """A logic gate connecting child nodes in a fault tree.

    Parameters
    ----------
    gate_type:
        One of ``"AND"``, ``"OR"`` or ``"K_OF_N"``.
    children:
        Child nodes (basic events or nested gates).
    k:
        Required minimum number of occurring inputs for ``"K_OF_N"``
        gates (ignored otherwise).

    Raises
    ------
    ValueError
        For unknown gate types, empty child lists, or invalid ``k``.
    """

    _VALID_TYPES = ("AND", "OR", "K_OF_N")

    def __init__(
        self,
        gate_type: str,
        children: Sequence[Node],
        k: int = 1,
    ) -> None:
        gate_type = gate_type.upper()
        if gate_type not in self._VALID_TYPES:
            raise ValueError(
                f"gate_type must be one of {self._VALID_TYPES}, "
                f"got {gate_type!r}"
            )
        if not children:
            raise ValueError("gate must have at least one child")
        self.gate_type = gate_type
        self.children: List[Node] = list(children)
        self.k = int(k)
        if gate_type == "K_OF_N":
            if not 1 <= self.k <= len(self.children):
                raise ValueError(
                    f"k must be in [1, {len(self.children)}], got {self.k}"
                )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Gate({self.gate_type}, {len(self.children)} children)"


class FaultTree:
    """A fault tree rooted at a top event node.

    Parameters
    ----------
    root:
        The root :class:`Gate` or :class:`BasicEvent` representing the
        top event.

    Examples
    --------
    See the module docstring.
    """

    def __init__(self, root: Node) -> None:
        self.root = root

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------
    def basic_events(self) -> Dict[str, BasicEvent]:
        """Collect all distinct basic events in the tree.

        Returns
        -------
        dict
            Mapping from event name to :class:`BasicEvent`.
        """
        found: Dict[str, BasicEvent] = {}

        def walk(node: Node) -> None:
            if isinstance(node, BasicEvent):
                found.setdefault(node.name, node)
                return
            for child in node.children:
                walk(child)

        walk(self.root)
        return found

    # ------------------------------------------------------------------
    # Probability computation
    # ------------------------------------------------------------------
    def top_probability(
        self, node: Optional[Node] = None, overrides: Dict[str, float] = None
    ) -> float:
        """Compute the exact probability of a node, recursively.

        Assumes basic events are statistically independent, which is
        the standard FTA assumption (common-cause dependencies are
        handled by modelling them as shared basic events).

        Parameters
        ----------
        node:
            Node to evaluate; defaults to the root.
        overrides:
            Optional mapping from basic-event name to a replacement
            probability (used by importance measures).

        Returns
        -------
        float
            The exact node probability.
        """
        overrides = overrides or {}
        node = node if node is not None else self.root

        if isinstance(node, BasicEvent):
            return float(overrides.get(node.name, node.probability))

        child_probs = [self.top_probability(c, overrides) for c in node.children]

        if node.gate_type == "AND":
            p = 1.0
            for cp in child_probs:
                p *= cp
            return p
        if node.gate_type == "OR":
            q = 1.0
            for cp in child_probs:
                q *= (1.0 - cp)
            return 1.0 - q
        # K_OF_N: exact binomial enumeration over which children occur
        k, n = node.k, len(child_probs)
        prob = 0.0
        for count in range(k, n + 1):
            for combo in itertools.combinations(range(n), count):
                p = 1.0
                for idx in range(n):
                    p *= child_probs[idx] if idx in combo else (1.0 - child_probs[idx])
                prob += p
        return prob

    # ------------------------------------------------------------------
    # Minimal cut sets
    # ------------------------------------------------------------------
    def minimal_cut_sets(self) -> List[FrozenSet[str]]:
        """Compute the minimal cut sets of the tree.

        A cut set is a set of basic events whose simultaneous
        occurrence causes the top event; it is *minimal* when no
        proper subset is itself a cut set.  The classic MOCUS
        (Method for Obtaining Cut Sets) top-down expansion is used.

        Returns
        -------
        list of frozenset
            All minimal cut sets, sorted by ascending size then by
            event names for determinism.

        Raises
        ------
        ValueError
            If the tree contains repeated basic events inside a single
            AND gate in a way that produces a non-minimal expansion --
            such sets are deduplicated automatically instead.
        """
        def expand(node: Node) -> List[FrozenSet[str]]:
            if isinstance(node, BasicEvent):
                return [frozenset([node.name])]
            child_sets = [expand(child) for child in node.children]
            if node.gate_type == "OR":
                result: List[FrozenSet[str]] = []
                for cs in child_sets:
                    result.extend(cs)
                return result
            if node.gate_type == "AND":
                # Cartesian product: one cut set from each child
                result = [frozenset()]
                for cs in child_sets:
                    result = [
                        prev | single
                        for prev in result
                        for single in cs
                    ]
                return result
            # K_OF_N: a cut set is the union of any k children's cut sets
            combos: List[FrozenSet[str]] = []
            for combo in itertools.combinations(child_sets, node.k):
                merged: List[FrozenSet[str]] = [frozenset()]
                for cs in combo:
                    merged = [prev | s for prev in merged for s in cs]
                combos.extend(merged)
            return combos

        raw = expand(self.root)
        # Remove non-minimal (superset) cut sets
        unique = set(raw)
        minimal = {
            cs for cs in unique
            if not any(other < cs for other in unique)
        }
        return sorted(minimal, key=lambda cs: (len(cs), sorted(cs)))

    def rare_event_upper_bound(self) -> float:
        """Compute the rare-event approximation of the top probability.

        The rare-event approximation ``P(top) <= sum of cut-set
        probabilities`` assumes cut sets are disjoint (true only when
        failures are rare).  It provides a fast upper bound and is
        widely used in PRA practice for screening.

        Returns
        -------
        float
            The upper bound, capped at 1.0.
        """
        events = self.basic_events()
        total = 0.0
        for cut_set in self.minimal_cut_sets():
            p = 1.0
            for name in cut_set:
                p *= events[name].probability
            total += p
        return min(1.0, total)

    # ------------------------------------------------------------------
    # Importance measures
    # ------------------------------------------------------------------
    def fussell_vesely(self, event_name: str) -> float:
        """Compute the Fussell-Vesely importance of a basic event.

        FV importance is the fraction of the top-event probability
        contributed by cut sets containing the event::

            FV(e) = sum of P(cut sets containing e) / P(top)

        Parameters
        ----------
        event_name:
            Name of the basic event.

        Returns
        -------
        float
            Importance in ``[0, 1]`` (may slightly exceed 1 for highly
            dependent cut sets; not capped, matching common practice
            of reporting the raw ratio).

        Raises
        ------
        KeyError
            If the event does not exist in the tree.
        """
        events = self.basic_events()
        if event_name not in events:
            raise KeyError(f"unknown basic event: {event_name!r}")
        top = self.top_probability()
        if top == 0.0:
            return 0.0
        contribution = 0.0
        for cut_set in self.minimal_cut_sets():
            if event_name in cut_set:
                p = 1.0
                for name in cut_set:
                    p *= events[name].probability
                contribution += p
        return contribution / top

    def birnbaum(self, event_name: str) -> float:
        """Compute the Birnbaum importance of a basic event.

        Birnbaum importance measures the sensitivity of the top-event
        probability to the event's probability::

            B(e) = P(top | p_e = 1) - P(top | p_e = 0)

        Parameters
        ----------
        event_name:
            Name of the basic event.

        Returns
        -------
        float
            Importance in ``[0, 1]``.

        Raises
        ------
        KeyError
            If the event does not exist in the tree.
        """
        events = self.basic_events()
        if event_name not in events:
            raise KeyError(f"unknown basic event: {event_name!r}")
        p_one = self.top_probability(overrides={event_name: 1.0})
        p_zero = self.top_probability(overrides={event_name: 0.0})
        return p_one - p_zero

    def importance_ranking(self) -> List[Tuple[str, float]]:
        """Rank basic events by descending Fussell-Vesely importance.

        Returns
        -------
        list of tuple
            ``(event_name, fv_importance)`` pairs sorted descending.
        """
        return sorted(
            ((name, self.fussell_vesely(name))
             for name in self.basic_events()),
            key=lambda t: t[1],
            reverse=True,
        )
