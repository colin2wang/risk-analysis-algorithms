"""Decision-tree analysis with Expected Monetary Value (EMV).

Decision-tree analysis is a classic technique for decisions under
uncertainty (PMI PMBOK "Expected Monetary Value analysis").  The tree
contains three node types:

* **Decision nodes** -- points where the decision maker chooses among
  mutually exclusive alternatives.
* **Chance nodes** -- random events with probabilities and outcomes.
* **Leaf nodes** -- terminal payoffs (costs are negative, gains
  positive).

The tree is evaluated bottom-up: every chance node computes the
probability-weighted expected value (EMV) of its branches, and every
decision node selects the alternative with the highest EMV.

Example
-------
>>> from risk_analysis.quantitative import DecisionNode, DecisionTree
>>> tree = DecisionTree(
...     DecisionNode.decision("Build or Buy", [
...         DecisionNode.chance("Build", [
...             (0.7, DecisionNode.leaf("On time", 500_000)),
...             (0.3, DecisionNode.leaf("Delayed", 100_000)),
...         ]),
...         DecisionNode.chance("Buy", [
...             (0.9, DecisionNode.leaf("Works", 400_000)),
...             (0.1, DecisionNode.leaf("Vendor fails", -50_000)),
...         ]),
...     ])
... )
>>> tree.evaluate()
310000.0
>>> tree.best_choice
'Buy'
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from dataclasses import dataclass, field


@dataclass
class DecisionNode:
    """A node in a decision tree.

    Use the factory methods :meth:`decision`, :meth:`chance` and
    :meth:`leaf` to construct nodes; the constructor arguments depend
    on the node kind:

    * ``kind="decision"`` -- ``options`` is a sequence of child nodes.
    * ``kind="chance"`` -- ``branches`` is a sequence of
      ``(probability, child_node)`` tuples whose probabilities must
      sum to 1 (within a small tolerance).
    * ``kind="leaf"`` -- ``payoff`` is the terminal monetary value.
    """

    kind: str
    label: str
    payoff: Optional[float] = None
    branches: Optional[Sequence[Tuple[float, "DecisionNode"]]] = None
    options: Optional[Sequence["DecisionNode"]] = None
    emv: Optional[float] = field(default=None, init=False)
    chosen_option: Optional["DecisionNode"] = field(
        default=None, init=False
    )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @staticmethod
    def decision(label: str, options: Sequence["DecisionNode"]) -> "DecisionNode":
        """Create a decision node with mutually exclusive alternatives.

        Parameters
        ----------
        label:
            Human-readable decision name.
        options:
            Child nodes, one per alternative.

        Returns
        -------
        DecisionNode
            A ``kind="decision"`` node.
        """
        return DecisionNode(kind="decision", label=label, options=options)

    @staticmethod
    def chance(
        label: str, branches: Sequence[Tuple[float, "DecisionNode"]]
    ) -> "DecisionNode":
        """Create a chance (random event) node.

        Parameters
        ----------
        label:
            Human-readable event name.
        branches:
            ``(probability, child)`` tuples.  Probabilities must be in
            ``[0, 1]`` and sum to 1.

        Returns
        -------
        DecisionNode
            A ``kind="chance"`` node.

        Raises
        ------
        ValueError
            If probabilities are invalid or do not sum to 1.
        """
        total = sum(p for p, _ in branches)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"branch probabilities must sum to 1, got {total!r}"
            )
        for p, _ in branches:
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"probabilities must be in [0, 1], got {p!r}")
        return DecisionNode(kind="chance", label=label, branches=branches)

    @staticmethod
    def leaf(label: str, payoff: float) -> "DecisionNode":
        """Create a terminal payoff node.

        Parameters
        ----------
        label:
            Human-readable outcome name.
        payoff:
            Terminal monetary value (costs negative, gains positive).

        Returns
        -------
        DecisionNode
            A ``kind="leaf"`` node.
        """
        return DecisionNode(kind="leaf", label=label, payoff=payoff)


class DecisionTree:
    """An evaluator for decision trees using Expected Monetary Value.

    The evaluation is recursive and bottom-up.  After
    :meth:`evaluate` completes, every node carries its computed EMV in
    ``node.emv`` and every decision node records its optimal
    alternative in ``node.chosen_option``.

    Parameters
    ----------
    root:
        The root :class:`DecisionNode` (usually a decision node, but
        chance or leaf roots are also accepted).

    Examples
    --------
    See the module docstring for a complete example.
    """

    def __init__(self, root: DecisionNode) -> None:
        self.root = root

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, node: Optional[DecisionNode] = None) -> float:
        """Evaluate the tree and return the root EMV.

        Chance nodes compute the probability-weighted sum of branch
        EMVs.  Decision nodes take the maximum of option EMVs and
        record the winning option.  Leaf nodes return their payoff.

        Parameters
        ----------
        node:
            Node to evaluate; defaults to the root.

        Returns
        -------
        float
            The EMV of the evaluated node.

        Raises
        ------
        ValueError
            If the node structure is inconsistent (e.g. a chance node
            without branches).
        """
        node = node or self.root

        if node.kind == "leaf":
            if node.payoff is None:
                raise ValueError(f"leaf {node.label!r} has no payoff")
            node.emv = float(node.payoff)
            return node.emv

        if node.kind == "chance":
            if not node.branches:
                raise ValueError(f"chance node {node.label!r} has no branches")
            node.emv = sum(
                p * self.evaluate(child) for p, child in node.branches
            )
            return node.emv

        if node.kind == "decision":
            if not node.options:
                raise ValueError(f"decision node {node.label!r} has no options")
            values = [(self.evaluate(opt), opt) for opt in node.options]
            best_value, best_option = max(values, key=lambda t: t[0])
            node.emv = best_value
            node.chosen_option = best_option
            return best_value

        raise ValueError(f"unknown node kind: {node.kind!r}")

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    @property
    def best_choice(self) -> Optional[str]:
        """Label of the optimal first-level alternative.

        Returns
        -------
        str or None
            The label of the root's chosen option, or ``None`` if the
            root is not a decision node (or has not been evaluated).
        """
        if self.root.kind == "decision" and self.root.chosen_option:
            return self.root.chosen_option.label
        return None

    def flatten_leaves(self) -> List[tuple]:
        """Enumerate all root-to-leaf paths with their joint probability.

        Returns
        -------
        list of tuple
            ``(path_labels, joint_probability, payoff)`` tuples, where
            ``path_labels`` is the chain of node labels from the root
            to the leaf.  Useful for building EMV sensitivity tables.
        """
        paths: List[tuple] = []

        def walk(node: DecisionNode, labels: List[str], prob: float) -> None:
            label_list = labels + [node.label]
            if node.kind == "leaf":
                paths.append((label_list, prob, float(node.payoff)))
                return
            if node.kind == "chance":
                for p, child in node.branches or []:
                    walk(child, label_list, prob * p)
                return
            for opt in node.options or []:
                walk(opt, label_list, prob)

        walk(self.root, [], 1.0)
        return paths
