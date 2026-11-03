"""Unit tests for decision-tree EMV analysis."""

import unittest

from risk_analysis.quantitative import DecisionNode, DecisionTree


class TestDecisionTree(unittest.TestCase):
    def setUp(self):
        self.tree = DecisionTree(
            DecisionNode.decision("Build or Buy", [
                DecisionNode.chance("Build", [
                    (0.7, DecisionNode.leaf("On time", 500_000)),
                    (0.3, DecisionNode.leaf("Delayed", 100_000)),
                ]),
                DecisionNode.chance("Buy", [
                    (0.9, DecisionNode.leaf("Works", 400_000)),
                    (0.1, DecisionNode.leaf("Vendor fails", -50_000)),
                ]),
            ])
        )

    def test_emv_chance_node(self):
        build = DecisionNode.chance("B", [
            (0.5, DecisionNode.leaf("win", 100)),
            (0.5, DecisionNode.leaf("lose", -50)),
        ])
        self.assertEqual(DecisionTree(build).evaluate(), 25.0)

    def test_root_emv_and_best_choice(self):
        emv = self.tree.evaluate()
        # Build EMV = 0.7*500k + 0.3*100k = 380k
        # Buy   EMV = 0.9*400k + 0.1*(-50k) = 355k
        self.assertEqual(emv, 380_000.0)
        self.assertEqual(self.tree.best_choice, "Build")

    def test_node_annotations_populated(self):
        self.tree.evaluate()
        self.assertAlmostEqual(self.tree.root.options[0].emv, 380_000.0)
        self.assertIsNotNone(self.tree.root.chosen_option)

    def test_nested_decision_tree(self):
        tree = DecisionTree(
            DecisionNode.decision("Outer", [
                DecisionNode.decision("Inner", [
                    DecisionNode.leaf("sure thing", 100),
                ]),
                DecisionNode.leaf("fallback", 60),
            ])
        )
        self.assertEqual(tree.evaluate(), 100.0)
        self.assertEqual(tree.best_choice, "Inner")

    def test_probabilities_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            DecisionNode.chance("Bad", [
                (0.5, DecisionNode.leaf("a", 1)),
                (0.2, DecisionNode.leaf("b", 2)),
            ])

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            DecisionNode.chance("Bad", [
                (1.5, DecisionNode.leaf("a", 1)),
            ])

    def test_flatten_leaves_paths(self):
        self.tree.evaluate()
        paths = self.tree.flatten_leaves()
        self.assertEqual(len(paths), 4)
        labels, prob, payoff = paths[0]
        self.assertEqual(labels[-1], "On time")
        self.assertAlmostEqual(prob, 0.7)
        self.assertEqual(payoff, 500_000.0)
        total_prob = sum(p for _, p, _ in paths)
        self.assertAlmostEqual(total_prob, 2.0)  # two alternatives x 1.0

    def test_chance_node_requires_branches(self):
        empty = DecisionNode(kind="chance", label="Empty", branches=[])
        with self.assertRaises(ValueError):
            DecisionTree(empty).evaluate()


if __name__ == "__main__":
    unittest.main()
