"""Unit tests for Fault Tree Analysis."""

import unittest

from risk_analysis.structural import BasicEvent, FaultTree, Gate


def build_sample_tree():
    """Two-channel protection system: top fails if both channels fail.

    Top = AND(OR(A, B), OR(C, D))
    """
    a = BasicEvent("A", 0.10)
    b = BasicEvent("B", 0.20)
    c = BasicEvent("C", 0.30)
    d = BasicEvent("D", 0.40)
    top = Gate("AND", [
        Gate("OR", [a, b]),
        Gate("OR", [c, d]),
    ])
    return FaultTree(top)


class TestFaultTreeProbability(unittest.TestCase):
    def test_single_and_gate(self):
        tree = FaultTree(Gate("AND", [
            BasicEvent("x", 0.5), BasicEvent("y", 0.5)]))
        self.assertAlmostEqual(tree.top_probability(), 0.25)

    def test_single_or_gate(self):
        tree = FaultTree(Gate("OR", [
            BasicEvent("x", 0.5), BasicEvent("y", 0.2)]))
        self.assertAlmostEqual(tree.top_probability(), 1 - 0.5 * 0.8)

    def test_sample_tree_exact_value(self):
        tree = build_sample_tree()
        # Channel1 = 1 - 0.9*0.8 = 0.28 ; Channel2 = 1 - 0.7*0.6 = 0.58
        expected = 0.28 * 0.58
        self.assertAlmostEqual(tree.top_probability(), expected)

    def test_overrides(self):
        tree = build_sample_tree()
        p_zero = tree.top_probability(overrides={"A": 0.0})
        # Channel1 = 1 - 0*0.8 = 1 -> top = Channel2 = 0.58
        self.assertAlmostEqual(p_zero, 0.58)

    def test_probability_validation(self):
        with self.assertRaises(ValueError):
            BasicEvent("bad", 1.5)

    def test_basic_event_collection(self):
        tree = build_sample_tree()
        events = tree.basic_events()
        self.assertEqual(set(events), {"A", "B", "C", "D"})


class TestMinimalCutSets(unittest.TestCase):
    def test_sample_tree_cut_sets(self):
        tree = build_sample_tree()
        cut_sets = tree.minimal_cut_sets()
        expected = {frozenset({"A", "C"}), frozenset({"A", "D"}),
                    frozenset({"B", "C"}), frozenset({"B", "D"})}
        self.assertEqual(set(cut_sets), expected)

    def test_or_tree_singletons(self):
        tree = FaultTree(Gate("OR", [
            BasicEvent("x", 0.1), BasicEvent("y", 0.1)]))
        self.assertEqual(tree.minimal_cut_sets(),
                         [frozenset({"x"}), frozenset({"y"})])

    def test_non_minimal_supersets_removed(self):
        # AND(A, OR(A, B)) -> cut sets {A}, {A, B}; minimal is {A}
        tree = FaultTree(Gate("AND", [
            BasicEvent("A", 0.1),
            Gate("OR", [BasicEvent("A", 0.1), BasicEvent("B", 0.2)]),
        ]))
        self.assertEqual(tree.minimal_cut_sets(), [frozenset({"A"})])

    def test_k_of_n_gate_cut_sets(self):
        tree = FaultTree(Gate("K_OF_N", [
            BasicEvent("x", 0.1), BasicEvent("y", 0.1),
            BasicEvent("z", 0.1)], k=2))
        cut_sets = tree.minimal_cut_sets()
        expected = {frozenset({"x", "y"}), frozenset({"x", "z"}),
                    frozenset({"y", "z"})}
        self.assertEqual(set(cut_sets), expected)


class TestApproximationAndImportance(unittest.TestCase):
    def test_rare_event_bound_is_upper_bound(self):
        tree = build_sample_tree()
        exact = tree.top_probability()
        bound = tree.rare_event_upper_bound()
        self.assertGreaterEqual(bound, exact - 1e-12)

    def test_fussell_vesely_ordering(self):
        # D has the largest probability, so it should dominate FV
        tree = build_sample_tree()
        ranking = tree.importance_ranking()
        self.assertEqual(ranking[0][0], "D")
        for _, fv in ranking:
            self.assertGreaterEqual(fv, 0.0)

    def test_birnbaum_range(self):
        tree = build_sample_tree()
        for name in ("A", "B", "C", "D"):
            b = tree.birnbaum(name)
            self.assertGreaterEqual(b, 0.0)
            self.assertLessEqual(b, 1.0)

    def test_birnbaum_of_essential_event_is_high(self):
        # In AND(top, x): if x=1, top = 1 -> Birnbaum = 1 - 0 = 1
        tree = FaultTree(Gate("AND", [
            BasicEvent("top_in", 0.5), BasicEvent("x", 0.3)]))
        self.assertAlmostEqual(tree.birnbaum("x"), 1.0)

    def test_unknown_event_raises(self):
        tree = build_sample_tree()
        with self.assertRaises(KeyError):
            tree.fussell_vesely("nope")
        with self.assertRaises(KeyError):
            tree.birnbaum("nope")


class TestGateValidation(unittest.TestCase):
    def test_unknown_gate_type(self):
        with self.assertRaises(ValueError):
            Gate("XOR", [BasicEvent("a", 0.1)])

    def test_empty_children(self):
        with self.assertRaises(ValueError):
            Gate("AND", [])

    def test_k_of_n_bounds(self):
        with self.assertRaises(ValueError):
            Gate("K_OF_N", [BasicEvent("a", 0.1)], k=5)

    def test_k_of_n_probability(self):
        # 2-of-2 equals AND
        tree = FaultTree(Gate("K_OF_N", [
            BasicEvent("x", 0.5), BasicEvent("y", 0.5)], k=2))
        self.assertAlmostEqual(tree.top_probability(), 0.25)


if __name__ == "__main__":
    unittest.main()
