"""Unit tests for Event Tree Analysis."""

import unittest

from risk_analysis.structural import Branch, EventTree


class TestEventTree(unittest.TestCase):
    def setUp(self):
        self.tree = EventTree("Pipe rupture", initiating_frequency=1e-4)
        self.tree.add_function(Branch(
            success_prob=0.95, success="OK", failure="Core damage"),
            name="HPI")
        self.tree.add_function(Branch(
            success_prob=0.90, success="OK", failure="Release"),
            name="Spray")

    def test_sequence_count(self):
        # Two binary headings -> 4 sequences
        self.assertEqual(len(self.tree.sequences()), 4)

    def test_frequencies_conserve_total(self):
        total = sum(s["frequency"] for s in self.tree.sequences())
        self.assertAlmostEqual(total, 1e-4)

    def test_all_success_sequence(self):
        seqs = self.tree.sequences()
        best = max(seqs, key=lambda s: s["probability"])
        self.assertAlmostEqual(best["probability"], 0.95 * 0.90)
        self.assertEqual(best["consequence"], "OK")

    def test_consequence_propagation(self):
        # HPI failure followed by Spray failure -> "Core damage"
        # (first non-OK consequence sticks)
        seqs = self.tree.sequences()
        worst = min(seqs, key=lambda s: s["probability"])
        self.assertAlmostEqual(worst["probability"], 0.05 * 0.10)
        self.assertEqual(worst["consequence"], "Core damage")

    def test_total_risk_aggregation(self):
        profile = self.tree.total_risk()
        self.assertAlmostEqual(sum(profile.values()), 1e-4)
        self.assertIn("OK", profile)
        self.assertIn("Core damage", profile)

    def test_probability_invariant(self):
        self.assertTrue(self.tree.check_probabilities())

    def test_negative_frequency_rejected(self):
        with self.assertRaises(ValueError):
            EventTree("X", initiating_frequency=-1)

    def test_invalid_success_probability(self):
        with self.assertRaises(ValueError):
            Branch(success_prob=1.2)

    def test_single_heading(self):
        tree = EventTree("Initiator", 1.0)
        tree.add_function(Branch(success_prob=0.8, failure="Damage"),
                          name="Guard")
        seqs = tree.sequences()
        self.assertEqual(len(seqs), 2)
        by_freq = {s["consequence"]: s["frequency"] for s in seqs}
        self.assertAlmostEqual(by_freq["OK"], 0.8)
        self.assertAlmostEqual(by_freq["Damage"], 0.2)


if __name__ == "__main__":
    unittest.main()
