"""Unit tests for Bow-Tie barrier analysis."""

import unittest

from risk_analysis.structural import BowTie, Barrier


class TestBowTie(unittest.TestCase):
    def setUp(self):
        self.bt = BowTie(top_event="Loss of containment")
        self.bt.add_threat("Corrosion", frequency=0.10,
                           barriers=[Barrier("Inspection", pfd=0.05)])
        self.bt.add_threat("Human error", frequency=0.20)
        self.bt.add_consequence("Fire", fraction=0.4,
                                barriers=[Barrier("Deluge", pfd=0.02)])
        self.bt.add_consequence("Benign release", fraction=0.6)

    def test_threat_frequency_reduction(self):
        # 0.10 * 0.05 = 0.005
        self.assertAlmostEqual(
            self.bt.threat_frequency_after("Corrosion"), 0.005)
        # No barriers -> unchanged
        self.assertAlmostEqual(
            self.bt.threat_frequency_after("Human error"), 0.20)

    def test_top_event_frequency_is_sum(self):
        self.assertAlmostEqual(self.bt.top_event_frequency(), 0.205)

    def test_explicit_top_frequency_overrides(self):
        bt = BowTie("Top", hazardous_event_frequency=0.5)
        self.assertAlmostEqual(bt.top_event_frequency(), 0.5)

    def test_consequence_frequency(self):
        # top 0.205 * fraction 0.4 * pfd 0.02
        expected = 0.205 * 0.4 * 0.02
        self.assertAlmostEqual(
            self.bt.consequence_frequency("Fire"), expected)

    def test_multiple_barriers_multiply(self):
        bt = BowTie("Top")
        bt.add_threat("T", frequency=1.0, barriers=[
            Barrier("B1", pfd=0.1), Barrier("B2", pfd=0.1)])
        self.assertAlmostEqual(bt.threat_frequency_after("T"), 0.01)

    def test_risk_profile_structure(self):
        profile = self.bt.risk_profile()
        self.assertEqual(profile["top_event"], "Loss of containment")
        self.assertAlmostEqual(profile["threats"]["Corrosion"], 0.005)
        self.assertAlmostEqual(profile["consequences"]["Fire"],
                               0.205 * 0.4 * 0.02)

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            self.bt.add_threat("Corrosion", frequency=0.1)
        with self.assertRaises(ValueError):
            self.bt.add_consequence("Fire", fraction=0.1)

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.bt.add_threat("Neg", frequency=-1)
        with self.assertRaises(ValueError):
            self.bt.add_consequence("Bad", fraction=1.5)
        with self.assertRaises(ValueError):
            Barrier("Bad", pfd=-0.1)

    def test_barrier_effectiveness(self):
        barrier = Barrier("Relief valve", pfd=0.01)
        self.assertAlmostEqual(barrier.effectiveness, 0.99)


if __name__ == "__main__":
    unittest.main()
