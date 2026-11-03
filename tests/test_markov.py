"""Unit tests for Markov chain risk state models."""

import unittest

import numpy as np

from risk_analysis.statistical import MarkovRiskModel


def build_two_state():
    return MarkovRiskModel(
        states=["Safe", "Breached"],
        matrix=[[0.9, 0.1],
                [0.0, 1.0]],  # Breached is absorbing
    )


def build_three_state():
    return MarkovRiskModel(
        states=["Healthy", "At Risk", "Compromised"],
        matrix=[[0.70, 0.25, 0.05],
                [0.40, 0.50, 0.10],
                [0.30, 0.30, 0.40]],
    )


class TestConstruction(unittest.TestCase):
    def test_row_stochastic_enforced(self):
        with self.assertRaises(ValueError):
            MarkovRiskModel(["a", "b"], [[0.5, 0.4], [0.5, 0.5]])

    def test_square_enforced(self):
        with self.assertRaises(ValueError):
            MarkovRiskModel(["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0]])

    def test_negative_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            MarkovRiskModel(["a", "b"], [[1.2, -0.2], [0.5, 0.5]])


class TestEvolution(unittest.TestCase):
    def test_distribution_sums_to_one(self):
        model = build_three_state()
        dist = model.evolve([1.0, 0.0, 0.0], steps=5)
        self.assertAlmostEqual(dist.sum(), 1.0, places=9)

    def test_absorbing_state_capture(self):
        model = build_two_state()
        dist = model.evolve([1.0, 0.0], steps=100)
        # After 100 steps, almost surely absorbed
        self.assertGreater(dist[1], 0.9999)

    def test_one_step_manual_check(self):
        model = build_three_state()
        dist = model.evolve([1.0, 0.0, 0.0], steps=1)
        np.testing.assert_allclose(dist, [0.70, 0.25, 0.05])

    def test_zero_steps_returns_start(self):
        model = build_three_state()
        dist = model.evolve([0.2, 0.3, 0.5], steps=0)
        np.testing.assert_allclose(dist, [0.2, 0.3, 0.5])

    def test_most_likely_state(self):
        model = build_two_state()
        # After 10 steps, P(Breached) = 1 - 0.9^10 ~ 0.65 -> index 1
        self.assertEqual(
            model.most_likely_state([1.0, 0.0], 10), "Breached")

    def test_state_index(self):
        model = build_two_state()
        self.assertEqual(model.state_index("Safe"), 0)
        with self.assertRaises(KeyError):
            model.state_index("Unknown")


class TestStationary(unittest.TestCase):
    def test_irreducible_stationary_sums_to_one(self):
        model = build_three_state()
        stationary = model.stationary_distribution()
        self.assertAlmostEqual(stationary.sum(), 1.0, places=9)
        self.assertTrue((stationary >= 0).all())

    def test_stationary_is_fixed_point(self):
        model = build_three_state()
        stationary = model.stationary_distribution()
        fixed = stationary @ model.matrix
        np.testing.assert_allclose(fixed, stationary, atol=1e-8)

    def test_evolution_converges_to_stationary(self):
        model = build_three_state()
        stationary = model.stationary_distribution()
        dist = model.evolve([1.0, 0.0, 0.0], steps=500)
        np.testing.assert_allclose(dist, stationary, atol=1e-8)


class TestAbsorption(unittest.TestCase):
    def test_identify_absorbing_states(self):
        model = build_two_state()
        self.assertEqual(model.absorbing_states(), [1])

    def test_no_absorbing_states(self):
        model = build_three_state()
        self.assertEqual(model.absorbing_states(), [])

    def test_expected_steps_finite(self):
        model = build_two_state()
        steps = model.expected_steps_to_absorption([1.0, 0.0])
        self.assertIsNotNone(steps)
        # Geometric with p=0.1 -> expected 10 steps
        self.assertAlmostEqual(steps, 10.0, places=6)

    def test_expected_steps_none_without_absorber(self):
        model = build_three_state()
        self.assertIsNone(model.expected_steps_to_absorption([1, 0, 0]))

    def test_expected_steps_from_absorbed_state(self):
        model = build_two_state()
        self.assertIsNone(
            model.expected_steps_to_absorption([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
