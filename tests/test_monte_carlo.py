"""Unit tests for the Monte Carlo simulation engine."""

import unittest

import numpy as np

from risk_analysis.statistical import (
    Discrete,
    Empirical,
    Lognormal,
    MonteCarloEngine,
    Normal,
    PERT,
    Triangular,
    Uniform,
)


class TestDistributions(unittest.TestCase):
    def test_uniform_bounds(self):
        rng = np.random.default_rng(42)
        samples = Uniform(2, 5).sample(rng, 10_000)
        self.assertGreaterEqual(samples.min(), 2.0)
        self.assertLessEqual(samples.max(), 5.0)
        self.assertAlmostEqual(samples.mean(), 3.5, delta=0.05)

    def test_triangular_mean_near_expected(self):
        rng = np.random.default_rng(1)
        samples = Triangular(1, 5, 2).sample(rng, 100_000)
        self.assertAlmostEqual(samples.mean(), (1 + 2 + 5) / 3, delta=0.02)

    def test_pert_mode_respected(self):
        rng = np.random.default_rng(3)
        samples = PERT(10, 50, 20).sample(rng, 100_000)
        # PERT mean = (a + 4m + b) / 6 = (10 + 80 + 50) / 6
        self.assertAlmostEqual(samples.mean(), 140 / 6, delta=0.7)

    def test_normal_moments(self):
        rng = np.random.default_rng(7)
        samples = Normal(10, 2).sample(rng, 100_000)
        self.assertAlmostEqual(samples.mean(), 10.0, delta=0.05)
        self.assertAlmostEqual(samples.std(), 2.0, delta=0.05)

    def test_lognormal_positive(self):
        rng = np.random.default_rng(9)
        samples = Lognormal(1.0, 0.5).sample(rng, 10_000)
        self.assertTrue((samples > 0).all())

    def test_discrete_values_and_weights(self):
        rng = np.random.default_rng(11)
        dist = Discrete([1, 2, 3], [0.2, 0.5, 0.3])
        samples = dist.sample(rng, 100_000)
        self.assertAlmostEqual((samples == 2).mean(), 0.5, delta=0.01)

    def test_discrete_weight_sum_validated(self):
        with self.assertRaises(ValueError):
            Discrete([1, 2], [0.5, 0.2])

    def test_empirical_resample(self):
        rng = np.random.default_rng(13)
        data = [1.0, 2.0, 3.0, 4.0]
        samples = Empirical(data).sample(rng, 1000)
        self.assertTrue(set(np.unique(samples)).issubset(set(data)))

    def test_empirical_requires_data(self):
        with self.assertRaises(ValueError):
            Empirical([])

    def test_out_of_order_parameters_rejected(self):
        with self.assertRaises(ValueError):
            Triangular(5, 1, 3)
        with self.assertRaises(ValueError):
            PERT(10, 1, 5)
        with self.assertRaises(ValueError):
            Normal(1, -2)


class TestMonteCarloEngine(unittest.TestCase):
    def test_reproducibility_with_seed(self):
        engine = MonteCarloEngine(
            inputs={"x": Uniform(0, 1)},
            model=lambda x: x["x"] * 10,
            n_samples=500, seed=123)
        r1, r2 = engine.run(), engine.run()
        np.testing.assert_array_equal(r1.samples, r2.samples)

    def test_linear_model_exact_mean(self):
        engine = MonteCarloEngine(
            inputs={"x": Uniform(2, 4)},
            model=lambda x: x["x"],
            n_samples=200_000, seed=5)
        result = engine.run()
        self.assertAlmostEqual(result.mean(), 3.0, delta=0.01)

    def test_summary_percentiles_ordered(self):
        engine = MonteCarloEngine(
            inputs={"x": Normal(0, 1)},
            model=lambda x: x["x"],
            n_samples=50_000, seed=2)
        result = engine.run()
        summary = result.summary()
        self.assertLess(summary["p5"], summary["p50"])
        self.assertLess(summary["p50"], summary["p95"])
        self.assertAlmostEqual(summary["p5"], -1.645, delta=0.05)

    def test_exceedance_probability(self):
        engine = MonteCarloEngine(
            inputs={"x": Uniform(0, 1)},
            model=lambda x: x["x"],
            n_samples=100_000, seed=4)
        result = engine.run()
        self.assertAlmostEqual(result.exceedance(0.8), 0.2, delta=0.01)

    def test_probability_exceeded(self):
        engine = MonteCarloEngine(
            inputs={"x": Uniform(0, 1)},
            model=lambda x: x["x"],
            n_samples=50_000, seed=6)
        result = engine.run()
        self.assertAlmostEqual(result.probability_exceeded(0.25),
                               0.75, delta=0.01)

    def test_histogram_bins_sum_to_n(self):
        engine = MonteCarloEngine(
            inputs={"x": Uniform(0, 1)},
            model=lambda x: x["x"],
            n_samples=1000, seed=8)
        result = engine.run()
        counts, _ = result.histogram(bins=10)
        self.assertEqual(int(sum(counts)), 1000)

    def test_two_input_model(self):
        engine = MonteCarloEngine(
            inputs={"a": Uniform(1, 1), "b": Uniform(2, 2)},
            model=lambda x: x["a"] + x["b"],
            n_samples=100, seed=1)
        result = engine.run()
        self.assertAlmostEqual(result.mean(), 3.0, places=6)

    def test_keep_inputs(self):
        engine = MonteCarloEngine(
            inputs={"x": Uniform(0, 1)},
            model=lambda x: x["x"],
            n_samples=10, seed=1)
        result = engine.run(keep_inputs=True)
        self.assertIn("x", result.inputs_sampled)
        self.assertEqual(len(result.inputs_sampled["x"]), 10)

    def test_validation(self):
        with self.assertRaises(ValueError):
            MonteCarloEngine({"x": Uniform(0, 1)}, lambda x: 1,
                             n_samples=0)


if __name__ == "__main__":
    unittest.main()
