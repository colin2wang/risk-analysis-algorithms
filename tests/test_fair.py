"""Unit tests for the FAIR cyber risk quantification model."""

import unittest

from risk_analysis.statistical import FAIRAnalysis


class TestFAIRAnalysis(unittest.TestCase):
    def setUp(self):
        self.fair = FAIRAnalysis(
            tef_min=50, tef_ml=100, tef_max=200,
            vuln_min=0.01, vuln_ml=0.03, vuln_max=0.10,
            lm_min=100_000, lm_ml=200_000, lm_max=1_000_000,
            n_samples=20_000, seed=42)

    def test_returns_all_keys(self):
        result = self.fair.run()
        for key in ("annual_loss_mean", "annual_loss_p50",
                    "annual_loss_p95", "annual_loss_p99",
                    "loss_events_mean"):
            self.assertIn(key, result)

    def test_reproducible_with_seed(self):
        r1, r2 = self.fair.run(), self.fair.run()
        self.assertEqual(r1["annual_loss_mean"], r2["annual_loss_mean"])
        self.assertEqual(r1["annual_loss_p95"], r2["annual_loss_p95"])

    def test_percentiles_monotonic(self):
        r = self.fair.run()
        self.assertLessEqual(r["annual_loss_p50"], r["annual_loss_p95"])
        self.assertLessEqual(r["annual_loss_p95"], r["annual_loss_p99"])

    def test_loss_magnitude_scales(self):
        small = FAIRAnalysis(
            50, 100, 200, 0.01, 0.03, 0.10,
            10_000, 20_000, 100_000, n_samples=20_000, seed=42)
        big = FAIRAnalysis(
            50, 100, 200, 0.01, 0.03, 0.10,
            100_000, 200_000, 1_000_000, n_samples=20_000, seed=42)
        self.assertLess(
            small.run()["annual_loss_mean"],
            big.run()["annual_loss_mean"])

    def test_mean_within_theoretical_bound(self):
        # Max loss per year <= max events * max magnitude
        r = self.fair.run()
        self.assertLessEqual(r["annual_loss_mean"], 200 * 1_000_000)

    def test_vulnerability_zero_no_loss(self):
        fair = FAIRAnalysis(
            50, 100, 200, 0.0, 0.0, 0.0,
            100_000, 200_000, 1_000_000, n_samples=1000, seed=1)
        r = fair.run()
        self.assertEqual(r["annual_loss_mean"], 0.0)

    def test_vulnerability_sensitivity_reduction(self):
        reduction = self.fair.sensitivity_to_vulnerability(
            vuln_min=0.001, vuln_ml=0.003, vuln_max=0.01)
        self.assertGreater(reduction["mean_reduction"], 0)
        self.assertGreaterEqual(reduction["p95_reduction"], 0)
        # Object state restored
        self.assertEqual(self.fair.vuln_ml, 0.03)

    def test_validation(self):
        with self.assertRaises(ValueError):
            FAIRAnalysis(200, 100, 50, 0, 0, 0, 1, 2, 3)
        with self.assertRaises(ValueError):
            FAIRAnalysis(1, 2, 3, 0.5, 0.2, 0.1, 1, 2, 3)
        with self.assertRaises(ValueError):
            FAIRAnalysis(1, 2, 3, 0, 0, 0, 5, 2, 1)
        with self.assertRaises(ValueError):
            FAIRAnalysis(1, 2, 3, 0.2, 0, 0, 1, 2, 3)  # vuln_ml < 0


if __name__ == "__main__":
    unittest.main()
