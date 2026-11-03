"""Unit tests for financial risk metrics (VaR / CVaR / ratios)."""

import unittest

import numpy as np

from risk_analysis.statistical import (
    cvar,
    historical_var,
    max_drawdown,
    monte_carlo_var,
    parametric_var,
    sharpe_ratio,
    sortino_ratio,
)


class TestHistoricalVar(unittest.TestCase):
    def test_known_quantile(self):
        # 100 returns; the 5% linear-interpolated quantile lands
        # between -0.05 and 0.01 -> VaR = 0.007
        returns = [0.01] * 95 + [-0.05, -0.06, -0.07, -0.08, -0.10]
        var = historical_var(returns, confidence_level=0.95)
        self.assertAlmostEqual(var, 0.007, places=6)

    def test_validation(self):
        with self.assertRaises(ValueError):
            historical_var([0.01, 0.02], confidence_level=1.0)


class TestParametricVar(unittest.TestCase):
    def test_matches_normal_quantile(self):
        np.random.seed(0)
        returns = np.random.normal(0.0, 0.01, 10_000).tolist()
        var = parametric_var(returns, confidence_level=0.95,
                             mean_return=0.0)
        # sigma of the sample is ~0.01 -> z_0.95 * sigma ~ 0.0164
        self.assertAlmostEqual(var, 0.01645, delta=0.0005)

    def test_validation(self):
        with self.assertRaises(ValueError):
            parametric_var([0.0, 0.0], confidence_level=0.0)


class TestMonteCarloVar(unittest.TestCase):
    def test_close_to_parametric(self):
        np.random.seed(1)
        returns = np.random.normal(0.0, 0.02, 5000).tolist()
        var_hist = monte_carlo_var(returns, confidence_level=0.95,
                                   n_simulations=100_000, seed=7)
        var_param = parametric_var(returns, confidence_level=0.95)
        self.assertAlmostEqual(var_hist, var_param, delta=0.003)

    def test_validation(self):
        with self.assertRaises(ValueError):
            monte_carlo_var([0.0, 0.0], n_simulations=0)


class TestCVaR(unittest.TestCase):
    def test_cvar_geq_var(self):
        returns = [0.01] * 95 + [-0.05, -0.06, -0.07, -0.08, -0.10]
        var = historical_var(returns, 0.95)
        es = cvar(returns, 0.95)
        self.assertGreaterEqual(es, var)
        # floor(100 * 0.05) = 5 worst returns
        self.assertAlmostEqual(es, (0.05 + 0.06 + 0.07 + 0.08 + 0.10) / 5,
                               places=9)

    def test_all_positive_returns(self):
        self.assertEqual(cvar([0.01, 0.02, 0.03], 0.95), 0.0)


class TestRatios(unittest.TestCase):
    def test_sharpe_known_value(self):
        # mean 0.02, std(ddof=1) 0.0141421, rf 0 -> sqrt(2)
        returns = [0.01, 0.03]
        self.assertAlmostEqual(sharpe_ratio(returns), 2 ** 0.5, places=6)

    def test_sharpe_zero_volatility(self):
        self.assertEqual(sharpe_ratio([0.0, 0.0, 0.0]), 0.0)

    def test_sortino_ignores_upside(self):
        returns = [0.05, -0.01, 0.05, -0.01]
        ratio = sortino_ratio(returns, risk_free_rate=0.0)
        self.assertGreater(ratio, 0.0)

    def test_sortino_no_downside(self):
        self.assertEqual(sortino_ratio([0.02, 0.03]), 0.0)


class TestMaxDrawdown(unittest.TestCase):
    def test_simple_decline(self):
        # 100 -> 110 -> 88 -> 99 : peak 110, trough 88, dd = 20%
        self.assertAlmostEqual(
            max_drawdown([0.10, -0.20, 0.125]), 0.20, places=9)

    def test_monotonic_growth_no_drawdown(self):
        self.assertEqual(max_drawdown([0.01, 0.02, 0.03]), 0.0)

    def test_all_losses(self):
        self.assertAlmostEqual(max_drawdown([-0.5, -0.5]), 0.75)


if __name__ == "__main__":
    unittest.main()
