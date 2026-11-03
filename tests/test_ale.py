"""Unit tests for SLE / ALE / ROSI quantitative metrics."""

import unittest

from risk_analysis.quantitative import (
    ale,
    cost_benefit_analysis,
    multi_year_ale,
    ros,
    sle,
)


class TestSLE(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(sle(1_000_000, 0.4), 400_000.0)

    def test_zero_exposure(self):
        self.assertEqual(sle(500_000, 0.0), 0.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            sle(-1, 0.5)
        with self.assertRaises(ValueError):
            sle(100, 1.5)


class TestALE(unittest.TestCase):
    def test_annual(self):
        self.assertEqual(ale(sle_value=100_000, aro=2.0), 200_000.0)

    def test_sub_annual(self):
        # Once per 20 years
        self.assertEqual(ale(sle_value=1_000_000, aro=0.05), 50_000.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ale(sle_value=100, aro=-0.5)


class TestROSI(unittest.TestCase):
    def test_justified_control(self):
        result = ros(ale_before=100_000, ale_after=20_000,
                     annual_control_cost=30_000)
        self.assertTrue(result["justified"])
        self.assertEqual(result["net_benefit"], 50_000.0)
        self.assertAlmostEqual(result["rosi_ratio"], 80000 / 30000)

    def test_unjustified_control(self):
        result = ros(ale_before=10_000, ale_after=9_000,
                     annual_control_cost=50_000)
        self.assertFalse(result["justified"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            ros(100_000, 20_000, annual_control_cost=0)


class TestCostBenefit(unittest.TestCase):
    def test_ranking(self):
        cba = cost_benefit_analysis([
            {"name": "A", "ale_before": 100_000,
             "ale_after": 50_000, "annual_control_cost": 10_000},
            {"name": "B", "ale_before": 100_000,
             "ale_after": 10_000, "annual_control_cost": 25_000},
        ])
        self.assertEqual(cba["baseline_ale"], 100_000.0)
        # A net: 40000; B net: 65000 -> B first
        self.assertEqual(cba["ranked"][0]["name"], "B")
        self.assertEqual(cba["ranked"][1]["name"], "A")

    def test_missing_keys_rejected(self):
        with self.assertRaises(ValueError):
            cost_benefit_analysis([{"name": "X"}])


class TestMultiYear(unittest.TestCase):
    def test_undiscounted_total(self):
        result = multi_year_ale(200_000, 0.5, years=3)
        self.assertEqual(result["annual_ale"], 100_000.0)
        self.assertEqual(result["total_ale"], 300_000.0)
        self.assertEqual(result["total_pv"], 300_000.0)

    def test_discounting_reduces_pv(self):
        result = multi_year_ale(200_000, 1.0, years=10, discount_rate=0.05)
        self.assertLess(result["total_pv"], result["total_ale"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            multi_year_ale(100, 1, years=0)


if __name__ == "__main__":
    unittest.main()
