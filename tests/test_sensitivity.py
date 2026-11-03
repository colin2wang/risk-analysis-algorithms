"""Unit tests for one-way sensitivity analysis and tornado data."""

import unittest

from risk_analysis.quantitative import one_way_sensitivity, tornado_data


class TestOneWaySensitivity(unittest.TestCase):
    def setUp(self):
        self.model = lambda x: x["revenue"] - x["cost"]

    def test_known_extremes(self):
        result = one_way_sensitivity(
            base_inputs={"revenue": 1000.0, "cost": 600.0},
            model=self.model,
            vary_pct=0.2,
        )
        revenue = result["revenue"]
        # revenue 800 -> 200 ; revenue 1200 -> 600
        self.assertAlmostEqual(revenue["low"], 200.0)
        self.assertAlmostEqual(revenue["high"], 600.0)
        self.assertAlmostEqual(revenue["swing"], 400.0)

    def test_base_output_unchanged(self):
        result = one_way_sensitivity(
            base_inputs={"a": 10.0},
            model=lambda x: x["a"] * 2,
        )
        self.assertAlmostEqual(result["a"]["base_output"], 20.0)

    def test_most_sensitive_input_has_largest_swing(self):
        result = one_way_sensitivity(
            base_inputs={"revenue": 1000.0, "cost": 600.0},
            model=self.model,
            vary_pct=0.2,
        )
        # 20% of 1000 (revenue) > 20% of 600 (cost)
        self.assertGreater(result["revenue"]["swing"],
                           result["cost"]["swing"])

    def test_tornado_sorting(self):
        result = one_way_sensitivity(
            base_inputs={"revenue": 1000.0, "cost": 600.0},
            model=self.model,
            vary_pct=0.2,
        )
        rows = tornado_data(result)
        self.assertEqual(rows[0][0], "revenue")
        swings = [high - low for _, low, high in rows]
        self.assertEqual(swings, sorted(swings, reverse=True))

    def test_validation(self):
        with self.assertRaises(ValueError):
            one_way_sensitivity({"a": 1.0}, self.model, vary_pct=0)
        with self.assertRaises(ValueError):
            one_way_sensitivity({"a": 1.0}, self.model, steps=0)
        with self.assertRaises(ValueError):
            one_way_sensitivity({}, self.model)


if __name__ == "__main__":
    unittest.main()
