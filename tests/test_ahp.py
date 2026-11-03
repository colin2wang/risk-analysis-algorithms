"""Unit tests for the Analytic Hierarchy Process (AHP)."""

import unittest

from risk_analysis.structural import AHP


class TestAHPWeights(unittest.TestCase):
    def setUp(self):
        # Saaty's textbook 3x3 criteria example
        self.matrix = [
            [1,   3,   9],
            [1/3, 1,   3],
            [1/9, 1/3, 1],
        ]
        self.ahp = AHP(self.matrix, criteria=["Cost", "Quality", "Speed"])

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(self.ahp.weights), 1.0, places=9)

    def test_weight_ordering(self):
        w = self.ahp.weights
        self.assertGreater(w[0], w[1])
        self.assertGreater(w[1], w[2])

    def test_perfectly_consistent_matrix_weights(self):
        # Exact geometric-mean weights: [3, 1, 1/3] normalised
        ahp = AHP([[1, 3, 9], [1/3, 1, 3], [1/9, 1/3, 1]])
        expected = [3 / (3 + 1 + 1 / 3), 1 / (3 + 1 + 1 / 3),
                    (1 / 3) / (3 + 1 + 1 / 3)]
        for got, exp in zip(ahp.weights, expected):
            self.assertAlmostEqual(got, exp, places=6)

    def test_consistent_matrix_cr_near_zero(self):
        self.assertLess(self.ahp.consistency_ratio, 0.01)

    def test_identity_matrix_perfectly_consistent(self):
        ahp = AHP([[1, 1], [1, 1]])
        self.assertAlmostEqual(ahp.consistency_ratio, 0.0)
        self.assertTrue(ahp.is_consistent())

    def test_inconsistent_matrix_detected(self):
        inconsistent = [
            [1, 9, 1],
            [1/9, 1, 9],
            [1, 1/9, 1],
        ]
        ahp = AHP(inconsistent)
        self.assertGreater(ahp.consistency_ratio, 0.10)
        self.assertFalse(ahp.is_consistent())


class TestAHPValidation(unittest.TestCase):
    def test_non_square_rejected(self):
        with self.assertRaises(ValueError):
            AHP([[1, 2, 3], [1 / 2, 1, 4]])

    def test_diagonal_must_be_one(self):
        with self.assertRaises(ValueError):
            AHP([[2, 1], [1, 1]])

    def test_reciprocity_enforced(self):
        with self.assertRaises(ValueError):
            AHP([[1, 3], [4, 1]])

    def test_scale_value_rejected(self):
        with self.assertRaises(ValueError):
            AHP([[1, 2.5], [0.4, 1]])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            AHP([[1, float("nan")], [float("nan"), 1]])


class TestAHPRanking(unittest.TestCase):
    def test_alternative_ranking(self):
        ahp = AHP([[1, 3], [1 / 3, 1]], criteria=["Risk", "Benefit"])
        ranked = ahp.rank_alternatives({
            "Vendor A": [0.9, 0.3],
            "Vendor B": [0.2, 0.8],
        })
        self.assertEqual(ranked[0][0], "Vendor A")
        self.assertAlmostEqual(ranked[0][1] + ranked[1][1], 1.0, places=9)

    def test_alternative_score_width_validated(self):
        ahp = AHP([[1, 3], [1 / 3, 1]])
        with self.assertRaises(ValueError):
            ahp.rank_alternatives({"A": [0.5, 0.5, 0.5]})


if __name__ == "__main__":
    unittest.main()
