"""Unit tests for the qualitative risk matrix."""

import unittest

from risk_analysis.core import RiskItem, RiskLevel
from risk_analysis.qualitative import RiskMatrix


class TestRiskMatrixDefaults(unittest.TestCase):
    """Tests for the standard 5x5 matrix behaviour."""

    def setUp(self):
        self.matrix = RiskMatrix.default_5x5()

    def test_multiplicative_scoring(self):
        self.assertEqual(self.matrix.score(3, 4), 12.0)

    def test_classify_critical_corner(self):
        cell = self.matrix.classify(5, 5)
        self.assertEqual(cell.level, RiskLevel.CRITICAL)
        self.assertEqual(cell.score, 25.0)

    def test_classify_low_corner(self):
        self.assertEqual(self.matrix.classify(1, 1).level, RiskLevel.LOW)

    def test_out_of_bounds_rejected(self):
        with self.assertRaises(ValueError):
            self.matrix.classify(0, 3)
        with self.assertRaises(ValueError):
            self.matrix.classify(3, 6)

    def test_invalid_size_rejected(self):
        with self.assertRaises(ValueError):
            RiskMatrix(size=1)

    def test_heat_map_counts_all_bands(self):
        risks = [
            RiskItem("a", 1, 1),   # Low
            RiskItem("b", 3, 3),   # Medium
            RiskItem("c", 4, 4),   # High
            RiskItem("d", 5, 5),   # Critical
        ]
        counts = self.matrix.heat_map(risks)
        self.assertEqual(counts[RiskLevel.LOW], 1)
        self.assertEqual(counts[RiskLevel.MEDIUM], 1)
        self.assertEqual(counts[RiskLevel.HIGH], 1)
        self.assertEqual(counts[RiskLevel.CRITICAL], 1)

    def test_sort_by_severity_descending(self):
        risks = [
            RiskItem("minor", 1, 2),
            RiskItem("severe", 5, 5),
            RiskItem("middle", 3, 3),
        ]
        pairs = self.matrix.sort_by_severity(risks)
        self.assertEqual(pairs[0][0].name, "severe")
        self.assertEqual(pairs[-1][0].name, "minor")

    def test_all_cells_size(self):
        self.assertEqual(len(self.matrix.all_cells()), 25)


class TestCustomMatrix(unittest.TestCase):
    """Tests for custom scoring / banding / naming configurations."""

    def test_custom_score_fn(self):
        matrix = RiskMatrix(size=4, score_fn=lambda l, i: (l + i) / 2.0)
        self.assertEqual(matrix.score(2, 4), 3.0)

    def test_level_names(self):
        names = ["Rare", "Unlikely", "Possible", "Likely", "Certain"]
        matrix = RiskMatrix.default_5x5()
        matrix_custom = RiskMatrix(size=5, level_names=names)
        self.assertEqual(matrix_custom.level_name(3), "Possible")
        self.assertEqual(matrix.level_name(3), "L3")

    def test_level_names_length_validated(self):
        with self.assertRaises(ValueError):
            RiskMatrix(size=5, level_names=["a", "b"])


if __name__ == "__main__":
    unittest.main()
