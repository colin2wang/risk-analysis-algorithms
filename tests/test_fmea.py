"""Unit tests for FMEA with Risk Priority Numbers."""

import unittest

from risk_analysis.qualitative import FMEA, FMEAItem


def make_item(severity, occurrence, detection, function="Valve"):
    return FMEAItem(
        function=function,
        failure_mode="Stuck closed",
        effect="Overpressure",
        cause="Corrosion",
        severity=severity,
        occurrence=occurrence,
        detection=detection,
    )


class TestFMEAItem(unittest.TestCase):
    """Tests for a single FMEA entry."""

    def test_rpn_product(self):
        item = make_item(8, 5, 6)
        self.assertEqual(item.rpn, 240)
        self.assertEqual(item.criticality, 40)

    def test_rating_validation(self):
        with self.assertRaises(ValueError):
            make_item(11, 5, 6)
        with self.assertRaises(ValueError):
            make_item(8, 0, 6)
        with self.assertRaises(ValueError):
            make_item(8, 5, -1)

    def test_extreme_ratings_accepted(self):
        item = make_item(1, 1, 1)
        self.assertEqual(item.rpn, 1)
        item = make_item(10, 10, 10)
        self.assertEqual(item.rpn, 1000)

    def test_to_risk_item_mapping(self):
        item = make_item(9, 7, 3, function="Pump")
        generic = item.to_risk_item()
        # occurrence 7 -> likelihood 4, severity 9 -> impact 5
        self.assertEqual(generic.likelihood, 4)
        self.assertEqual(generic.impact, 5)
        self.assertEqual(generic.attributes["rpn"], 189)


class TestFMEAWorksheet(unittest.TestCase):
    """Tests for the worksheet-level aggregation."""

    def setUp(self):
        self.fmea = FMEA(rpn_threshold=200)

    def test_add_and_rank(self):
        self.fmea.add_item(make_item(5, 4, 4))    # RPN 80
        self.fmea.add_item(make_item(9, 8, 9))    # RPN 648
        self.fmea.add_item(make_item(6, 6, 3))    # RPN 108
        ranked = self.fmea.ranked()
        self.assertEqual(ranked[0].rpn, 648)
        self.assertEqual(ranked[-1].rpn, 80)

    def test_critical_filtering(self):
        self.fmea.add_item(make_item(5, 4, 4))    # 80 -> not critical
        self.fmea.add_item(make_item(9, 8, 9))    # 648 -> critical
        critical = self.fmea.critical_items()
        self.assertEqual(len(critical), 1)
        self.assertEqual(critical[0].rpn, 648)

    def test_statistics(self):
        self.fmea.add_item(make_item(5, 4, 4))
        self.fmea.add_item(make_item(9, 8, 9))
        stats = self.fmea.statistics()
        self.assertEqual(stats["total_items"], 2)
        self.assertEqual(stats["max_rpn"], 648)
        self.assertEqual(stats["min_rpn"], 80)
        self.assertEqual(stats["critical_count"], 1)

    def test_empty_statistics(self):
        stats = self.fmea.statistics()
        self.assertEqual(stats["total_items"], 0)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            FMEA(rpn_threshold=0)
        with self.assertRaises(ValueError):
            FMEA(rpn_threshold=2000)


if __name__ == "__main__":
    unittest.main()
