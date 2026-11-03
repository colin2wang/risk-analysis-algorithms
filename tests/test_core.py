"""Unit tests for risk_analysis.core."""

import unittest

from risk_analysis.core import (
    ImpactScale,
    LikelihoodScale,
    RiskItem,
    RiskLevel,
)


class TestRiskLevel(unittest.TestCase):
    """Tests for the four-band risk classification."""

    def test_from_score_bands(self):
        self.assertEqual(RiskLevel.from_score(3, 25), RiskLevel.LOW)
        self.assertEqual(RiskLevel.from_score(6, 25), RiskLevel.MEDIUM)
        self.assertEqual(RiskLevel.from_score(10, 25), RiskLevel.MEDIUM)
        self.assertEqual(RiskLevel.from_score(13, 25), RiskLevel.HIGH)
        self.assertEqual(RiskLevel.from_score(19, 25), RiskLevel.HIGH)
        self.assertEqual(RiskLevel.from_score(20, 25), RiskLevel.CRITICAL)

    def test_from_score_boundary_low(self):
        self.assertEqual(RiskLevel.from_score(6.25, 25), RiskLevel.LOW)
        self.assertEqual(RiskLevel.from_score(6.26, 25), RiskLevel.MEDIUM)

    def test_invalid_max_score(self):
        with self.assertRaises(ValueError):
            RiskLevel.from_score(1, 0)


class TestScales(unittest.TestCase):
    """Tests for the ordinal likelihood and impact scales."""

    def test_likelihood_levels(self):
        self.assertEqual(LikelihoodScale.RARE.level, 1)
        self.assertEqual(LikelihoodScale.ALMOST_CERTAIN.level, 5)

    def test_impact_descriptions(self):
        for scale in ImpactScale:
            self.assertTrue(len(scale.description) > 0)


class TestRiskItem(unittest.TestCase):
    """Tests for the shared RiskItem dataclass."""

    def test_default_score_is_product(self):
        risk = RiskItem(name="Fire", likelihood=4, impact=5)
        self.assertEqual(risk.risk_score, 20)
        self.assertEqual(risk.risk_level, RiskLevel.CRITICAL)

    def test_invalid_likelihood_rejected(self):
        with self.assertRaises(ValueError):
            RiskItem(name="Bad", likelihood=6, impact=3)
        with self.assertRaises(ValueError):
            RiskItem(name="Bad", likelihood=0, impact=3)

    def test_invalid_impact_rejected(self):
        with self.assertRaises(ValueError):
            RiskItem(name="Bad", likelihood=3, impact=0)

    def test_to_dict_round_trip_fields(self):
        risk = RiskItem(name="Breach", likelihood=2, impact=5,
                        owner="Security")
        data = risk.to_dict()
        self.assertEqual(data["risk_score"], 10)
        self.assertEqual(data["risk_level"], "High")
        self.assertEqual(data["owner"], "Security")

    def test_automatic_id(self):
        a = RiskItem(name="A", likelihood=1, impact=1)
        b = RiskItem(name="B", likelihood=1, impact=1)
        self.assertNotEqual(a.risk_id, b.risk_id)


if __name__ == "__main__":
    unittest.main()
