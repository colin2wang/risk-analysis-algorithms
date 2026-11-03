"""Unit tests for scenario analysis and stress testing."""

import unittest

from risk_analysis.quantitative import Scenario, ScenarioAnalysis


class TestScenarioAnalysis(unittest.TestCase):
    def setUp(self):
        self.model = lambda x: x["revenue"] * x["margin"]

    def test_multiplicative_shocks(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={
            "revenue": 1000.0, "margin": 0.2})
        analysis.add_scenario(Scenario("Recession", shocks={
            "revenue": -0.4, "margin": -0.25}))
        results = analysis.run()
        self.assertAlmostEqual(results["Recession"]["output"], 120.0)
        self.assertAlmostEqual(results["__baseline__"]["output"], 200.0)

    def test_absolute_overrides(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={
            "revenue": 1000.0, "margin": 0.2})
        analysis.add_scenario(Scenario("Pinned", overrides={
            "revenue": 500.0, "margin": 0.1}))
        results = analysis.run()
        self.assertAlmostEqual(results["Pinned"]["output"], 50.0)

    def test_override_wins_over_shock(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={
            "revenue": 1000.0, "margin": 0.2})
        analysis.add_scenario(Scenario(
            "Both", shocks={"revenue": -0.5},
            overrides={"revenue": 800.0}))
        results = analysis.run()
        self.assertAlmostEqual(results["Both"]["inputs"]["revenue"], 800.0)

    def test_duplicate_names_rejected(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={"a": 1.0})
        analysis.add_scenario(Scenario("S", shocks={"a": 0.1}))
        with self.assertRaises(ValueError):
            analysis.add_scenario(Scenario("S", shocks={"a": 0.2}))

    def test_unknown_input_rejected(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={"a": 1.0})
        analysis.add_scenario(Scenario("S", shocks={"bogus": 0.1}))
        with self.assertRaises(ValueError):
            analysis.run()

    def test_scenario_must_define_change(self):
        with self.assertRaises(ValueError):
            Scenario("Empty")

    def test_worst_case_is_minimum_output(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={
            "revenue": 1000.0, "margin": 0.2})
        analysis.add_scenario(Scenario("Mild", shocks={"revenue": -0.1}))
        analysis.add_scenario(Scenario("Severe", shocks={"revenue": -0.5}))
        worst = analysis.worst_case()
        self.assertEqual(worst["name"], "Severe")
        self.assertAlmostEqual(worst["output"], 100.0)

    def test_weighted_expectation(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={
            "revenue": 1000.0, "margin": 0.2})
        analysis.add_scenario(Scenario("Good", shocks={"revenue": 0.2},
                                       probability=0.5))
        analysis.add_scenario(Scenario("Bad", shocks={"revenue": -0.2},
                                       probability=0.5))
        expected = analysis.weighted_expectation()
        self.assertAlmostEqual(expected, 200.0)

    def test_weighted_expectation_requires_probabilities(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={"a": 1.0})
        self.assertIsNone(analysis.weighted_expectation())

    def test_weighted_expectation_validates_sum(self):
        analysis = ScenarioAnalysis(self.model, base_inputs={"a": 1.0})
        analysis.add_scenario(Scenario("A", shocks={"a": 0.1}, probability=0.4))
        analysis.add_scenario(Scenario("B", shocks={"a": -0.1}, probability=0.4))
        with self.assertRaises(ValueError):
            analysis.weighted_expectation()


if __name__ == "__main__":
    unittest.main()
