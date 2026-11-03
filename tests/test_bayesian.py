"""Unit tests for Bayesian inference and Naive Bayes classification."""

import unittest

from risk_analysis.statistical import NaiveBayesRiskClassifier, bayes_update


class TestBayesUpdate(unittest.TestCase):
    def test_textbook_ids_example(self):
        # P(H)=0.01, P(alert|H)=0.90, P(alert|~H)=0.05
        posterior = bayes_update(0.01, 0.90, 0.05)
        expected = (0.01 * 0.90) / (0.01 * 0.90 + 0.99 * 0.05)
        self.assertAlmostEqual(posterior, expected, places=12)
        self.assertAlmostEqual(posterior, 0.1538, places=3)

    def test_certain_evidence_gives_one(self):
        self.assertEqual(bayes_update(0.3, 1.0, 0.0), 1.0)

    def test_non_discriminating_evidence_keeps_prior(self):
        # likelihood_true == likelihood_false -> posterior == prior
        self.assertAlmostEqual(bayes_update(0.4, 0.2, 0.2), 0.4)

    def test_invalid_prior_rejected(self):
        with self.assertRaises(ValueError):
            bayes_update(1.5, 0.5, 0.5)
        with self.assertRaises(ValueError):
            bayes_update(-0.1, 0.5, 0.5)

    def test_invalid_likelihoods_rejected(self):
        with self.assertRaises(ValueError):
            bayes_update(0.5, 1.2, 0.5)
        with self.assertRaises(ValueError):
            bayes_update(0.5, -0.1, 0.5)


class TestNaiveBayesRiskClassifier(unittest.TestCase):
    def setUp(self):
        # Synthetic dataset: patch status and exposure -> risk level
        self.dataset = [
            ({"patched": "no", "exposed": "yes"}, "high"),
            ({"patched": "no", "exposed": "yes"}, "high"),
            ({"patched": "no", "exposed": "yes"}, "high"),
            ({"patched": "no", "exposed": "no"}, "medium"),
            ({"patched": "no", "exposed": "no"}, "medium"),
            ({"patched": "yes", "exposed": "yes"}, "medium"),
            ({"patched": "yes", "exposed": "yes"}, "medium"),
            ({"patched": "yes", "exposed": "no"}, "low"),
            ({"patched": "yes", "exposed": "no"}, "low"),
            ({"patched": "yes", "exposed": "no"}, "low"),
        ]
        self.clf = NaiveBayesRiskClassifier().fit(self.dataset)

    def test_fit_returns_self(self):
        self.assertIsInstance(self.clf.fit(self.dataset),
                              NaiveBayesRiskClassifier)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            NaiveBayesRiskClassifier().predict({"patched": "no"})

    def test_obvious_high_risk(self):
        prediction = self.clf.predict(
            {"patched": "no", "exposed": "yes"})
        self.assertEqual(prediction, "high")

    def test_obvious_low_risk(self):
        prediction = self.clf.predict(
            {"patched": "yes", "exposed": "no"})
        self.assertEqual(prediction, "low")

    def test_predict_proba_sums_to_one(self):
        probs = self.clf.predict_proba(
            {"patched": "no", "exposed": "yes"})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)

    def test_predict_proba_all_classes_present(self):
        probs = self.clf.predict_proba(
            {"patched": "yes", "exposed": "no"})
        self.assertEqual(set(probs.keys()), {"high", "medium", "low"})

    def test_laplace_smoothing_handles_unseen_value(self):
        # "maybe" never appears in training data -> smoothed, no crash
        probs = self.clf.predict_proba(
            {"patched": "maybe", "exposed": "yes"})
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=9)

    def test_zero_alpha_disallowed(self):
        with self.assertRaises(ValueError):
            NaiveBayesRiskClassifier(alpha=0.0)

    def test_training_accuracy_perfect(self):
        correct = sum(
            1 for features, label in self.dataset
            if self.clf.predict(features) == label
        )
        self.assertEqual(correct, len(self.dataset))


if __name__ == "__main__":
    unittest.main()
