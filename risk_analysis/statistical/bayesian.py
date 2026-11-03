"""Bayesian inference for risk updating and Naive Bayes risk scoring.

Bayesian methods update a prior belief about risk in light of new
evidence via Bayes' theorem::

    P(H|E) = P(E|H) * P(H) / P(E)

Applications in risk analysis include:

* Updating failure probabilities after inspection results
  (risk-informed inspection, API 581).
* Combining expert priors with incident data.
* Classifying risks from observed indicators (Naive Bayes).

This module provides:

* :func:`bayes_update` -- a single binary-hypothesis update with
  likelihood ratio / Bayes factor semantics.
* :class:`NaiveBayesRiskClassifier` -- a from-scratch categorical
  Naive Bayes classifier with Laplace (add-one) smoothing, trained on
  labelled risk factors, producing calibrated posterior probabilities
  for risk levels.

Example
-------
>>> from risk_analysis.statistical import bayes_update
>>> # Prior: 1% chance of compromise; IDS alert raises it
>>> posterior = bayes_update(
...     prior=0.01,
...     likelihood_if_hypothesis_true=0.90,    # P(alert | breached)
...     likelihood_if_hypothesis_false=0.05,   # P(alert | clean) FPR
... )
>>> round(posterior, 4)
0.1538
"""

from __future__ import annotations

from collections import defaultdict
from math import log
from typing import Dict, Hashable, List, Sequence, Tuple


def bayes_update(
    prior: float,
    likelihood_if_hypothesis_true: float,
    likelihood_if_hypothesis_false: float,
) -> float:
    """Perform one binary-hypothesis Bayesian update.

    Parameters
    ----------
    prior:
        Prior probability of the hypothesis H, in ``[0, 1]``.
    likelihood_if_hypothesis_true:
        ``P(E | H)`` -- probability of observing the evidence E when
        the hypothesis holds (e.g. detection rate / sensitivity).
    likelihood_if_hypothesis_false:
        ``P(E | not H)`` -- probability of the evidence when the
        hypothesis does not hold (e.g. false positive rate).

    Returns
    -------
    float
        Posterior probability ``P(H | E)``.  When the evidence is
        impossible under both hypotheses (both likelihoods 0), the
        prior is returned unchanged.

    Raises
    ------
    ValueError
        If any probability is outside ``[0, 1]``.

    Examples
    --------
    >>> round(bayes_update(0.5, 0.8, 0.2), 6)
    0.8
    """
    for name, p in (
        ("prior", prior),
        ("likelihood_if_hypothesis_true", likelihood_if_hypothesis_true),
        ("likelihood_if_hypothesis_false", likelihood_if_hypothesis_false),
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {p!r}")
    numerator = likelihood_if_hypothesis_true * prior
    denominator = (
        likelihood_if_hypothesis_true * prior
        + likelihood_if_hypothesis_false * (1.0 - prior)
    )
    if denominator == 0.0:
        return prior
    return numerator / denominator


class NaiveBayesRiskClassifier:
    """Categorical Naive Bayes classifier for risk-level prediction.

    Implements the classic "naive" independence assumption: given the
    risk class (e.g. Low / High), every categorical feature (risk
    indicator) is conditionally independent.  Parameters are estimated
    with Laplace (add-one) smoothing so unseen feature values do not
    produce zero probabilities.

    Attributes
    ----------
    classes:
        Sorted list of class labels discovered during :meth:`fit`.
    fitted:
        ``True`` once the model has been trained.

    Examples
    --------
    >>> clf = NaiveBayesRiskClassifier()
    >>> data = [
    ...     ({"patched": "no",  "exposed": "yes"}, "High"),
    ...     ({"patched": "yes", "exposed": "no"},  "Low"),
    ...     ({"patched": "no",  "exposed": "no"},  "High"),
    ...     ({"patched": "yes", "exposed": "yes"}, "Low"),
    ... ]
    >>> clf.fit(data)
    >>> clf.predict({"patched": "no", "exposed": "yes"})
    'High'
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialise the classifier.

        Parameters
        ----------
        alpha:
            Laplace smoothing constant (>= 0).  ``1.0`` is the
            standard add-one smoothing.
        """
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.classes: List[Hashable] = []
        self._class_priors: Dict[Hashable, float] = {}
        self._feature_value_probs: Dict[str, Dict[Hashable, Dict[Hashable, float]]] = {}
        self.fitted = False

    def fit(self, dataset: Sequence[Tuple[Dict[str, Hashable], Hashable]]) -> "NaiveBayesRiskClassifier":
        """Train the classifier on labelled examples.

        Parameters
        ----------
        dataset:
            Sequence of ``(features, label)`` pairs where ``features``
            maps feature names to categorical values.

        Returns
        -------
        NaiveBayesRiskClassifier
            ``self`` for chaining.

        Raises
        ------
        ValueError
            If the dataset is empty or contains inconsistent feature
            sets.
        """
        if not dataset:
            raise ValueError("dataset must not be empty")
        feature_names = set(dataset[0][0].keys())
        for features, _ in dataset:
            if set(features.keys()) != feature_names:
                raise ValueError("all rows must share the same feature set")

        self.classes = sorted({label for _, label in dataset},
                              key=repr)
        n_rows = len(dataset)

        # Class priors
        class_counts: Dict[Hashable, int] = defaultdict(int)
        for _, label in dataset:
            class_counts[label] += 1
        self._class_priors = {
            c: class_counts[c] / n_rows for c in self.classes
        }

        # Per-feature, per-class value counts -> smoothed probabilities
        value_counts: Dict[str, Dict[Hashable, Dict[Hashable, int]]] = {
            f: {c: defaultdict(int) for c in self.classes}
            for f in feature_names
        }
        for features, label in dataset:
            for f, v in features.items():
                value_counts[f][label][v] += 1

        self._feature_value_probs = {}
        for f in feature_names:
            # Global vocabulary of distinct values for this feature
            vocabulary = {v for features_row, _ in dataset
                          for v in [features_row[f]]}
            self._feature_value_probs[f] = {}
            for c in self.classes:
                counts = value_counts[f][c]
                n_class = class_counts[c]
                denominator = n_class + self.alpha * len(vocabulary)
                probs = {
                    v: (count + self.alpha) / denominator
                    for v, count in counts.items()
                }
                # Unseen values (appearing only at prediction time) get
                # the smoothing floor alpha / denominator
                self._feature_value_probs[f][c] = {
                    "probs": probs,
                    "floor": self.alpha / denominator,
                }
        self.fitted = True
        return self

    def predict_proba(
        self, features: Dict[str, Hashable]
    ) -> Dict[Hashable, float]:
        """Posterior class probabilities for one example (log-space).

        Parameters
        ----------
        features:
            Mapping from feature name to categorical value.  Values
            unseen in training receive the smoothing floor.

        Returns
        -------
        dict
            Mapping from class label to posterior probability; the
            probabilities sum to 1.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        ValueError
            If feature names differ from the training set.
        """
        if not self.fitted:
            raise RuntimeError("classifier must be fitted before prediction")
        if set(features.keys()) != set(self._feature_value_probs.keys()):
            raise ValueError("feature names differ from the training set")

        log_posteriors: Dict[Hashable, float] = {}
        for c in self.classes:
            log_p = log(self._class_priors[c])
            for f, v in features.items():
                model = self._feature_value_probs[f][c]
                prob = model["probs"].get(v, model["floor"])
                log_p += log(prob)
            log_posteriors[c] = log_p

        # Normalise via log-sum-exp for numerical stability
        max_log = max(log_posteriors.values())
        exp_shifted = {
            c: pow(2.718281828459045, lp - max_log)
            for c, lp in log_posteriors.items()
        }
        total = sum(exp_shifted.values())
        return {c: val / total for c, val in exp_shifted.items()}

    def predict(self, features: Dict[str, Hashable]) -> Hashable:
        """Predict the most probable risk class for one example.

        Parameters
        ----------
        features:
            Mapping from feature name to categorical value.

        Returns
        -------
        Hashable
            The class label with the highest posterior probability.
        """
        probabilities = self.predict_proba(features)
        return max(probabilities.items(), key=lambda t: t[1])[0]
