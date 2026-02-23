"""
naive_bayes/walkthrough.py
--------------------------
Responsible for the step-by-step manual Bayes' Theorem
posterior calculation, making the math fully transparent.

No scikit-learn calls are made here — everything is computed
from first principles using the raw frequency counts.
"""

import numpy as np
import pandas as pd

from naive_bayes.logger import get_logger

log = get_logger(__name__)


class PosteriorWalkthrough:
    """
    Manually calculates and prints the posterior probability for a
    given sample using Bayes' Theorem and Laplacian smoothing.

    Steps shown:
      1. log-prior for each class
      2. log-likelihood for each feature value per class
      3. Summed log-score per class
      4. Normalised posterior probabilities
      5. Final prediction

    Args:
        df       : Training DataFrame.
        features : Feature column names.
        target   : Target column name.
        classes  : Ordered class labels.
        alpha    : Laplacian smoothing constant (default 1.0).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        classes: list[str],
        alpha: float = 1.0,
    ):
        log.debug("Initialising PosteriorWalkthrough (alpha=%.1f)", alpha)
        self.df           = df
        self.features     = features
        self.target       = target
        self.classes      = classes
        self.alpha        = alpha
        self.class_counts = df[target].value_counts().reindex(classes)
        n                 = len(df)
        self.priors       = {cls: self.class_counts[cls] / n for cls in classes}
        log.debug("Priors: %s", self.priors)

    # ----------------------------------------------------------
    # Public
    # ----------------------------------------------------------

    def run(self, sample: dict) -> None:
        """
        Print the full manual posterior calculation for one sample.

        Args:
            sample : Dict mapping feature name -> feature value.
        """
        log.info("Running manual posterior walkthrough for: %s", sample)
        print("\n" + "-" * 65)
        print("  WALKTHROUGH: Manual Posterior for Sample 1")
        print(f"  {sample}")
        print("-" * 65)

        scores = self._compute_log_scores(sample)
        self._print_normalised_posteriors(scores)

    # ----------------------------------------------------------
    # Private
    # ----------------------------------------------------------

    def _compute_log_scores(self, sample: dict) -> dict[str, float]:
        """
        Compute the unnormalised log-posterior score for each class.

        Returns:
            Dict mapping class label -> log-score.
        """
        scores: dict[str, float] = {}

        for cls in self.classes:
            log_prob = np.log(self.priors[cls])
            print(f"\n  Class = {cls}")
            print(f"    log P({cls}) = log({self.priors[cls]:.4f}) = {log_prob:.4f}")

            for feat in self.features:
                val       = sample[feat]
                freq_tbl  = self._feature_counts(feat)
                k         = len(freq_tbl)
                raw_count = freq_tbl.loc[val, cls] if val in freq_tbl.index else 0
                smoothed  = (raw_count + self.alpha) / (self.class_counts[cls] + self.alpha * k)
                log_lik   = np.log(smoothed)
                log_prob += log_lik

                log.debug(
                    "P(%s|%s) = (%d+%.1f)/(%d+%.1f*%d) = %.4f",
                    val, cls, raw_count, self.alpha,
                    self.class_counts[cls], self.alpha, k, smoothed,
                )
                print(
                    f"    x P({val}|{cls}) = ({int(raw_count)}+{self.alpha})/"
                    f"({self.class_counts[cls]}+{self.alpha}x{k})"
                    f" = {smoothed:.4f}  log={log_lik:.4f}"
                )

            scores[cls] = log_prob
            print(f"    {'─' * 37}")
            print(f"    log score = {log_prob:.4f}")

        return scores

    def _print_normalised_posteriors(self, scores: dict[str, float]) -> None:
        """Normalise log-scores and print the final posterior probabilities."""
        max_log  = max(scores.values())
        exp_vals = {cls: np.exp(scores[cls] - max_log) for cls in self.classes}
        total    = sum(exp_vals.values())

        print(f"\n  Normalised posterior probabilities:")
        for cls in self.classes:
            posterior = exp_vals[cls] / total
            log.debug("P(%s | features) = %.4f", cls, posterior)
            print(f"    P({cls} | features) = {posterior:.4f}")

        winner = max(exp_vals, key=exp_vals.get)
        log.info("Manual walkthrough prediction: %s", winner)
        print(f"\n  Prediction -> Play Tennis? {winner}")

    def _feature_counts(self, feature: str) -> pd.DataFrame:
        """Return a pivot of counts: rows=values, cols=classes."""
        return (
            self.df.groupby([feature, self.target])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(columns=self.classes, fill_value=0)
        )

