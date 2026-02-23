"""
naive_bayes/tables.py
---------------------
Responsible for building and printing:
  - The raw Frequency Table (counts per feature-value per class)
  - The smoothed Likelihood Table (Laplacian-corrected probabilities)

No model training or prediction logic lives here.
"""

import pandas as pd
from tabulate import tabulate

from naive_bayes.logger import get_logger

log = get_logger(__name__)


class FrequencyTable:
    """
    Builds and prints the raw frequency table and the
    Laplacian-smoothed likelihood table for a given dataset.

    Args:
        df          : Training DataFrame.
        features    : List of feature column names.
        target      : Target column name.
        classes     : Ordered list of class labels.
        alpha       : Laplacian smoothing constant (default 1.0).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        classes: list[str],
        alpha: float = 1.0,
    ):
        log.debug("Initialising FrequencyTable (alpha=%.1f)", alpha)
        self.df           = df
        self.features     = features
        self.target       = target
        self.classes      = classes
        self.alpha        = alpha
        self.class_counts = df[target].value_counts().reindex(classes)
        log.debug("Class counts: %s", self.class_counts.to_dict())

    # ----------------------------------------------------------
    # Public helpers
    # ----------------------------------------------------------

    def print_frequency_table(self) -> None:
        """Print raw counts of each feature-value per class."""
        log.info("Building frequency table")
        print("\n" + "-" * 65)
        print("  FREQUENCY TABLE  (raw counts per feature-value per class)")
        print("-" * 65)

        for feat in self.features:
            freq = self._feature_counts(feat)
            freq["Total"] = freq.sum(axis=1)
            log.debug("Frequency counts for '%s':\n%s", feat, freq)
            print(f"\n  Feature: {feat}")
            print(tabulate(freq, headers="keys", tablefmt="rounded_outline"))

        print(
            f"\n  Class totals  ->  "
            f"No: {self.class_counts['No']}   Yes: {self.class_counts['Yes']}"
        )

    def print_likelihood_table(self) -> None:
        """
        Print Laplacian-smoothed conditional probabilities P(value | class).

        Formula:
            P(value | class) = (count + alpha)
                               ────────────────────────────────
                               (class_total + alpha * k)

        where k is the number of unique values for that feature.
        alpha=1 (Laplacian correction) prevents zero probabilities
        for feature values that never appeared with a given class.
        """
        log.info("Building likelihood table (alpha=%.1f)", self.alpha)
        print("\n" + "-" * 65)
        print(f"  LIKELIHOOD TABLE  (Laplacian smoothing, alpha={self.alpha})")
        print("-" * 65)

        for feat in self.features:
            freq       = self._feature_counts(feat)
            num_values = len(freq)
            rows       = []

            for val in freq.index:
                row = {"Feature": feat, "Value": val}
                for cls in self.classes:
                    raw  = freq.loc[val, cls]
                    prob = self._smooth(raw, self.class_counts[cls], num_values)
                    row[f"P({val}|{cls})"] = f"{prob:.4f}  (raw={int(raw)})"
                    log.debug(
                        "P(%s|%s) = (%.0f + %.1f) / (%d + %.1f*%d) = %.4f",
                        val, cls, raw, self.alpha,
                        self.class_counts[cls], self.alpha, num_values, prob,
                    )
                rows.append(row)

            print(f"\n  Feature: {feat}   (k={num_values} unique values)")
            print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _feature_counts(self, feature: str) -> pd.DataFrame:
        """Return a pivot of counts: rows=values, cols=classes."""
        return (
            self.df.groupby([feature, self.target])
                   .size()
                   .unstack(fill_value=0)
                   .reindex(columns=self.classes, fill_value=0)
        )

    def _smooth(self, count: int, class_total: int, num_values: int) -> float:
        """Apply Laplacian (additive) smoothing to a single count."""
        return (count + self.alpha) / (class_total + self.alpha * num_values)

