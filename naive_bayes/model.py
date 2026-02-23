"""
naive_bayes/model.py
--------------------
Responsible for encoding, training, and predicting with
scikit-learn's CategoricalNB.

No display / table logic lives here — only ML concerns.
"""

import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

from naive_bayes.logger import get_logger

log = get_logger(__name__)


class NaiveBayesModel:
    """
    Thin wrapper around scikit-learn CategoricalNB.

    Responsibilities:
      - Ordinal-encode categorical features.
      - Fit the model with Laplacian smoothing (alpha).
      - Expose a predict() method that accepts a plain dict
        and returns (p_no, p_yes, label).

    Args:
        alpha : Laplacian smoothing constant (default 1.0).
    """

    def __init__(self, alpha: float = 1.0):
        log.debug("Initialising NaiveBayesModel (alpha=%.1f)", alpha)
        self.alpha    = alpha
        self.encoder  = OrdinalEncoder()
        self.model    = CategoricalNB(alpha=alpha)
        self.features: list[str] = []
        self._trained = False

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def train(self, df: pd.DataFrame, features: list[str], target: str) -> None:
        """
        Encode features with OrdinalEncoder and fit CategoricalNB.

        Args:
            df       : Training DataFrame.
            features : Feature column names.
            target   : Target column name (class label).
        """
        log.info("Training CategoricalNB (alpha=%.1f) on %d samples", self.alpha, len(df))
        self.features = features

        X = self.encoder.fit_transform(df[features])
        y = (df[target] == "Yes").astype(int)   # No=0, Yes=1
        self.model.fit(X, y)
        self._trained = True

        log.info(
            "Training complete. Classes: %s  |  Feature categories: %s",
            self.model.classes_,
            [list(c) for c in self.encoder.categories_],
        )
        self._print_training_summary()

    def _print_training_summary(self) -> None:
        """Print a human-readable training summary to stdout."""
        print("\n" + "-" * 65)
        print("  TRAINING  scikit-learn  CategoricalNB")
        print("-" * 65)
        print(f"\n  Model   : CategoricalNB(alpha={self.alpha})")
        print(f"  Classes : {self.model.classes_}  (0=No, 1=Yes)")
        print(f"  alpha={self.alpha} is the Laplacian correction — zero-count cells")
        print(f"  receive a small non-zero probability (avoids zero-probability errors).")

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def predict(self, sample: dict) -> tuple[float, float, str]:
        """
        Predict class probabilities for a single sample.

        Args:
            sample : Dict mapping feature name -> feature value.

        Returns:
            (p_no, p_yes, label) where label is "Yes" or "No".

        Raises:
            RuntimeError: if called before train().
        """
        if not self._trained:
            log.error("predict() called before train()")
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        log.debug("Predicting for sample: %s", sample)
        enc   = self.encoder.transform(pd.DataFrame([sample]))
        proba = self.model.predict_proba(enc)[0]
        label = "Yes" if self.model.predict(enc)[0] == 1 else "No"

        log.debug("Result -> P(No)=%.4f  P(Yes)=%.4f  Prediction=%s",
                  proba[0], proba[1], label)
        return proba[0], proba[1], label

