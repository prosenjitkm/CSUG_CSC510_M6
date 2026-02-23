"""
naive_bayes/classifier.py
-------------------------
Top-level orchestrator that wires together all components:
  - PlayTennisDataset
  - FrequencyTable
  - NaiveBayesModel
  - PosteriorWalkthrough

Only orchestration logic lives here — no data, math, or display
details are defined in this module.
"""

import pandas as pd
from tabulate import tabulate

from naive_bayes.dataset     import PlayTennisDataset
from naive_bayes.logger      import get_logger
from naive_bayes.model       import NaiveBayesModel
from naive_bayes.tables      import FrequencyTable
from naive_bayes.walkthrough import PosteriorWalkthrough

log = get_logger(__name__)


class NaiveBayesClassifier:
    """
    Orchestrates the full Naive Bayes classification pipeline.

    Pipeline steps:
      1. Load and display the dataset.
      2. Build and print the Frequency Table.
      3. Build and print the Likelihood Table (Laplacian-smoothed).
      4. Print prior probabilities.
      5. Train the CategoricalNB model.
      6. Predict posterior probabilities for a set of test samples.
      7. Run the manual Bayes' Theorem walkthrough for the first sample.
      8. Classify a user-defined custom input.

    Args:
        alpha : Laplacian smoothing constant (default 1.0).
    """

    def __init__(self, alpha: float = 1.0):
        log.info("Initialising NaiveBayesClassifier (alpha=%.1f)", alpha)
        self.alpha   = alpha
        self.dataset = PlayTennisDataset()

        df       = self.dataset.df
        features = self.dataset.features
        target   = self.dataset.target
        classes  = self.dataset.classes

        self.freq_table  = FrequencyTable(df, features, target, classes, alpha)
        self.nb_model    = NaiveBayesModel(alpha)
        self.walkthrough = PosteriorWalkthrough(df, features, target, classes, alpha)

    # ----------------------------------------------------------
    # Public
    # ----------------------------------------------------------

    def run(self, test_samples: list[dict], custom_test: dict) -> None:
        """
        Execute the full classification pipeline.

        Args:
            test_samples : List of feature dicts to classify.
            custom_test  : A single feature dict for the custom test.
        """
        log.info("Starting classification pipeline")
        df      = self.dataset.df
        target  = self.dataset.target
        classes = self.dataset.classes

        self._print_header()
        self.dataset.print_dataset()

        # Steps 1-2: Frequency & Likelihood tables
        self.freq_table.print_frequency_table()
        self.freq_table.print_likelihood_table()

        # Step 3: Priors
        self._print_priors(df, target, classes)

        # Step 4: Train
        self.nb_model.train(df, self.dataset.features, target)

        # Step 5: Predict test samples
        self._print_predictions(test_samples)

        # Step 6: Manual walkthrough
        self.walkthrough.run(test_samples[0])

        # Step 7: Custom test
        self._run_custom_test(custom_test)

        log.info("Pipeline complete")

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _print_header(self) -> None:
        print("=" * 65)
        print("  NAIVE BAYES CLASSIFIER  -  Play Tennis Dataset")
        print("=" * 65)

    def _print_priors(
        self,
        df: pd.DataFrame,
        target: str,
        classes: list[str],
    ) -> None:
        """Calculate and print prior probabilities for each class."""
        n_total      = len(df)
        class_counts = df[target].value_counts().reindex(classes)
        priors       = {cls: class_counts[cls] / n_total for cls in classes}

        log.debug("Priors: %s", priors)
        print(f"\n  Prior probabilities:")
        print(f"    P(No)  = {class_counts['No']}/{n_total} = {priors['No']:.4f}")
        print(f"    P(Yes) = {class_counts['Yes']}/{n_total} = {priors['Yes']:.4f}")

    def _print_predictions(self, test_samples: list[dict]) -> None:
        """Run predictions on all test samples and display a results table."""
        log.info("Predicting %d test samples", len(test_samples))
        print("\n" + "-" * 65)
        print("  POSTERIOR PROBABILITY PREDICTIONS")
        print("-" * 65)

        result_rows = []
        for s in test_samples:
            p_no, p_yes, label = self.nb_model.predict(s)
            result_rows.append({
                "Outlook":    s["Outlook"],
                "Temp":       s["Temperature"],
                "Humidity":   s["Humidity"],
                "Wind":       s["Wind"],
                "P(No)":      f"{p_no:.4f}",
                "P(Yes)":     f"{p_yes:.4f}",
                "Prediction": label,
            })
            log.debug(
                "Sample %s -> P(No)=%.4f  P(Yes)=%.4f  %s",
                s, p_no, p_yes, label,
            )

        print()
        print(tabulate(result_rows, headers="keys", tablefmt="rounded_outline"))

    def _run_custom_test(self, custom_test: dict) -> None:
        """Classify a single user-defined input and display results."""
        log.info("Running custom test: %s", custom_test)
        print("\n" + "=" * 65)
        print("  CUSTOM TEST  -  Edit the values in main.py and re-run")
        print("=" * 65)

        p_no, p_yes, label = self.nb_model.predict(custom_test)
        print(f"\n  Input      : {custom_test}")
        print(f"  P(No)      : {p_no:.4f}")
        print(f"  P(Yes)     : {p_yes:.4f}")
        print(f"  Prediction : Play Tennis? {label}\n")

