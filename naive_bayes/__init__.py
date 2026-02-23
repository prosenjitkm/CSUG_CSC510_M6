"""
naive_bayes
-----------
Public API for the Naive Bayes Classifier package.

Typical usage:
    from naive_bayes import NaiveBayesClassifier
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.run(test_samples, custom_test)
"""

# Import individual modules directly to avoid circular imports.
# Each sub-module only imports from naive_bayes.logger, never from naive_bayes directly.
from naive_bayes.classifier  import NaiveBayesClassifier   # noqa: E402
from naive_bayes.dataset     import PlayTennisDataset       # noqa: E402
from naive_bayes.model       import NaiveBayesModel         # noqa: E402
from naive_bayes.tables      import FrequencyTable          # noqa: E402
from naive_bayes.walkthrough import PosteriorWalkthrough    # noqa: E402

__all__ = [
    "NaiveBayesClassifier",
    "PlayTennisDataset",
    "NaiveBayesModel",
    "FrequencyTable",
    "PosteriorWalkthrough",
]

