"""
main.py  –  CSUG CSC510 Module 6
---------------------------------
Entry point for the Naive Bayes Classifier.

To test with your own input, edit `custom_test` below and run:
    python main.py

Valid values:
    Outlook     : "Sunny", "Overcast", "Rainy"
    Temperature : "Hot", "Mild", "Cool"
    Humidity    : "High", "Normal"
    Wind        : "Weak", "Strong"
"""

from naive_bayes import NaiveBayesClassifier
from naive_bayes.logger import get_logger

log = get_logger(__name__)


def main() -> None:
    # ----------------------------------------------------------
    # Test samples – fed through the trained model
    # ----------------------------------------------------------
    test_samples = [
        {"Outlook": "Sunny",    "Temperature": "Hot",  "Humidity": "High",   "Wind": "Weak"},
        {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong"},
        {"Outlook": "Rainy",    "Temperature": "Mild", "Humidity": "High",   "Wind": "Strong"},
        {"Outlook": "Sunny",    "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak"},
        {"Outlook": "Rainy",    "Temperature": "Hot",  "Humidity": "Normal", "Wind": "Weak"},
    ]

    # ----------------------------------------------------------
    # Custom test – change these values and re-run to classify
    # your own input
    # ----------------------------------------------------------
    custom_test = {
        "Outlook":     "Sunny",
        "Temperature": "Mild",
        "Humidity":    "Normal",
        "Wind":        "Weak",
    }

    log.info("Launching Naive Bayes Classifier")
    classifier = NaiveBayesClassifier(alpha=1.0)
    classifier.run(test_samples, custom_test)


if __name__ == "__main__":
    main()
