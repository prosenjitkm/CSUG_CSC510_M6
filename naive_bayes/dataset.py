"""
naive_bayes/dataset.py
----------------------
Responsible for defining and exposing the Play Tennis training dataset.
Swap this class (or sub-class it) to use a different dataset without
touching any other module.
"""

import pandas as pd
from tabulate import tabulate

from naive_bayes.logger import get_logger

log = get_logger(__name__)


class PlayTennisDataset:
    """
    Holds the classic 14-sample Play Tennis dataset.

    Attributes:
        df       : Raw DataFrame with all features and target column.
        features : List of feature column names.
        target   : Name of the target/label column.
        classes  : Ordered list of class labels.
    """

    FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]
    TARGET   = "Play"
    CLASSES  = ["No", "Yes"]

    _RAW_DATA = {
        "Outlook":     ["Sunny","Sunny","Overcast","Rainy","Rainy","Rainy","Overcast",
                        "Sunny","Sunny","Rainy","Sunny","Overcast","Overcast","Rainy"],
        "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool",
                        "Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
        "Humidity":    ["High","High","High","High","Normal","Normal","Normal",
                        "High","Normal","Normal","Normal","High","Normal","High"],
        "Wind":        ["Weak","Strong","Weak","Weak","Weak","Strong","Strong",
                        "Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
        "Play":        ["No","No","Yes","Yes","Yes","No","Yes",
                        "No","Yes","Yes","Yes","Yes","Yes","No"],
    }

    def __init__(self):
        log.debug("Initialising PlayTennisDataset")
        self.df       = pd.DataFrame(self._RAW_DATA)
        self.features = self.FEATURES
        self.target   = self.TARGET
        self.classes  = self.CLASSES
        log.info("Dataset loaded: %d samples, %d features", len(self.df), len(self.features))

    def print_dataset(self) -> None:
        """Pretty-print the full dataset to stdout."""
        log.debug("Printing dataset table")
        print(f"\nDataset ({len(self.df)} samples):\n")
        print(tabulate(self.df, headers="keys", tablefmt="rounded_outline", showindex=True))

