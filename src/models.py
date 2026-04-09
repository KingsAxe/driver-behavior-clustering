"""
models.py
---------
Cluster model wrapper that standardises fit / predict / save / load
across all algorithms used in this project.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


class ClusterModel:
    """
    Thin wrapper around sklearn clustering estimators.
    Provides a consistent interface for the notebook and the Streamlit app.
    """

    SUPPORTED = {
        "kmeans": KMeans,
        "dbscan": DBSCAN,
        "agglomerative": AgglomerativeClustering,
    }

    def __init__(self, algorithm: str = "kmeans", **kwargs):
        if algorithm not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from: {list(self.SUPPORTED.keys())}"
            )
        self.algorithm = algorithm
        self.params = kwargs
        self.model = self.SUPPORTED[algorithm](**kwargs)
        self.labels_ = None
        self.is_fitted = False

    def fit(self, X) -> "ClusterModel":
        self.labels_ = self.model.fit_predict(X)
        self.is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict cluster assignments for new data.
        Note: DBSCAN and Agglomerative do not support out-of-sample prediction.
        For those algorithms, this returns the training labels (in-sample only).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        if self.algorithm == "kmeans":
            return self.model.predict(X)
        raise NotImplementedError(
            f"Out-of-sample prediction is not supported for '{self.algorithm}'. "
            "Use K-Means for the production prediction pipeline."
        )

    def save(self, path: str) -> None:
        """Serialise the fitted model to disk using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str, algorithm: str = "kmeans") -> "ClusterModel":
        """Load a previously saved model from disk."""
        instance = cls.__new__(cls)
        instance.algorithm = algorithm
        instance.model = joblib.load(path)
        instance.labels_ = None
        instance.is_fitted = True
        return instance


def predict_segment(feature_values: pd.DataFrame, model_path: str) -> np.ndarray:
    """
    Production-style prediction function.
    Loads the saved K-Means model and returns cluster assignments.

    Parameters
    ----------
    feature_values : pd.DataFrame
        New driver data with the same 5 features as the training set.
    model_path : str
        Path to the saved joblib model file.

    Returns
    -------
    np.ndarray of cluster labels.
    """
    model = ClusterModel.load(model_path, algorithm="kmeans")
    return model.predict(feature_values)
