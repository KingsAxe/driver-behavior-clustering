"""
evaluate.py
-----------
Reusable metric functions for clustering evaluation.
Used by the notebook and the Streamlit app so logic is never duplicated.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def compute_all_metrics(X: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Compute the three standard internal clustering validation metrics.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The feature matrix (scaled).
    labels : np.ndarray
        Cluster label assignments. Labels of -1 (DBSCAN noise) are excluded
        before computing silhouette and DB scores.

    Returns
    -------
    dict with keys: silhouette, davies_bouldin, calinski_harabasz
    """
    # Filter noise points for density-based methods (DBSCAN label = -1)
    mask = labels != -1
    X_valid = X[mask] if isinstance(X, pd.DataFrame) else X[mask]
    labels_valid = labels[mask]

    n_unique = len(set(labels_valid))

    if n_unique < 2:
        # Cannot compute meaningful metrics with fewer than 2 clusters
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
        }

    return {
        "silhouette": round(silhouette_score(X_valid, labels_valid), 4),
        "davies_bouldin": round(davies_bouldin_score(X_valid, labels_valid), 4),
        "calinski_harabasz": round(calinski_harabasz_score(X_valid, labels_valid), 4),
    }


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Build a formatted comparison DataFrame from a list of result dicts.

    Each dict should have: algorithm, n_clusters, silhouette,
    davies_bouldin, calinski_harabasz.

    Returns
    -------
    pd.DataFrame sorted by silhouette score descending.
    """
    df = pd.DataFrame(results)
    df.columns = [
        "Algorithm",
        "Clusters Found",
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index",
    ]
    return df.sort_values("Silhouette Score", ascending=False).reset_index(drop=True)
