"""
clustering.py
-------------
K-Means and optional DBSCAN clustering on PCA-reduced data.
Includes silhouette scoring, cluster-summary generation, and
all safety guards for small / degenerate datasets.

SIH26051 — PCA-Based Compliance Risk Clustering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """Output of a clustering run."""
    labels: np.ndarray                        # Cluster label per sample (-1 = noise)
    n_clusters: int = 0                       # Distinct cluster IDs (excl. noise)
    algorithm: str = "kmeans"
    silhouette: float | None = None           # None when not computable
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 10,
) -> ClusterResult:
    """
    Run K-Means clustering on the PCA-reduced matrix *X*.

    Parameters
    ----------
    X : np.ndarray  shape (n_samples, n_components)
    n_clusters : int
    random_state : int
    max_iter : int
    n_init : int

    Returns
    -------
    ClusterResult
    """
    warnings: list[str] = []
    n_samples = len(X)

    # Guard: adjust k if too many clusters for the sample size
    if n_clusters >= n_samples:
        n_clusters = max(2, n_samples - 1)
        warnings.append(
            f"n_clusters reduced to {n_clusters} because dataset has only "
            f"{n_samples} samples."
        )

    if n_clusters < 2:
        # Single-cluster degenerate case
        labels = np.zeros(n_samples, dtype=int)
        warnings.append(
            "Only 1 cluster possible with this dataset size. "
            "Silhouette score is undefined."
        )
        return ClusterResult(
            labels=labels,
            n_clusters=1,
            algorithm="kmeans",
            silhouette=None,
            cluster_sizes={0: n_samples},
            warnings=warnings,
            params={"n_clusters": 1},
        )

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )
    labels = km.fit_predict(X)
    unique_labels = np.unique(labels)
    actual_k = len(unique_labels)

    if actual_k < n_clusters:
        warnings.append(
            f"K-Means produced {actual_k} non-empty clusters "
            f"(requested {n_clusters})."
        )

    sil = _safe_silhouette(X, labels, warnings)
    sizes = {int(k): int((labels == k).sum()) for k in unique_labels}

    logger.info(
        "K-Means: k=%d, actual_clusters=%d, silhouette=%s",
        n_clusters, actual_k, f"{sil:.4f}" if sil is not None else "N/A",
    )

    return ClusterResult(
        labels=labels,
        n_clusters=actual_k,
        algorithm="kmeans",
        silhouette=sil,
        cluster_sizes=sizes,
        warnings=warnings,
        params={"n_clusters": n_clusters, "random_state": random_state,
                "max_iter": max_iter, "n_init": n_init},
    )


# ---------------------------------------------------------------------------
# DBSCAN (optional)
# ---------------------------------------------------------------------------

def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> ClusterResult:
    """
    Run DBSCAN clustering on *X*.

    Noise points are labelled -1. Returns a ClusterResult; silhouette is
    computed only when at least 2 non-noise clusters exist and there are
    fewer than 10 000 samples (to avoid memory issues).

    Parameters
    ----------
    X : np.ndarray
    eps : float
        DBSCAN neighbourhood radius.
    min_samples : int
        Minimum points to form a dense region.
    """
    warnings: list[str] = []
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    unique_non_noise = np.unique(labels[labels >= 0])
    n_clusters = len(unique_non_noise)
    n_noise = int((labels == -1).sum())

    if n_noise > 0:
        warnings.append(f"DBSCAN: {n_noise} noise point(s) labelled -1.")

    if n_clusters == 0:
        warnings.append(
            "DBSCAN found no clusters. Try increasing eps or decreasing min_samples."
        )

    # Silhouette only on non-noise points when >= 2 clusters
    sil: float | None = None
    if n_clusters >= 2:
        mask = labels >= 0
        if mask.sum() >= n_clusters + 1:
            sil = _safe_silhouette(X[mask], labels[mask], warnings)

    sizes = {int(k): int((labels == k).sum()) for k in np.unique(labels)}

    logger.info(
        "DBSCAN: eps=%.3f, min_samples=%d, clusters=%d, noise=%d, sil=%s",
        eps, min_samples, n_clusters, n_noise,
        f"{sil:.4f}" if sil is not None else "N/A",
    )

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        algorithm="dbscan",
        silhouette=sil,
        cluster_sizes=sizes,
        warnings=warnings,
        params={"eps": eps, "min_samples": min_samples},
    )


# ---------------------------------------------------------------------------
# Elbow / silhouette sweep for optimal k
# ---------------------------------------------------------------------------

def sweep_k(
    X: np.ndarray,
    k_range: range | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute inertia and silhouette score for a range of k values.

    Parameters
    ----------
    X : np.ndarray
    k_range : range | None
        Default: range(2, min(11, n_samples))
    random_state : int

    Returns
    -------
    pd.DataFrame  columns: k, inertia, silhouette
    """
    n_samples = len(X)
    if k_range is None:
        k_range = range(2, min(11, n_samples))

    rows = []
    for k in k_range:
        if k >= n_samples:
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X)
        sil = _safe_silhouette(X, lbl, [])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cluster summary
# ---------------------------------------------------------------------------

def build_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing cluster groups on numeric features.

    Parameters
    ----------
    df : pd.DataFrame  (original, not PCA-reduced)
    labels : np.ndarray  cluster labels (same length as df)
    numeric_cols : list[str] | None

    Returns
    -------
    pd.DataFrame  indexed by cluster label
    """
    df = df.copy()
    df["_cluster"] = labels

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "_cluster"]

    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if not numeric_cols:
        return pd.DataFrame({"cluster": sorted(df["_cluster"].unique())})

    summary = (
        df.groupby("_cluster")[numeric_cols]
        .mean()
        .round(3)
        .reset_index()
        .rename(columns={"_cluster": "cluster"})
    )
    size_map = df["_cluster"].value_counts().rename("size").reset_index()
    size_map.columns = ["cluster", "size"]
    summary = summary.merge(size_map, on="cluster", how="left")
    summary = summary.sort_values("cluster").reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    warnings: list[str],
) -> float | None:
    """Compute silhouette score safely; return None if not computable."""
    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2:
        warnings.append("Silhouette score requires at least 2 clusters; skipped.")
        return None
    n_valid = (labels >= 0).sum()
    if n_valid <= len(unique):
        warnings.append("Not enough samples for silhouette score; skipped.")
        return None
    try:
        # Cap at 10 000 samples for speed
        if n_valid > 10_000:
            idx = np.random.default_rng(42).choice(n_valid, size=10_000, replace=False)
            mask = labels >= 0
            valid_X = X[mask][idx]
            valid_labels = labels[mask][idx]
        else:
            mask = labels >= 0
            valid_X = X[mask]
            valid_labels = labels[mask]
        return float(silhouette_score(valid_X, valid_labels))
    except Exception as exc:
        warnings.append(f"Silhouette computation failed: {exc}")
        return None
