"""
pca_analysis.py
---------------
Genuine PCA on the dense feature matrix produced by feature_engineering.
Includes automatic component-count selection, explained-variance reporting,
and shape-safety guards.

SIH26051 — PCA-Based Compliance Risk Clustering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    """Holds the outcome of a PCA fit."""
    X_reduced: np.ndarray                      # (n_samples × n_components)
    pca: PCA                                   # Fitted sklearn PCA object
    n_components: int = 0
    original_n_features: int = 0
    explained_variance_ratio: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    cumulative_variance: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    component_labels: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_pca(
    X: np.ndarray,
    n_components: int | None = None,
    variance_threshold: float | None = 0.90,
    random_state: int = 42,
) -> PCAResult:
    """
    Fit PCA on the dense matrix *X*.

    Component selection priority:
    1. If *n_components* is given explicitly, use it (clamped to valid range).
    2. Else if *variance_threshold* is given, find the minimum number of
       components that reach the target cumulative explained variance.
    3. Otherwise default to min(10, max_valid).

    Parameters
    ----------
    X : np.ndarray
        Dense, finite feature matrix (n_samples × n_features).
    n_components : int | None
        Explicit number of principal components.
    variance_threshold : float | None
        Target cumulative explained variance (0 < v ≤ 1). Used when
        n_components is None.
    random_state : int

    Returns
    -------
    PCAResult

    Raises
    ------
    ValueError
        If X is empty or has no valid features.
    """
    warnings: list[str] = []

    # ------------------------------------------------------------------ #
    # Input validation                                                     #
    # ------------------------------------------------------------------ #
    if X is None or X.size == 0:
        raise ValueError("Feature matrix X is empty — cannot run PCA.")

    n_samples, n_features = X.shape
    if n_features == 0:
        raise ValueError("Feature matrix has zero columns — cannot run PCA.")

    # Replace any remaining non-finite values
    if not np.all(np.isfinite(X)):
        n_bad = np.sum(~np.isfinite(X))
        X = np.where(np.isfinite(X), X, 0.0)
        warnings.append(f"Replaced {n_bad} non-finite value(s) with 0 before PCA.")

    # Remove constant columns (zero variance causes NaN in PCA loadings)
    col_std = X.std(axis=0)
    zero_var_mask = col_std == 0
    if zero_var_mask.any():
        n_dropped = int(zero_var_mask.sum())
        X = X[:, ~zero_var_mask]
        warnings.append(
            f"Dropped {n_dropped} constant feature column(s) before PCA."
        )
        n_features = X.shape[1]

    if n_features == 0:
        raise ValueError("All features have zero variance; PCA is not possible.")

    # Maximum valid number of PCA components
    max_components = min(n_samples, n_features)

    # ------------------------------------------------------------------ #
    # Determine n_components                                               #
    # ------------------------------------------------------------------ #
    if n_components is not None:
        n_comp = int(n_components)
        if n_comp < 1:
            n_comp = 1
            warnings.append("n_components < 1; clamped to 1.")
        if n_comp > max_components:
            warnings.append(
                f"n_components={n_comp} exceeds max ({max_components}); "
                f"clamped to {max_components}."
            )
            n_comp = max_components
    else:
        # First fit with all components to find threshold
        n_comp = _find_n_components_for_variance(
            X, variance_threshold or 0.90, max_components, random_state
        )

    logger.info("Running PCA: n_samples=%d, n_features=%d, n_components=%d",
                n_samples, n_features, n_comp)

    # ------------------------------------------------------------------ #
    # Fit PCA                                                              #
    # ------------------------------------------------------------------ #
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_
    cum_var = np.cumsum(evr)
    component_labels = [f"PC{i+1}" for i in range(n_comp)]

    if warnings:
        for w in warnings:
            logger.warning(w)

    return PCAResult(
        X_reduced=X_reduced,
        pca=pca,
        n_components=n_comp,
        original_n_features=n_features,
        explained_variance_ratio=evr,
        cumulative_variance=cum_var,
        component_labels=component_labels,
        warnings=warnings,
    )


def _find_n_components_for_variance(
    X: np.ndarray,
    target: float,
    max_components: int,
    random_state: int,
) -> int:
    """Return the smallest n that explains >= target of variance."""
    # Use a full PCA fit (faster than incremental for moderate sizes)
    try:
        probe = PCA(n_components=max_components, random_state=random_state)
        probe.fit(X)
        cum = np.cumsum(probe.explained_variance_ratio_)
        candidates = np.where(cum >= target)[0]
        if len(candidates) == 0:
            n = max_components
        else:
            n = int(candidates[0]) + 1
    except Exception as exc:
        logger.warning("Probe PCA failed (%s); defaulting to n_components=min(10, max).", exc)
        n = min(10, max_components)

    return max(1, n)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_variance_table(result: PCAResult) -> pd.DataFrame:
    """Return a DataFrame with per-component variance statistics."""
    return pd.DataFrame({
        "component": result.component_labels,
        "explained_variance_ratio": result.explained_variance_ratio,
        "cumulative_variance": result.cumulative_variance,
        "explained_variance_pct": (result.explained_variance_ratio * 100).round(2),
        "cumulative_variance_pct": (result.cumulative_variance * 100).round(2),
    })


def get_pca_scores_dataframe(
    result: PCAResult,
    df_original: pd.DataFrame,
    id_col: str = "product_id",
) -> pd.DataFrame:
    """
    Return a DataFrame of PCA scores (one row per listing).

    Columns: product_id (if present), PC1, PC2, ..., PCn.
    """
    scores = pd.DataFrame(
        result.X_reduced,
        columns=result.component_labels,
    )
    if id_col in df_original.columns:
        scores.insert(0, id_col, df_original[id_col].values)
    return scores
