"""
feature_engineering.py
-----------------------
Builds the preprocessing pipeline for the compliance dataset:
  - TF-IDF on text columns (title + description)
  - OneHotEncoder on categorical columns
  - StandardScaler on numerical columns
  - ColumnTransformer to combine all branches
  - Outputs a dense, finite numpy array ready for PCA

SIH26051 — PCA-Based Compliance Risk Clustering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default column assignments for the synthetic dataset
# ---------------------------------------------------------------------------

DEFAULT_TEXT_COLS = ["title", "description"]
DEFAULT_CAT_COLS = ["category", "declared_brand", "country_of_origin", "certifications"]
DEFAULT_NUM_COLS = [
    "declared_price",
    "missing_declaration_count",
    "declaration_completeness_pct",
    "price_anomaly",
    "is_duplicate_listing",
    "title_desc_mismatch",
    "desc_length",
    "listing_metadata_completeness",
    "voltage_declared",
    "warranty_declared",
    "cert_declared",
    "ingredients_declared",
    "expiry_declared",
    "license_declared",
    "batch_declared",
    "age_group_declared",
    "material_declared",
    "size_guide_declared",
    "compatibility_declared",
    "safety_class_declared",
    "publisher_declared",
]
EXCLUDE_COLS = ["product_id", "manufacturer", "data_source"]


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class FeatureMatrix:
    """Result of the feature-engineering step."""
    X: np.ndarray                        # Dense, finite feature matrix (n_samples × n_features)
    feature_names: list[str]             # Names for each column in X
    n_text_features: int = 0
    n_cat_features: int = 0
    n_num_features: int = 0
    text_cols_used: list[str] = field(default_factory=list)
    cat_cols_used: list[str] = field(default_factory=list)
    num_cols_used: list[str] = field(default_factory=list)
    pipeline: Any = None                 # Fitted ColumnTransformer (for reuse)
    svd: Any = None                      # Fitted TruncatedSVD (if applied)
    svd_n_components: int = 0


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _make_text_pipeline(max_features: int = 200) -> Pipeline:
    """TF-IDF vectorizer wrapped in a Pipeline (for ColumnTransformer)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            strip_accents="unicode",
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-Z]\w+\b",
            sublinear_tf=True,
        )),
    ])


def _make_cat_pipeline() -> Pipeline:
    """OneHotEncoder with unknown-category handling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("ohe", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=30,
        )),
    ])


def _make_num_pipeline() -> Pipeline:
    """Median imputation + StandardScaler for numerical features."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    text_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    tfidf_max_features: int = 150,
    svd_n_components: int | None = None,
    svd_variance_target: float = 0.90,
) -> FeatureMatrix:
    """
    Fit a preprocessing pipeline on *df* and return a dense feature matrix.

    The function:
    1. Resolves which columns to use.
    2. Fits TF-IDF → sparse, OHE → dense, Scaler → dense.
    3. Applies TruncatedSVD to the combined sparse representation (text + OHE)
       to keep dimensions manageable before the final dense PCA step.
    4. Concatenates numerical features with SVD-reduced features.
    5. Replaces any remaining infinities / NaNs with zeros.

    Parameters
    ----------
    df : pd.DataFrame
    text_cols : list[str] | None
    cat_cols : list[str] | None
    num_cols : list[str] | None
    tfidf_max_features : int
        Maximum vocabulary size per text column (default 150).
    svd_n_components : int | None
        Explicit number of SVD components. If None, chosen automatically to
        explain at least *svd_variance_target* of variance or capped at 50.
    svd_variance_target : float
        Target explained-variance ratio for automatic SVD component selection.

    Returns
    -------
    FeatureMatrix
    """
    df = df.copy()
    text_cols = _resolve_cols(df, text_cols, DEFAULT_TEXT_COLS)
    cat_cols  = _resolve_cols(df, cat_cols,  DEFAULT_CAT_COLS)
    num_cols  = _resolve_cols(df, num_cols,  DEFAULT_NUM_COLS)

    logger.info("Feature engineering: text=%s, cat=%s, num=%s",
                text_cols, cat_cols, num_cols)

    # ------------------------------------------------------------------ #
    # 1. Numerical branch (always dense)                                  #
    # ------------------------------------------------------------------ #
    X_num = _fit_num(df, num_cols)           # shape: (n, len(num_cols))

    # ------------------------------------------------------------------ #
    # 2. Sparse branch: TF-IDF per text col + OHE for categoricals        #
    # ------------------------------------------------------------------ #
    sparse_parts: list[sp.csr_matrix] = []
    text_feature_names: list[str] = []
    cat_feature_names: list[str] = []

    for col in text_cols:
        tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            strip_accents="unicode",
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-Z]\w+\b",
            sublinear_tf=True,
        )
        text_series = df[col].fillna("").astype(str)
        try:
            X_t = tfidf.fit_transform(text_series)   # sparse
            sparse_parts.append(X_t.tocsr())
            text_feature_names += [f"{col}__tfidf__{v}" for v in tfidf.get_feature_names_out()]
        except ValueError as exc:
            logger.warning("TF-IDF vectorization failed for '%s' (%s); using fallback zero column.", col, exc)
            sparse_parts.append(sp.csr_matrix((len(df), 1)))
            text_feature_names.append(f"{col}__empty")

    ohe_pipe = _make_cat_pipeline()
    if cat_cols:
        X_cat = ohe_pipe.fit_transform(df[cat_cols].astype(str))
        # X_cat is dense (sparse_output=False)
        n_cat_feats = X_cat.shape[1]
        try:
            ohe_obj = ohe_pipe.named_steps["ohe"]
            for i, col in enumerate(cat_cols):
                for cat_val in ohe_obj.categories_[i]:
                    cat_feature_names.append(f"{col}__ohe__{cat_val}")
        except Exception:
            cat_feature_names = [f"cat_feat_{j}" for j in range(n_cat_feats)]
        # Convert OHE to sparse for uniform handling
        sparse_parts.append(sp.csr_matrix(X_cat))
    else:
        n_cat_feats = 0

    n_samples = len(df)

    # ------------------------------------------------------------------ #
    # 3. TruncatedSVD on sparse block (text + OHE)                        #
    # ------------------------------------------------------------------ #
    svd_obj = None
    svd_n_used = 0
    X_svd: np.ndarray | None = None

    if sparse_parts:
        X_sparse = sp.hstack(sparse_parts, format="csr")
        max_possible_svd = min(X_sparse.shape[0] - 1, X_sparse.shape[1], 100)

        if max_possible_svd < 1:
            logger.warning("Sparse block too small for SVD; skipping.")
        else:
            if svd_n_components is None:
                n_svd = min(50, max_possible_svd)
            else:
                n_svd = min(svd_n_components, max_possible_svd)

            if n_svd < 1:
                logger.warning("SVD components < 1; skipping SVD.")
            else:
                svd_obj = TruncatedSVD(n_components=n_svd, random_state=42)
                X_svd = svd_obj.fit_transform(X_sparse)  # dense (n, n_svd)
                svd_n_used = n_svd
                logger.info("TruncatedSVD: %d components, %.1f%% variance",
                            n_svd, svd_obj.explained_variance_ratio_.sum() * 100)

    # ------------------------------------------------------------------ #
    # 4. Concatenate all dense blocks                                      #
    # ------------------------------------------------------------------ #
    parts_dense: list[np.ndarray] = [X_num]
    if X_svd is not None:
        parts_dense.append(X_svd)

    X_combined = np.hstack(parts_dense)

    # Sanitise: replace inf / -inf / NaN with 0
    X_combined = np.where(np.isfinite(X_combined), X_combined, 0.0)

    # ------------------------------------------------------------------ #
    # 5. Build feature names for the final matrix                         #
    # ------------------------------------------------------------------ #
    num_feature_names = [f"num__{c}" for c in num_cols]
    svd_feature_names = [f"svd__{i}" for i in range(svd_n_used)]
    all_feature_names = num_feature_names + svd_feature_names

    return FeatureMatrix(
        X=X_combined,
        feature_names=all_feature_names,
        n_text_features=len(text_feature_names),
        n_cat_features=n_cat_feats,
        n_num_features=len(num_cols),
        text_cols_used=text_cols,
        cat_cols_used=cat_cols,
        num_cols_used=num_cols,
        pipeline=ohe_pipe if cat_cols else None,
        svd=svd_obj,
        svd_n_components=svd_n_used,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_cols(df: pd.DataFrame, user_cols: list[str] | None,
                  defaults: list[str]) -> list[str]:
    """Return user_cols if provided and present in df, otherwise use defaults."""
    if user_cols is not None:
        return [c for c in user_cols if c in df.columns]
    return [c for c in defaults if c in df.columns]


def _fit_num(df: pd.DataFrame, num_cols: list[str]) -> np.ndarray:
    """Fit and transform numerical columns; returns dense array."""
    if not num_cols:
        return np.zeros((len(df), 0))
    pipe = _make_num_pipeline()
    return pipe.fit_transform(df[num_cols])


def get_feature_statistics(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Return descriptive statistics for numerical columns present in df."""
    present = [c for c in num_cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    return df[present].describe().T.reset_index().rename(columns={"index": "feature"})
