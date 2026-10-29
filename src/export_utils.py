"""
export_utils.py
---------------
CSV export helpers and joblib model persistence for SIH26051.

Saves / loads:
  - Enriched results DataFrame (CSV)
  - PCA variance table (CSV)
  - Cluster summary (CSV)
  - Fitted PCA model + preprocessing artefacts (joblib)
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default model directory (relative to project root)
DEFAULT_MODEL_DIR = Path("models")

# Required export columns (product_id may not always be present)
REQUIRED_EXPORT_COLS = ["cluster_label", "risk_score", "risk_label", "risk_reasons"]


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def build_results_dataframe(
    df_original: pd.DataFrame,
    labels: np.ndarray,
    risk_df: pd.DataFrame,
    pca_scores: pd.DataFrame | None = None,
    id_col: str = "product_id",
) -> pd.DataFrame:
    """
    Merge original data, cluster labels, risk scores, and PCA scores into
    one enriched results DataFrame.

    Parameters
    ----------
    df_original : pd.DataFrame  original (pre-PCA) dataset
    labels : np.ndarray  cluster labels
    risk_df : pd.DataFrame  must contain risk_score, risk_label, risk_reasons
    pca_scores : pd.DataFrame | None  PC1, PC2, … columns (optional)
    id_col : str  identifier column name

    Returns
    -------
    pd.DataFrame
    """
    result = df_original.copy().reset_index(drop=True)

    # Add cluster labels
    result["cluster_label"] = labels

    # Merge risk columns
    risk_cols = [c for c in ["risk_score", "risk_label", "risk_reasons"]
                 if c in risk_df.columns]
    for col in risk_cols:
        result[col] = risk_df[col].values

    # Optionally add PCA scores
    if pca_scores is not None:
        pc_cols = [c for c in pca_scores.columns
                   if c.startswith("PC") and c != id_col]
        for col in pc_cols:
            result[col] = pca_scores[col].values

    # Move id_col to front
    if id_col in result.columns:
        cols = [id_col] + [c for c in result.columns if c != id_col]
        result = result[cols]

    return result


def results_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes (for Streamlit download)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_variance_table(variance_df: pd.DataFrame) -> bytes:
    """Return PCA variance table as CSV bytes."""
    buf = io.StringIO()
    variance_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_cluster_summary(cluster_summary_df: pd.DataFrame) -> bytes:
    """Return cluster summary as CSV bytes."""
    buf = io.StringIO()
    cluster_summary_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    artefacts: dict[str, Any],
    filename: str = "compliance_model.joblib",
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> Path:
    """
    Save a dictionary of model artefacts using joblib.

    Parameters
    ----------
    artefacts : dict  e.g. {"pca": pca_obj, "svd": svd_obj, ...}
    filename : str
    model_dir : str | Path

    Returns
    -------
    Path  full path to the saved file
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / filename
    joblib.dump(artefacts, save_path)
    logger.info("Model artefacts saved to %s", save_path)
    return save_path


def load_model(
    filename: str = "compliance_model.joblib",
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """
    Load model artefacts from a joblib file.

    Returns
    -------
    dict[str, Any]

    Raises
    ------
    FileNotFoundError  if the file does not exist
    ValueError         if the file is corrupt or unpicklable
    """
    model_dir = Path(model_dir)
    load_path = model_dir / filename
    if not load_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {load_path}. "
            "Run the full pipeline first to generate and save a model."
        )
    try:
        artefacts = joblib.load(load_path)
    except Exception as exc:
        raise ValueError(f"Failed to load model from {load_path}: {exc}") from exc
    logger.info("Model artefacts loaded from %s", load_path)
    return artefacts


def model_exists(
    filename: str = "compliance_model.joblib",
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> bool:
    """Return True if the model file exists on disk."""
    return (Path(model_dir) / filename).exists()


# ---------------------------------------------------------------------------
# Validation helpers for exports
# ---------------------------------------------------------------------------

def validate_export_dataframe(df: pd.DataFrame) -> list[str]:
    """
    Check that the results DataFrame has all required export columns.

    Returns
    -------
    list[str]  list of missing column names (empty if all present)
    """
    return [c for c in REQUIRED_EXPORT_COLS if c not in df.columns]
