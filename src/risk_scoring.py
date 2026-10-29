"""
risk_scoring.py
---------------
Transparent, rule-based estimated-risk scoring for product listings.

IMPORTANT DISCLAIMER
--------------------
This module assigns estimated-risk labels based on configurable,
human-readable rules derived from availability of declared information.
It does NOT claim to detect legal violations, regulatory breaches, or
confirmed fraud. Labels such as "Higher estimated risk" indicate that
a listing shows multiple incomplete or anomalous signals — they must be
treated as a prompt for further human review, not as a definitive finding.

SIH26051 — PCA-Based Compliance Risk Clustering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskScoringConfig:
    """
    Configurable weights and thresholds for the risk scorer.

    All weights are non-negative floats; higher weight → larger impact.
    Thresholds are applied to the normalised score (0.0 – 1.0).
    """
    # --- Signal weights (each signal contributes `weight` to raw score) ---
    weight_missing_declaration: float = 0.30   # per missing required declaration
    weight_price_anomaly: float = 0.20
    weight_duplicate_listing: float = 0.15
    weight_title_desc_mismatch: float = 0.15
    weight_low_completeness: float = 0.10      # when completeness < 50 %
    weight_low_metadata: float = 0.10          # when listing_metadata < 50 %

    # --- Thresholds for label assignment (normalised 0–1 score) ----------
    threshold_moderate: float = 0.30
    threshold_high: float = 0.60

    # --- Labels ---
    label_low: str = "Lower estimated risk"
    label_moderate: str = "Moderate estimated risk"
    label_high: str = "Higher estimated risk"

    # Max possible raw score (used for normalisation)
    _max_raw: float = 1.0   # recomputed on init


DEFAULT_CONFIG = RiskScoringConfig()


# ---------------------------------------------------------------------------
# Signal columns expected in the DataFrame
# ---------------------------------------------------------------------------

SIGNAL_COLS = {
    "missing_declaration_count":    "Missing required declaration fields",
    "price_anomaly":                "Unusual declared price",
    "is_duplicate_listing":         "Possible duplicate listing",
    "title_desc_mismatch":          "Title-description mismatch detected",
    "declaration_completeness_pct": "Declaration completeness below 50%",
    "listing_metadata_completeness": "Listing metadata completeness below 50%",
}


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_risk(
    df: pd.DataFrame,
    config: RiskScoringConfig | None = None,
    max_missing_decl: int | None = None,
) -> pd.DataFrame:
    """
    Compute an estimated-risk score and label for every row in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least some of the SIGNAL_COLS columns.
    config : RiskScoringConfig | None
        Use DEFAULT_CONFIG when None.
    max_missing_decl : int | None
        Maximum possible missing_declaration_count (used for normalisation).
        Auto-detected from data when None.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with added columns:
        - ``risk_score``     : float in [0.0, 1.0]
        - ``risk_label``     : str  (one of the three label strings)
        - ``risk_reasons``   : str  (pipe-separated human-readable reasons)
    """
    if config is None:
        config = DEFAULT_CONFIG

    df = df.copy()
    n = len(df)
    if n == 0:
        df["risk_score"] = pd.Series(dtype=float)
        df["risk_label"] = pd.Series(dtype=object)
        df["risk_reasons"] = pd.Series(dtype=object)
        return df

    raw_score = np.zeros(n, dtype=float)
    reasons: list[list[str]] = [[] for _ in range(n)]

    # ------------------------------------------------------------------ #
    # 1. Missing declaration count                                         #
    # ------------------------------------------------------------------ #
    if "missing_declaration_count" in df.columns:
        miss = pd.to_numeric(df["missing_declaration_count"], errors="coerce").fillna(0)
        if max_missing_decl is None:
            max_val = miss.max()
            max_missing_decl = max(int(max_val) if pd.notna(max_val) else 1, 1)
        # Normalise 0–1, then weight
        contrib = (miss / max_missing_decl).clip(0, 1) * config.weight_missing_declaration
        raw_score += contrib.values
        # Flag rows where any declarations are missing
        flagged = miss > 0
        for i in np.where(flagged)[0]:
            reasons[i].append(
                f"Missing {int(miss.iloc[i])} required declaration field(s)"
            )

    # ------------------------------------------------------------------ #
    # 2. Price anomaly                                                     #
    # ------------------------------------------------------------------ #
    if "price_anomaly" in df.columns:
        pa = pd.to_numeric(df["price_anomaly"], errors="coerce").fillna(0).clip(0, 1)
        raw_score += pa.values * config.weight_price_anomaly
        for i in np.where(pa > 0)[0]:
            reasons[i].append("Unusual declared price detected")

    # ------------------------------------------------------------------ #
    # 3. Duplicate listing                                                 #
    # ------------------------------------------------------------------ #
    if "is_duplicate_listing" in df.columns:
        dup = pd.to_numeric(df["is_duplicate_listing"], errors="coerce").fillna(0).clip(0, 1)
        raw_score += dup.values * config.weight_duplicate_listing
        for i in np.where(dup > 0)[0]:
            reasons[i].append("Possible duplicate listing")

    # ------------------------------------------------------------------ #
    # 4. Title–description mismatch                                        #
    # ------------------------------------------------------------------ #
    if "title_desc_mismatch" in df.columns:
        mis = pd.to_numeric(df["title_desc_mismatch"], errors="coerce").fillna(0).clip(0, 1)
        raw_score += mis.values * config.weight_title_desc_mismatch
        for i in np.where(mis > 0)[0]:
            reasons[i].append("Title-description mismatch detected")

    # ------------------------------------------------------------------ #
    # 5. Low declaration completeness                                      #
    # ------------------------------------------------------------------ #
    if "declaration_completeness_pct" in df.columns:
        comp = pd.to_numeric(df["declaration_completeness_pct"], errors="coerce").fillna(100)
        low_comp = comp < 50
        raw_score += low_comp.values.astype(float) * config.weight_low_completeness
        for i in np.where(low_comp)[0]:
            reasons[i].append(
                f"Low declaration completeness ({comp.iloc[i]:.0f}%)"
            )

    # ------------------------------------------------------------------ #
    # 6. Low metadata completeness                                         #
    # ------------------------------------------------------------------ #
    if "listing_metadata_completeness" in df.columns:
        meta = pd.to_numeric(df["listing_metadata_completeness"], errors="coerce").fillna(100)
        low_meta = meta < 50
        raw_score += low_meta.values.astype(float) * config.weight_low_metadata
        for i in np.where(low_meta)[0]:
            reasons[i].append(
                f"Low listing metadata completeness ({meta.iloc[i]:.0f}%)"
            )

    # ------------------------------------------------------------------ #
    # Normalise raw score to [0, 1]                                        #
    # ------------------------------------------------------------------ #
    max_possible = (
        config.weight_missing_declaration
        + config.weight_price_anomaly
        + config.weight_duplicate_listing
        + config.weight_title_desc_mismatch
        + config.weight_low_completeness
        + config.weight_low_metadata
    )
    if max_possible > 0:
        norm_score = (raw_score / max_possible).clip(0.0, 1.0)
    else:
        norm_score = raw_score.clip(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Assign labels                                                        #
    # ------------------------------------------------------------------ #
    labels = np.where(
        norm_score >= config.threshold_high,
        config.label_high,
        np.where(
            norm_score >= config.threshold_moderate,
            config.label_moderate,
            config.label_low,
        ),
    )

    df["risk_score"] = np.round(norm_score, 4)
    df["risk_label"] = labels
    df["risk_reasons"] = [" | ".join(r) if r else "No signals detected" for r in reasons]

    logger.info(
        "Risk scoring complete. Distribution: %s",
        df["risk_label"].value_counts().to_dict(),
    )

    return df


# ---------------------------------------------------------------------------
# Cluster-level risk interpretation
# ---------------------------------------------------------------------------

def summarise_cluster_risk(
    df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Summarise risk distribution per cluster.

    Parameters
    ----------
    df : pd.DataFrame  (must contain 'risk_label' and 'risk_score')
    labels : np.ndarray  cluster labels

    Returns
    -------
    pd.DataFrame  one row per cluster
    """
    df = df.copy()
    df["_cluster"] = labels

    agg = (
        df.groupby("_cluster")
        .agg(
            size=("risk_score", "count"),
            mean_risk_score=("risk_score", "mean"),
            pct_high=("risk_label", lambda x: (x == "Higher estimated risk").mean() * 100),
            pct_moderate=("risk_label", lambda x: (x == "Moderate estimated risk").mean() * 100),
            pct_low=("risk_label", lambda x: (x == "Lower estimated risk").mean() * 100),
        )
        .round(2)
        .reset_index()
        .rename(columns={"_cluster": "cluster"})
    )

    # Assign an interpretive cluster-level label
    def _cluster_label(row: pd.Series) -> str:
        if row["mean_risk_score"] >= 0.6:
            return "Higher estimated risk cluster"
        if row["mean_risk_score"] >= 0.3:
            return "Moderate estimated risk cluster"
        return "Lower estimated risk cluster"

    agg["cluster_risk_label"] = agg.apply(_cluster_label, axis=1)
    return agg


def get_flagged_listings(
    df: pd.DataFrame,
    risk_col: str = "risk_label",
    flag_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return subset of df where risk_label is in flag_labels."""
    if flag_labels is None:
        flag_labels = ["Higher estimated risk", "Moderate estimated risk"]
    if risk_col not in df.columns:
        return df.iloc[0:0]
    return df[df[risk_col].isin(flag_labels)].copy()
