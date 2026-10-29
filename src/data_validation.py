"""
data_validation.py
------------------
Schema validation, missing-value analysis, duplicate detection,
and data-quality reporting for the SIH26051 compliance project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for validation results
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Holds the result of a full data-quality check."""
    row_count: int = 0
    column_count: int = 0
    duplicate_row_count: int = 0
    constant_columns: list[str] = field(default_factory=list)
    all_null_columns: list[str] = field(default_factory=list)
    missing_summary: dict[str, int] = field(default_factory=dict)   # col → count
    dtype_map: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def total_missing(self) -> int:
        return sum(self.missing_summary.values())


# ---------------------------------------------------------------------------
# Known schema for the synthetic / expected dataset
# ---------------------------------------------------------------------------

EXPECTED_NUMERICAL_COLS = [
    "declared_price",
    "missing_declaration_count",
    "declaration_completeness_pct",
    "price_anomaly",
    "is_duplicate_listing",
    "title_desc_mismatch",
    "desc_length",
    "listing_metadata_completeness",
    "voltage_declared", "warranty_declared", "cert_declared",
    "ingredients_declared", "expiry_declared", "license_declared",
    "batch_declared", "age_group_declared", "material_declared",
    "size_guide_declared", "compatibility_declared",
    "safety_class_declared", "publisher_declared",
]

EXPECTED_CATEGORICAL_COLS = [
    "category", "declared_brand", "manufacturer",
    "country_of_origin", "certifications",
]

EXPECTED_TEXT_COLS = ["title", "description"]

ID_COLS = ["product_id"]


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    """
    Run a complete data-quality check on *df*.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    report.row_count = len(df)
    report.column_count = len(df.columns)
    report.dtype_map = {c: str(df[c].dtype) for c in df.columns}

    # --- Fatal checks -------------------------------------------------------
    if report.row_count == 0:
        report.errors.append("DataFrame has zero rows.")
        return report

    if report.column_count == 0:
        report.errors.append("DataFrame has zero columns.")
        return report

    # --- Missing values ------------------------------------------------------
    missing = df.isnull().sum()
    report.missing_summary = {c: int(v) for c, v in missing.items() if v > 0}

    # --- All-null columns ----------------------------------------------------
    report.all_null_columns = [c for c in df.columns if df[c].isnull().all()]
    if report.all_null_columns:
        report.warnings.append(
            f"All-null columns (will be dropped): {report.all_null_columns}"
        )

    # --- Constant columns (only one unique non-null value) -------------------
    report.constant_columns = [
        c for c in df.columns
        if df[c].dropna().nunique() <= 1 and c not in report.all_null_columns
    ]
    if report.constant_columns:
        report.warnings.append(
            f"Constant columns (zero variance, may be excluded from PCA): "
            f"{report.constant_columns}"
        )

    # --- Duplicate rows ------------------------------------------------------
    report.duplicate_row_count = int(df.duplicated().sum())
    if report.duplicate_row_count:
        report.warnings.append(
            f"{report.duplicate_row_count} fully duplicate row(s) detected."
        )

    # --- Minimum row warning for PCA/clustering ------------------------------
    if report.row_count < 10:
        report.warnings.append(
            f"Only {report.row_count} rows — PCA and clustering may not be "
            "meaningful. A minimum of 30 rows is recommended."
        )

    logger.info(
        "Validation complete: %d rows, %d cols, %d missing values, "
        "%d duplicate rows, %d errors.",
        report.row_count, report.column_count,
        report.total_missing, report.duplicate_row_count, len(report.errors),
    )
    return report


def infer_column_roles(
    df: pd.DataFrame,
    user_id_cols: list[str] | None = None,
    user_text_cols: list[str] | None = None,
    user_cat_cols: list[str] | None = None,
    user_num_cols: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Automatically infer which columns are IDs, text, categorical, or numerical.

    User-supplied overrides take precedence over auto-detection.

    Returns
    -------
    dict with keys: 'id', 'text', 'categorical', 'numerical'
    """
    id_cols: list[str] = list(user_id_cols or [])
    text_cols: list[str] = list(user_text_cols or [])
    cat_cols: list[str] = list(user_cat_cols or [])
    num_cols: list[str] = list(user_num_cols or [])

    already_assigned = set(id_cols + text_cols + cat_cols + num_cols)

    for col in df.columns:
        if col in already_assigned:
            continue
        # Auto-detection heuristics
        if col.lower().endswith("_id") or col.lower() == "id":
            id_cols.append(col)
        elif df[col].dtype == object:
            # Long strings → text; short strings → categorical
            sample = df[col].dropna().astype(str)
            if sample.empty:
                cat_cols.append(col)
            elif sample.str.len().mean() > 40:
                text_cols.append(col)
            else:
                n_unique = df[col].nunique(dropna=True)
                ratio = n_unique / max(len(df), 1)
                # High-cardinality → treat as text (e.g. free-form IDs)
                if ratio > 0.5 and n_unique > 50:
                    text_cols.append(col)
                else:
                    cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    return {
        "id": id_cols,
        "text": text_cols,
        "categorical": cat_cols,
        "numerical": num_cols,
    }


def clean_dataframe(
    df: pd.DataFrame,
    drop_all_null_cols: bool = True,
    drop_duplicate_rows: bool = False,
) -> pd.DataFrame:
    """
    Apply basic cleaning steps.

    Parameters
    ----------
    df : pd.DataFrame
    drop_all_null_cols : bool
        Drop columns where every value is null (default True).
    drop_duplicate_rows : bool
        Drop fully duplicate rows (default False — preserve for analysis).

    Returns
    -------
    pd.DataFrame
        A cleaned copy; the original is never mutated.
    """
    df = df.copy()

    if drop_all_null_cols:
        all_null = [c for c in df.columns if df[c].isnull().all()]
        if all_null:
            df = df.drop(columns=all_null)
            logger.info("Dropped all-null columns: %s", all_null)

    if drop_duplicate_rows:
        before = len(df)
        df = df.drop_duplicates()
        logger.info("Dropped %d duplicate rows.", before - len(df))

    return df


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Coerce the specified columns to numeric, replacing non-parseable values
    with NaN (rather than raising errors).

    Returns a copy of the DataFrame.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_missing_value_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame: column, missing_count, missing_pct."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_counts.values,
        "missing_pct": missing_pct.values,
    })
    return summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    ).reset_index(drop=True)
