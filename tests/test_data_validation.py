"""
test_data_validation.py
-----------------------
Tests for src/data_validation.py

SIH26051 — PCA-Based Compliance Risk Clustering
"""

import numpy as np
import pandas as pd
import pytest

from src.data_validation import (
    ValidationReport,
    clean_dataframe,
    coerce_numeric_columns,
    get_missing_value_heatmap_data,
    infer_column_roles,
    validate_dataframe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product_id": [f"P{i}" for i in range(10)],
        "category": ["Electronics"] * 5 + ["Toys"] * 5,
        "declared_price": [100.0, 200.0, None, 150.0, 50.0,
                           80.0, 90.0, 110.0, 130.0, 75.0],
        "description": ["A good product"] * 10,
    })


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------

class TestValidateDataframe:
    def test_normal_df(self, sample_df):
        report = validate_dataframe(sample_df)
        assert report.is_valid
        assert report.row_count == 10
        assert report.column_count == 4

    def test_empty_df_gives_error(self):
        report = validate_dataframe(pd.DataFrame())
        assert not report.is_valid
        assert any("zero" in e.lower() for e in report.errors)

    def test_detects_missing_values(self, sample_df):
        report = validate_dataframe(sample_df)
        assert "declared_price" in report.missing_summary
        assert report.missing_summary["declared_price"] == 1

    def test_detects_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        report = validate_dataframe(df)
        assert report.duplicate_row_count == 1

    def test_detects_constant_columns(self):
        df = pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3]})
        report = validate_dataframe(df)
        assert "x" in report.constant_columns

    def test_detects_all_null_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        report = validate_dataframe(df)
        assert "b" in report.all_null_columns

    def test_warns_on_small_dataset(self):
        df = pd.DataFrame({"a": range(5), "b": range(5)})
        report = validate_dataframe(df)
        assert any("30" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# infer_column_roles
# ---------------------------------------------------------------------------

class TestInferColumnRoles:
    def test_numeric_detected(self):
        df = pd.DataFrame({"price": [1.0, 2.0], "flag": [0, 1]})
        roles = infer_column_roles(df)
        assert "price" in roles["numerical"]
        assert "flag" in roles["numerical"]

    def test_id_col_detected(self):
        df = pd.DataFrame({"product_id": ["P1", "P2"], "val": [1, 2]})
        roles = infer_column_roles(df)
        assert "product_id" in roles["id"]

    def test_user_override_respected(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        roles = infer_column_roles(df, user_text_cols=["a"])
        assert "a" in roles["text"]

    def test_short_strings_categorical(self):
        df = pd.DataFrame({"cat": ["Electronics", "Toys"] * 5})
        roles = infer_column_roles(df)
        assert "cat" in roles["categorical"]


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------

class TestCleanDataframe:
    def test_drops_all_null_cols(self):
        df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
        cleaned = clean_dataframe(df, drop_all_null_cols=True)
        assert "b" not in cleaned.columns

    def test_drops_duplicates_when_requested(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        cleaned = clean_dataframe(df, drop_duplicate_rows=True)
        assert len(cleaned) == 2

    def test_original_not_mutated(self):
        df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
        _ = clean_dataframe(df)
        assert "b" in df.columns   # original intact


# ---------------------------------------------------------------------------
# coerce_numeric_columns
# ---------------------------------------------------------------------------

class TestCoerceNumericColumns:
    def test_coerces_mixed_column(self):
        df = pd.DataFrame({"price": ["100", "abc", "200"]})
        result = coerce_numeric_columns(df, ["price"])
        assert pd.api.types.is_numeric_dtype(result["price"])
        assert result["price"].isna().sum() == 1

    def test_ignores_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = coerce_numeric_columns(df, ["nonexistent"])
        assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# get_missing_value_heatmap_data
# ---------------------------------------------------------------------------

class TestMissingHeatmap:
    def test_only_returns_cols_with_missing(self, sample_df):
        result = get_missing_value_heatmap_data(sample_df)
        assert set(result["column"]).issubset(set(sample_df.columns))
        assert (result["missing_count"] > 0).all()

    def test_empty_when_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = get_missing_value_heatmap_data(df)
        assert result.empty
