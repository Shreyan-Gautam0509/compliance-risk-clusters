"""
test_feature_engineering.py
----------------------------
Tests for src/feature_engineering.py

SIH26051 — PCA-Based Compliance Risk Clustering
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_sample_data
from src.feature_engineering import (
    DEFAULT_CAT_COLS,
    DEFAULT_NUM_COLS,
    DEFAULT_TEXT_COLS,
    FeatureMatrix,
    build_feature_matrix,
    get_feature_statistics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    """20-row synthetic dataset for quick tests."""
    return generate_sample_data(n=20, random_seed=1)


@pytest.fixture(scope="module")
def full_df():
    """500-row synthetic dataset."""
    return generate_sample_data(n=500, random_seed=42)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_returns_feature_matrix(self, small_df):
        fm = build_feature_matrix(small_df)
        assert isinstance(fm, FeatureMatrix)

    def test_output_shape(self, small_df):
        fm = build_feature_matrix(small_df)
        assert fm.X.shape[0] == len(small_df)
        assert fm.X.shape[1] > 0

    def test_no_inf_or_nan(self, full_df):
        fm = build_feature_matrix(full_df)
        assert np.all(np.isfinite(fm.X)), "Feature matrix contains non-finite values"

    def test_feature_names_length_matches(self, small_df):
        fm = build_feature_matrix(small_df)
        assert len(fm.feature_names) == fm.X.shape[1]

    def test_works_without_text_cols(self, small_df):
        fm = build_feature_matrix(small_df, text_cols=[])
        assert fm.X.shape[0] == len(small_df)
        assert fm.n_text_features == 0

    def test_works_without_cat_cols(self, small_df):
        fm = build_feature_matrix(small_df, cat_cols=[])
        assert fm.n_cat_features == 0

    def test_only_num_cols(self, small_df):
        fm = build_feature_matrix(small_df, text_cols=[], cat_cols=[])
        assert fm.X.shape[1] > 0

    def test_empty_text_column_handled(self, small_df):
        df = small_df.copy()
        df["title"] = ""
        fm = build_feature_matrix(df)
        assert np.all(np.isfinite(fm.X))

    def test_all_nan_numeric_column_handled(self, small_df):
        df = small_df.copy()
        df["declared_price"] = np.nan
        # Should not raise; imputer fills NaN
        fm = build_feature_matrix(df)
        assert np.all(np.isfinite(fm.X))

    def test_large_dataset(self, full_df):
        fm = build_feature_matrix(full_df)
        assert fm.X.shape[0] == 500
        assert fm.X.shape[1] >= 5


# ---------------------------------------------------------------------------
# get_feature_statistics
# ---------------------------------------------------------------------------

class TestGetFeatureStatistics:
    def test_returns_dataframe(self, small_df):
        stats = get_feature_statistics(small_df, DEFAULT_NUM_COLS)
        assert isinstance(stats, pd.DataFrame)

    def test_columns_present(self, small_df):
        stats = get_feature_statistics(small_df, DEFAULT_NUM_COLS)
        assert "feature" in stats.columns

    def test_handles_missing_columns(self, small_df):
        stats = get_feature_statistics(small_df, ["nonexistent_col"])
        assert stats.empty
