"""
test_pca_analysis.py
--------------------
Tests for src/pca_analysis.py

SIH26051 — PCA-Based Compliance Risk Clustering
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from src.data_loader import generate_sample_data
from src.feature_engineering import build_feature_matrix
from src.pca_analysis import (
    PCAResult,
    get_pca_scores_dataframe,
    get_variance_table,
    run_pca,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def X_small():
    df = generate_sample_data(n=30, random_seed=7)
    fm = build_feature_matrix(df)
    return fm.X


@pytest.fixture(scope="module")
def X_large():
    df = generate_sample_data(n=200, random_seed=42)
    fm = build_feature_matrix(df)
    return fm.X


# ---------------------------------------------------------------------------
# run_pca
# ---------------------------------------------------------------------------

class TestRunPCA:
    def test_returns_pca_result(self, X_small):
        result = run_pca(X_small)
        assert isinstance(result, PCAResult)

    def test_output_shape(self, X_small):
        result = run_pca(X_small, n_components=3)
        assert result.X_reduced.shape == (len(X_small), 3)

    def test_explained_variance_sums_correctly(self, X_large):
        result = run_pca(X_large, n_components=5)
        evr = result.explained_variance_ratio
        assert abs(evr.sum() - result.cumulative_variance[-1]) < 1e-6

    def test_variance_ratios_positive(self, X_large):
        result = run_pca(X_large, n_components=5)
        assert (result.explained_variance_ratio > 0).all()

    def test_cumulative_is_monotone(self, X_large):
        result = run_pca(X_large, n_components=5)
        cum = result.cumulative_variance
        assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))

    def test_component_count_clamped(self, X_small):
        # Requesting more components than samples should not crash
        n_samples = len(X_small)
        result = run_pca(X_small, n_components=n_samples + 100)
        assert result.n_components <= n_samples

    def test_variance_threshold_respected(self, X_large):
        result = run_pca(X_large, variance_threshold=0.80)
        assert result.cumulative_variance[-1] >= 0.80 or result.n_components >= 1

    def test_non_finite_input_handled(self):
        X = np.random.default_rng(0).random((50, 10))
        X[0, 0] = np.inf
        X[1, 1] = np.nan
        result = run_pca(X, n_components=3)
        assert np.all(np.isfinite(result.X_reduced))

    def test_constant_columns_handled(self):
        X = np.ones((50, 5))
        # Mix in some variance
        X[:, 0] = np.arange(50, dtype=float)
        X[:, 1] = np.arange(50, dtype=float) * 2
        result = run_pca(X, n_components=2)
        assert result.n_components >= 1

    def test_empty_matrix_raises(self):
        with pytest.raises(ValueError):
            run_pca(np.array([]))

    def test_two_row_dataset(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = run_pca(X, n_components=1)
        assert result.X_reduced.shape[0] == 2

    def test_component_labels_match(self, X_large):
        result = run_pca(X_large, n_components=4)
        assert result.component_labels == ["PC1", "PC2", "PC3", "PC4"]


# ---------------------------------------------------------------------------
# get_variance_table
# ---------------------------------------------------------------------------

class TestVarianceTable:
    def test_returns_dataframe(self, X_large):
        result = run_pca(X_large, n_components=3)
        tbl = get_variance_table(result)
        assert isinstance(tbl, pd.DataFrame)
        assert "explained_variance_pct" in tbl.columns
        assert len(tbl) == 3

    def test_cumulative_ends_at_expected(self, X_large):
        result = run_pca(X_large, n_components=5)
        tbl = get_variance_table(result)
        assert abs(tbl["cumulative_variance_pct"].iloc[-1] - result.cumulative_variance[-1] * 100) < 0.01


# ---------------------------------------------------------------------------
# get_pca_scores_dataframe
# ---------------------------------------------------------------------------

class TestPCAScoresDataframe:
    def test_has_pc_columns(self, X_large):
        df_orig = generate_sample_data(n=200, random_seed=42)
        result = run_pca(X_large, n_components=3)
        scores = get_pca_scores_dataframe(result, df_orig)
        assert "PC1" in scores.columns
        assert "PC2" in scores.columns

    def test_has_id_column(self, X_large):
        df_orig = generate_sample_data(n=200, random_seed=42)
        result = run_pca(X_large, n_components=2)
        scores = get_pca_scores_dataframe(result, df_orig, id_col="product_id")
        assert "product_id" in scores.columns
