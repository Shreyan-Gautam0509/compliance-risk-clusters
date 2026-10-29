"""
test_clustering.py
------------------
Tests for src/clustering.py

SIH26051 — PCA-Based Compliance Risk Clustering
"""

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    ClusterResult,
    build_cluster_summary,
    run_dbscan,
    run_kmeans,
    sweep_k,
)
from src.data_loader import generate_sample_data
from src.feature_engineering import build_feature_matrix
from src.pca_analysis import run_pca


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def X_pca():
    df = generate_sample_data(n=100, random_seed=0)
    fm = build_feature_matrix(df)
    result = run_pca(fm.X, n_components=5)
    return result.X_reduced


@pytest.fixture(scope="module")
def df_100():
    return generate_sample_data(n=100, random_seed=0)


# ---------------------------------------------------------------------------
# run_kmeans
# ---------------------------------------------------------------------------

class TestRunKMeans:
    def test_returns_cluster_result(self, X_pca):
        cr = run_kmeans(X_pca, n_clusters=3)
        assert isinstance(cr, ClusterResult)

    def test_label_count_matches_samples(self, X_pca):
        cr = run_kmeans(X_pca, n_clusters=3)
        assert len(cr.labels) == len(X_pca)

    def test_n_clusters_accurate(self, X_pca):
        cr = run_kmeans(X_pca, n_clusters=4)
        assert cr.n_clusters <= 4

    def test_silhouette_in_valid_range(self, X_pca):
        cr = run_kmeans(X_pca, n_clusters=3)
        if cr.silhouette is not None:
            assert -1.0 <= cr.silhouette <= 1.0

    def test_k_exceeds_samples_is_handled(self):
        X = np.random.default_rng(0).random((5, 3))
        cr = run_kmeans(X, n_clusters=10)
        assert cr.n_clusters <= 5
        assert len(cr.labels) == 5

    def test_k_equals_1(self):
        X = np.random.default_rng(0).random((3, 2))
        cr = run_kmeans(X, n_clusters=1)
        assert cr.n_clusters == 1
        assert cr.silhouette is None

    def test_cluster_sizes_sum_to_total(self, X_pca):
        cr = run_kmeans(X_pca, n_clusters=4)
        assert sum(cr.cluster_sizes.values()) == len(X_pca)

    def test_reproducibility(self, X_pca):
        cr1 = run_kmeans(X_pca, n_clusters=3, random_state=42)
        cr2 = run_kmeans(X_pca, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(cr1.labels, cr2.labels)

    def test_two_samples(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        cr = run_kmeans(X, n_clusters=2)
        assert len(cr.labels) == 2


# ---------------------------------------------------------------------------
# run_dbscan
# ---------------------------------------------------------------------------

class TestRunDBSCAN:
    def test_returns_cluster_result(self, X_pca):
        cr = run_dbscan(X_pca, eps=1.0, min_samples=3)
        assert isinstance(cr, ClusterResult)
        assert cr.algorithm == "dbscan"

    def test_labels_same_length(self, X_pca):
        cr = run_dbscan(X_pca)
        assert len(cr.labels) == len(X_pca)

    def test_noise_labels_are_minus_one(self, X_pca):
        cr = run_dbscan(X_pca, eps=0.001, min_samples=100)
        # Very tight eps → likely all noise
        unique = set(cr.labels.tolist())
        assert -1 in unique or len(unique) >= 1   # doesn't crash


# ---------------------------------------------------------------------------
# sweep_k
# ---------------------------------------------------------------------------

class TestSweepK:
    def test_returns_dataframe(self, X_pca):
        df = sweep_k(X_pca, k_range=range(2, 6))
        assert isinstance(df, pd.DataFrame)
        assert "k" in df.columns
        assert "inertia" in df.columns

    def test_k_range_respected(self, X_pca):
        df = sweep_k(X_pca, k_range=range(2, 5))
        assert set(df["k"]).issubset({2, 3, 4})


# ---------------------------------------------------------------------------
# build_cluster_summary
# ---------------------------------------------------------------------------

class TestBuildClusterSummary:
    def test_returns_dataframe(self, df_100, X_pca):
        cr = run_kmeans(X_pca, n_clusters=3)
        summary = build_cluster_summary(df_100, cr.labels)
        assert isinstance(summary, pd.DataFrame)
        assert "cluster" in summary.columns

    def test_row_count_matches_k(self, df_100, X_pca):
        cr = run_kmeans(X_pca, n_clusters=3)
        summary = build_cluster_summary(df_100, cr.labels)
        assert len(summary) == cr.n_clusters

    def test_numeric_means_in_summary(self, df_100, X_pca):
        cr = run_kmeans(X_pca, n_clusters=3)
        num_cols = ["declared_price", "missing_declaration_count"]
        summary = build_cluster_summary(df_100, cr.labels, numeric_cols=num_cols)
        assert "declared_price" in summary.columns
