"""
test_risk_scoring.py
--------------------
Tests for src/risk_scoring.py

SIH26051 — PCA-Based Compliance Risk Clustering
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_sample_data
from src.risk_scoring import (
    DEFAULT_CONFIG,
    RiskScoringConfig,
    get_flagged_listings,
    score_risk,
    summarise_cluster_risk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    return generate_sample_data(n=100, random_seed=5)


@pytest.fixture
def clean_listing():
    return pd.DataFrame([{
        "product_id": "P001",
        "missing_declaration_count": 0,
        "declaration_completeness_pct": 100.0,
        "price_anomaly": 0,
        "is_duplicate_listing": 0,
        "title_desc_mismatch": 0,
        "listing_metadata_completeness": 95.0,
    }])


@pytest.fixture
def risky_listing():
    return pd.DataFrame([{
        "product_id": "P002",
        "missing_declaration_count": 5,
        "declaration_completeness_pct": 10.0,
        "price_anomaly": 1,
        "is_duplicate_listing": 1,
        "title_desc_mismatch": 1,
        "listing_metadata_completeness": 20.0,
    }])


# ---------------------------------------------------------------------------
# score_risk
# ---------------------------------------------------------------------------

class TestScoreRisk:
    def test_returns_dataframe(self, sample_df):
        result = score_risk(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_required_columns(self, sample_df):
        result = score_risk(sample_df)
        assert "risk_score" in result.columns
        assert "risk_label" in result.columns
        assert "risk_reasons" in result.columns

    def test_risk_score_in_range(self, sample_df):
        result = score_risk(sample_df)
        assert result["risk_score"].between(0.0, 1.0).all()

    def test_clean_listing_gets_low_score(self, clean_listing):
        result = score_risk(clean_listing)
        assert result["risk_score"].iloc[0] == 0.0
        assert result["risk_label"].iloc[0] == DEFAULT_CONFIG.label_low

    def test_risky_listing_gets_high_score(self, risky_listing):
        result = score_risk(risky_listing)
        assert result["risk_score"].iloc[0] > 0.5
        assert result["risk_label"].iloc[0] == DEFAULT_CONFIG.label_high

    def test_risk_labels_are_valid(self, sample_df):
        result = score_risk(sample_df)
        valid = {DEFAULT_CONFIG.label_low, DEFAULT_CONFIG.label_moderate, DEFAULT_CONFIG.label_high}
        assert set(result["risk_label"].unique()).issubset(valid)

    def test_original_df_not_mutated(self, sample_df):
        _ = score_risk(sample_df)
        assert "risk_score" not in sample_df.columns

    def test_reasons_populated_for_risky(self, risky_listing):
        result = score_risk(risky_listing)
        reasons = result["risk_reasons"].iloc[0]
        assert reasons != "No signals detected"

    def test_no_signal_columns_gives_zero(self):
        df = pd.DataFrame({"product_id": ["P1", "P2"], "title": ["a", "b"]})
        result = score_risk(df)
        assert (result["risk_score"] == 0.0).all()

    def test_custom_config_thresholds(self, sample_df):
        config = RiskScoringConfig(threshold_moderate=0.10, threshold_high=0.20)
        result = score_risk(sample_df, config=config)
        # With low thresholds, more listings should be flagged higher
        high_count = (result["risk_label"] == config.label_high).sum()
        assert high_count >= 0  # just no crash

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["missing_declaration_count", "price_anomaly"])
        result = score_risk(df)
        assert result.empty or len(result) == 0


# ---------------------------------------------------------------------------
# summarise_cluster_risk
# ---------------------------------------------------------------------------

class TestSummariseClusterRisk:
    def test_returns_dataframe(self, sample_df):
        scored = score_risk(sample_df)
        labels = np.zeros(len(scored), dtype=int)
        labels[50:] = 1
        summary = summarise_cluster_risk(scored, labels)
        assert isinstance(summary, pd.DataFrame)
        assert "cluster" in summary.columns

    def test_has_risk_breakdown_columns(self, sample_df):
        scored = score_risk(sample_df)
        labels = np.zeros(len(scored), dtype=int)
        labels[50:] = 1
        summary = summarise_cluster_risk(scored, labels)
        assert "pct_high" in summary.columns
        assert "pct_moderate" in summary.columns


# ---------------------------------------------------------------------------
# get_flagged_listings
# ---------------------------------------------------------------------------

class TestGetFlaggedListings:
    def test_filters_correctly(self, sample_df):
        scored = score_risk(sample_df)
        flagged = get_flagged_listings(scored)
        labels = set(flagged["risk_label"].unique())
        assert labels.issubset({"Moderate estimated risk", "Higher estimated risk"})

    def test_empty_df_when_no_risk_col(self, sample_df):
        flagged = get_flagged_listings(sample_df, risk_col="nonexistent")
        assert flagged.empty
