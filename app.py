"""
app.py
------
PCA-Based Compression of Product-Listing Compliance Data
for E-Commerce Risk-Cluster Identification

SIH Problem ID: SIH26051
Executive Web Application & Machine Learning Dashboard

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.clustering import build_cluster_summary, run_dbscan, run_kmeans, sweep_k
from src.data_loader import generate_sample_data, get_sample_csv_bytes, load_csv
from src.data_validation import (
    clean_dataframe,
    coerce_numeric_columns,
    get_missing_value_heatmap_data,
    validate_dataframe,
)
from src.export_utils import (
    build_results_dataframe,
    export_cluster_summary,
    export_variance_table,
    load_model,
    model_exists,
    results_to_csv_bytes,
    save_model,
    validate_export_dataframe,
)
from src.feature_engineering import (
    DEFAULT_CAT_COLS,
    DEFAULT_NUM_COLS,
    DEFAULT_TEXT_COLS,
    build_feature_matrix,
    get_feature_statistics,
)
from src.pca_analysis import get_pca_scores_dataframe, get_variance_table, run_pca
from src.risk_scoring import get_flagged_listings, score_risk, summarise_cluster_risk
from src.visualization import (
    plot_category_distribution,
    plot_cluster_distribution,
    plot_cluster_risk_heatmap,
    plot_compliance_gauge,
    plot_cumulative_variance,
    plot_elbow,
    plot_missing_values,
    plot_numeric_histogram,
    plot_pca_scatter,
    plot_pca_scatter_3d,
    plot_risk_distribution,
    plot_scree,
)

# ---------------------------------------------------------------------------
# Logging & Page Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("app")

st.set_page_config(
    page_title="SIH26051 | Compliance Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Modern Executive Dark Glassmorphism CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    color: #F8FAFC;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0A0F1D 0%, #030712 90%);
}

/* Glassmorphism Card Container */
.glass-card {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.25);
    box-shadow: 0 15px 35px -5px rgba(2, 132, 199, 0.15);
}

/* KPI Metric Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 24px -6px rgba(0,0,0,0.4);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38BDF8, #818CF8);
}
.kpi-card.warning::before {
    background: linear-gradient(90deg, #F59E0B, #EF4444);
}
.kpi-card.success::before {
    background: linear-gradient(90deg, #10B981, #06B6D4);
}
.kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 6px;
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38BDF8;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 8px 0;
    letter-spacing: -0.03em;
}
.hero-desc {
    font-size: 0.95rem;
    color: #94A3B8;
    margin: 0;
    max-width: 820px;
    line-height: 1.5;
}

/* Status Pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-pill.online {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34D399;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 8px #10B981;
}

/* Custom Risk Badges */
.badge-low {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34D399;
    padding: 2px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
}
.badge-mod {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #FBBF24;
    padding: 2px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
}
.badge-high {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #F87171;
    padding: 2px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.8rem;
}

/* Styled Streamlit Controls */
div[data-testid="stSidebar"] {
    background: #090E1A !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    font-weight: 600;
    color: #94A3B8;
}
.stTabs [aria-selected="true"] {
    color: #38BDF8 !important;
    border-bottom: 2px solid #38BDF8 !important;
    background: rgba(56, 189, 248, 0.05);
}

/* Hide default streamlit header elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization & Auto-Pipeline Runner
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "df_raw": None,
    "df_processed": None,
    "feature_matrix": None,
    "pca_result": None,
    "cluster_result": None,
    "risk_df": None,
    "results_df": None,
    "cluster_summary_df": None,
    "cluster_risk_summary": None,
    "pca_scores_df": None,
    "pipeline_ran": False,
    "data_source_label": "",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_pipeline() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def run_full_pipeline(
    df: pd.DataFrame,
    n_clusters: int = 4,
    n_pca: int | None = None,
    var_target: float = 0.90,
) -> None:
    """Executes Feature Extraction -> PCA Reduction -> Clustering -> Risk Scoring."""
    try:
        # 1. Clean
        df_clean = clean_dataframe(df, drop_all_null_cols=True, drop_duplicate_rows=False)
        num_cols = [c for c in DEFAULT_NUM_COLS if c in df_clean.columns]
        df_clean = coerce_numeric_columns(df_clean, num_cols)
        st.session_state.df_processed = df_clean

        # 2. Feature engineering (TF-IDF + SVD, OHE, Scaler)
        fm = build_feature_matrix(df_clean)
        st.session_state.feature_matrix = fm

        # 3. PCA Analysis
        pca_result = run_pca(
            fm.X,
            n_components=n_pca,
            variance_threshold=var_target,
        )
        st.session_state.pca_result = pca_result
        pca_scores = get_pca_scores_dataframe(pca_result, df_clean)
        st.session_state.pca_scores_df = pca_scores

        # 4. Clustering (K-Means)
        cr = run_kmeans(pca_result.X_reduced, n_clusters=n_clusters)
        st.session_state.cluster_result = cr

        # 5. Risk Scoring
        risk_df = score_risk(df_clean)
        st.session_state.risk_df = risk_df

        # 6. Enriched Results
        results_df = build_results_dataframe(df_clean, cr.labels, risk_df, pca_scores)
        st.session_state.results_df = results_df

        # 7. Cluster Profiles & Aggregates
        cluster_summary = build_cluster_summary(
            df_clean, cr.labels,
            numeric_cols=["declared_price", "missing_declaration_count",
                          "declaration_completeness_pct",
                          "listing_metadata_completeness"],
        )
        st.session_state.cluster_summary_df = cluster_summary
        st.session_state.cluster_risk_summary = summarise_cluster_risk(risk_df, cr.labels)

        # 8. Model persistence
        save_model({
            "pca": pca_result.pca,
            "svd": fm.svd,
            "n_components": pca_result.n_components,
            "feature_names": fm.feature_names,
        })
        st.session_state.pipeline_ran = True

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        logger.exception("Pipeline execution failed")


# Auto-bootstrap with sample data so the dashboard is immediately rich & populated on first visit!
if st.session_state.df_raw is None:
    df_init = generate_sample_data(n=500, random_seed=42)
    st.session_state.df_raw = df_init
    st.session_state.data_source_label = "SYNTHETIC_DEMO"
    run_full_pipeline(df_init, n_clusters=4, var_target=0.90)


# ---------------------------------------------------------------------------
# Sidebar Navigation & Platform Identity
# ---------------------------------------------------------------------------
PAGES = [
    "📊 Executive Overview",
    "📂 Data Studio & Ingestion",
    "⚙️ Feature Engineering",
    "🔬 PCA Dimensionality Space",
    "🎯 Cluster Intelligence",
    "🔎 Listing Risk Explorer",
    "💾 Export & Model Registry",
]

with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="background: linear-gradient(135deg, #0284C7, #38BDF8); width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px;">🛡️</div>
            <div>
                <div style="font-weight:800; font-size:1.1rem; color:#FFFFFF; letter-spacing:-0.02em;">COMPLY-AI</div>
                <div style="font-size:0.7rem; color:#38BDF8; font-weight:700; letter-spacing:0.06em;">SIH PROBLEM ID: SIH26051</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation Menu", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.7); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:14px; margin-bottom:14px;">
        <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:700; margin-bottom:6px;">ENGINE STATUS</div>
        <div class="status-pill online"><div class="pulse-dot"></div> ACTIVE & TRAINED</div>
        <div style="font-size:0.75rem; color:#64748B; margin-top:8px;">Model: PCA + K-Means (k=4)</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "⚖️ **Regulatory Notice:** Machine learning signals provide estimated compliance risk indicators for human audit prioritization, not certified legal findings."
    )


# ---------------------------------------------------------------------------
# Global Header Hero Banner
# ---------------------------------------------------------------------------
df_active = st.session_state.df_raw
cr_active = st.session_state.cluster_result
rdf_active = st.session_state.risk_df
pca_active = st.session_state.pca_result
fm_active = st.session_state.feature_matrix

n_listings = len(df_active) if df_active is not None else 0
flagged_cnt = len(get_flagged_listings(rdf_active)) if rdf_active is not None else 0
flagged_pct = (flagged_cnt / max(n_listings, 1)) * 100
orig_feats = fm_active.X.shape[1] if fm_active is not None else 0
n_comp = pca_active.n_components if pca_active is not None else 0
pct_var = (pca_active.cumulative_variance[-1] * 100) if pca_active is not None else 0.0

st.markdown(f"""
<div class="hero-header">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div>
            <div class="hero-badge">Smart India Hackathon 2026 • Problem ID SIH26051</div>
            <h1 class="hero-title">Product Compliance & Risk Cluster Intelligence</h1>
            <p class="hero-desc">
                High-dimensional compression and unsupervised cluster discovery for e-commerce declaration compliance, audit prioritization, and risk pattern mitigation.
            </p>
        </div>
        <div style="text-align:right;">
            <div class="status-pill online"><div class="pulse-dot"></div> Live Engine Connected</div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:6px;">Source: <b>{st.session_state.data_source_label or 'SYNTHETIC'}</b></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Strip
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card success">
        <div class="kpi-title">Listings Ingested</div>
        <div class="kpi-value">{n_listings:,}</div>
        <div class="kpi-sub">Total product records loaded</div>
    </div>
    <div class="kpi-card warning">
        <div class="kpi-title">Flagged for Audit</div>
        <div class="kpi-value">{flagged_cnt:,}</div>
        <div class="kpi-sub">{flagged_pct:.1f}% moderate / higher risk</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Discovered Clusters</div>
        <div class="kpi-value">{cr_active.n_clusters if cr_active else 0}</div>
        <div class="kpi-sub">K-Means compliance cohorts</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Dimensionality Compression</div>
        <div class="kpi-value">{orig_feats} → {n_comp}</div>
        <div class="kpi-sub">{((1 - n_comp/max(orig_feats, 1))*100):.1f}% feature reduction</div>
    </div>
    <div class="kpi-card success">
        <div class="kpi-title">Explained Variance</div>
        <div class="kpi-value">{pct_var:.1f}%</div>
        <div class="kpi-sub">Information preserved via PCA</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ===========================================================================
if page == PAGES[0]:
    st.markdown("### 📊 Executive Compliance Dashboard")

    col_gauge, col_pie = st.columns([1, 1])
    with col_gauge:
        compliance_score = max(0.0, min(100.0, 100.0 - flagged_pct))
        st.plotly_chart(plot_compliance_gauge(compliance_score), use_container_width=True)

    with col_pie:
        if rdf_active is not None:
            st.plotly_chart(plot_risk_distribution(rdf_active), use_container_width=True)

    st.markdown("---")

    col_heat, col_scatter = st.columns([1, 1])
    with col_heat:
        if st.session_state.cluster_risk_summary is not None:
            st.plotly_chart(plot_cluster_risk_heatmap(st.session_state.cluster_risk_summary), use_container_width=True)
    with col_scatter:
        if pca_active is not None and cr_active is not None:
            st.plotly_chart(
                plot_pca_scatter(
                    pca_active.X_reduced,
                    labels=cr_active.labels,
                    risk_labels=rdf_active["risk_label"] if rdf_active is not None else None,
                    title="2D Principal Component Space (PC1 vs PC2)",
                ),
                use_container_width=True,
            )

    # Key Insights Box
    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#38BDF8; margin-top:0;">💡 Executive Audit Summary & Insights</h4>
        <ul style="color:#CBD5E1; line-height:1.7; margin-bottom:0;">
            <li><b>Automated Dimensionality Reduction:</b> Compresses high-dimensional declaration matrices, multi-language titles, and categorical attributes into orthogonal principal axes while retaining <b>90%+ total variance</b>.</li>
            <li><b>Risk Segregation:</b> Distinct cluster cohorts immediately segregate compliant mass-market items from listings with systemic missing certifications or severe price anomalies.</li>
            <li><b>Human-in-the-Loop Prioritization:</b> High-risk listings are mapped directly to transparent explainability tags (e.g. <i>Missing certifications, Title-description mismatch, Price anomaly</i>).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 2: DATA STUDIO & INGESTION
# ===========================================================================
elif page == PAGES[1]:
    st.markdown("### 📂 Data Studio & Compliance Ingestion")

    tab_upload, tab_synth, tab_quality = st.tabs(["📤 Upload Custom Dataset", "🔬 Synthetic Generator", "🛡️ Data Quality Diagnostic"])

    with tab_upload:
        st.markdown("Upload any enterprise product catalog CSV. The system automatically inspects data types, maps known compliance declarations, and identifies anomalies.")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Supported format: CSV with product metadata and declarations")

        if uploaded_file is not None:
            try:
                df_up = load_csv(uploaded_file)
                st.session_state.df_raw = df_up
                st.session_state.data_source_label = "USER_UPLOAD"
                run_full_pipeline(df_up)
                st.success(f"✅ Ingested {len(df_up):,} listings from uploaded file! Pipeline updated.")
                st.rerun()
            except ValueError as e:
                st.error(f"Upload error: {e}")

    with tab_synth:
        st.markdown("Generate realistic demonstration e-commerce compliance datasets with configurable volume, declaration completeness ratios, and anomalous distributions.")
        col_s1, col_s2, col_s3 = st.columns(3)
        row_count = col_s1.slider("Record Volume", 100, 2500, 500, step=100)
        seed_val = col_s2.number_input("Random Seed", value=42, step=1)
        k_val = col_s3.slider("Target Clusters", 2, 8, 4)

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            if st.button("🚀 Re-Generate & Retrain Pipeline", type="primary", use_container_width=True):
                with st.spinner("Generating and training..."):
                    df_new = generate_sample_data(n=row_count, random_seed=int(seed_val))
                    st.session_state.df_raw = df_new
                    st.session_state.data_source_label = "SYNTHETIC_DEMO"
                    run_full_pipeline(df_new, n_clusters=k_val)
                    st.success(f"✅ Generated {row_count} listings and retrained pipeline!")
                    st.rerun()
        with col_b2:
            st.download_button(
                "⬇️ Download Demonstration CSV",
                data=get_sample_csv_bytes(n=row_count, random_seed=int(seed_val)),
                file_name="compliance_demonstration_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_quality:
        if df_active is not None:
            report = validate_dataframe(df_active)
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Row Count", f"{report.row_count:,}")
            q2.metric("Column Count", report.column_count)
            q3.metric("Duplicate Rows", report.duplicate_row_count)
            q4.metric("Missing Data Points", report.total_missing)

            miss_df = get_missing_value_heatmap_data(df_active)
            st.plotly_chart(plot_missing_values(miss_df), use_container_width=True)

    if df_active is not None:
        st.markdown("#### 📋 Ingested Catalog Preview (Top 25 Records)")
        st.dataframe(df_active.head(25), use_container_width=True, height=350)


# ===========================================================================
# PAGE 3: FEATURE ENGINEERING
# ===========================================================================
elif page == PAGES[2]:
    st.markdown("### ⚙️ Multi-Modal Feature Extraction Pipeline")

    if fm_active is None:
        st.info("Pipeline not yet initialized.")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Standardized Numerical Features", fm_active.n_num_features)
        col_f2.metric("Text TF-IDF & SVD Components", fm_active.svd_n_components)
        col_f3.metric("Total Dense Transformed Columns", fm_active.X.shape[1])

        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#38BDF8; margin-top:0;">Pipeline Architecture & Transformation Strategy</h4>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:16px;">
                <div style="background:rgba(30,41,59,0.5); padding:16px; border-radius:10px;">
                    <b style="color:#F8FAFC;">1. Text TF-IDF + TruncatedSVD</b><br>
                    <span style="font-size:0.85rem; color:#94A3B8;">Product titles and descriptions are tokenized with sublinear term-frequency weighting, then reduced through TruncatedSVD to prevent sparse explosion.</span>
                </div>
                <div style="background:rgba(30,41,59,0.5); padding:16px; border-radius:10px;">
                    <b style="color:#F8FAFC;">2. Categorical One-Hot Encoding</b><br>
                    <span style="font-size:0.85rem; color:#94A3B8;">Categories, origin countries, and certifications are encoded with unknown category fallbacks and missing value handling.</span>
                </div>
                <div style="background:rgba(30,41,59,0.5); padding:16px; border-radius:10px;">
                    <b style="color:#F8FAFC;">3. Numerical Imputation & StandardScaler</b><br>
                    <span style="font-size:0.85rem; color:#94A3B8;">Compliance ratios, price deviations, and declaration flags are median-imputed and z-score normalized to achieve zero mean and unit variance.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            if "category" in df_active.columns:
                st.plotly_chart(plot_category_distribution(df_active), use_container_width=True)
        with col_right:
            numeric_opts = [c for c in fm_active.num_cols_used if c in df_active.columns]
            if numeric_opts:
                sel_num = st.selectbox("Inspect Empirical Distribution", numeric_opts)
                st.plotly_chart(plot_numeric_histogram(df_active, sel_num), use_container_width=True)

        with st.expander("🔬 View Feature Numerical Descriptive Statistics"):
            st.dataframe(get_feature_statistics(df_active, fm_active.num_cols_used), use_container_width=True)


# ===========================================================================
# PAGE 4: PCA DIMENSIONALITY SPACE
# ===========================================================================
elif page == PAGES[3]:
    st.markdown("### 🔬 Principal Component Analysis (PCA) Decomposition")

    if pca_active is None:
        st.info("Run the pipeline to inspect PCA decomposition.")
    else:
        tab_2d, tab_3d, tab_variance, tab_data = st.tabs([
            "🌐 2D Projection Space",
            "✨ 3D Cluster Topology",
            "📉 Scree & Variance Curves",
            "📋 Variance Statistics Table",
        ])

        with tab_2d:
            c1, c2 = st.columns(2)
            pc_x_choice = c1.selectbox("Horizontal Axis", pca_active.component_labels, index=0)
            pc_y_choice = c2.selectbox("Vertical Axis", pca_active.component_labels, index=min(1, len(pca_active.component_labels) - 1))

            idx_x = pca_active.component_labels.index(pc_x_choice)
            idx_y = pca_active.component_labels.index(pc_y_choice)

            st.plotly_chart(
                plot_pca_scatter(
                    pca_active.X_reduced,
                    labels=cr_active.labels if cr_active else None,
                    risk_labels=rdf_active["risk_label"] if rdf_active is not None else None,
                    pc_x=idx_x,
                    pc_y=idx_y,
                    title=f"PCA Latent Manifold Projection ({pc_x_choice} vs {pc_y_choice})",
                ),
                use_container_width=True,
            )

        with tab_3d:
            st.markdown("Interactive 3D visualization of the top 3 principal components. Click and drag to rotate, pan, and inspect cluster separations.")
            st.plotly_chart(
                plot_pca_scatter_3d(
                    pca_active.X_reduced,
                    labels=cr_active.labels if cr_active else None,
                    risk_labels=rdf_active["risk_label"] if rdf_active is not None else None,
                ),
                use_container_width=True,
            )

        with tab_variance:
            col_v1, col_v2 = st.columns([1, 1])
            with col_v1:
                st.plotly_chart(plot_scree(pca_active.explained_variance_ratio, pca_active.component_labels), use_container_width=True)
            with col_v2:
                st.plotly_chart(plot_cumulative_variance(pca_active.cumulative_variance, pca_active.component_labels), use_container_width=True)

        with tab_data:
            st.dataframe(get_variance_table(pca_active), use_container_width=True)


# ===========================================================================
# PAGE 5: CLUSTER INTELLIGENCE
# ===========================================================================
elif page == PAGES[4]:
    st.markdown("### 🎯 Unsupervised Compliance Risk Clustering")

    if cr_active is None:
        st.info("Run the pipeline to inspect clustering intelligence.")
    else:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Active Clusters", cr_active.n_clusters)
        col_m2.metric("Algorithm", cr_active.algorithm.upper())
        sil_display = f"{cr_active.silhouette:.4f}" if cr_active.silhouette is not None else "N/A"
        col_m3.metric("Silhouette Separation Score", sil_display)

        tab_pop, tab_elbow_view, tab_table_view = st.tabs([
            "📊 Cohort Populations",
            "📉 Elbow Optimization Curve",
            "📋 Cluster Attribute Signatures",
        ])

        with tab_pop:
            col_c1, col_c2 = st.columns([1, 1])
            with col_c1:
                st.plotly_chart(plot_cluster_distribution(cr_active.cluster_sizes), use_container_width=True)
            with col_c2:
                if st.session_state.cluster_risk_summary is not None:
                    st.plotly_chart(plot_cluster_risk_heatmap(st.session_state.cluster_risk_summary), use_container_width=True)

        with tab_elbow_view:
            st.markdown("Calculates the sum of squared distances (Inertia) across candidate $k$ values to find the optimal cluster elbow.")
            with st.spinner("Computing elbow sweep..."):
                sweep_df = sweep_k(pca_active.X_reduced)
            st.plotly_chart(plot_elbow(sweep_df), use_container_width=True)

        with tab_table_view:
            if st.session_state.cluster_summary_df is not None:
                st.dataframe(st.session_state.cluster_summary_df, use_container_width=True)


# ===========================================================================
# PAGE 6: LISTING RISK EXPLORER
# ===========================================================================
elif page == PAGES[5]:
    st.markdown("### 🔎 Listing Audit & Risk Explorer")

    results_df = st.session_state.results_df
    if results_df is None:
        st.info("Pipeline not executed yet.")
    else:
        # Search & Filters Strip
        f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
        search_query = f1.text_input("🔍 Search Listings", placeholder="Filter by Product ID or Title...")

        all_cats = ["All Categories"] + sorted(results_df["category"].dropna().unique().tolist()) if "category" in results_df.columns else ["All"]
        sel_cat = f2.selectbox("Category", all_cats)

        all_clusters = ["All Clusters"] + [f"Cluster {c}" for c in sorted(results_df["cluster_label"].dropna().unique())] if "cluster_label" in results_df.columns else ["All"]
        sel_cl = f3.selectbox("Cluster Cohort", all_clusters)

        all_risks = ["All Risk Tiers", "Higher estimated risk", "Moderate estimated risk", "Lower estimated risk"]
        sel_risk = f4.selectbox("Risk Level", all_risks)

        # Filtering logic
        df_filtered = results_df.copy()
        if search_query:
            mask = pd.Series(False, index=df_filtered.index)
            for col in ["title", "product_id"]:
                if col in df_filtered.columns:
                    mask |= df_filtered[col].astype(str).str.contains(search_query, case=False, na=False)
            df_filtered = df_filtered[mask]

        if sel_cat != "All Categories" and "category" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["category"] == sel_cat]

        if sel_cl != "All Clusters" and "cluster_label" in df_filtered.columns:
            cl_num = int(sel_cl.replace("Cluster ", ""))
            df_filtered = df_filtered[df_filtered["cluster_label"] == cl_num]

        if sel_risk != "All Risk Tiers" and "risk_label" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["risk_label"] == sel_risk]

        st.caption(f"Showing **{len(df_filtered):,}** matching product listings out of {len(results_df):,} total.")

        # Show Table
        primary_cols = [c for c in [
            "product_id", "category", "title", "declared_price",
            "cluster_label", "risk_label", "risk_score", "risk_reasons",
            "declaration_completeness_pct", "missing_declaration_count",
        ] if c in df_filtered.columns]

        st.dataframe(df_filtered[primary_cols], use_container_width=True, height=450)

        if len(df_filtered) > 0:
            with st.expander("🔬 Deep Audit Inspector (Single Listing Full Profile)"):
                row_idx = st.number_input("Select Record Index", 0, len(df_filtered) - 1, 0)
                selected_row = df_filtered.iloc[row_idx]
                
                col_d1, col_d2 = st.columns(2)
                row_items = list(selected_row.items())
                half = len(row_items) // 2
                with col_d1:
                    for k, v in row_items[:half]:
                        st.markdown(f"**{k}:** `{v}`")
                with col_d2:
                    for k, v in row_items[half:]:
                        st.markdown(f"**{k}:** `{v}`")


# ===========================================================================
# PAGE 7: EXPORT & MODEL REGISTRY
# ===========================================================================
elif page == PAGES[6]:
    st.markdown("### 💾 Export Outputs & Model Registry")

    results_df = st.session_state.results_df

    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#38BDF8; margin-top:0;">Download Enriched Compliance Records & Analysis Artifacts</h4>
        <p style="color:#94A3B8; font-size:0.9rem;">
            Export processed datasets complete with assigned cluster IDs, estimated risk scores, and granular reason codes for regulatory audit logs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if results_df is not None:
        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            st.download_button(
                "⬇️ Enriched Compliance Dataset (CSV)",
                data=results_to_csv_bytes(results_df),
                file_name="compliance_audit_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_exp2:
            if pca_active is not None:
                st.download_button(
                    "⬇️ PCA Variance Table (CSV)",
                    data=export_variance_table(get_variance_table(pca_active)),
                    file_name="pca_variance_table.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with c_exp3:
            if st.session_state.cluster_summary_df is not None:
                st.download_button(
                    "⬇️ Cluster Signatures Summary (CSV)",
                    data=export_cluster_summary(st.session_state.cluster_summary_df),
                    file_name="cluster_profiles.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("---")
    st.markdown("#### 🤖 Saved Model Artifact Registry")
    if model_exists("compliance_model.joblib"):
        st.success("✅ Serialized Model Artifact Located: `models/compliance_model.joblib`")
        if st.button("📂 Load and Inspect Joblib Metadata"):
            try:
                arts = load_model("compliance_model.joblib")
                st.json({k: str(v) for k, v in arts.items() if k not in ("pca", "svd")})
            except Exception as e:
                st.error(f"Error loading model: {e}")
    else:
        st.info("No saved model found on disk yet.")
