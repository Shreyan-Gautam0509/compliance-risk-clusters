"""
app.py
------
PCA-Based Compression of Product-Listing Compliance Data
for E-Commerce Risk-Cluster Identification

SIH Problem ID: SIH26051

Streamlit multi-page dashboard with seven sections:
  A. Home Dashboard
  B. Dataset Management
  C. Feature Engineering
  D. PCA Analysis
  E. Risk Clustering
  F. Listing Explorer
  G. Export & Model Management

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
    plot_cumulative_variance,
    plot_elbow,
    plot_missing_values,
    plot_numeric_histogram,
    plot_pca_scatter,
    plot_risk_distribution,
    plot_scree,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SIH26051 — Compliance Risk Clustering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
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


# ---------------------------------------------------------------------------
# Full pipeline runner (module-level so every page can call it)
# ---------------------------------------------------------------------------

def run_full_pipeline(
    df: pd.DataFrame,
    n_clusters: int,
    n_pca: int | None,
    var_target: float,
) -> None:
    """Run Feature Engineering → PCA → Clustering → Risk Scoring and store results."""
    progress = st.progress(0, text="Starting pipeline…")
    try:
        # 1. Clean
        progress.progress(10, text="Cleaning data…")
        df_clean = clean_dataframe(df, drop_all_null_cols=True, drop_duplicate_rows=False)
        num_cols = [c for c in DEFAULT_NUM_COLS if c in df_clean.columns]
        df_clean = coerce_numeric_columns(df_clean, num_cols)
        st.session_state.df_processed = df_clean

        # 2. Feature engineering
        progress.progress(25, text="Engineering features…")
        fm = build_feature_matrix(df_clean)
        st.session_state.feature_matrix = fm

        # 3. PCA
        progress.progress(50, text="Running PCA…")
        pca_result = run_pca(
            fm.X,
            n_components=n_pca,
            variance_threshold=var_target,
        )
        st.session_state.pca_result = pca_result
        pca_scores = get_pca_scores_dataframe(pca_result, df_clean)
        st.session_state.pca_scores_df = pca_scores

        # 4. Clustering
        progress.progress(70, text="Clustering…")
        cr = run_kmeans(pca_result.X_reduced, n_clusters=n_clusters)
        st.session_state.cluster_result = cr

        # 5. Risk scoring
        progress.progress(85, text="Scoring risk…")
        risk_df = score_risk(df_clean)
        st.session_state.risk_df = risk_df

        # 6. Build enriched results
        progress.progress(92, text="Building results…")
        results_df = build_results_dataframe(df_clean, cr.labels, risk_df, pca_scores)
        st.session_state.results_df = results_df

        # 7. Summaries
        cluster_summary = build_cluster_summary(
            df_clean, cr.labels,
            numeric_cols=["declared_price", "missing_declaration_count",
                          "declaration_completeness_pct",
                          "listing_metadata_completeness"],
        )
        st.session_state.cluster_summary_df = cluster_summary
        st.session_state.cluster_risk_summary = summarise_cluster_risk(risk_df, cr.labels)

        # 8. Save model artefacts
        save_model({
            "pca": pca_result.pca,
            "svd": fm.svd,
            "n_components": pca_result.n_components,
            "feature_names": fm.feature_names,
        })

        st.session_state.pipeline_ran = True
        progress.progress(100, text="✅ Pipeline complete!")
        st.success(
            f"Pipeline complete! Reduced to **{pca_result.n_components}** PCA components "
            f"({pca_result.cumulative_variance[-1]*100:.1f}% variance explained). "
            f"**{cr.n_clusters}** clusters found."
        )
        for w in pca_result.warnings + cr.warnings:
            st.warning(w)

    except Exception as exc:
        progress.empty()
        st.error(f"❌ Pipeline failed: {exc}")
        logger.exception("Pipeline error")


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "🏠 Home Dashboard",
    "📂 Dataset Management",
    "🔧 Feature Engineering",
    "📊 PCA Analysis",
    "🎯 Risk Clustering",
    "🔎 Listing Explorer",
    "💾 Export & Model Management",
]

with st.sidebar:
    st.markdown("## 🔍 SIH26051")
    st.markdown("**Compliance Risk Clustering**")
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption(
        "⚠️ Risk labels are **estimated** signals for human review. "
        "They do NOT constitute verified legal findings."
    )


# ===========================================================================
# PAGE A — Home Dashboard
# ===========================================================================
if page == PAGES[0]:
    st.title("🔍 PCA-Based Compliance Risk Clustering")
    st.markdown(
        "**SIH Problem ID: SIH26051** — Smart India Hackathon 2026 | "
        "E-Commerce Product-Listing Compliance Analysis"
    )

    col_desc, col_flow = st.columns([3, 1])

    with col_desc:
        st.markdown("""
### Project Overview

This application helps e-commerce platform operators identify product listings
that may require compliance review. It:

1. **Extracts features** from product-listing and declaration data.
2. **Reduces dimensionality** using Principal Component Analysis (PCA) to
   find the most informative compliance patterns.
3. **Clusters** listings into groups with similar compliance profiles.
4. **Scores** each listing using transparent, configurable rules that flag
   signals such as missing declarations, unusual prices, or duplicate entries.

> ⚠️ **Disclaimer:** This tool provides *estimated* risk signals based on
> available data fields. It does not detect or confirm legal violations.
> All flags must be reviewed by a qualified compliance officer.
        """)

    with col_flow:
        st.markdown("### Workflow")
        st.markdown("""
```
📤 Upload / Generate
        ↓
🔧 Feature Extraction
        ↓
📊 PCA Reduction
        ↓
🎯 Clustering
        ↓
⚠️  Risk Scoring
        ↓
💾 Export Results
```
        """)

    st.markdown("---")

    # KPI metrics
    df_kpi   = st.session_state.df_raw
    cr_kpi   = st.session_state.cluster_result
    rdf_kpi  = st.session_state.risk_df
    pca_kpi  = st.session_state.pca_result
    fm_kpi   = st.session_state.feature_matrix

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📋 Listings Loaded",
              f"{len(df_kpi):,}" if df_kpi is not None else "—")
    m2.metric("🎯 Clusters",
              str(cr_kpi.n_clusters) if cr_kpi is not None else "—")
    m3.metric("⚠️ Flagged for Review",
              f"{len(get_flagged_listings(rdf_kpi)):,}" if rdf_kpi is not None else "—")
    m4.metric("📐 Feature Columns",
              str(fm_kpi.X.shape[1]) if fm_kpi is not None else "—")
    m5.metric("🔻 PCA Components",
              str(pca_kpi.n_components) if pca_kpi is not None else "—")

    if st.session_state.pipeline_ran and pca_kpi is not None:
        reduction_pct = (1 - pca_kpi.n_components / max(fm_kpi.X.shape[1], 1)) * 100
        st.success(
            f"✅ Pipeline complete — dimensionality reduced by "
            f"**{reduction_pct:.1f}%** "
            f"({fm_kpi.X.shape[1]} → {pca_kpi.n_components} features), "
            f"explaining **{pca_kpi.cumulative_variance[-1]*100:.1f}%** of variance."
        )
    else:
        st.info(
            "ℹ️ No results yet. Go to **📂 Dataset Management** to load data "
            "and run the pipeline."
        )


# ===========================================================================
# PAGE B — Dataset Management
# ===========================================================================
elif page == PAGES[1]:
    st.title("📂 Dataset Management")

    tab_gen, tab_upload = st.tabs(
        ["🔬 Generate Sample Data", "📤 Upload Your CSV"]
    )

    # ---- Generate tab ----
    with tab_gen:
        st.markdown("""
Generate a **synthetic demonstration dataset** of product listings.

> 🧪 This data is entirely artificial. It does NOT represent real products,
> brands, or companies, and is provided solely for demonstration purposes.
        """)
        col_n, col_s = st.columns(2)
        n_rows   = col_n.slider("Number of rows", 50, 2000, 500, step=50)
        rng_seed = col_s.number_input("Random seed", value=42, step=1)

        col_dl, col_gen = st.columns(2)
        with col_dl:
            csv_bytes = get_sample_csv_bytes(n=n_rows, random_seed=int(rng_seed))
            st.download_button(
                "⬇️ Download Sample CSV",
                data=csv_bytes,
                file_name="sample_compliance_data.csv",
                mime="text/csv",
            )
        with col_gen:
            if st.button("🚀 Generate & Load into App", type="primary"):
                with st.spinner("Generating…"):
                    reset_pipeline()
                    df_gen = generate_sample_data(n=n_rows, random_seed=int(rng_seed))
                    st.session_state.df_raw = df_gen
                    st.session_state.data_source_label = "SYNTHETIC_DEMO"
                st.success(f"✅ Generated {len(df_gen):,} rows.")
                st.rerun()

    # ---- Upload tab ----
    with tab_upload:
        st.markdown("Upload your own CSV. Columns are auto-mapped to known compliance features.")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df_up = load_csv(uploaded_file)
                reset_pipeline()
                st.session_state.df_raw = df_up
                st.session_state.data_source_label = "USER_UPLOAD"
                st.success(
                    f"✅ Loaded {len(df_up):,} rows, {len(df_up.columns)} columns."
                )
            except ValueError as exc:
                st.error(f"❌ {exc}")

    st.markdown("---")

    # ---- Dataset preview + quality report ----
    df_cur = st.session_state.df_raw
    if df_cur is not None:
        src_label = st.session_state.data_source_label
        if src_label == "SYNTHETIC_DEMO":
            st.info(
                "🧪 Showing **SYNTHETIC DEMONSTRATION** data — "
                "not real product records."
            )

        st.subheader("📋 Data Preview")
        st.dataframe(df_cur.head(20), use_container_width=True)
        st.caption(
            f"Shape: {df_cur.shape[0]:,} rows × {df_cur.shape[1]} columns"
        )

        # Quality report
        st.subheader("🔍 Data Quality Report")
        report = validate_dataframe(df_cur)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total rows",       f"{report.row_count:,}")
        q2.metric("Total columns",    report.column_count)
        q3.metric("Duplicate rows",   report.duplicate_row_count)
        q4.metric("Missing values",   report.total_missing)

        for w in report.warnings:
            st.warning(w)
        for e in report.errors:
            st.error(e)

        miss_df = get_missing_value_heatmap_data(df_cur)
        if not miss_df.empty:
            st.plotly_chart(
                plot_missing_values(miss_df), use_container_width=True
            )

        with st.expander("📐 Column Summary"):
            dtype_df = pd.DataFrame({
                "Column":   df_cur.columns,
                "dtype":    [str(df_cur[c].dtype)        for c in df_cur.columns],
                "Non-null": [int(df_cur[c].notna().sum()) for c in df_cur.columns],
                "Unique":   [int(df_cur[c].nunique())     for c in df_cur.columns],
            })
            st.dataframe(dtype_df, use_container_width=True)

        # Pipeline controls
        st.markdown("---")
        st.subheader("▶️ Run Full Pipeline")

        p1, p2, p3 = st.columns(3)
        n_clusters_in = p1.number_input(
            "K-Means clusters", min_value=2, max_value=20, value=4, step=1
        )
        n_pca_in = p2.number_input(
            "PCA components (0 = auto)", min_value=0, max_value=50, value=0, step=1
        )
        var_target_in = p3.slider(
            "Target variance % (when components=0)", 70, 99, 90, step=1
        )

        if st.button(
            "🚀 Run Pipeline  (Feature Engineering → PCA → Cluster → Risk Score)",
            type="primary",
        ):
            run_full_pipeline(
                df=df_cur,
                n_clusters=int(n_clusters_in),
                n_pca=int(n_pca_in) if n_pca_in > 0 else None,
                var_target=var_target_in / 100.0,
            )
    else:
        st.info(
            "ℹ️ No data loaded yet. Generate sample data or upload a CSV above."
        )


# ===========================================================================
# PAGE C — Feature Engineering
# ===========================================================================
elif page == PAGES[2]:
    st.title("🔧 Feature Engineering")

    fm_page = st.session_state.feature_matrix
    df_page = st.session_state.df_processed or st.session_state.df_raw

    if fm_page is None:
        st.info("ℹ️ Run the pipeline from **📂 Dataset Management** first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Numerical features",         fm_page.n_num_features)
        c2.metric("Text-derived (SVD) features", fm_page.svd_n_components)
        c3.metric("Total feature-matrix columns", fm_page.X.shape[1])

        st.markdown(f"""
### Pipeline Summary

| Step | Details |
|------|---------|
| Text columns (TF-IDF) | `{', '.join(fm_page.text_cols_used) or 'None'}` |
| Categorical columns (OHE) | `{', '.join(fm_page.cat_cols_used) or 'None'}` |
| Numerical columns (StandardScaler) | {len(fm_page.num_cols_used)} columns |
| TF-IDF raw vocabulary features | {fm_page.n_text_features} |
| TruncatedSVD reduction components | {fm_page.svd_n_components} |
| Final dense feature matrix | {fm_page.X.shape[0]} × {fm_page.X.shape[1]} |
        """)

        if df_page is not None:
            st.subheader("📊 Numerical Feature Statistics")
            stats_df = get_feature_statistics(df_page, fm_page.num_cols_used)
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True)

            st.subheader("📈 Feature Distributions")
            num_options = [c for c in fm_page.num_cols_used if c in df_page.columns]
            if num_options:
                chosen = st.selectbox(
                    "Select a numerical feature to visualise", num_options
                )
                st.plotly_chart(
                    plot_numeric_histogram(df_page, chosen),
                    use_container_width=True,
                )

            if "category" in df_page.columns:
                st.subheader("📦 Product Category Distribution")
                st.plotly_chart(
                    plot_category_distribution(df_page, "category"),
                    use_container_width=True,
                )


# ===========================================================================
# PAGE D — PCA Analysis
# ===========================================================================
elif page == PAGES[3]:
    st.title("📊 PCA Analysis")

    pca_page = st.session_state.pca_result
    fm_page  = st.session_state.feature_matrix

    if pca_page is None:
        st.info("ℹ️ Run the pipeline from **📂 Dataset Management** first.")
    else:
        for w in pca_page.warnings:
            st.warning(w)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Original features",    fm_page.X.shape[1])
        d2.metric("PCA components",        pca_page.n_components)
        pct_red = (1 - pca_page.n_components / max(fm_page.X.shape[1], 1)) * 100
        d3.metric("Dimensionality reduction", f"{pct_red:.1f}%")
        d4.metric("Variance explained",
                  f"{pca_page.cumulative_variance[-1]*100:.1f}%")

        st.markdown("---")
        t_scree, t_cumvar, t_table, t_scatter = st.tabs([
            "📉 Scree Plot",
            "📈 Cumulative Variance",
            "📋 Variance Table",
            "🔵 PCA Scatter",
        ])

        with t_scree:
            st.plotly_chart(
                plot_scree(
                    pca_page.explained_variance_ratio,
                    pca_page.component_labels,
                ),
                use_container_width=True,
            )

        with t_cumvar:
            st.plotly_chart(
                plot_cumulative_variance(
                    pca_page.cumulative_variance,
                    pca_page.component_labels,
                ),
                use_container_width=True,
            )

        with t_table:
            var_tbl = get_variance_table(pca_page)
            st.dataframe(var_tbl, use_container_width=True)

        with t_scatter:
            cr_page  = st.session_state.cluster_result
            rdf_page = st.session_state.risk_df
            hover_df = None
            df_proc  = st.session_state.df_processed
            if df_proc is not None and "product_id" in df_proc.columns:
                hover_df = df_proc[["product_id"]].copy()

            sel_x, sel_y = st.columns(2)
            pc_x_lbl = sel_x.selectbox(
                "X-axis", pca_page.component_labels, index=0
            )
            pc_y_lbl = sel_y.selectbox(
                "Y-axis",
                pca_page.component_labels,
                index=min(1, len(pca_page.component_labels) - 1),
            )
            pc_x_idx = pca_page.component_labels.index(pc_x_lbl)
            pc_y_idx = pca_page.component_labels.index(pc_y_lbl)

            st.plotly_chart(
                plot_pca_scatter(
                    pca_page.X_reduced,
                    labels=(cr_page.labels if cr_page else None),
                    risk_labels=(
                        rdf_page["risk_label"] if rdf_page is not None else None
                    ),
                    pc_x=pc_x_idx,
                    pc_y=pc_y_idx,
                    hover_data=hover_df,
                    title=f"PCA Scatter — {pc_x_lbl} vs {pc_y_lbl}",
                ),
                use_container_width=True,
            )


# ===========================================================================
# PAGE E — Risk Clustering
# ===========================================================================
elif page == PAGES[4]:
    st.title("🎯 Risk Clustering")

    cr_page  = st.session_state.cluster_result
    rdf_page = st.session_state.risk_df
    pca_page = st.session_state.pca_result
    cs_page  = st.session_state.cluster_summary_df
    crs_page = st.session_state.cluster_risk_summary

    if cr_page is None:
        st.info("ℹ️ Run the pipeline from **📂 Dataset Management** first.")
    else:
        for w in cr_page.warnings:
            st.warning(w)

        e1, e2, e3 = st.columns(3)
        e1.metric("Clusters",   cr_page.n_clusters)
        e2.metric("Algorithm",  cr_page.algorithm.upper())
        e3.metric(
            "Silhouette Score",
            f"{cr_page.silhouette:.4f}" if cr_page.silhouette is not None else "N/A",
        )
        if cr_page.silhouette is not None:
            quality = (
                "Good"  if cr_page.silhouette > 0.50 else
                "Fair"  if cr_page.silhouette > 0.25 else "Weak"
            )
            st.caption(
                f"Silhouette interpretation: **{quality}** "
                "(range −1 to +1; higher = better-separated clusters)"
            )

        st.markdown("---")
        t_dist, t_elbow, t_comp, t_risk = st.tabs([
            "📊 Cluster Distribution",
            "📉 Elbow Curve",
            "🔬 Cluster Comparison",
            "⚠️ Risk Distribution",
        ])

        with t_dist:
            st.plotly_chart(
                plot_cluster_distribution(cr_page.cluster_sizes),
                use_container_width=True,
            )
            if pca_page is not None:
                st.plotly_chart(
                    plot_pca_scatter(
                        pca_page.X_reduced,
                        labels=cr_page.labels,
                        title="Cluster Scatter (PC1 vs PC2)",
                    ),
                    use_container_width=True,
                )

        with t_elbow:
            st.markdown(
                "The elbow curve helps choose an optimal k. "
                "Look for the point where inertia drops less sharply."
            )
            if pca_page is not None:
                with st.spinner("Computing elbow sweep…"):
                    sweep_df = sweep_k(pca_page.X_reduced)
                st.plotly_chart(plot_elbow(sweep_df), use_container_width=True)
            else:
                st.info("PCA result not available.")

        with t_comp:
            if cs_page is not None:
                st.dataframe(cs_page, use_container_width=True)
                if crs_page is not None:
                    st.plotly_chart(
                        plot_cluster_risk_heatmap(crs_page),
                        use_container_width=True,
                    )
            else:
                st.info("No cluster comparison data.")

        with t_risk:
            if rdf_page is not None:
                st.plotly_chart(
                    plot_risk_distribution(rdf_page),
                    use_container_width=True,
                )
                flagged = get_flagged_listings(rdf_page)
                st.metric("⚠️ Listings flagged for review", len(flagged))
                if not flagged.empty:
                    show_cols = [
                        c for c in [
                            "product_id", "category", "risk_label",
                            "risk_score", "risk_reasons",
                        ] if c in flagged.columns
                    ]
                    st.dataframe(
                        flagged[show_cols].head(50), use_container_width=True
                    )
            else:
                st.info("No risk data available.")


# ===========================================================================
# PAGE F — Listing Explorer
# ===========================================================================
elif page == PAGES[5]:
    st.title("🔎 Listing Explorer")

    rdf_exp = st.session_state.results_df

    if rdf_exp is None:
        st.info("ℹ️ Run the pipeline from **📂 Dataset Management** first.")
    else:
        st.markdown(
            "Browse, search, and filter individual product listings. "
            "Risk labels are **estimates** — not verified compliance decisions."
        )

        # Filters
        f1, f2, f3 = st.columns(3)

        cat_options = ["All"]
        if "category" in rdf_exp.columns:
            cat_options += sorted(rdf_exp["category"].dropna().unique().tolist())
        selected_cat = f1.selectbox("Category", cat_options)

        cluster_options = ["All"]
        if "cluster_label" in rdf_exp.columns:
            cluster_options += sorted(
                [str(x) for x in rdf_exp["cluster_label"].dropna().unique()]
            )
        selected_cluster = f2.selectbox("Cluster", cluster_options)

        risk_options = [
            "All",
            "Higher estimated risk",
            "Moderate estimated risk",
            "Lower estimated risk",
        ]
        selected_risk = f3.selectbox("Risk Level", risk_options)

        search_term = st.text_input("🔍 Search title or product ID", "")

        # Apply filters
        filtered = rdf_exp.copy()
        if selected_cat != "All" and "category" in filtered.columns:
            filtered = filtered[filtered["category"] == selected_cat]
        if selected_cluster != "All" and "cluster_label" in filtered.columns:
            filtered = filtered[
                filtered["cluster_label"].astype(str) == selected_cluster
            ]
        if selected_risk != "All" and "risk_label" in filtered.columns:
            filtered = filtered[filtered["risk_label"] == selected_risk]
        if search_term:
            mask = pd.Series(False, index=filtered.index)
            for col in ["title", "product_id"]:
                if col in filtered.columns:
                    mask |= (
                        filtered[col]
                        .astype(str)
                        .str.contains(search_term, case=False, na=False)
                    )
            filtered = filtered[mask]

        st.markdown(f"**{len(filtered):,}** listings match your filters.")

        display_cols = [
            c for c in [
                "product_id", "category", "title", "declared_price",
                "cluster_label", "risk_label", "risk_score", "risk_reasons",
                "declaration_completeness_pct", "missing_declaration_count",
            ] if c in filtered.columns
        ]
        st.dataframe(filtered[display_cols], use_container_width=True)

        if len(filtered) > 0:
            with st.expander("🔬 View Full Details for One Listing"):
                idx_max = len(filtered) - 1
                row_idx = st.number_input("Row index", 0, idx_max, 0)
                row = filtered.iloc[row_idx]
                col_a, col_b = st.columns(2)
                items = list(row.items())
                mid = len(items) // 2
                for k, v in items[:mid]:
                    col_a.markdown(f"**{k}:** {v}")
                for k, v in items[mid:]:
                    col_b.markdown(f"**{k}:** {v}")


# ===========================================================================
# PAGE G — Export & Model Management
# ===========================================================================
elif page == PAGES[6]:
    st.title("💾 Export & Model Management")

    rdf_exp  = st.session_state.results_df
    pca_exp  = st.session_state.pca_result
    cs_exp   = st.session_state.cluster_summary_df

    # ---- Download section ----
    st.subheader("📤 Download Results")

    if rdf_exp is None:
        st.info("ℹ️ No results to export — run the pipeline first.")
    else:
        missing_export_cols = validate_export_dataframe(rdf_exp)
        if missing_export_cols:
            st.warning(f"Results are missing columns: {missing_export_cols}")

        g1, g2, g3 = st.columns(3)

        with g1:
            st.download_button(
                "⬇️ Enriched Results (CSV)",
                data=results_to_csv_bytes(rdf_exp),
                file_name="compliance_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with g2:
            if pca_exp is not None:
                st.download_button(
                    "⬇️ PCA Variance Table (CSV)",
                    data=export_variance_table(get_variance_table(pca_exp)),
                    file_name="pca_variance.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with g3:
            if cs_exp is not None:
                st.download_button(
                    "⬇️ Cluster Summary (CSV)",
                    data=export_cluster_summary(cs_exp),
                    file_name="cluster_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # Preview
        st.subheader("👀 Results Preview (first 20 rows)")
        preview_cols = [
            c for c in [
                "product_id", "cluster_label",
                "risk_score", "risk_label", "risk_reasons",
            ] if c in rdf_exp.columns
        ]
        st.dataframe(rdf_exp[preview_cols].head(20), use_container_width=True)

    st.markdown("---")

    # ---- Model management ----
    st.subheader("🤖 Saved Model")
    model_file = "compliance_model.joblib"

    if model_exists(model_file):
        st.success(f"✅ Model found: `models/{model_file}`")
        if st.button("📂 Load & Inspect Model Artefacts"):
            try:
                artefacts = load_model(model_file)
                st.json(
                    {k: str(v) for k, v in artefacts.items()
                     if k not in ("pca", "svd")}
                )
                st.info(
                    f"PCA components in saved model: "
                    f"{artefacts.get('n_components', '?')}"
                )
            except Exception as exc:
                st.error(f"❌ {exc}")
    else:
        st.info(
            "No saved model found. Run the pipeline at least once to create one."
        )

    st.markdown("---")
    st.caption(
        "**Disclaimer:** All exported risk labels are ESTIMATED signals derived "
        "from rule-based scoring. They must be reviewed by qualified compliance "
        "personnel and must not be treated as legal determinations."
    )
