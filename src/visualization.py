"""
visualization.py
----------------
Plotly-based chart builders for the SIH26051 dashboard.

Modern corporate dark glassmorphism theme designed for high-impact
presentation, executive compliance reporting, and technical depth.

SIH26051 — PCA-Based Compliance Risk Clustering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Theme & Palette Constants
# ---------------------------------------------------------------------------

THEME_BG = "rgba(15, 23, 42, 0.5)"
PAPER_BG = "rgba(0, 0, 0, 0)"
GRID_COLOR = "rgba(148, 163, 184, 0.12)"
TEXT_COLOR = "#F1F5F9"
MUTED_TEXT = "#94A3B8"

RISK_COLOURS = {
    "Lower estimated risk":    "#10B981",  # Vibrant Emerald
    "Moderate estimated risk": "#F59E0B",  # Vibrant Amber
    "Higher estimated risk":   "#EF4444",  # Crimson Coral
}

CLUSTER_PALETTE = [
    "#38BDF8",  # Sky Cyan
    "#818CF8",  # Indigo
    "#C084FC",  # Purple
    "#34D399",  # Mint
    "#FB923C",  # Orange
    "#F472B6",  # Pink
    "#2DD4BF",  # Teal
    "#A3E635",  # Lime
]


def _apply_dark_theme(fig: go.Figure, height: int = 440) -> go.Figure:
    """Helper to apply uniform executive dark styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=THEME_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                  color=TEXT_COLOR, size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        height=height,
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showline=True,
            linecolor="rgba(148, 163, 184, 0.2)",
            tickfont=dict(color=MUTED_TEXT),
            titlefont=dict(color=TEXT_COLOR, size=13),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showline=True,
            linecolor="rgba(148, 163, 184, 0.2)",
            tickfont=dict(color=MUTED_TEXT),
            titlefont=dict(color=TEXT_COLOR, size=13),
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
            font=dict(color=TEXT_COLOR),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# PCA Visualisations
# ---------------------------------------------------------------------------

def plot_scree(evr: np.ndarray, component_labels: list[str]) -> go.Figure:
    """
    Scree plot: individual explained-variance ratio per principal component.
    """
    fig = go.Figure()
    pcts = evr * 100
    fig.add_trace(go.Bar(
        x=component_labels,
        y=pcts,
        name="Individual Variance",
        marker=dict(
            color=pcts,
            colorscale=[[0, "#1E293B"], [0.5, "#0284C7"], [1, "#38BDF8"]],
            line=dict(color="#38BDF8", width=1),
        ),
        text=[f"{v:.1f}%" for v in pcts],
        textposition="outside",
        textfont=dict(color="#E2E8F0", size=11),
    ))
    fig.update_layout(
        title=dict(text="<b>Scree Plot — Individual Explained Variance</b>", font=dict(size=16, color="#F8FAFC")),
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        yaxis=dict(range=[0, max(pcts) * 1.25 + 2]),
    )
    return _apply_dark_theme(fig, height=430)


def plot_cumulative_variance(
    cumulative: np.ndarray,
    component_labels: list[str],
    threshold: float | None = 0.90,
) -> go.Figure:
    """Cumulative explained-variance curve with gradient area and threshold line."""
    fig = go.Figure()
    cum_pct = cumulative * 100
    
    fig.add_trace(go.Scatter(
        x=component_labels,
        y=cum_pct,
        mode="lines+markers",
        name="Cumulative Variance",
        line=dict(color="#818CF8", width=3, shape="spline"),
        marker=dict(size=8, color="#C084FC", line=dict(color="#FFFFFF", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.12)",
    ))
    
    if threshold is not None:
        fig.add_hline(
            y=threshold * 100,
            line_dash="dash",
            line_color="#EF4444",
            line_width=2,
            annotation_text=f"Target: {threshold*100:.0f}%",
            annotation_position="bottom right",
            annotation_font=dict(color="#FCA5A5", size=12),
        )
        
    fig.update_layout(
        title=dict(text="<b>Cumulative Explained Variance Curve</b>", font=dict(size=16, color="#F8FAFC")),
        xaxis_title="Principal Component",
        yaxis_title="Cumulative Variance (%)",
        yaxis=dict(range=[0, 105]),
    )
    return _apply_dark_theme(fig, height=430)


def plot_pca_scatter(
    X_reduced: np.ndarray,
    labels: np.ndarray | None = None,
    risk_labels: pd.Series | None = None,
    pc_x: int = 0,
    pc_y: int = 1,
    hover_data: pd.DataFrame | None = None,
    title: str = "PCA Projection Space",
) -> go.Figure:
    """
    2-D scatter of PCA scores coloured by cluster or risk label.
    """
    n_components = X_reduced.shape[1] if X_reduced.ndim > 1 else 1
    x_idx = min(pc_x, n_components - 1)
    y_idx = min(pc_y, n_components - 1) if n_components > 1 else 0

    plot_df = pd.DataFrame({
        f"PC{x_idx+1}": X_reduced[:, x_idx] if n_components > 1 else X_reduced.ravel(),
        f"PC{y_idx+1}": (X_reduced[:, y_idx] if n_components > 1 else np.zeros(len(X_reduced))),
    })

    if labels is not None:
        plot_df["Cluster"] = [f"Cluster {l}" if str(l) != "-1" else "Noise" for l in labels]
    if risk_labels is not None:
        plot_df["Risk Level"] = risk_labels.values

    colour_col = "Risk Level" if risk_labels is not None else ("Cluster" if labels is not None else None)
    colour_map = RISK_COLOURS if colour_col == "Risk Level" else None

    if hover_data is not None:
        for col in hover_data.columns:
            plot_df[col] = hover_data[col].values

    hover_cols = [c for c in plot_df.columns if c not in (f"PC{x_idx+1}", f"PC{y_idx+1}")]

    fig = px.scatter(
        plot_df,
        x=f"PC{x_idx+1}",
        y=f"PC{y_idx+1}",
        color=colour_col,
        color_discrete_map=colour_map,
        color_discrete_sequence=CLUSTER_PALETTE,
        hover_data=hover_cols if hover_cols else None,
        title=f"<b>{title}</b>",
        height=520,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="#FFFFFF")))
    return _apply_dark_theme(fig, height=520)


def plot_pca_scatter_3d(
    X_reduced: np.ndarray,
    labels: np.ndarray | None = None,
    risk_labels: pd.Series | None = None,
    hover_data: pd.DataFrame | None = None,
) -> go.Figure:
    """Interactive 3D scatter plot of top 3 principal components."""
    if X_reduced.shape[1] < 3:
        return plot_pca_scatter(X_reduced, labels, risk_labels, title="PCA Projection (2D Fallback)")

    plot_df = pd.DataFrame({
        "PC1": X_reduced[:, 0],
        "PC2": X_reduced[:, 1],
        "PC3": X_reduced[:, 2],
    })

    if labels is not None:
        plot_df["Cluster"] = [f"Cluster {l}" for l in labels]
    if risk_labels is not None:
        plot_df["Risk Level"] = risk_labels.values

    colour_col = "Risk Level" if risk_labels is not None else ("Cluster" if labels is not None else None)
    colour_map = RISK_COLOURS if colour_col == "Risk Level" else None

    if hover_data is not None:
        for col in hover_data.columns:
            plot_df[col] = hover_data[col].values

    fig = px.scatter_3d(
        plot_df,
        x="PC1", y="PC2", z="PC3",
        color=colour_col,
        color_discrete_map=colour_map,
        color_discrete_sequence=CLUSTER_PALETTE,
        title="<b>3D Principal Component Cluster Topology</b>",
        height=580,
    )
    fig.update_traces(marker=dict(size=5, opacity=0.85))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        scene=dict(
            xaxis=dict(backgroundcolor=THEME_BG, gridcolor=GRID_COLOR),
            yaxis=dict(backgroundcolor=THEME_BG, gridcolor=GRID_COLOR),
            zaxis=dict(backgroundcolor=THEME_BG, gridcolor=GRID_COLOR),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Clustering Visualisations
# ---------------------------------------------------------------------------

def plot_cluster_distribution(cluster_sizes: dict[int, int]) -> go.Figure:
    """Bar chart of cluster sizes with modern gradient styling."""
    clusters = [str(k) for k in sorted(cluster_sizes)]
    sizes = [cluster_sizes[int(c)] for c in clusters]
    labels = [f"Cluster {c}" if c != "-1" else "Noise (-1)" for c in clusters]

    fig = go.Figure(go.Bar(
        x=labels,
        y=sizes,
        marker=dict(
            color=sizes,
            colorscale="Viridis",
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        text=sizes,
        textposition="outside",
        textfont=dict(color="#F1F5F9", size=12, family="Inter"),
    ))
    fig.update_layout(
        title=dict(text="<b>Cluster Population Distribution</b>", font=dict(size=16, color="#F8FAFC")),
        xaxis_title="Cluster Assignment",
        yaxis_title="Total Product Listings",
    )
    return _apply_dark_theme(fig, height=430)


def plot_risk_distribution(df: pd.DataFrame, risk_col: str = "risk_label") -> go.Figure:
    """Modern Donut chart of estimated-risk label distribution."""
    if risk_col not in df.columns:
        return go.Figure()
    counts = df[risk_col].value_counts()
    
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker=dict(
            colors=[RISK_COLOURS.get(l, "#64748B") for l in counts.index],
            line=dict(color="#0F172A", width=2.5),
        ),
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(color="#FFFFFF", size=12),
        hoverinfo="label+value+percent",
    ))
    fig.update_layout(
        title=dict(text="<b>Platform Risk Breakdown</b>", font=dict(size=16, color="#F8FAFC")),
        showlegend=True,
        annotations=[
            dict(text=f"<b>{len(df):,}</b><br><span style='font-size:10px;color:#94A3B8'>Total Listings</span>",
                 x=0.5, y=0.5, font_size=15, showarrow=False, font_color="#F8FAFC")
        ],
    )
    return _apply_dark_theme(fig, height=430)


def plot_compliance_gauge(compliance_rate: float) -> go.Figure:
    """Executive gauge meter showing platform compliance index (0 to 100%)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=compliance_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>Platform Compliance Score</b>", 'font': {'size': 16, 'color': '#F8FAFC'}},
        number={'suffix': "%", 'font': {'size': 36, 'color': '#F8FAFC'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#38BDF8", 'thickness': 0.28},
            'bgcolor': "rgba(15, 23, 42, 0.8)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.2)'},
            ],
            'threshold': {
                'line': {'color': "#10B981", 'width': 4},
                'thickness': 0.75,
                'value': 85.0
            }
        }
    ))
    return _apply_dark_theme(fig, height=320)


def plot_elbow(sweep_df: pd.DataFrame) -> go.Figure:
    """Elbow curve (inertia vs k) for K-Means tuning."""
    if sweep_df.empty:
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=sweep_df["k"],
        y=sweep_df["inertia"],
        mode="lines+markers",
        name="Inertia",
        line=dict(color="#38BDF8", width=2.5),
        marker=dict(size=7, color="#0284C7"),
    ), secondary_y=False)
    if "silhouette" in sweep_df.columns:
        sil_valid = sweep_df.dropna(subset=["silhouette"])
        fig.add_trace(go.Scatter(
            x=sil_valid["k"],
            y=sil_valid["silhouette"],
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#F43F5E", width=2, dash="dash"),
            marker=dict(size=7, color="#FB7185"),
        ), secondary_y=True)
    fig.update_layout(
        title=dict(text="<b>K-Means Optimization: Inertia & Silhouette Trade-off</b>", font=dict(size=16, color="#F8FAFC")),
        xaxis_title="Number of Clusters (k)",
    )
    _apply_dark_theme(fig, height=430)
    fig.update_yaxes(title_text="Inertia (Within-Cluster Sum of Squares)", secondary_y=False, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True, gridcolor="rgba(0,0,0,0)")
    return fig


def plot_cluster_risk_heatmap(cluster_risk_df: pd.DataFrame) -> go.Figure:
    """Heatmap of risk-percentage breakdown per cluster."""
    if cluster_risk_df.empty:
        return go.Figure()

    cols = ["pct_low", "pct_moderate", "pct_high"]
    present = [c for c in cols if c in cluster_risk_df.columns]
    if not present:
        return go.Figure()

    z = cluster_risk_df[present].values
    x_labels = [c.replace("pct_", "").replace("_", " ").title() + " Risk (%)" for c in present]
    y_labels = [f"Cluster {c}" for c in cluster_risk_df["cluster"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, "#0F172A"],
            [0.35, "#065F46"],
            [0.7, "#B45309"],
            [1.0, "#DC2626"],
        ],
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="<b>%{text}</b>",
        textfont=dict(color="#FFFFFF", size=13),
        showscale=True,
        colorbar=dict(title="% in Tier", tickcolor="#94A3B8", tickfont=dict(color="#94A3B8")),
    ))
    fig.update_layout(
        title=dict(text="<b>Cluster Compliance Risk Profile Heatmap</b>", font=dict(size=16, color="#F8FAFC")),
    )
    return _apply_dark_theme(fig, height=max(320, len(y_labels) * 55 + 120))


# ---------------------------------------------------------------------------
# Data Quality / EDA Visualisations
# ---------------------------------------------------------------------------

def plot_missing_values(missing_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of columns with missing values."""
    if missing_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="✨ No Missing Values Detected in Dataset",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#10B981")
        )
        return _apply_dark_theme(fig, height=280)

    fig = px.bar(
        missing_df.sort_values("missing_pct", ascending=True),
        x="missing_pct",
        y="column",
        orientation="h",
        text="missing_count",
        title="<b>Missing Values by Feature Attribute</b>",
        labels={"missing_pct": "Missing Ratio (%)", "column": "Feature"},
        color="missing_pct",
        color_continuous_scale="Reds",
        height=max(320, len(missing_df) * 32 + 80),
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#F8FAFC"))
    return _apply_dark_theme(fig, height=max(320, len(missing_df) * 32 + 80))


def plot_category_distribution(df: pd.DataFrame, col: str = "category") -> go.Figure:
    """Bar chart of value counts for a categorical column."""
    if col not in df.columns:
        return go.Figure()
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(
        counts,
        x=col,
        y="count",
        title=f"<b>Product Catalog Distribution by {col.title()}</b>",
        text="count",
        color="count",
        color_continuous_scale="Teal",
        height=420,
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#F8FAFC"))
    return _apply_dark_theme(fig, height=420)


def plot_numeric_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    """Histogram for a single numeric column."""
    if col not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df, x=col,
        nbins=35,
        title=f"<b>Empirical Distribution of {col}</b>",
        color_discrete_sequence=["#38BDF8"],
        height=400,
    )
    fig.update_traces(marker=dict(line=dict(color="#0F172A", width=1)))
    return _apply_dark_theme(fig, height=400)
