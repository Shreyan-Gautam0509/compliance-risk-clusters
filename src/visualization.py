"""
visualization.py
----------------
Plotly-based chart builders for the SIH26051 dashboard.

All functions return plotly Figure objects; they do NOT call st.plotly_chart
themselves so they can be tested without a running Streamlit server.

SIH26051 — PCA-Based Compliance Risk Clustering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

RISK_COLOURS = {
    "Lower estimated risk":    "#2ecc71",
    "Moderate estimated risk": "#f39c12",
    "Higher estimated risk":   "#e74c3c",
}

CLUSTER_PALETTE = px.colors.qualitative.Plotly


# ---------------------------------------------------------------------------
# PCA Visualisations
# ---------------------------------------------------------------------------

def plot_scree(evr: np.ndarray, component_labels: list[str]) -> go.Figure:
    """
    Scree plot: individual explained-variance ratio per principal component.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=component_labels,
        y=evr * 100,
        name="Individual",
        marker_color="#3498db",
        text=[f"{v*100:.1f}%" for v in evr],
        textposition="outside",
    ))
    fig.update_layout(
        title="Scree Plot — Individual Explained Variance (%)",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        plot_bgcolor="white",
        yaxis=dict(range=[0, max(evr * 100) * 1.20 + 2]),
        height=420,
    )
    return fig


def plot_cumulative_variance(
    cumulative: np.ndarray,
    component_labels: list[str],
    threshold: float | None = 0.90,
) -> go.Figure:
    """Cumulative explained-variance curve with optional threshold line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=component_labels,
        y=cumulative * 100,
        mode="lines+markers",
        name="Cumulative variance",
        line=dict(color="#8e44ad", width=2),
        marker=dict(size=7),
    ))
    if threshold is not None:
        fig.add_hline(
            y=threshold * 100,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"{threshold*100:.0f}% target",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title="Cumulative Explained Variance (%)",
        xaxis_title="Principal Component",
        yaxis_title="Cumulative Variance (%)",
        plot_bgcolor="white",
        yaxis=dict(range=[0, 105]),
        height=420,
    )
    return fig


def plot_pca_scatter(
    X_reduced: np.ndarray,
    labels: np.ndarray | None = None,
    risk_labels: pd.Series | None = None,
    pc_x: int = 0,
    pc_y: int = 1,
    hover_data: pd.DataFrame | None = None,
    title: str = "PCA Scatter — PC1 vs PC2",
) -> go.Figure:
    """
    2-D scatter of PCA scores coloured by cluster or risk label.

    Parameters
    ----------
    X_reduced : np.ndarray  (n, ≥2)
    labels : np.ndarray | None  cluster labels
    risk_labels : pd.Series | None  risk label strings
    pc_x, pc_y : int  indices of components to plot (0-based)
    hover_data : pd.DataFrame | None  extra columns for hover tooltip
    """
    n_components = X_reduced.shape[1] if X_reduced.ndim > 1 else 1

    x_idx = min(pc_x, n_components - 1)
    y_idx = min(pc_y, n_components - 1) if n_components > 1 else 0

    plot_df = pd.DataFrame({
        f"PC{x_idx+1}": X_reduced[:, x_idx] if n_components > 1 else X_reduced.ravel(),
        f"PC{y_idx+1}": (X_reduced[:, y_idx] if n_components > 1 else
                         np.zeros(len(X_reduced))),
    })

    if labels is not None:
        plot_df["Cluster"] = labels.astype(str)
    if risk_labels is not None:
        plot_df["Risk"] = risk_labels.values

    colour_col = "Risk" if risk_labels is not None else ("Cluster" if labels is not None else None)
    colour_map = RISK_COLOURS if colour_col == "Risk" else None

    if hover_data is not None:
        for col in hover_data.columns:
            plot_df[col] = hover_data[col].values

    hover_cols = [c for c in plot_df.columns
                  if c not in (f"PC{x_idx+1}", f"PC{y_idx+1}")]

    fig = px.scatter(
        plot_df,
        x=f"PC{x_idx+1}",
        y=f"PC{y_idx+1}",
        color=colour_col,
        color_discrete_map=colour_map,
        hover_data=hover_cols if hover_cols else None,
        title=title,
        height=500,
    )
    fig.update_layout(plot_bgcolor="white")
    return fig


# ---------------------------------------------------------------------------
# Clustering Visualisations
# ---------------------------------------------------------------------------

def plot_cluster_distribution(cluster_sizes: dict[int, int]) -> go.Figure:
    """Bar chart of cluster sizes."""
    clusters = [str(k) for k in sorted(cluster_sizes)]
    sizes = [cluster_sizes[int(c)] for c in clusters]
    colours = [CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(len(clusters))]

    fig = go.Figure(go.Bar(
        x=[f"Cluster {c}" for c in clusters],
        y=sizes,
        marker_color=colours,
        text=sizes,
        textposition="outside",
    ))
    fig.update_layout(
        title="Cluster Size Distribution",
        xaxis_title="Cluster",
        yaxis_title="Number of Listings",
        plot_bgcolor="white",
        height=420,
    )
    return fig


def plot_risk_distribution(df: pd.DataFrame, risk_col: str = "risk_label") -> go.Figure:
    """Pie chart of estimated-risk label distribution."""
    if risk_col not in df.columns:
        return go.Figure()
    counts = df[risk_col].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker=dict(colors=[RISK_COLOURS.get(l, "#95a5a6") for l in counts.index]),
        hole=0.35,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Estimated Risk Distribution",
        height=420,
    )
    return fig


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
        line=dict(color="#3498db"),
    ), secondary_y=False)
    if "silhouette" in sweep_df.columns:
        sil_valid = sweep_df.dropna(subset=["silhouette"])
        fig.add_trace(go.Scatter(
            x=sil_valid["k"],
            y=sil_valid["silhouette"],
            mode="lines+markers",
            name="Silhouette",
            line=dict(color="#e74c3c", dash="dash"),
        ), secondary_y=True)
    fig.update_layout(
        title="K-Means Elbow Curve",
        xaxis_title="Number of Clusters (k)",
        plot_bgcolor="white",
        height=420,
    )
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
    return fig


def plot_cluster_risk_heatmap(cluster_risk_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of risk-percentage breakdown per cluster.
    """
    if cluster_risk_df.empty:
        return go.Figure()

    cols = ["pct_high", "pct_moderate", "pct_low"]
    present = [c for c in cols if c in cluster_risk_df.columns]
    if not present:
        return go.Figure()

    z = cluster_risk_df[present].values
    x_labels = [c.replace("pct_", "% ").replace("_", " ").title() for c in present]
    y_labels = [f"Cluster {c}" for c in cluster_risk_df["cluster"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, "#2ecc71"],
            [0.5, "#f39c12"],
            [1.0, "#e74c3c"],
        ],
        text=[[f"{v:.1f}%" for v in row] for row in z],
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="% of cluster"),
    ))
    fig.update_layout(
        title="Risk Breakdown per Cluster (%)",
        height=max(300, len(y_labels) * 60 + 100),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Data quality / EDA visualisations
# ---------------------------------------------------------------------------

def plot_missing_values(missing_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of columns with missing values."""
    if missing_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Missing Values Found 🎉")
        return fig
    fig = px.bar(
        missing_df.sort_values("missing_pct", ascending=True),
        x="missing_pct",
        y="column",
        orientation="h",
        text="missing_count",
        title="Missing Values by Column",
        labels={"missing_pct": "Missing (%)", "column": "Column"},
        color="missing_pct",
        color_continuous_scale="Reds",
        height=max(300, len(missing_df) * 30 + 80),
    )
    fig.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
    return fig


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
        title=f"Distribution of {col}",
        text="count",
        color="count",
        color_continuous_scale="Blues",
        height=420,
    )
    fig.update_layout(plot_bgcolor="white", coloraxis_showscale=False)
    return fig


def plot_numeric_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    """Histogram for a single numeric column."""
    if col not in df.columns:
        return go.Figure()
    fig = px.histogram(
        df, x=col,
        nbins=40,
        title=f"Distribution of {col}",
        color_discrete_sequence=["#3498db"],
        height=380,
    )
    fig.update_layout(plot_bgcolor="white")
    return fig
