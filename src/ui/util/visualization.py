import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap as _umap
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, export_text

from src.analysis.dim_reducer import local_2d

PCA_VARIANCE_LABEL_THRESHOLD = 4.0
KDE_MIN_PTS = 5  # minimum points to render a KDE blob


def cluster_gauss_kde(
    df_points: pd.DataFrame,
    X_scaled_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    *,
    kde_dr_method: str = "PCA",
) -> go.Figure:
    centroids_2d = layout_df[["x", "y"]].to_numpy()
    size_map = layout_df.set_index("cluster")["size"]

    fig = go.Figure()

    for i, c in enumerate(layout_df["cluster"]):
        pts = X_scaled_df.loc[df_points["cluster"] == c].to_numpy()
        if len(pts) < KDE_MIN_PTS:
            continue

        pts_2d = local_2d(pts, kde_dr_method)

        # KDE on the local 2D points
        kde = gaussian_kde(pts_2d.T, bw_method="scott")
        pad = 0.5 * pts_2d.std(axis=0).max()
        lx_min, lx_max = pts_2d[:, 0].min() - pad, pts_2d[:, 0].max() + pad
        ly_min, ly_max = pts_2d[:, 1].min() - pad, pts_2d[:, 1].max() + pad
        lx = np.linspace(lx_min, lx_max, 60)
        ly = np.linspace(ly_min, ly_max, 60)
        LX, LY = np.meshgrid(lx, ly)
        Z = kde(np.vstack([LX.ravel(), LY.ravel()])).reshape(LX.shape)

        # scale the local footprint and place at the cluster's MDS position
        cx, cy = centroids_2d[i]
        local_extent = max(lx_max - lx_min, ly_max - ly_min)
        target_size = 0.8 * np.sqrt(size_map[c] / size_map.max()) + 0.3  # tunable
        scale = target_size / local_extent

        plot_x = (lx - (lx_min + lx_max) / 2) * scale + cx
        plot_y = (ly - (ly_min + ly_max) / 2) * scale + cy

        fig.add_trace(
            go.Contour(
                x=plot_x,
                y=plot_y,
                z=Z,
                colorscale="Viridis",
                showscale=False,
                contours={"coloring": "heatmap", "showlines": True, "start": Z.max() * 0.05, "size": Z.max() * 0.15},
                line_smoothing=0.85,
                hoverinfo="skip",
            )
        )

    # clickable centroid markers — Plotly selection events fire on Scatter traces only
    fig.add_trace(
        go.Scatter(
            x=layout_df["x"],
            y=layout_df["y"],
            mode="markers+text",
            marker={"size": 18, "color": "white", "opacity": 0.7, "line": {"width": 2, "color": "black"}},
            text=[f"C{c}" for c in layout_df["cluster"]],
            textposition="top center",
            customdata=layout_df[["cluster", "size"]].to_numpy(),
            hovertemplate="<b>Cluster %{customdata[0]}</b><br>Size: %{customdata[1]}<extra></extra>",
            name="clusters",
            showlegend=False,
        )
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Cluster topography — global MDS layout, local KDE per cluster",
        plot_bgcolor="white",
    )

    return fig


def cluster_characteristics(cluster_id, df, X_scaled_df, feature_cols, tree_depth: int = 3):
    in_cluster = df["cluster"] == cluster_id
    pts = X_scaled_df.loc[in_cluster, feature_cols]

    z_mean = pts.mean()
    z_std = pts.std()
    order = z_mean.abs().sort_values(ascending=False).index.tolist()
    z_mean, z_std = z_mean[order], z_std[order]

    fig = go.Figure()

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.6)
    fig.add_trace(
        go.Bar(
            x=order,
            y=z_mean.to_numpy(),
            error_y={"type": "data", "array": z_std.to_numpy(), "visible": True, "color": "rgba(0,0,0,0.35)", "thickness": 1.5},
            marker_color=["crimson" if v < 0 else "steelblue" for v in z_mean],
            hovertemplate="<b>%{x}</b><br>z-score: %{y:.2f}<br>within std: %{error_y.array:.2f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"Cluster {cluster_id}  •  n={in_cluster.sum()}",
        yaxis_title="z-score vs. global mean",
        xaxis_tickangle=-40,
        height=520,
    )

    # predicate rules — train on ORIGINAL units so thresholds are interpretable
    tree = DecisionTreeClassifier(max_depth=tree_depth, class_weight="balanced", random_state=0)
    tree.fit(df[feature_cols].to_numpy(), in_cluster.to_numpy().astype(int))
    rules = export_text(tree, feature_names=list(feature_cols))

    return fig, rules


def make_scatter_fig(
    plot_df: pd.DataFrame,
    method: str,
    *,
    cluster_mode: bool,
    interactive_mode: bool,
) -> go.Figure:
    has_clusters = cluster_mode and "cluster_label" in plot_df.columns
    has_interactive = interactive_mode and "interactive_group" in plot_df.columns

    if has_clusters and has_interactive:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="interactive_group",
            color_discrete_map={"Matches filters": "#1f77b4", "Other": "#bdbdbd"},
            symbol="cluster_label",
            hover_data=["row_id"],  # quality was here make it configurable?
            title=f"{method} projection - interactive filters + clusters in original space",
            height=700,
        )
    elif has_clusters:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="cluster_label",
            hover_data=["row_id"],
            title=f"{method} projection - clusters in original space",
            height=700,
        )
    elif has_interactive:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="interactive_group",
            color_discrete_map={"Matches filters": "#1f77b4", "Other": "#bdbdbd"},
            hover_data=["row_id"],
            title=f"{method} projection - interactive feature range filtering",
            height=700,
        )
    else:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            hover_data=["row_id"],
            title=f"{method} projection",
            height=700,
        )

    fig.update_traces(marker={"size": 8, "opacity": 0.85})
    fig.update_layout(dragmode="lasso")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def make_pca_variance_fig(explained_ratio: np.ndarray) -> go.Figure:
    pc_labels = [f"PC{i + 1}" for i in range(len(explained_ratio))]
    explained_pct = explained_ratio * 100.0
    cumulative_pct = np.cumsum(explained_pct)

    fig = go.Figure()
    for label, pct, cumulative in zip(pc_labels, explained_pct, cumulative_pct, strict=True):
        fig.add_trace(
            go.Bar(
                y=["Variance"],
                x=[pct],
                orientation="h",
                name=label,
                text=[f"{pct:.1f}%"] if pct >= PCA_VARIANCE_LABEL_THRESHOLD else None,
                textposition="inside",
                customdata=[[cumulative]],
                hovertemplate=(f"{label}<br>Variance: %{{x:.2f}}%<br>Cumulative: %{{customdata[0]:.2f}}%<extra></extra>"),
            ),
        )

    fig.update_layout(
        barmode="stack",
        height=150,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis_title="Share of total variance (%)",
        yaxis_title=None,
        yaxis={"showticklabels": False},
        legend={"orientation": "h", "y": 1.5, "x": 0},
    )
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    return fig


def make_feature_range_fig(range_df_full: pd.DataFrame, range_df_trimmed: pd.DataFrame) -> go.Figure:
    g_span = range_df_full["global_max"] - range_df_full["global_min"]

    def norm(series: pd.Series) -> pd.Series:
        return 2.0 * (series - range_df_full["global_min"]) / g_span - 1.0

    full_norm_min = norm(range_df_full["sel_min"])
    full_norm_max = norm(range_df_full["sel_max"])

    g_span_t = range_df_trimmed["global_max"] - range_df_trimmed["global_min"]
    trim_norm_min = 2.0 * (range_df_trimmed["sel_min"] - range_df_trimmed["global_min"]) / g_span_t - 1.0
    trim_norm_max = 2.0 * (range_df_trimmed["sel_max"] - range_df_trimmed["global_min"]) / g_span_t - 1.0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Global range",
            y=range_df_full["feature"],
            x=[2.0] * len(range_df_full),
            base=[-1.0] * len(range_df_full),
            orientation="h",
            marker={"color": "rgba(120, 120, 120, 0.35)"},
            hoverinfo="skip",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Selected range (threshold=1.0)",
            y=range_df_full["feature"],
            x=full_norm_max - full_norm_min,
            base=full_norm_min,
            customdata=range_df_full[["sel_min", "sel_max"]],
            orientation="h",
            marker={"color": "rgba(40, 130, 255, 0.75)"},
            hovertemplate=(
                "<b>%{y}</b><br>Selected min: %{customdata[0]:.2f}<br>Selected max: %{customdata[1]:.2f}<br>Selected range: %{x:.2f}<extra></extra>"
            ),
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Selected range (threshold=0.9)",
            y=range_df_trimmed["feature"],
            x=trim_norm_max - trim_norm_min,
            base=trim_norm_min,
            customdata=range_df_trimmed[["sel_min", "sel_max"]],
            orientation="h",
            marker={"color": "rgba(255, 140, 60, 0.8)"},
            hovertemplate=(
                "<b>%{y}</b><br>Trimmed min: %{customdata[0]:.2f}<br>Trimmed max: %{customdata[1]:.2f}<br>Trimmed range: %{x:.2f}<extra></extra>"
            ),
        ),
    )
    fig.update_layout(
        title="Selected subset standardized ranges (RCM=1.0 and 0.9) inside global feature range",
        xaxis_title="Standardized feature value",
        yaxis_title="Feature",
        barmode="overlay",
        bargap=0.45,
        height=max(320, 28 * len(range_df_full) + 120),
        legend={"orientation": "h", "y": 1.1, "x": 0},
    )
    fig.update_xaxes(range=[-1, 1])
    return fig
