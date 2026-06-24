import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analysis.analysis_routine import HierarchyObject
from src.analysis.characteristics import fit_cluster_decision_tree
from src.ui.tree_nav import child_size

PCA_VARIANCE_LABEL_THRESHOLD = 4.0

_KDE_GRID = np.linspace(-0.5, 0.5, 60)


def cluster_gauss_kde(node: HierarchyObject) -> go.Figure:
    children = node["next_object_layer"] or []
    sizes = [child_size(c) for c in children]
    max_size = max(sizes) if sizes else 1

    fig = go.Figure()

    for i, child in enumerate(children):
        Z = child["kde"]
        if Z is None:
            continue
        cx, cy = child["rel_position"] or (0.0, 0.0)
        size = sizes[i]

        target_size = 0.8 * np.sqrt(size / max_size) + 0.3
        plot_x = _KDE_GRID * target_size + cx
        plot_y = _KDE_GRID * target_size + cy

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
            ),
        )

    # clickable centroid markers — index i maps directly to next_object_layer[i]
    fig.add_trace(
        go.Scatter(
            x=[c["rel_position"][0] if c["rel_position"] else 0.0 for c in children],
            y=[c["rel_position"][1] if c["rel_position"] else 0.0 for c in children],
            mode="markers+text",
            marker={"size": 18, "color": "white", "opacity": 0.7, "line": {"width": 2, "color": "black"}},
            text=[f"C{i}" for i in range(len(children))],
            textposition="top center",
            customdata=[[i, sizes[i]] for i in range(len(children))],
            hovertemplate="<b>Cluster %{customdata[0]}</b><br>Size: %{customdata[1]}<extra></extra>",
            name="clusters",
            showlegend=False,
        ),
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Cluster topography — MDS layout, pre-computed KDE per cluster",
        plot_bgcolor="white",
    )
    return fig


_DEFAULT_TREE_DEPTH = 3


def cluster_characteristics_fig(
    characteristics: pd.DataFrame,
    n_points: int,
    df: pd.DataFrame,
    row_indices: np.ndarray,
    feature_cols: list[str],
) -> tuple[go.Figure, str]:
    order = characteristics.index.tolist()
    z_mean = characteristics["z_mean"]
    z_std = characteristics["z_std"]
    raw_mean = characteristics["raw_mean"]

    extra_cols = [
        c for c in df.columns
        if c not in feature_cols and c not in {"row_id"} and pd.api.types.is_numeric_dtype(df[c])
    ]

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.6)
    fig.add_trace(
        go.Bar(
            x=order,
            y=z_mean[order].to_numpy(),
            customdata=raw_mean[order].to_numpy(),
            error_y={
                "type": "data",
                "array": z_std[order].to_numpy(),
                "visible": True,
                "color": "rgba(0,0,0,0.35)",
                "thickness": 1.5,
            },
            marker_color=["crimson" if v < 0 else "steelblue" for v in z_mean[order]],
            hovertemplate=(
                "<b>%{x}</b><br>z-score: %{y:.2f}<br>within std: %{error_y.array:.2f}"
                "<br>cluster mean: %{customdata:.2f}<extra></extra>"
            ),
            name="feature cols",
            showlegend=len(extra_cols) > 0,
        ),
    )

    if extra_cols:
        extra_order = sorted(extra_cols)
        cluster_rows = df.iloc[row_indices]
        g_mean = df[extra_order].mean()
        g_std = df[extra_order].std().replace(0, 1)
        c_mean = cluster_rows[extra_order].mean()
        c_std = cluster_rows[extra_order].std()
        ez_mean = (c_mean - g_mean) / g_std
        fig.add_trace(
            go.Bar(
                x=extra_order,
                y=ez_mean.to_numpy(),
                customdata=c_mean.to_numpy(),
                error_y={
                    "type": "data",
                    "array": c_std.to_numpy(),
                    "visible": True,
                    "color": "rgba(0,0,0,0.35)",
                    "thickness": 1.5,
                },
                marker_color=["darkorange" if v < 0 else "mediumseagreen" for v in ez_mean],
                hovertemplate=(
                    "<b>%{x}</b><br>z-score: %{y:.2f}<br>within std: %{error_y.array:.2f}"
                    "<br>cluster mean: %{customdata:.2f}<extra></extra>"
                ),
                name="analysis cols",
                showlegend=True,
            ),
        )

    fig.update_layout(
        title=f"Cluster characteristics  •  n={n_points}",
        yaxis_title="z-score vs. global mean",
        xaxis_tickangle=-40,
        height=520,
        barmode="group",
    )

    in_cluster = pd.Series(data=False, index=df.index)
    in_cluster.iloc[row_indices] = True
    rules = fit_cluster_decision_tree(df, feature_cols, in_cluster, _DEFAULT_TREE_DEPTH)

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
            hover_data=["row_id"],
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
