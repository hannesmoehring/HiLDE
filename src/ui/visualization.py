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
    non_feature_only: bool = False,
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
    if not non_feature_only:
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


# Predicate-range chart palette (indigo for predicate clauses, grey for the rest).
_RANGE_TRACK = "rgba(140, 140, 150, 0.14)"
_PRED_EXTENT = "rgba(99, 110, 250, 0.30)"
_PRED_CORE = "rgba(99, 110, 250, 0.92)"
_OTHER_EXTENT = "rgba(150, 150, 150, 0.20)"
_OTHER_CORE = "rgba(150, 150, 150, 0.55)"


def make_feature_range_fig(range_df_full: pd.DataFrame, range_df_trimmed: pd.DataFrame) -> go.Figure:
    """Per-feature range bands placed within each feature's global range (0-100%).

    For every feature a faint full-width track shows the global extent; a translucent
    band shows the selection's full range (RCM=1.0) and a solid inner band its core
    (RCM=0.9). Predicate clauses are drawn in indigo, other features in muted grey.
    """
    full = range_df_full.reset_index(drop=True)
    trim = range_df_trimmed.reset_index(drop=True)

    has_predicate = "in_predicate" in full.columns
    in_pred = full["in_predicate"].to_numpy() if has_predicate else np.zeros(len(full), dtype=bool)
    clause_f1 = full["clause_f1"].to_numpy() if has_predicate else np.zeros(len(full))

    def to_pct(values: pd.Series, ref: pd.DataFrame) -> np.ndarray:
        span = (ref["global_max"] - ref["global_min"]).to_numpy()
        span = np.where(span == 0, 1.0, span)
        return (values.to_numpy() - ref["global_min"].to_numpy()) / span

    full_lo, full_hi = to_pct(full["sel_min"], full), to_pct(full["sel_max"], full)
    core_lo, core_hi = to_pct(trim["sel_min"], trim), to_pct(trim["sel_max"], trim)

    # Order so predicate clauses sit at the top, strongest F1 first.
    sort_key = np.lexsort((clause_f1, in_pred))  # ascending; Plotly puts index 0 at the bottom
    category_array = full["feature"].to_numpy()[sort_key].tolist()

    extent_color = [_PRED_EXTENT if f else _OTHER_EXTENT for f in in_pred]
    core_color = [_PRED_CORE if f else _OTHER_CORE for f in in_pred]

    core_custom = np.column_stack([trim["sel_min"].to_numpy(), trim["sel_max"].to_numpy(), clause_f1])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Global range",
            y=full["feature"],
            x=[1.0] * len(full),
            base=[0.0] * len(full),
            orientation="h",
            marker={"color": _RANGE_TRACK},
            hoverinfo="skip",
            showlegend=False,
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Full range",
            y=full["feature"],
            x=full_hi - full_lo,
            base=full_lo,
            orientation="h",
            marker={"color": extent_color, "line": {"width": 0}},
            hoverinfo="skip",
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Core range (RCM 0.9)",
            y=trim["feature"],
            x=core_hi - core_lo,
            base=core_lo,
            orientation="h",
            marker={"color": core_color, "line": {"width": 0}},
            customdata=core_custom,
            hovertemplate=(
                "<b>%{y}</b><br>Core range: %{customdata[0]:.2f} – %{customdata[1]:.2f}"
                "<br>Clause F1: %{customdata[2]:.2f}<extra></extra>"
            ),
        ),
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        barmode="overlay",
        bargap=0.55,
        height=max(220, 34 * len(full) + 90),
        margin={"l": 8, "r": 24, "t": 16, "b": 36},
        font={"size": 13},
        hoverlabel={"bgcolor": "white"},
    )
    fig.update_xaxes(
        range=[-0.02, 1.02],
        tickformat=".0%",
        tickvals=[0.0, 0.5, 1.0],
        title_text="Position within feature's global range",
        showgrid=True,
        gridcolor="rgba(140,140,150,0.18)",
        zeroline=False,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=category_array,
        title_text=None,
        showgrid=False,
        ticklabelposition="outside",
    )
    return fig
