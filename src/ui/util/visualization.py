import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text


def cluster_gauss_kde(df_points: pd.DataFrame, X_scaled_df: pd.DataFrame, layout_df: pd.DataFrame) -> go.Figure:
    centroids_2d = layout_df[["x", "y"]].to_numpy()
    size_map = layout_df.set_index("cluster")["size"]

    fig = go.Figure()

    for i, c in enumerate(layout_df["cluster"]):
        pts = X_scaled_df.loc[df_points["cluster"] == c].to_numpy()
        if len(pts) < 5:
            continue

        # local 2D embedding of just this cluster
        pca = PCA(n_components=2)
        pts_2d = pca.fit_transform(pts)

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
            customdata=layout_df["cluster"].to_numpy().reshape(-1, 1),
            hovertemplate="<b>Cluster %{customdata[0]}</b><br>Click to explore<extra></extra>",
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
