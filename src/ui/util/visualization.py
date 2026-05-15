import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, export_text


def cluster_gauss_kde(df: pd.DataFrame, X_scaled_df: pd.DataFrame, centroids_2d: np.ndarray) -> go.Figure:
    fig = go.Figure()

    for i, c in enumerate(df["cluster"].unique()):
        pts = X_scaled_df.loc[df["cluster"] == c].values
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
        target_size = 0.8 * np.sqrt(df["size"].loc[c] / df["size"].max()) + 0.3  # tunable
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
                contours=dict(coloring="heatmap", showlines=True, start=Z.max() * 0.05, size=Z.max() * 0.15),
                line_smoothing=0.85,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=centroids_2d[:, 0],
            y=centroids_2d[:, 1],
            mode="text",
            text=[f"C{c}" for c in df["cluster"].unique()],
            textposition="top center",
            showlegend=False,
        )
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title="Cluster topography — global MDS layout, local KDE per cluster",
        plot_bgcolor="white",
    )

    return fig


def cluster_characteristics(cluster_id, df, X_scaled_df, feature_cols, top_n=8, tree_depth=3):
    in_cluster = df["cluster"] == cluster_id
    pts = X_scaled_df.loc[in_cluster, feature_cols]

    # in scaled space, global mean=0 and global std=1, so:
    z_mean = pts.mean()  # signed z-score of cluster mean per dim
    z_std = pts.std()  # within-cluster std, in units of global std
    order = z_mean.abs().sort_values(ascending=False).index.tolist()
    z_mean, z_std = z_mean[order], z_std[order]

    R = max(3.0, z_mean.abs().max() + 1)  # radial extent in z-units

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "xy"}]],
        column_widths=[0.55, 0.45],
        subplot_titles=(f"Cluster {cluster_id} profile", "Top distinguishing dimensions"),
    )

    theta = order + [order[0]]

    # reference ring at global mean (z=0 → r=R)
    fig.add_trace(
        go.Scatterpolar(r=[R] * len(theta), theta=theta, mode="lines", line=dict(color="gray", dash="dot"), name="global mean", hoverinfo="skip"),
        row=1,
        col=1,
    )

    # within-cluster ±1σ band (the "uncertainty" of the cluster on each dim)
    band_hi = (z_mean + z_std).tolist() + [(z_mean + z_std).iloc[0]]
    band_lo = (z_mean - z_std).tolist() + [(z_mean - z_std).iloc[0]]
    fig.add_trace(
        go.Scatterpolar(r=[v + R for v in band_hi], theta=theta, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[v + R for v in band_lo],
            theta=theta,
            fill="tonext",
            fillcolor="rgba(70,130,200,0.25)",
            mode="lines",
            line=dict(width=0),
            name="±1σ within-cluster",
        ),
        row=1,
        col=1,
    )

    # cluster mean polygon — outside ring = above global, inside = below
    means = z_mean.tolist() + [z_mean.iloc[0]]
    fig.add_trace(
        go.Scatterpolar(r=[v + R for v in means], theta=theta, mode="lines+markers", line=dict(color="rgb(50,90,180)", width=2), name="cluster mean"),
        row=1,
        col=1,
    )

    fig.update_polars(radialaxis=dict(range=[0, 2 * R], showticklabels=False))

    # signed-z bar chart for the top-N most distinguishing dims
    top = z_mean.head(top_n)
    fig.add_trace(
        go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
            marker_color=["crimson" if v < 0 else "steelblue" for v in top.values],
            text=[f"within σ={z_std[d]:.2f}" for d in top.index],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title="z-score of cluster mean (vs. global)", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(title=f"Cluster {cluster_id}  •  n={in_cluster.sum()}", height=520)

    # predicate rules — train on ORIGINAL units so thresholds are interpretable
    tree = DecisionTreeClassifier(max_depth=tree_depth, class_weight="balanced", random_state=0)
    tree.fit(df[feature_cols].values, in_cluster.astype(int).values)
    rules = export_text(tree, feature_names=list(feature_cols))

    return fig, rules
