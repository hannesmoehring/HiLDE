from __future__ import annotations

import streamlit as st

from src.analysis.analysis_routine import NodeScores


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def render_node_scores(scores: NodeScores | None, *, title: str | None = None) -> None:
    """Render a compact row of DR-quality metric tiles for one analysis node."""
    if title:
        st.markdown(title)
    if scores is None:
        st.caption("Quality scores unavailable — re-run Save & Apply.")
        return

    cols = st.columns(4)
    cols[0].metric("Trustworthiness", _fmt(scores["trustworthiness"]), help="Higher is better (1 = perfect). Local neighborhood preservation.")
    cols[1].metric("Continuity", _fmt(scores["continuity"]), help="Higher is better (1 = perfect). Original neighbors kept close in 2D.")
    cols[2].metric("Stress", _fmt(scores["stress"]), help="Lower is better (0 = perfect). Global distance distortion.")
    cols[3].metric("CADI", _fmt(scores["cadi"]), help="Lower is better (0 = perfect). Class angular distortion across sub-clusters; N/A on leaves.")

    st.caption(
        f"MRRE false/missing: {_fmt(scores['mrre_false'])} / {_fmt(scores['mrre_missing'])}"
        f"  ·  k={scores['k'] if scores['k'] is not None else '—'}"
        f"  ·  n={scores['n_points']}",
    )
