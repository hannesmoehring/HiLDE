from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evaluation.evaluate import start_evaluation
from src.types import Config
from src.ui.state import current_config


def handle_hierarchical_save(df: pd.DataFrame, feature_columns: list[str]) -> None:
    config: Config = current_config()
    with st.spinner("Computing analysis tree & quality scores…"):
        tree = start_evaluation(df, feature_columns, config)

    st.session_state["analysis_tree"] = tree
    st.session_state["global_scaler"] = tree.get("scaler")
    st.session_state["tree_path"] = []
    st.session_state["selected_indices"] = []
    st.session_state["selected_df"] = pd.DataFrame()
