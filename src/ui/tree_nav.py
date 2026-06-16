from __future__ import annotations

from src.analysis.analysis_routine import AnalysisObject, HierarchyObject


def get_node_at_path(root: AnalysisObject, path: list[int] | tuple[int, ...]) -> AnalysisObject:
    node: AnalysisObject = root
    for idx in path:
        if "is_leaf" in node:
            break
        hier: HierarchyObject = node  # type: ignore[assignment]
        node = hier["next_object_layer"][idx]
    return node


def child_size(child: AnalysisObject) -> int:
    if "is_leaf" in child:
        return child["exploration_points"].shape[0]  # type: ignore[index]
    return child["cluster_points"].shape[0]  # type: ignore[index]
