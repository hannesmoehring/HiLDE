"""Capture `analysis_iris.json` and `movement_iris.json` from a LIVE backend.

The two fixtures under `frontend/src/fixtures/` are reference samples of the wire
format. The movement one carries an invariant a hand-written sample cannot be
trusted to satisfy: every ghost is the serialized source point plus
`visible_displacement` — one shared translation. Capturing both fixtures from ONE
run makes that checkable, and this script checks it before writing anything.

Usage — with the backend running on any port (run cache on or off):

    PYTHONPATH=. uv run uvicorn backend.app:app --port 8000
    python frontend/scripts/capture_movement_fixture.py --base-url http://127.0.0.1:8000

Writes:
    frontend/src/fixtures/analysis_iris.json   the `/api/analysis` payload ({meta, tree})
                                               of a two-layer PCA build on Iris
    frontend/src/fixtures/movement_iris.json   the `/api/movement` response for a PCA
                                               point target on `root/1`: source child 0,
                                               aimed at child 1's visible centroid

Standard library only, so it runs without the project's virtualenv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[1] / "src" / "fixtures"
CONFIG = {"hierarchical_layers": 2, "method": "PCA"}
POLL_SECONDS = 1.0


def _call(base: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            return json.load(res)
    except urllib.error.HTTPError as exc:
        sys.exit(f"{path}: HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}")


def _finished(base: str, first: dict[str, Any], jobs_path: str) -> dict[str, Any]:
    """Poll a job envelope to its terminal state, like the frontend does."""
    job = first
    while job.get("status") == "running":
        time.sleep(POLL_SECONDS)
        job = _call(base, f"{jobs_path}/{job['job_id']}")
    if job.get("status") == "error":
        sys.exit(f"job failed: {job.get('detail')}")
    return job


def _node(tree: dict[str, Any], node_id: str) -> dict[str, Any]:
    node = tree
    for part in node_id.split("/")[1:]:
        node = node["children"][int(part)]
    return node


def _child_visible_points(
    parent: dict[str, Any], child: dict[str, Any]
) -> list[list[float]]:
    position = {row: i for i, row in enumerate(parent["row_indices"])}
    return [parent["embedding_original"][position[row]] for row in child["row_indices"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--dataset", default="Iris (Low)")
    ap.add_argument("--node", default="root/1", help="parent node id (TreeNode.id)")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    columns = _call(base, f"/api/datasets/{urllib.parse.quote(args.dataset)}/columns")
    feature_cols = columns["default_feature_cols"]

    analysis = _finished(
        base,
        _call(
            base,
            "/api/analysis",
            {
                "dataset": args.dataset,
                "feature_cols": feature_cols,
                "config": CONFIG,
                "use_cache": True,
            },
        ),
        "/api/analysis/jobs",
    )
    payload = {"meta": analysis["meta"], "tree": analysis["tree"]}
    meta = payload["meta"]

    parent = _node(payload["tree"], args.node)
    children = parent["children"] or []
    if len(children) < 2:
        sys.exit(f"{args.node} has {len(children)} children; the capture needs two")
    source, destination = children[0], children[1]
    # A POINT target at the destination cluster's visible centroid: a free click
    # that happens to land on data, so the sample exercises the pinv path and
    # the min-norm sentence while still being a movement one can read.
    dest = _child_visible_points(parent, destination)
    target = {
        "kind": "point",
        "x": round(sum(p[0] for p in dest) / len(dest), 4),
        "y": round(sum(p[1] for p in dest) / len(dest), 4),
    }

    movement = _finished(
        base,
        _call(
            base,
            "/api/movement",
            {
                "analysis_id": meta["analysis_id"],
                "dataset": args.dataset,
                "feature_cols": meta["feature_cols"],
                "config": meta["config"],
                "node_id": args.node,
                "source_child_index": 0,
                "target": target,
                "strength": None,
            },
        ),
        "/api/movement/jobs",
    )

    # The invariant: ghost == serialized source point + visible_displacement.
    origin = dict(zip(source["row_indices"], _child_visible_points(parent, source)))
    dx, dy = movement["visible_displacement"]
    worst = max(
        max(
            abs(p["x"] - origin[p["row_id"]][0] - dx),
            abs(p["y"] - origin[p["row_id"]][1] - dy),
        )
        for p in movement["preview_points"]
    )
    if worst > 1e-9:
        sys.exit(
            f"preview points are not a shared translation (max deviation {worst:.3e})"
        )

    (FIXTURES / "analysis_iris.json").write_text(
        json.dumps(payload, separators=(",", ":"))
    )
    (FIXTURES / "movement_iris.json").write_text(
        json.dumps(movement, separators=(",", ":"))
    )
    print(
        f"wrote {FIXTURES / 'analysis_iris.json'} and {FIXTURES / 'movement_iris.json'}\n"
        f"  analysis_id {meta['analysis_id'][:12]}…  node {args.node}  "
        f"source C0 ({movement['source_size']} points) -> point {target['x']}, {target['y']}\n"
        f"  shared-translation deviation {worst:.2e}  "
        f"alignment RMSE {movement['metrics']['alignment_rmse']:.2e}"
    )


if __name__ == "__main__":
    main()
