# `src/` — the calculation layer

The whole method, with no UI attached. Give it a DataFrame, a list of feature columns and
a `Config`; get back a scored tree of nested regions.

**`src/` never imports from `backend/` or `frontend/`.** It is a plain Python library and
carries no web or Streamlit dependency. The FastAPI service is a wrapper over it
([`backend/README.md`](../backend/README.md)) and the research harnesses
([`src_research/README.md`](../src_research/README.md)) call the same entry point the
server calls — so an experiment and a served build run identical code.

---

## The one entry point

```python
from src.config_defaults import default_config
from src.datasets import DATASETS
from src.evaluation.evaluate import start_evaluation

df = DATASETS["Wine quality (Low)"]()
feature_cols = [c for c in df.columns if c != "row_id" and not c.startswith("target_")]

config = default_config()  # never share one instance — the layer mutates it
config["hierarchical_layers"] = 2
config["method"] = "UMAP"

tree = start_evaluation(df, feature_cols, config)
```

`start_evaluation` does two things in order: build the tree
(`analysis_routine.compute_analysis_tree`), then walk it attaching DR-quality scores to
every node. `print_tree(tree)` renders the result in the terminal.

---

## Layout

```
src/
├── types.py                    Config TypedDict + DRMethod
├── config_defaults.py          default_config() — the canonical Config
├── datasets.py                 the loader registry (10 datasets)
├── analysis/
│   ├── analysis_routine.py     compute_analysis_tree — the recursion
│   ├── clustering.py           HDBSCAN (+ KMeans / GMM / DBSCAN alternatives)
│   ├── dim_reducer.py          PCA / t-SNE / UMAP / MDS behind one interface
│   ├── characteristics.py      per-cluster z_mean / z_std / raw_mean
│   └── predicate_generator.py  axis-aligned range predicates
├── evaluation/
│   ├── evaluate.py             start_evaluation — build, then score every node
│   └── neighbor_metrics.py     chunked stress / trustworthiness-continuity / MRRE
└── util/
    ├── console.py              shared rich console for build progress
    └── datasets.py             MNIST IDX readers used by the registry
```

---

## The pipeline

```
df[feature_cols]
   │
   ├─ StandardScaler, fit ONCE at the root ────────────▶  X   (kept on tree["scaler"])
   │      (skipped when config["normalize"] is False)
   │
   ├─ UMAP pre-reduction to hclust_umap_n_components ──▶  X_reduced
   │      (only when that value is < the feature count)
   │
   └─ _build_next(X_reduced, depth=0)
          │
          ├─ depth >= hierarchical_layers, or too few points?  ──▶  ExplorationObject (leaf)
          │
          ├─ HDBSCAN on X_reduced  ──▶  labels (+ GLOSH outlier_scores)
          │       fewer than 2 real clusters?               ──▶  ExplorationObject (leaf)
          │
          ├─ per-cluster characteristics, and cluster centroids
          │  laid out in 2D by MDS  ──▶  rel_position
          │
          └─ recurse into each cluster  ──▶  HierarchyObject
```

Every node — internal or leaf — additionally gets its **own** 2D projection of its own
rows (`_embed_original`, using `config["method"]`). That is the embedding the UI displays
and the one the scores are computed on.

### Two node kinds

Both are TypedDicts, both carry `embedding_original`, `embedding_original_variance`,
`rel_characteristics`, `rel_position`, `row_indices` and (after scoring) `scores`.

| | `HierarchyObject` | `ExplorationObject` |
|---|---|---|
| when | it clustered into ≥ 2 real clusters | depth limit, too few points, or clustering found < 2 clusters |
| extra | `next_object_layer` (children), `cluster_points`, `outlier_scores` (GLOSH) | `is_leaf`, `depth`, `exploration_points` |

The root additionally carries `scaler`, fit once and reused by scoring and by the
predicate's global scope. `backend/serialize.py` drops it — it is server-side only.

### Two things worth knowing about the tree

**Noise rows are counted in a parent but appear in no child.** HDBSCAN labels points `-1`
when they belong to no cluster, and `_build_next` recurses only into real clusters. Child
sizes therefore do not sum to the parent's. Those rows are not lost — the UI reaches them
through *Explore entire layer* — but any code summing children must account for the gap.

**z-scores are whole-dataset relative at every depth.** The scaler is fit once at the
root and only row-masked thereafter, so a depth-3 cluster's `z_mean` still reads against
the whole dataset, not against its parent. Non-feature columns are the exception: they are
contrasted against the rows of the space the cluster was selected out of, standardised
with `ddof=0` so they land on the same axis as the feature columns.

---

## Modules

### `analysis/dim_reducer.py`

One interface over four reducers: `fit_dimensionality_reducer(method, X, config,
n_components)` returns a `ReductionResult` (embedding, the fitted reducer, and
`explained_variance_ratio` for PCA only). `reduce_dimensionality` is the same thing when
you only want the coordinates.

Two settings here exist purely for **determinism**, and both are load-bearing:

- **UMAP is constructed with `init="pca"`.** `random_state` alone is not enough: under
  the default `init="spectral"`, a disconnected fuzzy graph falls into
  `multi_component_layout`, which places the components outside the seeded path.
- **PCA uses `svd_solver="covariance_eigh"`**, which consults no RNG. The default
  `"auto"` picks the *randomized* solver on wide or small matrices (Olivetti's 400×4096
  root, for one) and draws from the unseeded process-global RNG. For tall/narrow shapes
  `"auto"` already chooses `covariance_eigh`, so those embeddings are unchanged.

t-SNE clamps `perplexity` below the sample count so it survives small sub-regions. MDS
uses `init="random"` with `n_init`/`max_iter` from the config.

> **t-SNE and PCA do not honour a replicate seed.** `_tsne` passes no `init`, and
> scikit-learn's default `init="pca"` never consults `random_state`; PCA takes no seed at
> all. Only UMAP and MDS produce genuinely different embeddings across seeds. This matters
> for the experiment harnesses — see `scripts/checks/05_h1a_replicate_collapse.py`.

### `analysis/clustering.py`

`compute_clusters(X, method, config)`. HDBSCAN is the only method the tree calls, and the
only one that returns a second value (`outlier_scores_`, i.e. GLOSH). KMeans, GMM and
DBSCAN are reachable through the same function; `src_research/hyperparameter_tuning.py`
sweeps DBSCAN and KMeans as a clustering factor (GMM is implemented but commented out of
its active grid).

### `analysis/predicate_generator.py`

`generate_predicate(method, df, X_scaled, threshold, selected_indices, tail_split)`.

The shipped method is **`"db"`** — DimBridge-style recursive predicate induction. It
explains the selection against the background dataset by greedily building a conjunction
of per-feature interval clauses, adding at each step the clause that most improves F1
between predicate membership and the selection labels, and stopping when no clause
improves it. Every feature gets a row back, whether or not it entered the conjunction:

| field | meaning |
|---|---|
| `sel_min`, `sel_max`, `sel_range` | the clause interval for this feature |
| `global_min`, `global_max` | that feature's range over the background, for drawing |
| `clause_f1`, `clause_precision`, `clause_recall` | this clause **alone** against the selection |
| `in_predicate`, `predicate_step` | did it enter the conjunction, and at which step |
| `predicate_f1` | F1 of the **final** conjunction (identical on every row) |

`threshold` is the range-coverage multiplier (RCM): the interval is trimmed to that
quantile coverage, so `1.0` is the selection's full range and `0.9` a trimmed core. The
trim is split between the two tails by `tail_split` — `"severity"` (default) in proportion
to how far each tail reaches from the median, or `"symmetric"` for an even split. The
backend draws both `1.0` and `0.9` for every selection; the RQ2 harness sweeps the
threshold down and measures what happens to stability.

`"threshold"` returns the per-feature marginal ranges with no scoring and no
conjunction — every feature gets an interval, so it is the **dense** predicate to `"db"`'s
sparse one. The app never uses it, but `benchmark_workflow.py` and `predicate_stability.py`
both run the two side by side, which is where the thesis's brevity comparison comes from.
`"hm"` is a third name that just delegates to `"threshold"`; nothing calls it.

### `evaluation/evaluate.py`

Scores each node on **its own embedding** — the same coordinates the UI shows — with the
root scaler applied to its rows, so the number describes exactly what is on screen.

Neighbourhood size is `k = min(20, (n-1)//2)`, and a node under 10 points is not scored at
all. Trustworthiness, continuity, both MRRE terms and stress come from `neighbor_scores`
in one pass; **CADI** (ZADU's Class Angular Distortion Index) comes from ZADU itself and
only exists on internal nodes, where the child cluster assignment supplies the labels.

Two failure modes are deliberately visible rather than absorbed:

- **A node that was never projected is not scored.** `embedding_original is None` (too
  small, or the reducer raised) returns an all-`None` `NodeScores`. Scoring fabricated
  coordinates would publish DR quality for a projection that does not exist.
- **A scoring failure leaves `k` as `None`**, so the node reads as unscored rather than
  advertising a neighbourhood size next to five missing numbers, and the reason is warned
  to the console. `MemoryError` lands here too — the one failure the chunked rewrite below
  exists to prevent.

### `evaluation/neighbor_metrics.py`

Stress, trustworthiness/continuity and MRRE. **The formulas and normalisation constants
are ZADU's, unchanged — this returns the same numbers.** What differs is bookkeeping.

ZADU materialises the pairwise distance matrix and then `argsort(argsort(...))` for
rankings, in both spaces: six N×N arrays live at once, which is ~40 GB at N = 30 000 and
gets the process OOM-killed. Every quantity those formulas actually read is row-local —
each point needs only the ranks of its own neighbours — so the rows are walked in chunks
and only the per-row values are kept, at O(chunk × N).

ZADU remains the right tool for everything else; this covers only the three measures whose
cost is quadratic in memory. `scripts/checks/02_zadu_equivalence.py` and
`03_zadu_chunked_and_degenerate.py` prove the two agree, including on the chunked path and
degenerate inputs. **The thesis's faithfulness numbers rest on that equivalence.**

### `datasets.py`

`DATASETS` maps a display key to a zero-argument loader, memoized with `functools.cache`
so each runs at most once per process. Six datasets work fully offline from a fresh clone;
four self-provision over the network on first use and cache to disk. The full table, with
sources and cache locations, is in the [root README](../README.md#datasets).

Every loader returns a DataFrame with a `row_id` column and label columns named
`target_*`. That naming is the contract: `backend/datasets.py::default_feature_cols` holds
`row_id` and every `target_*` out of the feature space, so labels never reach the
clustering, the projection or the predicate.

### `types.py` and `config_defaults.py`

`Config` is a flat `TypedDict` of every knob; `default_config()` returns a complete,
fresh instance. **The calc layer mutates config in place** (it clamps
`hclust_umap_n_components` and `umap_n_neighbors` against the data's shape), so callers
must not share one instance across builds.

The knobs the UI actually exposes:

| Key | Default | Effect |
|---|---|---|
| `normalize` | `True` | standardise the feature space at the root |
| `hierarchical_layers` | `1` | recursion depth |
| `hclust_umap_n_components` | `2` | UMAP pre-reduction dimensionality before clustering |
| `hclust_min_samples` | `5` | HDBSCAN `min_samples` |
| `hclust_min_cluster_size` | `25` | HDBSCAN `min_cluster_size`; also the leaf cutoff (`< 2×` this ⇒ leaf) |
| `method` | `"PCA"` | per-node exploration embedding — but see below |
| `tsne_*`, `umap_*`, `mds_*` | | per-reducer parameters |

Two traps in that table:

- **`method` redraws every node**, not just the leaf you are exploring: each layer's
  cluster projection is produced by the same `_embed_original`.
- **The app's default is not this default.** `frontend/src/config.ts` opens on
  `method: "UMAP"` while the Python default stays `"PCA"`, because
  `src_research/benchmark_workflow.py` builds "with shipped defaults" and reads its DR
  method straight off `default_config()`. Every request carries `method` explicitly, so
  the frontend value is what the served app runs on.

`init_state()` is a back-compat shim for research scripts that were written against the
removed Streamlit `src/ui/state.py`; it just returns the defaults.

---

## Known gaps

- **`pca_components` is dead configuration.** Declared in `types.py`, defaulted in
  `config_defaults.py` and mirrored in the frontend, but never read: every node embedding
  is hardcoded to 2D.
- **`characteristics.fit_cluster_decision_tree` is unused.** It fits a depth-3 decision
  tree separating a cluster from the rest and returns `export_text` output. Nothing calls
  it; it is kept from an earlier surface-level explanation attempt.
- **`util/datasets.train_val_split` is unused.** Only the MNIST IDX readers in that file
  are imported.
