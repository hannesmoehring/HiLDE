# Reproduction scripts for REVIEW.md

Self-contained, read-only. Nothing here writes into the repo, mutates git state, or touches
`.cache/`, `outputs/` or `datasets/`. Each script prints the evidence quoted in `REVIEW.md`.

## Running

Run from `/tmp` so that no CWD-relative dataset path resolves into the repo:

```bash
REPO=/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD
cd /tmp && PYTHONPATH=$REPO $REPO/.venv/bin/python $REPO/review_repro/<script>.py
```

**Exception:** `10_umap_still_nondeterministic.py` needs the wine CSVs, whose paths are
CWD-relative, so run it from the repo root instead:

```bash
cd $REPO && PYTHONPATH=. .venv/bin/python review_repro/10_umap_still_nondeterministic.py
```

`02`, `03` and `09` import `zadu`; `05` and `09` import from `src_research`. All are already
in `.venv`. Nothing requires network access. Scripts `01`, `04`, `05`, `06`, `08`, `10` print
progress lines from the pipeline's own logger — pipe through `grep -v "Dim reduction\|Clustering:"`
to suppress them.

## What each script demonstrates

| Script | REVIEW.md | Demonstrates | Headline number |
|---|---|---|---|
| `01_zero_embedding_is_scored.py` | B4 | A failed reducer becomes an all-zeros embedding that is then scored and published as real DR quality | forced failure → `stress=1.0 trust=0.5599`; real PCA control → `stress=0.378 trust=0.844` |
| `02_zadu_equivalence.py` | §6 (clean) | `neighbor_metrics` reproduces ZADU exactly across 8 (n,d,k) combinations incl. `k=1` and the `k=(n-1)//2` boundary | worst absolute difference `0.000e+00` |
| `03_zadu_chunked_and_degenerate.py` | §6 (clean), H1 | The chunked path is equivalent to ZADU (1/5/22/150 chunks) and ties are exact; **and** the degenerate case diverges | chunked worst `2.2e-16`; all-identical points → `ZeroDivisionError` where ZADU returns values |
| `04_characteristics_baseline_divergence.py` | B6 | The tree z-scores features against the **root**, the new endpoint against the **node** — same points, same parent | `f1`: tree `-0.515` vs endpoint `+1.205` (opposite signs) |
| `05_h1a_replicate_collapse.py` | B1 | The 24h seed change made all 5 H1a "replicates" byte-identical, inflating the Wilcoxon `n` fivefold | all replicates identical, `max|diff| = 0.000e+00`; R=20: honest `p=0.189` → inflated `p=0.0025` |
| `06_pca_is_unseeded.py` | B5 | `_pca` gets no `random_state`; sklearn's `auto` solver picks `randomized` for wide/small matrices | 400×3000: `identical across 3 runs: False`, `max|diff| = 2.426e+01` |
| `07_qm9_ring_labels_ok.py` | §6 (refuted) | The QM9 ring-label hypothesis is **wrong** — the docstring's premise is false but the labels are correct | 1439 SMILES carry bracket digits; **0 mislabels** across all 133,885 rows |
| `08_characteristics_units_and_noise.py` | B6, B7, H2 | Three defects: non-feature `z_std` in raw units; feature/extra `ddof` mismatch; HDBSCAN noise leaving the hierarchy | `z_std` off by **40.71×**; ddof ratio `1.008438968` = `sqrt(60/59)` exactly; Iris `root/1`: 100 points, 92 in children |
| `09_mrre_direction_inverted.py` | B2 | Both MRRE terms are higher-is-better, but `HIGHER_IS_BETTER` declares them lower-is-better | good vs bad embedding: `mrre_false 0.686 > 0.503` while declared `lower=better` |
| `10_umap_still_nondeterministic.py` | B0 | `random_state=42` does **not** make UMAP reproducible on a disconnected fuzzy graph; the whole hierarchy changes | 3 identical builds → **33 / 61 / 69** clusters; `init="pca"` → identical |

## Expected runtime

`02`, `07`, `09` are seconds. `01`, `03`, `06` are under a minute. `04`, `05`, `08` build real
trees (~1-3 min each). `10` runs UMAP on 6497×11 five times — allow ~5 minutes.

## Caveats

- `04` and `08` build trees with HDBSCAN + UMAP. Per finding **B0**, UMAP on the *default*
  dataset is nondeterministic, so exact cluster sizes may differ between your run and the numbers
  quoted in `REVIEW.md`. The *direction* of every finding is structural and does not depend on
  which clusters HDBSCAN happens to find — `04`'s sign flip and `08`'s 40× ratio reproduce
  regardless. `08` uses a hand-built frame for parts (1) and (2) precisely so those two numbers
  are exact.
- `06` section 1's solver probe prints `?` — that helper is unreliable and is not the evidence;
  section 2 (the behavioural test through the repo's own code path) is.
