# UI review — the Ranges tab and "Explore entire layer"

Adversarial, read-only review of the exploration-panel rewrite, scoped to the diff.

## Scope correction — this was never a pre-commit review

The task described the target as uncommitted work sitting on `HEAD = ebea1b2`, with
`RangeFilters.tsx` and `useDebounced.ts` as untracked files. **That precondition was already
false when the review started**, and most of it had been false for some time. Commit
timestamps:

```
6d2a34e  01:00:52  ExplorationPanel rewrite (234 lines), RangeFilters.tsx (253, "untracked"),
                   useDebounced.ts (15, "untracked"), styles.css, .gitignore, .wtest
14223c6  01:06:02  App.tsx (+64) — the "Explore entire layer" feature, styles.css
8d4399e  01:17:43  config.ts (PCA -> UMAP), styles.css      <- the only one during this review
ebea1b2            Correct a count in 6001d17's commit message   <- the stated HEAD
```

At the first command of this session `HEAD` was already `14223c6`, not `ebea1b2`, and
`git status` showed only `config.ts` and `styles.css` still modified. So 6d2a34e and 14223c6
— which carry the entire ExplorationPanel rewrite, both "untracked" files, and the whole-layer
feature — predate this session. Only 8d4399e landed during it, capturing the two files that
were genuinely uncommitted at the start.

**The reviewed content is nevertheless exactly the intended target, with nothing missed:**

```
git diff ebea1b2 HEAD  ==  6d2a34e + 14223c6 + 8d4399e
                       ==  (already committed on arrival) + (uncommitted on arrival)
```

This was not merely asserted. The stated precondition was **reconstructed** in `/tmp`
(`git archive ebea1b2` — read-only on this repo — into a scratch repo, then the three
commits' content applied as working-tree changes). That scratch tree reproduces the task's
description exactly:

```
 M .gitignore   D .wtest   M frontend/src/App.tsx   M .../ExplorationPanel.tsx
 M frontend/src/config.ts  M frontend/src/styles.css
?? frontend/src/components/RangeFilters.tsx     <- the two files the task called untracked
?? frontend/src/hooks/useDebounced.ts
```

and the equivalence holds byte-for-byte:

```
git diff HEAD (at HEAD=ebea1b2)                    ==  git diff ebea1b2 HEAD minus those two files   [IDENTICAL]
RangeFilters.tsx, useDebounced.ts as untracked     ==  the committed copies                          [byte-identical]
```

So `git diff ebea1b2 HEAD` **is** "`git diff HEAD` plus the two untracked files" — the target
as specified. The false precondition concerns the repository's *commit state*, not the content
under review; the content is provably the specified one.

Every file the task named is covered — the ExplorationPanel rewrite, `RangeFilters.tsx`,
`useDebounced.ts`, the interactive-predicate tab and the whole-layer feature — plus
`config.ts` and `styles.css`. The only content beyond the described scope is `.gitignore`
and `.wtest` (an empty *tracked* file that existed at `ebea1b2`, now deleted and ignored —
housekeeping, not new junk). `git status` now shows nothing but this report.

What changes is the **question**, not the evidence. `git branch -r --contains HEAD` →
`origin/main`: all three commits are **already pushed**. So this is not "safe to commit" but
**"safe to keep, or must this be reverted before the freeze"** — and a revert now needs a new
commit, not a rewrite or a force-push.

Target contents (`git diff ebea1b2 HEAD --numstat`): `App.tsx` +54/−10,
`ExplorationPanel.tsx` +79/−155 (234 lines touched), `RangeFilters.tsx` new (253 lines),
`useDebounced.ts` new (15 lines), `styles.css` +140/−15, `config.ts` +6/−2 (default DR
method PCA → UMAP). `.wtest` — an empty *tracked* file that existed at `ebea1b2` — is
deleted and gitignored; that is housekeeping, not new junk.

---

## Findings, ranked

**How they compose.** Four of these are separate doors into one room. **U2** (a dropped
clause), **U4** (any first checkbox tick), **U12** (a constant column — ~25 % of MNIST
columns) and the untouched-slider case all end with `selected == every row of the node`;
**U3** is what that end state then does — the exactly-zero characteristics chart the prior
review cycle existed to eliminate. **U9** and **U10** are why the user cannot get out of it
from that tab. Clearing `selected` on the `filtering` true→false edge (U3's missing `else`)
neutralizes the *misleading chart* for all four doors at once; it does not stop the doors
themselves from selecting the whole node while the Ranges tab is open, which is what U4's
duplicate full-node fetch costs. Both halves are worth fixing, but the `else` is the one that
stops the demo from lying.

### U1 — The Ranges tab on a whole layer returns an 807 MB response the browser cannot consume · **CONFIRMED** · NEW
`frontend/src/components/ExplorationPanel.tsx:129` · `frontend/src/App.tsx:401-403` · `backend/app.py:321-323`

The two new features compose into a request neither could produce alone. "Explore entire
layer" at layer 1 sets `explorationPath = []`, so the explored node is the **root** — a node
the panel could previously receive only if the root happened to be a leaf. The Ranges tab
then fetches *every row × every feature and target column* (`fetchRows(dataset,
node.row_indices, tableCols)`) merely to compute bounds and 28-bin histograms.

Measured against the live backend:

| request | status | size | time | server RSS |
|---|---|---|---|---|
| ids=5000, cols=794 (H8's original case) | 200 | 57.68 MB | 0.74 s | — |
| ids=20000, cols=794 | 200 | 230.82 MB | 2.94 s | — |
| **ids=70000, cols=794 (the new root case)** | 200 | **807.57 MB** | **10.55 s** | 1343 → **5348 MiB** |

`api.ts:27` consumes this with `res.json()`. V8's `MAX_STRING_LENGTH` on this machine is
**536,870,888** bytes, so an 807 MB body is past the limit; the response either throws
during decode or exhausts the tab parsing 70000 × 794 ≈ 55.6 M property slots. Either way
`.catch(() => setRangeData(null))` (`:131`) swallows it and `RangeFilters.tsx:93` renders a
permanent "Loading column values…", indistinguishable from an in-flight fetch.

**Failure scenario.** MNIST → Build → layer 1 → *Explore entire layer* → *Ranges* tab. The
tab never loads; the server spends 10.5 s and 5.3 GB producing a body that is discarded.

Most of that body is not data. `to_json(orient="records")` (`app.py:322`) repeats every
column *name* in every row, which accounts for ~69 % of the bytes:

```
records orient (what ships)          ~715 MB modelled   vs 807.57 MB measured
columnar (names once, values only)   ~222 MB            <- under V8's 536.9 MB cap
```

**Fix direction.** Compute bounds and histograms server-side (the tab needs 794 × [min,max]
plus 28 bin counts, not 55.6 M cells); failing that, switch this endpoint to a columnar
payload and cap the request. Either way, surface the `.catch` as an error state, not `null`.

---

### U2 — The `filtering` guard tests the wrong array, so `[].every` selects the whole node again · **CONFIRMED** · H7 mechanism, reachable by a new route
`ExplorationPanel.tsx:96` vs `:141-152`

`filtering = interactive && filterCols.length > 0` is computed from the *picked* columns, but
the conjunction at `:148-152` runs over `clauses` — `filterCols` after a
`.filter(c => c.j >= 0 && Number.isFinite(c.lo) && Number.isFinite(c.hi))` at `:147`. When
the two disagree, `clauses` is empty, `[].every(...)` is vacuously true for every row, and
`selected` becomes the entire node **while `filtering` is still true**.

`filterCols` is reset only on `node.id` (`:106`), and the config rail is never disabled, so
unticking a feature — or the one-click **None** button (`App.tsx:185`) — changes `tableCols`
(`:118`), refetches `rangeData` without that column, and drops the clause. Verified that
`toggleFeature` (`App.tsx:99`) does **not** null `analysis`: the panel stays mounted with an
unchanged `node.id`, so nothing resets the filter.

Measured with the component's verbatim expression over real `collectRangeData` output:

```
filterCols=['petal length'], column absent from rangeData
  clauses surviving = 0 · filtering = true · tab badge shows 1 · SELECTED = 4/4 rows
on a 98-row Iris node:  filterCols=1, surviving clauses=0, selected = 98/98
```

**Failure scenario.** Ranges tab → tick `alcohol`, narrow it to 12 of 1599 points → untick
`alcohol` in the feature picker. The selection silently jumps to all 1599 rows, no range row
is drawn (`RangeFilters.tsx:157` returns `null` for `j < 0`), yet the tab badge still reads
`1` and the header still reads "1599 of 1599 points match 1 range".

**Fix direction.** Hoist `clauses` out of the memo and derive `filtering` from
`clauses.length > 0`; prune `filterCols`/`ranges` against `tableCols` when it changes.

---

### U3 — Leaving the Ranges tab carries a whole-node selection into Characteristics → the z=0 chart · **CONFIRMED** · H7's end state, still reachable
`ExplorationPanel.tsx:157-165` (no `else`), `:96`, `:222-241`

Ticking one column with the sliders untouched makes the window the column's own
`[min, max]` (`:144` falls back to `rangeData.bounds[j]`), so **every** finite row matches.
Switching to Characteristics flips `interactive` → false, `filtering` → false and
`interactiveGroup` → `null`; the effect at `:157` is `if (interactiveGroup) setSelected(...)`
with no else branch, so `selected` keeps the whole node. Nothing else clears it — the reset
effect is keyed on `node.id` alone.

Measured, Iris, node = 98 rows, against the live backend:

```
ALL rows selected                        real 12-point lasso
  max|z_mean| non-feature = 0.000000e+00   max|z_mean| non-feature = 9.797959e-01
  max|z_std|  non-feature = 1.000000e+00   max|z_std|  non-feature = 0.000000e+00
  max|z_mean| feature     = 6.334241e-01
```

The non-feature/label bars are **exactly** zero with whiskers pinned at exactly ±1 —
REVIEW.md's H7 payload verbatim — because `raw = sub[...]` z-scores them against the node
(`backend/characteristics.py:76`), and the selection *is* the node. Feature bars survive only
because the scaler there is fit on the whole dataset (`characteristics.py:72`). With
**"Characteristics: non-feature columns only"** ticked, `shown` is exactly the collapsed set,
so every visible bar reads 0.00 under the title "Selection characteristics — vs. {pathLabel}".

SSR A/B of the real component with the exact stale state shows the two are
indistinguishable to the user:

```
B. view=characteristics, selected=all 40 rows -> "Selected points: 40", "Computing characteristics…"
D. view=characteristics, selected=12-row lasso -> "Selected points: 12", "Computing characteristics…"
```

**Severity calibration (the refutation pass narrowed this).** The chart is *uniformly* blank
only in two cases: when the explored node is the **root** — i.e. exactly the new "Explore
entire layer" on layer 1, or a leaf root — or when **"non-feature columns only"** is ticked
(it defaults to `false`, `App.tsx:24`). Measured on Wine:

```
root (6497 rows), all selected      max|z_mean| features 3.6e-15   non-features 0
3000-row sub-node, all selected     max|z_mean| features 0.447     non-features exactly 0
real 400-row subset                 max|z_mean| features 1.23      non-features 0.936
```

Feature bars survive on a proper sub-node because their scaler is fit on the whole dataset.
So on a sub-node the failure is not a blank chart but a **mislabelled duplicate**: the panel
silently redraws that node's own `rel_characteristics` under the title "Selection
characteristics — vs. {pathLabel}", i.e. it answers "how does this node differ from the
dataset?" while claiming to answer "how does this selection differ from this node?". Wrong in
a different way, and harder to notice than a flat chart. The blank case coincides exactly with
the new whole-layer feature.

**Fix direction.** Clear `selected` on the `filtering` true→false edge, or gate the
characteristics effect on `selected.length < node.row_indices.length`.

---

### U4 — The first checkbox tick selects 100% of the node and re-requests it · **CONFIRMED** · NEW, incomplete fix
`ExplorationPanel.tsx:96`, `:144`, `:157-165`, `:196-212`

The `filtering` guard defers "select everything" from tab-open to first-checkbox, but a fresh
column's default window is its full span, so the first tick still matches every row and fires
a second full-size `/api/rows` + `/api/targets`. At root scale that is **another 807.57 MB /
10.55 s / +4 GB RSS** — to render 50 table rows (`:441`). The header even says so:
"70000 of 70000 points match 1 range".

**Fix direction.** Treat a clause still at the column's own bounds as inactive; reuse the
already-fetched `rangeData` rows for the table instead of refetching.

---

### U5 — The selected-points table, its CSV and its image links come from different generations · **CONFIRMED** · M7 never fixed, now continuous
`ExplorationPanel.tsx:174-180`, `:419`, `:422`, `:441-448`

`predicate`/`targets`/`rows` are cleared only on the empty/stale branch, so during a refetch
the table paints generation-A row *values* while `rowId = node.row_indices[selected[i]]`
resolves against generation-B `selected`. The comment at `:439-440` asserts an invariant the
code does not hold.

A 2 s slider drag emits up to 200 change events; the 120 ms debounce collapses them to
**~16 selection updates per drag**, against 1 per lasso gesture before. Measured server-side:

```
Iris    7 cols  node=  98 | ranges scan   2.0 ms | table refetch    1.5 ms
Digits 74 cols  node= 500 | ranges scan  10.4 ms | table refetch    4.3 ms
MNIST 794 cols  node=3000 | ranges scan 461.7 ms | table refetch  166.8 ms (11.54 MB)
```

On MNIST the 167 ms refetch **exceeds the 120 ms debounce**, so during a sustained drag the
table never catches up — the desync is the steady state, not a flash. Clicking a row opens
the image of a point that is not the one whose values are displayed. Trailing rows give
`selected[i] === undefined` → `rowId` undefined → the click silently does nothing (benign by
accident, not design).

**Fix direction.** Clear `rows`/`targets` above the fetches, as `LayerSide` and `PointImage`
already do; or carry the ids in the response object and render `rowId` from those.

---

### U6 — "Explore entire layer" defeats the `staleSelection` guard by construction · **CONFIRMED** (wasted requests; no wrong render) · pre-existing guard, newly deterministic
`ExplorationPanel.tsx:171` · triggered by `App.tsx:362`, `:401-403`

`staleSelection` only tests `i >= node.row_indices.length`. The new button always swaps in an
**ancestor**, whose `row_indices` is a strict superset, so every stale index is in range and
the guard never fires. `<ExplorationPanel>` has no `key` (`App.tsx:428`), so it does not
remount; React batches the reset effect's `setSelected([])` until the flush ends, so the
fetch effect at `:174` — which re-runs because `node.id` is in its deps — closes over the
*previous* node's positions.

**Failure scenario.** Drill to a leaf, lasso 3000 points, click *Explore entire layer* →
`/api/predicate`, `/api/targets` and `/api/rows` all fire with child-relative indices resolved
against the root's `row_indices` (~35 MB on MNIST). The backend answers 200 — the ids are
perfectly in range and it cannot tell. Reproduced in a mounted-component harness (child
`[10..14]`, selection `[0,1,2]` → ancestor `[90,91,92,10,…]`): exactly three requests carry
the mismatch, including `rows rowIds=[90,91,92]` — rows the user never selected.

**Severity, corrected by the refutation pass.** The `cancelled` cleanup sets before any
response lands (final harness state: `Selected points: 0`, 0 table rows), so **nothing wrong
is ever rendered**. The cost is three discarded round-trips per superset navigation, one of
them a multi-second predicate, on an already-saturated threadpool. `staleSelection` (`:171`)
and the fetch effect's dep array are **byte-identical at `ebea1b2`** (old lines 177/222) — the
diff did not change the guard. What it adds is the app's first one-click control that
navigates deterministically to a superset node, which is what turns an occasional accident
into a repeatable one.

**Fix direction.** Store the node id the selection was made in and compare identity, rather
than inferring staleness from length — or put `key={node.id}` on `<ExplorationPanel>`.

---

### U7 — The UMAP default removes the only tell for H4's method mislabel · **CONFIRMED** · NEW
`frontend/src/config.ts:15` · `ExplorationPanel.tsx:246`, `:407` · `charts/ProjectionScatter.tsx:189-190`

Reproducibility is **not** the problem: both blockers are fixed and hold under the new
default — `dim_reducer.py:85` uses `svd_solver="covariance_eigh"` (no RNG) and
`dim_reducer.py:109` passes `init="pca"`. Verified identical embeddings across repeat runs
on five node shapes (`max|diff| = 0.0000e+00`, including wine 6497×11 and olivetti 400×4096).
The comment's justification also checks out: `src_research/benchmark_workflow.py:163` really
does call `default_config()`, so holding the Python default at PCA is deliberate.

The problem is what the flip does to H4. REVIEW.md's H4 bullet was "switch Method PCA→UMAP →
axes relabel over unchanged coordinates **and the variance strip vanishes**" — the vanishing
strip was the tell. `showVariance` requires `config.method === "PCA"` *and* the backend only
ever emits `explained_variance_ratio` for PCA (measured: UMAP build → `leaves_with_variance=0`).
So under the new default the strip is never present, and switching the live rail knob
UMAP→PCA relabels the axes to "PC1/PC2" over unchanged UMAP coordinates with **no visible
difference from a real PCA run**. `method={config.method}` at `:407` is live rail state, not
`shownAnalysis.meta`. A screenshot taken then is a PCA-captioned UMAP figure with no tell.

Knock-on: `PcaVarianceBar` has exactly one render site, so explained variance is now dead
everywhere on the default path; and build cost rises (wine warm: PCA 7.6 s → UMAP 23.3 s;
Digits 1.02 s → 4.98 s). The default experience is now a configuration no research artifact
characterizes — H1a/H2b/benchmark_workflow all describe PCA exploration embeddings.

**Fix direction.** Label the scatter from `shownAnalysis.meta.config.method`, not live
`config.method`; decide deliberately whether the frozen demo should default to a method the
thesis does not characterize.

---

### U8 — No AbortController + `config` in the request effect's deps → concurrent full-node fetches stack to the container limit · **CONFIRMED** · M6 never fixed, newly at scale
`ExplorationPanel.tsx:216` · `App.tsx:94-96` · `api.ts:17-28`

`config` is still in the fetch effect's dep array — **byte-identical to `ebea1b2`** (old line
222), so this is not a regression, but the Ranges tab routinely makes `selected` the whole
node, which is what turns it into a real cost. `patchConfig` mints a new object per change,
and `grep -rn "AbortController\|signal" frontend/src` → 0 hits, so nothing cancels the
in-flight request. Measured with the real component, 3 simulated keystrokes:

```
Predicate tab       9 requests  (predicate + targets + rows, x3)
Characteristics tab 9 requests  (the predicate still fires, though nothing renders it)
Ranges tab          6 requests  (predicate correctly skipped -> 2 per keystroke)
```

On the Ranges tab those two requests per keystroke each carry the entire node (U2/U4/U12).
Note the sibling characteristics effect was **already** narrowed to `config.normalize` at
`ebea1b2:247` — that half predates this diff and is not one of its improvements.

Measured, MNIST root:

| concurrency | per-request | server RSS peak |
|---|---|---|
| 1 | 10.55 s | 5348 MiB |
| 2 | 21.4 s | 8981 MiB |
| 3 | 31.9 s | **12083 MiB** |

`docker-compose.yml` sets `mem_limit: 12g` = 12288 MiB. Three spinner clicks put the process
at **98 % of the container limit**, before the tree cache or a build.

**Fix direction.** Thread an `AbortController` through `api.ts::post` and abort in each
cleanup; depend on the config fields actually read, as `:241` already does.

---

### U9 — The lasso is dead across the whole Ranges tab, so U2/U3's stuck selections cannot be cleared in-tab · **CONFIRMED** · **REGRESSION introduced by this diff**
`ExplorationPanel.tsx:409` vs `:410`

The diff changed one of two adjacent props and not the other. Verified against the old file:

```
ebea1b2:414  onSelect={interactive ? () => {} : setSelected}
ebea1b2:415  selected={interactive ? [] : selected}      <- both on `interactive`
HEAD:409     onSelect={interactive ? () => {} : setSelected}   <- unchanged
HEAD:410     selected={filtering  ? [] : selected}       <- moved to `filtering`
```

At `ebea1b2` the pair was consistent. Now, with the Ranges tab open and zero columns picked,
`selected` takes the else branch so the previous lasso is still ringed, while `onSelect` is
still the no-op. `ProjectionScatter` has no disabled state — the overlay rect is always
`cursor:"crosshair"; pointerEvents:"all"` (`:296`) and the drag handler always draws the
polygon — so the lasso looks live, and mouseup changes nothing. The "click empty space
clears" path at `:169` routes through the same no-op, and `Clear` never touches `selected`
(U10), so there is no way out of a stuck selection from that tab.

**Failure scenario.** Lasso 2 points → open Ranges → both stay ringed → draw a fresh lasso
around different points → "Selected points: 2" never changes and no error appears.

**Fix direction.** `onSelect={filtering ? () => {} : setSelected}`, matching the next line.

---

### U10 — `Clear` resets the filter UI but not the selection it produced · **CONFIRMED** · NEW
`ExplorationPanel.tsx:315-318` · `RangeFilters.tsx:111`, `:167`

`setFilterCols([]); setRanges({})` leaves `selected` populated, and `interactiveGroup` has
become `null` so `:157` refuses to clear it. SSR-rendered result: the header reads **"No
ranges yet"** while `<h3>Selected points: 47</h3>` and the 47-row table are untouched and the
plot flips those 47 back to selection styling. The Clear button is then
`disabled={active.length === 0}` so it cannot be pressed again, and the lasso is dead (U9) —
there is no in-tab way to clear the 47. The per-row `×` has the same effect when it removes
the last column.

**Fix direction.** Add `setSelected([])` to `onClear` — same fix as U3.

---

### U11 — The dual range slider becomes undraggable at `lo = hi = max` · **CONFIRMED** · NEW
`RangeFilters.tsx:227-244` · `styles.css` `.range-track input[type="range"]`

Both inputs are absolutely stacked on one 30 px track at full width, `pointer-events: none`
on the input and `auto` on the thumb, with no `z-index`. SSR confirms the **maximum** input is
DOM order 1, so at equal position it paints above the minimum and captures the pointer:

```
ranges={}          -> order 0: "a minimum" value=0   order 1: "a maximum" value=10
ranges={a:[10,10]} -> order 0: "a minimum" value=10  order 1: "a maximum" value=10
```

(Both are `position: absolute` with `z-index: auto`, so paint order follows tree order.)

Dragging the min thumb fully right yields `[Math.min(v, hi), hi]` = `[max, max]`. Both thumbs
now sit at 100 %, only the max thumb is reachable by pointer, and its handler computes
`[lo, Math.max(v, lo)]` = `[max, max]` for every `v` because `lo` is already `max`. The window
is frozen on a degenerate `[max, max]` and the selection collapses to points equal to the
column maximum. The mirror state `lo = hi = min` *is* recoverable (the max thumb can still
move right), so the trap is one-sided.

**Escape routes, checked.** The freeze is **pointer-only**: neither input carries a
`tabindex` override and `styles.css:545-546` explicitly styles `:focus-visible` on both
thumbs, so the min input stays keyboard-focusable even while visually covered, and moving it
recovers the window (verified: `11 – 11` → `3 – 11`). **Clear** also works. The row's `×`
does **not** — `onRemove` edits only `active`, so `ranges[col]` survives and re-ticking the
column restores `[11, 11]` (that is U16). There is no click-to-jump on the track either: the
inputs are `pointer-events: none` and only the thumbs accept the pointer, so clicking the bar
does nothing. So this is a mouse/touch dead-end with two non-obvious ways out — but a slider
that stops responding to dragging is the failure a user will actually hit. The old code laid
the two sliders out in separate grid columns, so they could never overlap.

**Fix direction.** Raise the min input's `z-index` when it sits at the top of the span, or
split the track's hit area at the midpoint between the thumbs.

---

### U12 — A constant column passes the `usable` gate, matches every row, and paints an empty highlight · **CONFIRMED** · NEW (the contradiction), pre-existing in kind (the selection)
`RangeFilters.tsx:131`, `:190-193`, `:214`

`usable` tests only `Number.isFinite(bounds[j][0])`, so a column whose values are all
identical is offered as filterable. Measured:

```
constant column   bounds=[7,7]      a:usable
all-null column   bounds=[NaN,NaN]  a:DISABLED   <- correctly gated
empty node        bounds=[NaN,NaN]  DISABLED     <- correctly gated
```

Replaying the component's verbatim expressions on a constant column:

```
bounds(px_0) = [0, 0]   usable (gate at RangeFilters.tsx:131) = true
surviving clauses = 1  ->  SELECTED = 4/4 rows
span=0  loPct=0  hiPct=0  step=1   (min===max slider: no thumb travel)
histogram bins drawn "is-in" = 0/28
```

`span = 0` makes `at()` return 0 for everything, so `loPct = hiPct = 0` and every bin's centre
`((i+0.5)/28)*100 ≥ 1.79` fails `centre >= loPct && centre <= hiPct`; `step` falls back to 1 on
a `min === max` slider so neither thumb can move. Meanwhile the clause is trivially true for
every row. The user sees "nothing is in range" and gets everything — then U3 applies.

Reachability on the image datasets is not a corner case — measured on the local MNIST IDX:

```
MNIST train (60000, 784): constant columns over the WHOLE dataset = 67/784
  a 3000-row cluster-sized subset: constant columns = 147/784  (19%)
  a  500-row cluster-sized subset: constant columns = 200/784  (26%)
```

So on a realistic MNIST node roughly **one in four offered columns** is a live instance of
this: ticking it silently selects the entire node, draws an empty in-range highlight, and
hands U3 its whole-node selection.

**Fix direction.** Treat `min === max` as unusable, or render it as a single-value chip
rather than a slider.

---

### U13 — Ordinary tab switching re-POSTs for an unchanged selection · **CONFIRMED** · pre-existing dep, newly activated
`ExplorationPanel.tsx:216`, `:92`

`interactive` is in the fetch effect's deps and is now `view === "ranges"` — i.e. ordinary tab
navigation. Clicking Predicate → Ranges re-issues `/api/targets` and `/api/rows` for an
unchanged selection and node and discards the predicate; clicking back refits it in full
(4.2 s on MNIST per M6's measurement). Re-entering Ranges with a live filter costs three
full-node-scale fetches.

**Fix direction.** Split the predicate effect (deps `interactive`, `scope`, config fields)
from targets/rows (deps `selected`, `node.id`, `tableCols`).

---

### U14 — A settled slider move that changes no point's membership still refetches and closes the open image · **CONFIRMED** · NEW
`ExplorationPanel.tsx:139-153`, `:157-165`, `:111-113`

`interactiveGroup` is a `useMemo` returning a fresh array and `:159` calls `setSelected` with
another fresh array, so `Object.is` fails at both the state-bailout and the effect-dep
comparison even when the contents are byte-identical. A simulated 40-event drag whose final
window keeps membership identical emitted **0 requests during the drag** (the debounce works)
but **2 after the settle** — `/api/targets` and `/api/rows` — with `selected` unchanged.
Widening a slider through a gap in the histogram is the everyday case. The same identity
change fires `:111`, so an open point image closes itself on a nudge that changed nothing.

**Fix direction.** Compare against the previous membership before calling `setSelected`.

---

### U15 — One painted frame of the previous node's filter membership on the new node's points · **PLAUSIBLE** · low
`ExplorationPanel.tsx:123-135`, `:139-153`, `:408`

**Failure scenario.** Filter inside cluster A, then click cluster B: B's scatter paints one
frame coloured by A's "Matches filters"/"Other" array, applied positionally, before
repainting. On the render that swaps `node`, `setRangeData(null)` has not landed yet and the
memo's deps are unchanged, so the cached old array is handed to the scatter; positions past
the old node's length give `undefined` → the "Other" hue. Passive effects run after paint in
React 18, so the frame is real, but it is sub-100 ms — hence PLAUSIBLE rather than CONFIRMED:
I could not observe the paint without a browser.

**Fix direction.** Include `node.id` in the memo, or gate the scatter on `rangeData` having
been fetched for the current node.

---

### U16 — Removing a column keeps its window, so re-ticking silently restores it · **CONFIRMED** · NEW, ambiguous intent
`RangeFilters.tsx:142-144`, `:167` vs `ExplorationPanel.tsx:314`

`onActive` edits only `filterCols`; the `ranges` entry survives. SSR-verified: with a
surviving `ranges={a:[2,3]}` the re-added row renders `a | 2 – 3` rather than the column's
full `1 – 5` bounds, so 2 of 5 points match with no visible reason why the window is not
full-width.

**Failure scenario.** Narrow `alcohol` to a tight band, remove it with the `×`, then re-tick
it from the column list: the selection immediately drops to that old band instead of the full
column, with nothing on screen explaining why. Genuinely ambiguous whether "remembers your
window" is intended; it is undiscoverable either way.

**Fix direction.** Delete the `ranges` key in `onActive`'s removal branch, or say so in the
row's `title` if the memory is deliberate.

---

### U17 — `config.ts` is now the sole source of truth for `method`, and nothing would catch a typo · **CONFIRMED** · H10 unfixed, newly load-bearing
`frontend/src/config.ts:15` · `backend/app.py:66-70`, `:186`

The shipped value is correct — measured, `method: "UMAP"` builds clean with 0 null
embeddings. The exposure is that `_merge_config` is still a bare `dict.update` and the
response echoes the client's own partial rather than the effective merge:

| config sent | result |
|---|---|
| `method: "UMAP"` (shipped) | 200, 0 null embeddings — clean |
| `method: "UMAPP"` (near-miss) | **200 `done`, 3/3 nodes `embedding_original: null`**, nothing rendered, no message |
| `methd: "UMAP"` (misspelled **field**) | **200 `done`** — silently ran PCA while the UI says UMAP |
| `hierarchical_layers: "1"` | 200 with `status: "error"` |

`pca_components` is confirmed still dead (M16): declared in four places, read nowhere —
`_pca` is only ever called with a hard-coded `n_components=2`.

**Fix direction.** Reject unknown keys and non-`Literal` methods in `_merge_config`; put the
merged config in `payload["meta"]["config"]`.

---

## Regression checklist

| Prior finding | Verdict | Evidence |
|---|---|---|
| **H2** (noise unreachable) | **n.a. — partially mitigated** | "Explore entire layer" explores `node.row_indices`, the parent's full row set, a strict superset of its clusters' union. "Exploring every point in this layer" is literally accurate, and this is the first UI path from which noise rows can be lassoed at all. No misclaim. |
| **H4** (panels off live rail state) | **NEVER FIXED; one bullet made worse** | `App.tsx:122` still reconciles only `meta.dataset`; `meta.feature_cols`/`meta.config` still referenced nowhere. `exploreWhole` itself is sound across every `nLayers` transition. But U7: the UMAP default removes the variance-strip tell, and U2: live `featureCols` now silently empties a filter conjunction. |
| **H6** (no ErrorBoundary + reachable throw) | **STILL FIXED** | `ErrorBoundary.tsx` exists with `getDerivedStateFromError`/`componentDidCatch`; `main.tsx:16-18` wraps `<App/>`. `CharacteristicsBar.tsx:155` hoists `hovered`, guarded at `:158` and `:279`. No new render throw: `ExplorationPanel.tsx:144`'s `?? [NaN, NaN]` covers `bounds[-1]`, `RangeFilters.tsx:157` guards `j < 0`. |
| **H7** (empty conjunction → whole node → z=0) | **PARTIALLY FIXED — end state still reachable, payload narrower** | The named route is genuinely closed: the tab *is* the mode, so the filter UI can no longer be off-screen while `interactive`. But U2, U4 and U12 all land on `selected == every row`. Non-feature bars still collapse to exactly `0.000000e+00` with `z_std` exactly `1.0`; feature bars now survive on a sub-node (scaler fit on the whole dataset), so the *uniformly* blank chart needs the root node or `charNonFeatureOnly` — and the new whole-layer feature supplies the root. The comment at `:93-95` claims the hole is closed; it is not. |
| **H8** (threadpool/payload saturation) | **NEVER FIXED; materially worse** | The slider half is genuinely fixed (0 requests during a drag). But U1 raises the ceiling 14× on the same dataset (57.68 MB → 807.57 MB) and U8 shows 3 concurrent fetches reaching 98 % of the 12 g container limit. |
| **H10** (config unvalidated) | **NEVER FIXED; newly load-bearing** | U17. `methd`/`UMAPP` both return 200. |
| **M2** (`/api/rows` columns) | **ALREADY FIXED before this diff** | `app.py:310` distinguishes `None` from `[]`; `:314-319` returns 400 for unknown columns. The diff is the first code to send `target_*` names there; had M2 not landed first, this diff would have turned it into a 500. |
| **M6** (whole `config` in deps) | **NEVER FIXED — byte-identical to `ebea1b2`** | `:216` still ends `…, tableCols, config]`. The `config.normalize` narrowing at `:241` was already there. Not a regression, but U8 makes it cost more. |
| **M7** (stale data during refetch) | **NEVER FIXED; materially worse** | U5 — ~16 selection updates per drag vs 1 per lasso, and on MNIST the refetch outruns the debounce so the desync is continuous. |
| **M8** (`.catch` swallows diagnostics) | **NEVER FIXED; count unchanged** | 5 catch sites before (`135/198/210/218/241`), 5 after (`131/192/204/212/235`). `setZData(null)` was renamed, not added. *(I initially mis-scored this as a new sixth site.)* |
| **M22** (F1 for an empty conjunction) | **STILL FIXED (frontend)** | Measured with all 98 rows selected: `scope=local → f1=1.0000, n_features_used=0/4`, so the guard at `:364` fires and the hint is shown. `scope=global → 2/4` is a legitimate predicate, correctly not suppressed. Only the Predicate tab renders a predicate, so the guard covers every tab that can. Backend still emits `f1=1.00` with 0 clauses, and local scope still legends the node's range as "Global range" — both unchanged. |
| **B0 / B5** (DR nondeterminism) | **FIXED, verified under the new default** | `init="pca"` and `svd_solver="covariance_eigh"`. Repeat runs identical (`max\|diff\| = 0.0000e+00`) on five node shapes. The PCA→UMAP flip does **not** reintroduce nondeterminism. |

---

## Checked and clean

Recorded so absence of findings is distinguishable from absence of looking.

- **Index space is sound.** "Explore entire layer" hands the panel a **single real `TreeNode`**
  (`explorationPath = parentPath`), never a synthetic union across siblings. No per-node
  remapping is required and none is missing. Every index crossing was traced: the `rangeData`
  fetch sends global ids to a global endpoint; `interactiveGroup → setSelected` emits
  node-relative positions; `/api/predicate`, `/api/targets`, `/api/characteristics` receive
  `row_indices` (global) + `selected_local_indices` (node-relative), exactly what
  `predicate.py:46` / `targets.py:53` / `characteristics.py:60-70` consume; the table maps to
  global via `node.row_indices[i]` before sending.
- **`/api/rows` preserves request order** (`df.iloc[req.ids][cols]` is positional), verified
  empirically including duplicates and a relabelled index — so `rangeData.values[i]` ↔
  `node.row_indices[i]` holds.
- **`showScores` / `scoresShownByLayer` is correct in every reachable configuration**,
  including the new whole-layer case: the explored node is the layer-above's selected child,
  whose `LayerSide` renders its tiles; only an empty path shows them in the panel.
- **`exploreWhole` cannot desync.** Every `setTreePath` goes through `navigate`, which always
  writes both values, so the flag cannot outlive its path. `exploringHere` is true for exactly
  one `L`. Lowering `hierarchical_layers` leaves it inert, converging on the same node the
  fall-through picks.
- **`useDebounced` is a correct trailing debounce**, and the debounced value is the *only*
  slider path to the network (`RangeFilters` gets raw `ranges` for thumb positions and the
  printed label only). H8's "one un-debounced request per slider input event" is genuinely
  fixed. Its `useState(value)` initializer is benign: at mount both hold the same object, so
  the first `setSettled` hits React's bailout.
- **`cancelled` flags are present and correct in all three async effects the diff touches**,
  including the `.catch` arms; `useDebounced` clears its timer on unmount. No response can
  land after a node change and overwrite fresh state.
- **No NaN reaches a request body.** `ranges`/`settledRanges` are consumed only by
  `interactiveGroup` and never serialised; `numeric()` matches the backend's NaN→`null`
  coercion exactly. `Number(true) === 1` is correct for one-hot label columns.
- **Negative indices are unreachable from the new code** — every index emitted is an array
  position. (H10's unguarded negative branch on `/api/targets`/`/api/characteristics` is still
  open, just not newly reachable.)
- **Degenerate bodies answer 400, not 500**: `feature_cols: []` (reachable via the *None*
  button), `row_indices: []`, `ids: []`.
- **Three prior LOW findings are fixed by the rewrite** — `[Infinity, -Infinity]` bounds on an
  empty node (now `[NaN, NaN]` + a disabled checkbox), `Math.min(...col)` `RangeError` past
  ~125 k points (now an explicit loop), and `Number(null) || 0` reading a missing cell as a
  real zero (now `numeric()` → NaN).
- **No new instance of the prior LOW "array-index keys on re-filtered arrays" finding.**
  `RangeFilters` keys the column list on `c`, the groups on `g.key` and the range rows on
  `col`; the only `key={i}` is the 28 histogram bins, a fixed-length array that never
  reorders.
- **`tsc --noEmit` is clean** (exit 0, strict), re-confirmed after the review.
- **`frontend/dist` is NOT stale** — built 01:13:30 against source last touched 01:12:17, and
  the bundle contains both new features. It remains untracked, so the container still serves
  whatever the last local build produced.

---

## How this was verified

No browser automation exists in this repo, so behaviour was established three ways, in
descending order of directness:

1. **Live backend measurement** — the server was started and the exact payload shapes the new
   code emits were POSTed: `/api/rows` at 5 000 / 20 000 / 70 000 ids, `/api/characteristics`
   with a whole-node vs a lasso selection, `/api/targets`/`/api/predicate` at root scale,
   config near-misses (`UMAPP`, `methd`), and repeat-run determinism for PCA and UMAP. All
   numbers quoted above are measured, not estimated. The repo was left untouched; the server
   was stopped afterwards.
2. **SSR A/B rendering of the real components** (React 18 `renderToStaticMarkup`, bundled with
   the repo's own esbuild) with the exact stale state injected — used for U3 (a whole-node
   selection is indistinguishable from a lasso), U10 ("No ranges yet" above a populated
   table), U11 (DOM order of the two range inputs) and U16.
3. **Replaying the components' verbatim expressions** over real `collectRangeData` output —
   used for U2, U12 and U14, where the defect is in pure derivation logic.

Effect *ordering* claims (U6, U15) rest on React 18's documented semantics — passive effects
run in hook-declaration order within one flush, and state set there lands in a later render —
plus the verified absence of a `key` on `<ExplorationPanel>`; there is no jsdom/vitest in
`frontend/node_modules` to execute them, and none was installed.

Four independent finders covered state/effect lifecycle, index-space correctness, regression
against the prior review, and the backend contract. A fifth pass then tried to **refute** the
six load-bearing findings, defaulting to REFUTED, and built a happy-dom + React 18 harness
that mounts the real `ExplorationPanel` with only `../api` and the chart children stubbed —
so U2, U6, U9 and the config-keystroke counts were observed on the actual component with
effects running, not modelled. None of the six was refutable; two were **narrowed**, and both
corrections are folded in above:

- **U3** — "every bar zero" holds only at the root node or with the non-feature toggle on;
  on a sub-node it is a mislabelled duplicate chart instead. Narrowed.
- **U6** — costs three discarded requests, never wrong pixels, and the guard itself is
  unchanged by the diff. Downgraded.

Three disagreements were settled against the actual old file rather than by majority:

- Finder 4 listed the stale-column case as "clean" because it cannot crash or emit a
  malformed clause — true, but it does not address the vacuous-conjunction consequence that
  four other lines of evidence confirm. **U2 stands.**
- The refutation pass credited the diff with narrowing the characteristics effect to
  `config.normalize`. `git show ebea1b2:...` line 247 shows it was **already** narrowed.
  Finder 3 was right; that is not an improvement of this diff.
- I mis-scored M8 as gaining a sixth `.catch` site; it did not — the site was renamed. The
  checklist reflects the corrected count.

Correspondingly, U9 was **upgraded** from "pre-existing, newly activated" to a regression once
`ebea1b2:414-415` showed both props were consistent before the diff and only one moved.

## Recommendation

### The verdict as asked: **NO-COMMIT**

The task asked for a commit / no-commit call on these changes entering the frozen build.
Stated in those terms, unambiguously: **NO-COMMIT.** Had this been a pre-commit gate, U1, U3
(with its four one-click doors) and U7 would each independently have been reason to hold it.

### The verdict as the repository actually stands: **revert or fix forward**

That gate no longer exists. Per the scope correction at the top, the rewrite, both new files
and the whole-layer feature were committed before this review began and all three commits are
pushed to `origin/main`, so "don't commit" is not an action anyone can still take. The live
choice is:

- **Revert** — `git revert` the three commits (a new commit; do not rewrite pushed history).
  Correct if the freeze is imminent and the two features are not required for it.
- **Fix forward** — cheaper, and the better option unless the freeze is hours away. The
  blocking set is small and localized (see below); U1 is the only one needing real design
  work, and even it has a narrow form (cap the Ranges fetch and surface the error) that
  unblocks the freeze without the server-side histogram rewrite.

Doing nothing is the one option the evidence does not support: the flagship demo path
(whole layer → Ranges) does not work on the image datasets, and the exactly-zero
characteristics chart is the specific misreading the previous review cycle was run to
eliminate.

Nothing here corrupts stored analysis results: the tree, the row-order alignment and the
determinism fixes are all intact, and the index space of the new feature is correct. The
defects are in what the UI *selects* and *claims*.

Three items should block the freeze:

1. **U1** — the headline combination of the two new features (whole layer + Ranges) does not
   work on the image datasets: an 807 MB response, past V8's 536.9 MB string cap, which
   either throws during decode or exhausts the tab, and which the `.catch` presents as a
   permanent loading spinner. This is the flagship demo path, and it costs the server 10.5 s
   and 5.3 GB to produce a body that is then discarded.
2. **U3, and the four doors into it (U2, U4, U12, the untouched slider)** — H7's payload is
   reachable again. A chart of exactly-zero bars titled "Selection characteristics" is the
   specific misreading the prior review cycle was run to eliminate, and the new comment at
   `ExplorationPanel.tsx:93-95` asserts it is fixed. U12 makes it a **single click** on
   roughly a quarter of the columns MNIST offers, and U9/U10 remove the ways out.
3. **U7** — flipping the default to UMAP is defensible on determinism grounds and correctly
   justified in its comment, but it silently removes the only visual tell for H4's
   method-mislabel and puts the default demo on a configuration no thesis experiment
   characterizes.

The cheapest fix-forward is small and well-localized: hoist `clauses` and derive `filtering`
from it (U2); `setSelected([])` on the `filtering` true→false edge and in `onClear` (U3, U10);
change `interactive` to `filtering` at `:409` — a one-word regression this diff introduced by
moving the adjacent prop and not this one (U9); cap or columnarize the Ranges fetch (U1);
and either label the scatter from `shownAnalysis.meta.config.method` or revert `config.ts`
(U7). U11 and U12 are two-line guards. The remainder (U5, U8, U13, U14) are pre-existing
inefficiencies the new tab amplifies and can be scheduled after the freeze.

Also worth doing before the freeze, though not defects: the three commits are all titled
"gogitgo shortcut" with no description of either feature; the features are undocumented in
`PLAN.md` and `README.md`; and `REVIEW.md`'s H7 entry still describes the deleted
`InteractiveFilters`/`filterFeatures` code, so the review artifact no longer matches the code
it certifies.
