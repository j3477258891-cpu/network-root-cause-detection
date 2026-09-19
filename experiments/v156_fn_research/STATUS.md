# V156 FN-Research — 24h Track

**Purpose.** Determine whether a fold-honest classifier can score *unselected*
alarms well enough that adding its top-K picks beats the strongest existing
control on the exact F1 objective. This is the only remaining in-pool route to a
larger gain. It consumes **zero submission quota** unless it passes the gate.

**Clock.** Deadline = start + 24h. Retries do **not** reset the deadline.

**Reuse (read-only).**
- V25 bundle `D:/zgyidong/experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz`
  (`train_v11/v13/v19`, `train_labels`, `train_alarm_ptr`, `train_station_folds`,
  `train_alarm_x`, `train_path_node_type/edge_type/length`).
- V152–V155 station-isolated folds (`train_station_folds`) and cross-fit baseline.
- V153 candidate pool — a new group of ≤16 nodes must **not** overlap the 17 candidates.

## Fixed comparisons

| # | Feature set | Notes |
|---|---|---|
| 1 | `control` | existing CatBoost / V38 add-action scores (v11,v13,v19) |
| 2 | `control+disagreement` | + pairwise abs-diffs and std across the three logits |
| 3 | `control+graph` | + `alarm_x` + `path_node_type` + `path_edge_type` + `path_length` |
| 4 | `combined` | 1+2+3 (semantic embeddings added only with full outer-fold isolation) |

Parameter / calibration / add-budget is chosen **only on the inner split**.
Budget sweep is fixed at **8 / 16 / 32 / 64 nodes per 546 orders** (never a
"70% precision" proxy).

## Gate (all must hold to earn a submission)

1. 5 folds, **≥ 3 folds** with positive F1 gain.
2. Pooled gain **>** the strongest old control.
3. Worst fold **≥** that control's worst fold.
4. **2000 paired bootstrap** (station-grouped): both the **absolute** gain and the
   **relative-to-control** gain have a **95% lower bound > 0**.
5. Budget chosen on the inner split; test actions pass the historical equations,
   confirmed labels and the risk pool.

Only **one** pure-add group of **≤ 16** nodes on **distinct orders** (disjoint from
the 17 candidates) may be built. The group must additionally beat the best
existing-pool extra probe on **conditional expectation** before it may take an
extra probe slot. If the gate fails, times out, or the calibration is not
comparable → **do not submit; quota returns to the existing pool.**

## 0.94 restated

0.94 must be recomputed by the **actual action count**: the p03 pure-add route
must satisfy `K > 12.71 + 0.47·N`. The fixed-prediction "net +13 TP" is *not* the
add threshold.

## Files
- `research.py` — harness (fold-honest cross-fit, budget sweep, bootstrap).
- `results.json` — per-comparison budgets and deltas.
- `STATUS.json` — gate decision once the run completes.
