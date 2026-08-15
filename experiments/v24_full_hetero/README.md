# V24 Full Heterogeneous Graph Probe

V24 keeps every topology node and valid edge inside a disjoint graph per work
order. It does not merge repeated physical entity IDs across orders and does
not use raw RID, station name, IP, title, or location as device features.

The first cloud phase is deliberately bounded:

1. Run a worst-order full forward/backward memory smoke test.
2. Train seed `20260803` with five honest outer folds. An inner fold chooses
   the epoch count; the untouched outer fold is evaluated once after refit.
3. Stop unless fixed-K rank-only OOF is positive in at least three folds and
   the worst fold is at least `-2 TP`.
4. Never generate a submission during the probe phase.

Cloud command:

```bash
python run_probe.py \
  --data-root /path/to/v24_dataset \
  --output /root/work/v24_outputs \
  --device auto
```

The online champion remains the immutable `0.906324` file until a later full
three-seed gate passes. V19 full is rejected after the probe scored `0.905373`
(`-1 TP`).

## Probe result

The Ascend 910B probe completed on 2026-08-04 for seed `20260803`.

- Full training graph: 279,836 nodes and 490,685 raw edges.
- Worst-order smoke: 3,079 nodes, 14,465 edges, finite gradients, and about
  588 MiB peak device memory.
- Honest fixed-K OOF delta: `-2 TP` over a 2,826 TP baseline.
- Fold deltas: `[-1, -1, -2, 0, +2]`; only one of five folds improved.
- Bootstrap 95% lower bound: `-12 TP`; 21 work orders changed.

The first-stage gate failed (`total >= +1`, at least three improved folds, and
no fold below `-2`). V24 is terminated before the remaining two seeds. No
submission file was generated and the `0.906324` champion remains unchanged.
The machine-readable result is in `probe_gate_report.json`.
