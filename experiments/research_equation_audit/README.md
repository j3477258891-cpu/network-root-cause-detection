# Research equation audit

This directory contains a read-only reconstruction of the real leaderboard
evidence. It does not create a submission CSV and does not overwrite V120 or
V121 artifacts.

The audit includes only:

- V27/V28 batches explicitly marked `scored`;
- V29 and V30 entries explicitly marked `verified` online;
- the scored records recorded by the V104/V117/V118/V119/V120/V121 ledgers;
- the one explicitly scored V75 block row (`v75_block_01_row_01`), whose state
  entry omits a path and is resolved from its probe id;
- the current V121 `probe_01_exploratory_best.csv` result.

Inferred branches, OOF reports, posterior files and offline simulations are
listed as excluded and are not equations. V30's `verified` entries do not
contain a path, so the script resolves them deterministically from
`probe_id`, then verifies the file hash.

## Run

```powershell
$env:PYTHONPATH = 'D:\zgyidong\.deps'
$py = 'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py experiments\research_equation_audit\audit_real_scores.py
```

Outputs:

- `real_scored_records.json`: unique scored files, P/F1/TP and provenance;
- `equation_system.json`: binary label equations and test-universe variables;
- `candidate_bounds.json`: MILP min/max delta for the V120/V121/V122 union;
- `fixed_labels.json`: fixed 0/1 labels for candidate nodes when the MILP can
  prove them;
- `observed_differences.json`: baseline diffs and directly fixed one-swap
  actions;
- `summary.json`: counts, positive-safe candidates and fixed negative/zero
  actions.
- `audit_v16_top20.py` / `v16_top20_bounds.json`: the 20 highest V16
  `seed_min_margin` swaps checked against the same scored equations and the
  current 1,035-root baseline.

The displayed leaderboard score is converted with
`TP = round(F1 * (1044 + P) / 2)`. A candidate is called equation-safe only
when its MILP `min_delta` is `+1`; model probabilities alone never qualify it.
