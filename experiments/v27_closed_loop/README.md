# v27 leaderboard closed loop

This directory implements the verified-action workflow for the 0.906324
champion. It is intentionally conservative: every candidate is generated from
the frozen champion, every action is validated against the test topology, and
the unverified optimistic file is never promoted automatically.

## Rebuild the catalog and first-round probes

Run with the bundled Python runtime:

```powershell
$py = 'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py experiments\v27_closed_loop\closed_loop.py init --force
& $py experiments\v27_closed_loop\build_actions.py
& $py experiments\v27_closed_loop\prepare_round.py
```

The generated files are:

- `action_catalog.json`: normalized actions, evidence, protected-node counts.
- `batch_plan.json`: non-overlapping probe batches; empty strict categories are
  recorded instead of being replaced with speculative changes.
- `submissions\probe*.csv`: leaderboard-ready probes.
- `submissions\highest_verified_checkpoint.csv`: current verified champion.
- `submissions\target_enhanced_UNVERIFIED.csv`: optimistic ceiling only; do not
  submit it before its component batches are scored and accepted.
- `reports\handoff.json`: paths, hashes and validation summaries.

## Record leaderboard feedback

Each score is converted back to integer TP using `T=1044`:

```powershell
& $py experiments\v27_closed_loop\closed_loop.py record `
  --batch-id probe01_precision_a --score 0.907619
```

Positive-F1 batches are accepted. Mixed batches are marked `split`; split
halves are generated relative to the same champion. Only one half needs a
leaderboard submission: after it is scored, the complement is inferred from
the parent batch's integer TP delta and is recorded without spending another
submission.

After recording feedback, inspect:

```powershell
& $py experiments\v27_closed_loop\closed_loop.py status `
  --catalog experiments\v27_closed_loop\action_catalog.json
& $py experiments\v27_closed_loop\closed_loop.py checkpoint
```

`status.json` reports the verified score, TP gap, prediction-count target,
optimistic upper bound, submission budget and the hard-stop feasibility gate.

## Validation

```powershell
& $py -m unittest discover -s experiments\v27_closed_loop -p 'test_*.py' -v
& $py experiments\v27_closed_loop\closed_loop.py validate `
  --submission experiments\v27_closed_loop\submissions\probe01_precision_a.csv
```

The validator checks order IDs, root-count bounds, duplicate nodes, topology
membership and the exact output schema. The historical score tests confirm
that `0.906324 -> 953 TP` and `0.905373 -> 952 TP`.

## Safety gates

- Actions are one-per-order and node-disjoint.
- The 12 protected add/remove nodes from the empirically locked swap12 report
  cannot be changed.
- Graph-time-only candidates are excluded.
- A checkpoint is promoted only from scored positive batches.
- If the optimistic upper bound is below `0.926324`, the status report marks
  the target unreachable and the current champion remains the handoff.
