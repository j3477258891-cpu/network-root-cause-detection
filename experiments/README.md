# Leaderboard experiment workflow

This directory is anchored to the verified `0.905373` submission. The original
champion is never overwritten.

## Current submission order

1. Submit `submissions/day01_probe01_v11_full.csv`.
2. Record the exact six-decimal score before using another file.
3. Use `probe02` and `probe03` only for attribution or when `probe01` is neutral
   or worse. Both are measured against the original champion.
4. Merge only batches whose inferred `delta_tp` is positive.

## Record a score

Run this without `--promote` when the result is only an experimental batch:

```powershell
python experiment_manager.py record `
  --root D:\zgyidong\experiments `
  --experiment day01_probe02_v11_local_a `
  --score 0.906324
```

Add `--promote` only when the candidate is being frozen as the new champion.
The command rejects scores that cannot map to an integer true-positive count.

## Split a neutral batch

```powershell
python experiment_manager.py split `
  --root D:\zgyidong\experiments `
  --experiment day01_probe02_v11_local_a `
  --test-dir D:\zgyidong\test
```

## Merge accepted batches

```powershell
python experiment_manager.py merge `
  --root D:\zgyidong\experiments `
  --id day01_merge_positive `
  --experiments day01_probe02_v11_local_a,day01_probe03_v11_local_b `
  --test-dir D:\zgyidong\test
```

Every generated CSV has a JSON manifest in `reports/`. `ledger.json` is the
source of truth; `ledger.csv` is a convenient human-readable projection.

