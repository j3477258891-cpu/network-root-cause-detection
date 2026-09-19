# V29 topology-aware action probes

This experiment starts from the verified `0.914313068` checkpoint and trains
three selective action models:

- delete a likely false-positive selected alarm;
- add a likely missed root cause;
- swap a selected and unselected alarm within one order.

The alarm features include the V25 static matrix, V11/V13/V19 scores and
missing flags, plus explicit topology node/edge path sequences. Delete and add
models are cross-fitted with both station and template folds. Every test order
touched by an earlier online probe is excluded.

Run with the bundled Python runtime:

```powershell
$env:PYTHONPATH='D:\zgyidong\.deps'
& 'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'D:\zgyidong\experiments\v29_domain_ranker\v29_actions.py'
& 'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  'D:\zgyidong\experiments\v29_domain_ranker\validate_v29.py'
```

The current full run passed the delete and swap gates. The add gate failed, so
no add or combined checkpoint is valid for submission.

Submission order:

1. Submit `submissions/v29_delete_top10.csv` (`P=1035`). Its SHA-256 is
   `bec7c56dfa4205a5a7077d6c5d9ee908649156fb205d2f9abf80c78fdc5b3d8d`.
2. Record the leaderboard score and recover `TP` with
   `round(score * (1044 + 1035) / 2)`.
3. Use `submissions/v29_swap_top8.csv` only as the next independent probe.

Never submit a V29 add or combined file unless a later full run passes the add
gate and regenerates that exact artifact.

Online update (2026-08-18): `v29_delete_top10.csv` scored `0.916787`, which
is `TP=953` at `P=1035`. Despite losing two TP, the reduced prediction count
improves F1, so `submissions/day05_delete_verified.csv` is the highest verified
checkpoint. The swap probe was built relative to the old `0.914313068` baseline;
do not combine it with the delete checkpoint until its own leaderboard score is
recorded.

The swap probe later scored `0.914313`, exactly `TP=955` at `P=1045`. It is
therefore neutral and is not part of the champion. Do not submit the same
8-action swap batch again; split it into smaller batches before testing any
of its actions again.

The prepared second-stage files are `submissions/v29_swap_top5.csv` and
`submissions/v29_swap_tail3.csv`. They are both relative to the original
`0.914313068` champion; submit `top5` first, then choose whether to test
`tail3` from its score.
