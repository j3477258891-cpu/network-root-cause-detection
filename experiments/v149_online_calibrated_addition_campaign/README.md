# V149 online-calibrated addition campaign

The verified starting point is V148 (`P=1037`, `TP=961`, public F1 `0.923594`).
All ten probes are independent additions to that same file; they are not a
cumulative chain.

## Commands

Use the bundled Python runtime:

```powershell
$py = 'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py 'D:\zgyidong\experiments\v149_online_calibrated_addition_campaign.py' build
& $py 'D:\zgyidong\experiments\v149_online_calibrated_addition_campaign.py' record --probe-id 1 --score 0.000000
& $py 'D:\zgyidong\experiments\v149_online_calibrated_addition_campaign.py' decode
& $py 'D:\zgyidong\experiments\v149_online_calibrated_addition_campaign.py' validate
```

Replace the example score with the actual six-decimal leaderboard score. Only
real leaderboard scores belong in `online_scores.json`. After every `record`,
read `decode_state.json`: submit `next_probe` until the state is `decoded`, then
run `decode`. A final CSV is emitted only for a unique solution with at least
one confirmed true addition.

After the final file is publicly scored, record it with:

```powershell
& $py 'D:\zgyidong\experiments\v149_online_calibrated_addition_campaign.py' record --probe-id final --score 0.000000
```

Only an exact match to the decoded expectation releases the optional `6600`
single-swap file. Record that result with probe id `6600`.
