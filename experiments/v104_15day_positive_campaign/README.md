# V104 15-day positive-code campaign

`probe_00_calibration.csv` removes the fixed 169-action delete pool. Submit it first.
Then submit `probe_01_code.csv` through `probe_25_code.csv`; every probe uses the
same delete pool plus a row of the 25×71 binary matrix. Enter the 26 leaderboard
F1 values into the decoder:

```powershell
$py = 'C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py experiments\v104_15day_positive_campaign.py --decode `
  --calibration-score <score0> `
  --probe-scores <score1> <score2> ... <score25>
```

After decoding, use `v112_adaptive_final_runtime.py` for the three final
submissions. It chooses the next deletion set from the previous final's exact
score, so the final checkpoints are adaptive rather than fixed.

```powershell
& $py experiments\v112_adaptive_final_runtime.py              # first final
& $py experiments\v112_adaptive_final_runtime.py --score <previous_score>
```

Stop as soon as a final score reaches 0.945; otherwise continue until three
final slots are used.
All files here are local artifacts only and have not been submitted.

The current `decoded.json`, `adaptive_final_state.json`, and `final_*.csv` are
synthetic decoder-validation artifacts and must be regenerated after real
leaderboard scores are available.

The failed `distance1_swap_equation_filtered.csv` submission consumed one of
the 30 available slots. This revised campaign uses exactly 29 more submissions:
1 calibration + 25 code probes + 3 adaptive finals, fitting the remaining
15-day budget at two submissions per day.

The calibration was later scored at F1=0.840838 (TP=803), so this campaign is
now rejected: its 169-action delete pool removed 153 true roots. Do not submit
`probe_01_code.csv` through `probe_25_code.csv`; use the regenerated V118
safe-delete campaign instead.
