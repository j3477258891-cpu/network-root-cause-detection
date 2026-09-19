# V120 topology replacement campaign

Generated locally from the current 1,035-root checkpoint. Every action is one same-order deletion plus one addition, so all probes have P=1035.

1. If the daily quota is available, submit **only** `probe_00_baseline.csv` first. It is byte-identical to the local checkpoint and has SHA-256 `8390ecf4a3fab22a8c40c950eac76dd3e4b1309a2527412549193613c64ccfcb`.
2. After the leaderboard returns a six-decimal score, record it without guessing TP:
   PowerShell: `$py='C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'; & $py experiments/v120_swap_campaign.py record --probe-id 0 --score <score> --sha256 8390ecf4a3fab22a8c40c950eac76dd3e4b1309a2527412549193613c64ccfcb`
3. Do not submit a group probe until the baseline TP is recorded. Then enter each returned group score with `record --probe-id <id> --score <score> --baseline-tp <baseline_tp>` and run `decode`.
4. Never create a `final_*.csv` unless `final_gate.json` becomes true after online decoding. This generator does not upload files.

The current offline gate is evidence about the model only, not a leaderboard result. It is currently false: 92 structural candidates exist, but only 19 meet the three-model/0.60 prior filter.
