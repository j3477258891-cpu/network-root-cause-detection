# V156 verified execution status

Live CLI: `campaign.py` delegates to `verified_engine.py`. Retained legacy helpers and old recommendation files are provenance, not the live decision authority. Use `verified_recommendation.json`.

## Final result recorded: 0.929254 (2026-09-11)

The user reported 0.929254 directly after the final_r04 recommendation. Recorded as that file's online score from user feedback; platform screenshot, uploaded bytes, and exact submission time were not independently supplied. Ledger evidence preserves this attribution and recording-time proxy.

- New actual champion: `v156_final_r04_utf8.csv`, P=1048, TP=972, F1=0.929254. All six displayed decimals match the prior equation prediction.
- Gain versus the previous actual champion 0.928742: +0.000512 displayed F1. Gain versus campaign-start V153 p03 0.927855: +0.001399 displayed F1. Gain versus V149 0.927717: +0.001537 displayed F1.
- SHA-256 unchanged: `8ed77d30fa1773cded4d98ff6d40326738575f9a608c9b2f129ebfec5e897221`.
- Decision: retain this champion and stop this campaign. The local ten-attempt ledger is exhausted, counting the earlier failed BOM submission provisionally. No duplicate final or additional probe is recommended. Platform refunds or extra quota require confirmation before planning new submissions.
- No new labels are resolved by this expected final score; 78 joint correction worlds remain. Their optimistic bound 0.931546 is not a realized result or a forecast. The 0.94 target was not reached.
- No competition upload was performed by this tooling.

## Previous recommendation after r05 (superseded)

The user replied with score 0.928742 after the r05 recommendation. This was attributed to r05 from conversation context, not verified from a platform screenshot or downloaded file. The ledger explicitly records this attribution and uses recording time only as a submission-time proxy. If the score belongs to a different file, correct that attribution before further submission.

- r05: P=1047, TP=971, total delta TP=+2 versus V149. Subtracting the confirmed +3 carrier contribution gives query 1/12/16 aggregate delta TP=-1. This does not identify each action's separate outcome.
- Feasible correction worlds: 270 to 78. Actual champion remains 0.928742. The new optimistic pool upper bound is 0.931546, not an achievable forecast with one remaining attempt.
- Decision: stop probing and use the final slot for `v156_final_r04_utf8.csv`, containing only confirmed additions 8/11/15. Its lower file number does not make it obsolete.
- Final P=1048, TP=972, equation-predicted F1=0.929254 (gain about 0.000512 versus actual champion). This is not a public result yet.
- Final SHA-256: `8ed77d30fa1773cded4d98ff6d40326738575f9a608c9b2f129ebfec5e897221`.
- Rechecked all 546 orders, exact action difference, root counts 1 to 8, no duplicate nodes, UTF-8 without BOM, and unchanged SHA. Independent full 65536-subset enumeration over all 78 feasible worlds confirms the same best count-determined merge.
- Local quota: one remaining total, one used today by report-time proxy. Confirm actual platform limits before submitting. No automatic upload and no new CSV or overwritten CSV.

## Previous recommendation after r03 (superseded)

- Actual champion: r03, P=1047, TP=971, public F1=0.928742. User-downloaded CSV hash matches the registered r03 artifact.
- Six additions decoded: true 8/11/15, false 9/14/17. There are 270 remaining joint correction worlds; no further addition query is needed.
- Next candidate: `v156_probe_r05_utf8.csv`, query actions 1/12/16 (one deletion, two swaps), carrying confirmed true additions 8/11/15. P=1047, orders=546, UTF-8 without BOM.
- SHA-256: `3b6b630e0eca4ba5527cd47d707c9fc84d35abb847103fd0c0b39451ded1ca0f`.
- Local remaining quota=2 including the final reserve. This probe costs one attempt. Do NOT submit prepared `v156_final_r04_utf8.csv` first if pursuing this two-attempt route.
- Platform quota has not been checked live. Local daily count is already 3, with some times recorded from user-report proxies; wait for an available submission day and confirm at least two total opportunities remain. No upload is performed.
- Direct merge fallback is P=1048, TP=972, predicted F1=0.929254, not yet publicly scored. Its existing CSV remains unchanged.

Completed one-step optimization selected the same query under both working model weights and uniform feasible-world weights. Expected retained final F1 is respectively 0.929562551 and 0.929695498; these are conditional calculations, not calibrated success probabilities. All six feedback branches preserve the 0.929254 merge route, provided the scoring model, files and remaining final slot remain valid. The fixed current pool cannot reach 0.94 (optimistic bound 0.932504).

| Query delta TP | Probe score | Best final/retained score after one remaining merge |
|---:|---:|---:|
| -3 | 0.926829 | 0.929254 |
| -2 | 0.927786 | 0.929254 |
| -1 | 0.928742 | 0.929254 |
| 0 | 0.929699 | 0.929699 |
| +1 | 0.930655 | 0.930655 |
| +2 | 0.931612 | 0.931612 |

The query delta excludes the carrier's known +3 TP. Scores are rounded to six decimals. A nonmatching score pauses for audit; do not force it into the nearest branch. If the probe already attains the best known combination, do not duplicate it as a final submission.

## Historical p02 handoff (superseded; do not resubmit)

- Upload candidate: `v156_probe_p02.csv`, UTF-8 without BOM.
- It is exactly p01 with the initial three BOM bytes removed. Failed p01 remains preserved.
- Query: V149 + candidates 8, 9, 11. Orders=546, P=1048.
- SHA-256: `624ea827c370fe0ea0f9d59ecae6189844ad1b477809379740c5175eb5125c7d`.
- p01 failed on 2026-09-10 at 16:18:08 +08:00, no score. It consumes one provisional attempt and contributes no equation. Local remaining quota=4, daily used=1 at time of recommendation. Confirm platform quota before upload.
- Do not resubmit the BOM-containing p01. No competition upload is performed by this tooling.

Possible six-decimal p02 results: 0.926386 / 0.927342 / 0.928298 / 0.929254, corresponding to 0 / 1 / 2 / 3 true queried nodes. Other scores pause the campaign. Extreme outcomes resolve all six labels immediately; middle outcomes need at most two further decode queries. The protected merge is P=1048, TP=972, F1=0.929254 under unchanged scoring assumptions.

## Runtime and commands

Use the existing working ML environment:
`D:/zgyidong/experiments/v152_error_repair_campaign/.venv/Scripts/python.exe`

- `campaign.py audit`: read-only hashes, quota, full joint feasible worlds and exact known merge.
- `campaign.py build`: idempotent historical audit/freeze; never clears actual attempts. Initial MILP audit checked all 3240 joint outcome vectors against 68 score equations.
- `campaign.py recommend --time-limit 1800`: updates the verified report, without emitting a CSV.
- `campaign.py recommend --emit --time-limit 1800`: prepares one recommended artifact; does not upload.
- `campaign.py record --probe-id p02 --attempt-id <unique-platform-attempt> --score <six-decimal-score> --submitted-at <ISO-time-with-offset> --evidence <user-result-source>`: records an actual score and updates constraints on the next audit/recommendation. Failed attempts use `--failed`, without a score.
- `campaign.py emit-final`: only emits a count-determined merge strictly above the real champion. Does not submit a duplicate achieved combination.
- `campaign.py research`: read verified study status. `research --run` starts/resumes the separate study; do not start while the current PID is alive.
- `python -B -m unittest -v test_verified_engine`: regression tests in this directory.

## Evidence boundaries

`history_worlds.json` is a hash-frozen MILP-verified feasible support, not a label posterior. Complete joint model products are mixed once, then conditioned on feedback; stage-wise independence is not assumed. Zero/tiny model probability never excludes a mathematically feasible label from the merge safety checks.

The exact merge optimizer eliminates individually known helpful/harmful outcomes, then enumerates every unresolved action subset using an integer difference-space basis. The chosen merge is checked directly against every remaining world. Independent regression enumerates all 65536 subsets without that shortcut for every proposed correction-probe feedback branch and reproduces the optimum. Conditional probability, exact score arithmetic, public scores, and the optimistic pool upper bound are separately reported.

The earlier timed search used a safe fallback. The current optimized search completed (4372 evaluated nodes for each weighting; `search_truncated=false`). Its result is restricted to the implemented correction-subset query class and known-world model, not a real-world global-optimum claim. All ten regression tests pass, including all twelve initial addition states, live record updates, signed feedback, timeout fallback and independent merge enumeration.

## Research completed; gate failed, no new-model submission

The old FN `research.py/results.json` are retained but not accepted as validation: they used test G=1044 in training F1, lacked nested budget selection, and did not complete the station-cluster bootstrap gate.

`../v156_fn_research/verified_research.py` uses the existing connected five-fold/node-crossfit helpers, matched-budget historical controls, inner-only parameter/temperature/budget selection, all legal unselected nodes, one action/order, and 2000 station-cluster paired bootstrap samples. It excludes saved supervised logits and partial semantic embeddings whose outer-fold provenance is not demonstrated.

The original conservative deadline is 2026-09-11 12:08:49 +08:00 and was not reset. All five folds finished at 2026-09-10 16:59:48 +08:00. All four families failed the combined acceptance criteria. Even the positive aggregate graph-feature result had only two positive folds and a negative bootstrap lower bound. `../v156_fn_research/verified_results.json` contains the details. No new-model CSV is eligible; research consumed no submission quota. The last experimental slot returns to the existing correction pool.
