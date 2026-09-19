# V157 joint correction campaign

New six-attempt allocation explicitly confirmed by the user after V156 final scored 0.929254. The previous ledger is preserved, not reset. CSVs are prepared locally only; the user uploads them.

## Current feedback: p02 = 0.927717; verified shortcut (2026-09-12)

Recorded from the user reply immediately following the p02 recommendation. Filename attribution remains conversation-derived; no platform screenshot or exact submission timestamp was provided. P=1045, TP=969, delta TP=-3 versus V156 final. Feasible worlds reduced from 21 to 4. Actual champion remains 0.929254. Four new attempts remain; report-time ledger has two attempts today.

All four worlds fix deletion1 at delta TP=0 (safe) and both swaps12/16 at -1 or 0 (no positive outcome). The best currently known merge deletes1, yielding predicted 0.929699. The conditional pool optimum is now exactly 0.930589 in every remaining world; 0.931546 is no longer attainable from this pool.

Do not blindly use the original third/fourth rows. Complete enumeration found a one-query shortcut: query deletion3/4/7 plus swap12, carrying safe deletion1. Swap12 is used ONLY as an information probe, not a final profitable action. Each of four possible integer feedbacks identifies a unique world. The live engine considers such exact shortcuts only with 2 to 6 worlds; it minimizes query action count, then maximizes worst probe score and model-weighted probe score. It does not claim a globally optimal policy. No safe fixed-carrier contribution is treated as unknown.

Next file: `v157_probe_adaptive_p03_utf8.csv`, P=1044. SHA-256: `c4096c31d9868b4f7304196070838e7e1922b4c1459bd81dab5ff780940985c8`.

| Reported score | TP | Final safe deletions (in addition to existing true additions8/11/15) |
|---|---:|---|
| 0.927203 | 968 | 1,2,6 |
| 0.928161 | 969 | 1,4,10 |
| 0.929119 | 970 | 1,2,3 |
| 0.930077 | 971 | 1,4,7 |

Every branch's final merge is P=1045, TP=972, expected F1=0.930589. It requires one information submission then one merge, leaving two contingency opportunities from the current four if no anomalies occur. Neither file is uploaded automatically. Verify daily platform availability; wait for the next day if today's two attempts are used. The unchanged default four-row matrix remains the fallback when no fully verified one-shot query exists.

## Previous feedback: p01 (superseded)

Recorded from user feedback after the p01 recommendation; filename attribution is inferred from conversation and the exact platform timestamp/uploaded bytes were not supplied. P=1045 implies unique TP=969, a -3 TP change versus the scored 972-TP champion. Remaining feasible worlds: 78 to 21. No scoring anomaly; the actual champion remains 0.929254 and the best currently count-determined merge is still that champion.

Decision: continue. Five new attempts remain. The three remaining default queries plus one final merge fit within this budget, with one contingency attempt. All 21 possible feedback paths were checked and still permit a final pool optimum of 0.930589 or 0.931546 under the unchanged scoring model. These are conditional outcomes, not actual scores.

Next: `v157_probe_p02_utf8.csv`, P=1045, independently from the V156 final baseline. Delete candidates 3/7/13 and execute swap16. Do not carry over p01 changes. SHA-256: `9c46515a1a0a5a7231421c5f29ab41f3ae23ada7d5b08a69b309775159d4f3fd`. Structure, order, metadata differences and no-BOM encoding passed. Its feedback leaves at most six worlds (and one world in either extreme). Local report-time accounting shows one attempt today; confirm the actual platform limit before upload. No upload performed.

## First file (already scored; do not resubmit)

`v157_probe_p01_utf8.csv`: start from scored V156 final, delete candidates 6/10/13 and execute swap 12. P=1045; 546 orders; original order retained; exact JSON node changes; UTF-8 without BOM. Candidate IDs retain their V153 meaning. Confirm actual daily availability before upload: report-time accounting already includes two September 11 scores from V156.

The first feedback can be 0.926759, 0.927717, 0.928674, 0.929631, 0.930589 or 0.931546. These map to query delta TP -4 through +1 relative to the scored 972-TP baseline. Other scores pause for audit. A lower probe score is not itself a failed experiment.

## Protected route

Fixed default queries: [6,10,12,13], [3,7,13,16], [1,2,13,16], [1,3,13]. Each uses the same scored baseline, never cumulative probes. The four count signatures distinguish all 78 historical feasible worlds. Stop early if a count-determined merge reaches the conditional pool upper bound.

Keep one final merge slot and one contingency slot. Correctly deleting the three false positives yields P=1045, TP=972, F1=0.930589. If a +1 swap is found, P=1045, TP=973, F1=0.931546. These are conditional mathematical outcomes, not achieved scores. No additional new-model pool has passed its gate. Do not spend the contingency slot merely to use up the allocation.

## Commands

Use the existing ML runtime at `D:/zgyidong/experiments/v152_error_repair_campaign/.venv/Scripts/python.exe` with `-B campaign.py` from this folder.

- `build`: idempotent initial freeze; no ledger clearing.
- `audit`: read-only source hashes, current support, champion, quota and next action.
- `recommend`: update derived recommendation without writing a CSV.
- `prepare`: validate and return the existing pending artifact, or author exactly the next safe probe/final using the bundled Node artifact builder. No uploads.
- `record --probe-id probe_p01 --attempt-id UNIQUE_ID --score SIX_DECIMALS --submitted-at ISO_TIMESTAMP --evidence SOURCE`: record actual user feedback. Use `--failed` without a score for a failed attempt. Do not invent platform timestamps: label a report-time proxy explicitly in evidence.
- `python -B -m unittest -v test_campaign`: temporary-fixture regression tests, including every one of the 78 initial label states and all six first-score branches. Synthetic scores never enter the live ledger.

Every accepted CSV has a registered SHA-256 and is revalidated on state load. Anomalies consume quota and pause exploration. The present fixed-matrix budget guard is conservative when failures leave too few slots: it does not claim an unproved adaptive shortcut. No automated refund, anomaly resolution, new-model probe or competition upload is implemented.
