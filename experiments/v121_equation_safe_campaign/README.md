# V121 equation-safe campaign

This campaign uses only real scored submissions to constrain binary test-alarm labels. It emits `probe_01_safe_single.csv` only when every feasible integer label assignment gives that replacement a +1 TP delta.

Current result: the equation system is feasible, but no candidate passes `min(delta)=+1`; no V121 submission file was emitted. The current online baseline remains TP=956, F1=0.919673.

To append an actual score after a future safe submission, use the `record` command. Do not enter OOF or simulated scores.
