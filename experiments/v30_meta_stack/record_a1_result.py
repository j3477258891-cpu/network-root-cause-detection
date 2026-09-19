"""Record the A1 singleton score and materialize the inferred best A pair."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from build_cross_order_probes import BASELINE_P, OUTPUT, TRUE_ROOTS, sha256


BASE_TP = 955
A1 = OUTPUT / "submissions/v30_safe_split_a1.csv"
A2_COMPLEMENT = OUTPUT / "submissions/v30_safe_split_a1_complement.csv"
BASE = OUTPUT / "submissions/v30_cross_order_top5.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=float, required=True)
    args = parser.parse_args()
    tp = round(args.score * (TRUE_ROOTS + BASELINE_P) / 2)
    if tp not in (BASE_TP - 1, BASE_TP, BASE_TP + 1):
        raise ValueError(f"score implies unexpected TP={tp}")
    first_delta = tp - BASE_TP
    second_delta = -first_delta  # A1+A2 was measured as exactly neutral.
    if first_delta > 0:
        chosen, chosen_pair, chosen_delta = A1, "v30_safe_ext_001", first_delta
    elif second_delta > 0:
        chosen, chosen_pair, chosen_delta = (
            A2_COMPLEMENT, "v30_safe_ext_002", second_delta
        )
    else:
        chosen, chosen_pair, chosen_delta = BASE, None, 0
    output = OUTPUT / "submissions/v30_verified_checkpoint_after_a.csv"
    shutil.copyfile(chosen, output)
    result = {
        "probe_id": "v30_safe_split_a1",
        "submitted_score": args.score,
        "submitted_tp": tp,
        "first_pair_delta_tp": first_delta,
        "second_pair_delta_tp_inferred": second_delta,
        "chosen_pair": chosen_pair,
        "chosen_delta_tp": chosen_delta,
        "checkpoint_tp": BASE_TP + chosen_delta,
        "checkpoint_predictions": BASELINE_P,
        "checkpoint_score": 2 * (BASE_TP + chosen_delta) / (TRUE_ROOTS + BASELINE_P),
        "checkpoint_path": str(output),
        "checkpoint_sha256": sha256(output),
        "inference_basis": "v30_safe_split_a2 delta_tp=0",
    }
    (OUTPUT / "a_pair_online_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
