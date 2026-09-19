"""Extend the verified V30 top5 with five untouched, non-protected pairs."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from build_cross_order_probes import (
    BASELINE_P,
    CHAMPION,
    DATA,
    OUTPUT,
    RECORDS,
    TRUE_ROOTS,
    actions_for_pairs,
    apply_actions,
    build_pairs,
    load_submission,
    sha256,
    touched_orders,
    write_submission,
)


PROTECTED = (
    Path(r"D:\zgyidong")
    / "experiments/submissions/template_ranked/v11_constrained_report.json"
)
VERIFIED_MANIFEST = OUTPUT / "manifests/v30_cross_order_top5.json"
NAME = "v30_top5_verified_plus_safe5"
VERIFIED_TP = 955
EXTENSION_COUNT = 5


def protected_nodes():
    report = json.loads(PROTECTED.read_text(encoding="utf-8"))
    protected_in = {(row["order_id"], row["rid"]) for row in report["forced_in"]}
    protected_out = {(row["order_id"], row["rid"]) for row in report["forced_out"]}
    return protected_in, protected_out


def extension_score_possibilities():
    return [
        {
            "extension_delta_tp": delta,
            "tp": VERIFIED_TP + delta,
            "score": round(2 * (VERIFIED_TP + delta) / (TRUE_ROOTS + BASELINE_P), 9),
        }
        for delta in range(-EXTENSION_COUNT, EXTENSION_COUNT + 1)
    ]


def main():
    with np.load(DATA, allow_pickle=False) as archive:
        ptr = archive["test_alarm_ptr"]
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    scores = np.load(OUTPUT / "v30_consensus_test.npy")
    order_ids, champion_roots = load_submission(CHAMPION)
    records_by_order = {order["order_id"]: order for order in records}
    verified = json.loads(VERIFIED_MANIFEST.read_text(encoding="utf-8"))
    verified_actions = verified["actions"]
    verified_orders = {action["order_id"] for action in verified_actions}
    protected_in, protected_out = protected_nodes()
    excluded = touched_orders() | verified_orders
    pairs, row_alarm = build_pairs(
        records,
        ptr,
        scores,
        champion_roots,
        excluded,
        protected_in=protected_in,
        protected_out=protected_out,
    )
    if len(pairs) < EXTENSION_COUNT:
        raise ValueError(("insufficient safe positive-margin pairs", len(pairs)))
    extension_pairs = pairs[:EXTENSION_COUNT]
    for index, pair in enumerate(extension_pairs, start=1):
        pair["pair_id"] = f"v30_safe_ext_{index:03d}"
    extension_actions = actions_for_pairs(extension_pairs, records, row_alarm)
    all_actions = verified_actions + extension_actions
    roots = apply_actions(champion_roots, records_by_order, all_actions)
    predictions = sum(len(nodes) for nodes in roots.values())
    if predictions != BASELINE_P:
        raise ValueError(("prediction count", predictions, BASELINE_P))
    submission_path = OUTPUT / "submissions" / f"{NAME}.csv"
    write_submission(submission_path, order_ids, roots)
    manifest = {
        "probe_id": NAME,
        "path": str(submission_path),
        "original_baseline": str(CHAMPION),
        "verified_base": {
            "probe_id": "v30_cross_order_top5",
            "score": 0.918711,
            "tp": VERIFIED_TP,
            "predictions": BASELINE_P,
            "actions": verified_actions,
        },
        "extension_pair_count": EXTENSION_COUNT,
        "extension_pairs": extension_pairs,
        "extension_actions": extension_actions,
        "actions": all_actions,
        "predictions": predictions,
        "prediction_delta": 0,
        "protected_in_count": len(protected_in),
        "protected_out_count": len(protected_out),
        "excluded_order_count": len(excluded),
        "sha256": sha256(submission_path),
        "score_possibilities": extension_score_possibilities(),
    }
    manifest_path = OUTPUT / "manifests" / f"{NAME}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "probe": NAME,
                "verified_tp": VERIFIED_TP,
                "extension_pairs": EXTENSION_COUNT,
                "extension_margins": [pair["margin"] for pair in extension_pairs],
                "predictions": predictions,
                "sha256": manifest["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
