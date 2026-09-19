"""Build adaptive subsets of the scored V30 safe5 extension."""

from __future__ import annotations

import gzip
import json

from build_cross_order_probes import (
    BASELINE_P,
    CHAMPION,
    OUTPUT,
    RECORDS,
    TRUE_ROOTS,
    apply_actions,
    load_submission,
    sha256,
    write_submission,
)


SOURCE = OUTPUT / "manifests/v30_top5_verified_plus_safe5.json"
VERIFIED_TP = 955
SPLITS = {
    "v30_safe_split_a2": ["v30_safe_ext_001", "v30_safe_ext_002"],
    "v30_safe_split_a1": ["v30_safe_ext_001"],
    "v30_safe_split_a1_complement": ["v30_safe_ext_002"],
    "v30_safe_split_b3": ["v30_safe_ext_003", "v30_safe_ext_004", "v30_safe_ext_005"],
    "v30_safe_split_b2": ["v30_safe_ext_003", "v30_safe_ext_004"],
    "v30_safe_split_b1": ["v30_safe_ext_003"],
}


def score_possibilities(pair_count):
    return [
        {
            "subset_delta_tp": delta,
            "tp": VERIFIED_TP + delta,
            "score": round(2 * (VERIFIED_TP + delta) / (TRUE_ROOTS + BASELINE_P), 9),
        }
        for delta in range(-pair_count, pair_count + 1)
    ]


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    verified_actions = source["verified_base"]["actions"]
    extension_by_pair = {}
    for action in source["extension_actions"]:
        extension_by_pair.setdefault(action["evidence"]["pair_id"], []).append(action)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {order["order_id"]: order for order in records}
    order_ids, champion_roots = load_submission(CHAMPION)
    summary = {}
    for name, pair_ids in SPLITS.items():
        subset_actions = [
            action for pair_id in pair_ids for action in extension_by_pair[pair_id]
        ]
        actions = verified_actions + subset_actions
        roots = apply_actions(champion_roots, records_by_order, actions)
        predictions = sum(len(nodes) for nodes in roots.values())
        if predictions != BASELINE_P:
            raise ValueError((name, predictions, BASELINE_P))
        path = OUTPUT / "submissions" / f"{name}.csv"
        write_submission(path, order_ids, roots)
        manifest = {
            "probe_id": name,
            "path": str(path),
            "original_baseline": str(CHAMPION),
            "verified_base_probe": "v30_cross_order_top5",
            "verified_base_tp": VERIFIED_TP,
            "verified_base_actions": verified_actions,
            "source_scored_probe": "v30_top5_verified_plus_safe5",
            "source_extension_delta_tp": -1,
            "subset_pair_ids": pair_ids,
            "subset_actions": subset_actions,
            "actions": actions,
            "predictions": predictions,
            "sha256": sha256(path),
            "score_possibilities": score_possibilities(len(pair_ids)),
        }
        manifest_path = OUTPUT / "manifests" / f"{name}.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary[name] = {
            "pairs": len(pair_ids),
            "predictions": predictions,
            "sha256": manifest["sha256"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
