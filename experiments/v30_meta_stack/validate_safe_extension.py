"""Independent validation for the verified-top5 safe extension probe."""

from __future__ import annotations

import json
from pathlib import Path

from validate_v30 import (
    CHAMPION,
    EXPERIMENT,
    independently_touched_orders,
    load_submission,
    sha256,
)


PROTECTED = (
    Path(r"D:\zgyidong")
    / "experiments/submissions/template_ranked/v11_constrained_report.json"
)
NAME = "v30_top5_verified_plus_safe5"


def main():
    manifest = json.loads(
        (EXPERIMENT / "manifests" / f"{NAME}.json").read_text(encoding="utf-8")
    )
    top5 = json.loads(
        (EXPERIMENT / "manifests/v30_cross_order_top5.json").read_text(encoding="utf-8")
    )
    protected = json.loads(PROTECTED.read_text(encoding="utf-8"))
    protected_in = {(row["order_id"], row["rid"]) for row in protected["forced_in"]}
    protected_out = {(row["order_id"], row["rid"]) for row in protected["forced_out"]}
    historical_orders = independently_touched_orders()
    verified_orders = {action["order_id"] for action in top5["actions"]}
    extension = manifest["extension_actions"]
    extension_orders = [action["order_id"] for action in extension]
    if manifest["verified_base"]["actions"] != top5["actions"]:
        raise ValueError("verified top5 actions changed")
    if len(extension_orders) != len(set(extension_orders)):
        raise ValueError("duplicate extension order")
    if set(extension_orders) & (historical_orders | verified_orders):
        raise ValueError("extension overlaps historical or verified order")
    pair_counts = {}
    for action in extension:
        pair_id = action["evidence"]["pair_id"]
        pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
        for rid in action["remove_rids"]:
            if (action["order_id"], rid) in protected_in:
                raise ValueError(("removes protected-in", action["action_id"]))
        for rid in action["add_rids"]:
            if (action["order_id"], rid) in protected_out:
                raise ValueError(("adds protected-out", action["action_id"]))
    if len(pair_counts) != 5 or set(pair_counts.values()) != {2}:
        raise ValueError(("incomplete extension pairs", pair_counts))
    base_predictions, base_roots = load_submission(CHAMPION)
    predictions, roots = load_submission(Path(manifest["path"]))
    if base_predictions != predictions or predictions != manifest["predictions"]:
        raise ValueError(("prediction count", base_predictions, predictions))
    if sha256(Path(manifest["path"])) != manifest["sha256"]:
        raise ValueError("hash mismatch")
    actions = {action["order_id"]: action for action in manifest["actions"]}
    if len(actions) != len(manifest["actions"]):
        raise ValueError("duplicate action order in combined probe")
    for order_id, base in base_roots.items():
        expected = set(base)
        action = actions.get(order_id)
        if action:
            remove = set(action["remove_rids"])
            add = set(action["add_rids"])
            if not remove <= expected or add & expected:
                raise ValueError(("invalid declared action", action["action_id"]))
            expected -= remove
            expected |= add
        if roots[order_id] != expected:
            raise ValueError(("undeclared difference", order_id))
    print(
        json.dumps(
            {
                "passed": True,
                "probe": NAME,
                "predictions": predictions,
                "verified_actions": len(top5["actions"]),
                "safe_extension_actions": len(extension),
                "sha256": manifest["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
