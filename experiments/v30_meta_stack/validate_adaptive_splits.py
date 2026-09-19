"""Validate adaptive subsets against the scored safe5 manifest."""

from __future__ import annotations

import json
from pathlib import Path

from build_adaptive_splits import SPLITS
from validate_v30 import CHAMPION, EXPERIMENT, load_submission, sha256


def main():
    source = json.loads(
        (EXPERIMENT / "manifests/v30_top5_verified_plus_safe5.json").read_text(
            encoding="utf-8"
        )
    )
    source_actions = {
        action["action_id"]: action for action in source["extension_actions"]
    }
    base_predictions, base_roots = load_submission(CHAMPION)
    results = {}
    for name, pair_ids in SPLITS.items():
        manifest = json.loads(
            (EXPERIMENT / "manifests" / f"{name}.json").read_text(encoding="utf-8")
        )
        if manifest["subset_pair_ids"] != pair_ids:
            raise ValueError((name, "pair declaration mismatch"))
        for action in manifest["subset_actions"]:
            if source_actions.get(action["action_id"]) != action:
                raise ValueError((name, "action differs from scored source", action["action_id"]))
        pair_counts = {}
        for action in manifest["subset_actions"]:
            pair_id = action["evidence"]["pair_id"]
            pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
        if set(pair_counts) != set(pair_ids) or set(pair_counts.values()) != {2}:
            raise ValueError((name, "incomplete pairs", pair_counts))
        path = Path(manifest["path"])
        predictions, roots = load_submission(path)
        if predictions != base_predictions or predictions != manifest["predictions"]:
            raise ValueError((name, "prediction count", predictions))
        if sha256(path) != manifest["sha256"]:
            raise ValueError((name, "hash mismatch"))
        actions = {action["order_id"]: action for action in manifest["actions"]}
        if len(actions) != len(manifest["actions"]):
            raise ValueError((name, "duplicate action order"))
        for order_id, base in base_roots.items():
            expected = set(base)
            action = actions.get(order_id)
            if action:
                expected -= set(action["remove_rids"])
                expected |= set(action["add_rids"])
            if roots[order_id] != expected:
                raise ValueError((name, "undeclared difference", order_id))
        results[name] = {
            "pairs": len(pair_ids),
            "predictions": predictions,
            "sha256": manifest["sha256"],
        }
    print(json.dumps({"passed": True, "probes": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
