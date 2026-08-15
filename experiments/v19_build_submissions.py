"""Build submit-ready V19 graph+time candidates from the online champion.

The full candidate applies the leakage-resistant V19 choice (40% graph+time,
count cap 7).  The probe applies only a small, seed-stable, globally balanced
subset of those order-level actions.  Both preserve the V11 lock orders and
historical online exclusions byte-for-byte relative to the champion.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
EXPERIMENTS = ROOT / "experiments"
V16_DIR = EXPERIMENTS / "v16"
V19_DIR = EXPERIMENTS / "v19_cloud"
V11_DIR = ROOT / "codexgz" / "v11"
SUBMISSIONS = EXPERIMENTS / "submissions"
CHAMPION = SUBMISSIONS / "champion_0.906324_day01_probe01_v11_full.csv"
LOCK_REPORT = V11_DIR / "v11_constrained_report.json"
EXCLUSIONS = EXPERIMENTS / "v15" / "v15b_exclusions.json"
LEDGER = EXPERIMENTS / "ledger.json"

WEIGHT_GRAPH = 0.40
COUNT_CAP = 7
TARGET_COUNT = 1059
MAX_ROOTS = 8
PROBE_MIN_ORDERS = 4
PROBE_MAX_ORDERS = 8
SEEDS = (20260803, 20260817, 20260831)

sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(V16_DIR))
import v19_nested_count_audit as v19  # noqa: E402
import v16_structure_signal as v16  # noqa: E402


def submission_mask(orders, slices, rows):
    mask = np.zeros(slices[-1].stop, dtype=bool)
    for order_index, order in enumerate(orders):
        selected = {item["@rid"] for item in rows[order["id"]]}
        for local_index, node in enumerate(order["alarms"]):
            mask[slices[order_index].start + local_index] = node["@rid"] in selected
    return mask


def protected_capped_mask(scores, slices, base_mask, excluded_indices):
    """V19 cap decoder with protected orders fixed to the champion."""
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for order_index, order_slice in enumerate(slices):
        if order_index in excluded_indices:
            selected[order_slice] = base_mask[order_slice]
            continue
        base_k = int(base_mask[order_slice].sum())
        minimum = max(1, base_k - COUNT_CAP)
        maximum = min(MAX_ROOTS, order_slice.stop - order_slice.start, base_k + COUNT_CAP)
        ranked = np.argsort(-scores[order_slice], kind="stable")
        selected[order_slice.start + ranked[:minimum]] = True
        optional.extend((order_slice.start + ranked[minimum:maximum]).tolist())

    remaining = TARGET_COUNT - int(selected.sum())
    if not 0 <= remaining <= len(optional):
        raise ValueError((remaining, len(optional)))
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    assert int(selected.sum()) == TARGET_COUNT
    return selected


def action_for_order(order, order_slice, before_mask, after_mask, scores_by_seed):
    before_local = set(np.flatnonzero(before_mask[order_slice]).tolist())
    after_local = set(np.flatnonzero(after_mask[order_slice]).tolist())
    if before_local == after_local:
        return None
    removed_local = sorted(before_local - after_local)
    added_local = sorted(after_local - before_local)
    before_rows = np.asarray(
        [order_slice.start + value for value in sorted(before_local)], dtype=np.int64
    )
    after_rows = np.asarray(
        [order_slice.start + value for value in sorted(after_local)], dtype=np.int64
    )
    seed_utilities = [
        float(scores[after_rows].sum() - scores[before_rows].sum())
        for scores in scores_by_seed
    ]
    return {
        "order_id": order["id"],
        "before_k": len(before_local),
        "after_k": len(after_local),
        "delta_k": len(after_local) - len(before_local),
        "removed": [order["alarms"][value]["@rid"] for value in removed_local],
        "added": [order["alarms"][value]["@rid"] for value in added_local],
        "changed_nodes": len(removed_local) + len(added_local),
        "seed_utilities": seed_utilities,
        "min_seed_utility": float(min(seed_utilities)),
        "mean_seed_utility": float(np.mean(seed_utilities)),
    }


def choose_probe(actions):
    """Choose a small stable action set with exact aggregate delta K of zero."""
    # Utility is only comparable for balanced action sets. Dynamic programming
    # retains the highest worst-seed/mean utility state for each (size, delta K).
    states = {(0, 0): (0.0, 0.0, ())}
    for index, action in enumerate(actions):
        updated = dict(states)
        for (size, delta), (min_sum, mean_sum, chosen) in states.items():
            if size >= PROBE_MAX_ORDERS:
                continue
            key = (size + 1, delta + action["delta_k"])
            value = (
                min_sum + action["min_seed_utility"],
                mean_sum + action["mean_seed_utility"],
                chosen + (index,),
            )
            current = updated.get(key)
            if current is None or value[:2] > current[:2]:
                updated[key] = value
        states = updated

    candidates = []
    for size in range(PROBE_MIN_ORDERS, PROBE_MAX_ORDERS + 1):
        state = states.get((size, 0))
        if state is not None:
            candidates.append((state[0], state[1], -size, state[2]))
    if not candidates:
        raise RuntimeError("No 4-8 order seed-stable balanced probe exists")
    chosen_indices = max(candidates)[3]
    return [actions[index] for index in chosen_indices]


def apply_actions(base_mask, orders, slices, full_mask, actions):
    result = base_mask.copy()
    chosen = {action["order_id"] for action in actions}
    for order_index, order in enumerate(orders):
        if order["id"] in chosen:
            result[slices[order_index]] = full_mask[slices[order_index]]
    assert int(result.sum()) == TARGET_COUNT
    return result


def write_submission(path, orders, slices, mask):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_index, order in enumerate(orders):
            rootcauses = []
            for local_index in np.flatnonzero(mask[slices[order_index]]):
                node = order["alarms"][int(local_index)]
                rootcauses.append(
                    {
                        "@rid": node["@rid"],
                        "title": node.get("title", ""),
                        "location": node.get("location", ""),
                        "reason": node.get("reason", ""),
                    }
                )
            writer.writerow(
                [order["id"], json.dumps({"rootcause": rootcauses}, ensure_ascii=False)]
            )


def validate_locks(mask, orders, slices, forced_in, forced_out):
    selected = set()
    for order_index, order in enumerate(orders):
        for local_index in np.flatnonzero(mask[slices[order_index]]):
            selected.add((order["id"], order["alarms"][int(local_index)]["@rid"]))
    assert forced_in <= selected
    assert not (forced_out & selected)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_metadata(path, variant, actions, validation, excluded_orders):
    digest = sha256(path)
    payload = {
        "variant": variant,
        "source_champion": str(CHAMPION),
        "source_champion_sha256": sha256(CHAMPION),
        "method": "0.60 V11 + 0.40 graph-time, V19 count-cap decoder",
        "weight_graph_time": WEIGHT_GRAPH,
        "count_cap": COUNT_CAP,
        "seed_stable_probe": variant == "probe01",
        "excluded_order_count": len(excluded_orders),
        "changed_order_count": len(actions),
        "delta_p": sum(action["delta_k"] for action in actions),
        "removed_count": sum(len(action["removed"]) for action in actions),
        "added_count": sum(len(action["added"]) for action in actions),
        "changes": actions,
        "offline_evidence": {
            "group_oof_tp_delta": 22,
            "fold_tp_deltas": [3, 7, 2, 3, 7],
            "seed_tp_deltas": [22, 20, 20],
            "bootstrap_95_lower": 3.0,
            "note": "Offline evidence is not an online score guarantee.",
        },
        "validation": validation,
        "sha256": digest,
    }
    path.with_suffix(".diff.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return payload


def main():
    if sha256(CHAMPION) != "6b59ad62b5d1c3d5153a91309f514532228f72a5498a591bf56c86913c703e93":
        raise RuntimeError("Champion SHA256 does not match the frozen baseline")

    orders = v19.load_orders(ROOT / "test", False)
    slices = v19.build_slices(orders)
    champion_rows = v16.read_submission(CHAMPION)
    champion_mask = submission_mask(orders, slices, champion_rows)
    assert int(champion_mask.sum()) == TARGET_COUNT

    lock_data = json.loads(LOCK_REPORT.read_text(encoding="utf-8"))
    forced_in = {(item["order_id"], item["rid"]) for item in lock_data["forced_in"]}
    forced_out = {(item["order_id"], item["rid"]) for item in lock_data["forced_out"]}
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    excluded_orders = set(exclusions["total"])
    excluded_orders.update(order_id for order_id, _ in forced_in | forced_out)
    excluded_indices = {
        index for index, order in enumerate(orders) if order["id"] in excluded_orders
    }

    v11_scores = np.load(V11_DIR / "v11_test_scores.npy")
    graph_mean = np.load(V16_DIR / "v16_graph_time_test_scores.npy")
    graph_seeds = [
        np.load(V16_DIR / f"v16_graph_time_test_seed_{seed}.npy") for seed in SEEDS
    ]
    blended_mean = (1.0 - WEIGHT_GRAPH) * v11_scores + WEIGHT_GRAPH * graph_mean
    blended_seeds = [
        (1.0 - WEIGHT_GRAPH) * v11_scores + WEIGHT_GRAPH * scores
        for scores in graph_seeds
    ]

    full_mask = protected_capped_mask(
        blended_mean, slices, champion_mask, excluded_indices
    )
    seed_masks = [
        protected_capped_mask(scores, slices, champion_mask, excluded_indices)
        for scores in blended_seeds
    ]

    full_actions = []
    stable_actions = []
    for order_index, order in enumerate(orders):
        action = action_for_order(
            order, slices[order_index], champion_mask, full_mask, blended_seeds
        )
        if action is None:
            continue
        full_actions.append(action)
        if all(
            np.array_equal(
                mask[slices[order_index]], full_mask[slices[order_index]]
            )
            for mask in seed_masks
        ):
            stable_actions.append(action)

    stable_actions.sort(
        key=lambda item: (item["min_seed_utility"], item["mean_seed_utility"]),
        reverse=True,
    )
    probe_actions = choose_probe(stable_actions)
    probe_mask = apply_actions(champion_mask, orders, slices, full_mask, probe_actions)

    outputs = []
    for variant, filename, mask, actions in (
        (
            "probe01",
            "v19_graph_time_probe01_p1059.csv",
            probe_mask,
            probe_actions,
        ),
        (
            "full_cap7",
            "v19_graph_time_full_cap7_p1059.csv",
            full_mask,
            full_actions,
        ),
    ):
        path = SUBMISSIONS / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing candidate: {path}")
        write_submission(path, orders, slices, mask)
        validation = v16.validate_submission(path, orders)
        validate_locks(mask, orders, slices, forced_in, forced_out)
        metadata = write_metadata(path, variant, actions, validation, excluded_orders)
        if metadata["sha256"] == sha256(CHAMPION):
            raise RuntimeError(f"{filename} is identical to champion")
        outputs.append(
            {
                "path": str(path),
                "variant": variant,
                "changed_orders": len(actions),
                "delta_p": metadata["delta_p"],
                "predictions": validation["predictions"],
                "distribution": validation["distribution"],
                "sha256": metadata["sha256"],
            }
        )

    known_hashes = set(json.loads(LEDGER.read_text(encoding="utf-8"))["scores"])
    assert all(item["sha256"] not in known_hashes for item in outputs)
    summary = EXPERIMENTS / "v19_submission_generation.json"
    summary.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
