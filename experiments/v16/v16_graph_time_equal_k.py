"""Build strict per-order equal-K V16 graph+time probes.

The online champion fixes the number of roots selected for every order. This
script only tests whether graph+time scores can choose better nodes within the
same order. Historical exclusions and V11 locked orders remain byte-equivalent
to the champion.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import v16_structure_signal as v16


WEIGHTS = (0.195, 0.010)


def build_slices(orders):
    slices = []
    cursor = 0
    for order in orders:
        slices.append(slice(cursor, cursor + len(order["alarms"])))
        cursor += len(order["alarms"])
    return slices


def submission_mask(orders, slices, submission_rows):
    mask = np.zeros(slices[-1].stop, dtype=bool)
    for order_index, order in enumerate(orders):
        selected = {item["@rid"] for item in submission_rows[order["id"]]}
        for local_index, node in enumerate(order["alarms"]):
            mask[slices[order_index].start + local_index] = node["@rid"] in selected
    return mask


def fixed_k_protected_mask(scores, orders, slices, champion_mask, excluded_orders):
    mask = champion_mask.copy()
    for order_index, order in enumerate(orders):
        if order["id"] in excluded_orders:
            continue
        order_slice = slices[order_index]
        count = int(np.sum(champion_mask[order_slice]))
        ranked = np.argsort(-scores[order_slice], kind="stable")[:count]
        mask[order_slice] = False
        mask[order_slice.start + ranked] = True
    return mask


def all_seed_stable_mask(
    mean_mask,
    seed_scores,
    orders,
    slices,
    champion_mask,
    excluded_orders,
):
    seed_masks = [
        fixed_k_protected_mask(
            scores, orders, slices, champion_mask, excluded_orders
        )
        for scores in seed_scores
    ]
    stable = champion_mask.copy()
    for order_index, order in enumerate(orders):
        if order["id"] in excluded_orders:
            continue
        order_slice = slices[order_index]
        target = mean_mask[order_slice]
        if all(np.array_equal(mask[order_slice], target) for mask in seed_masks):
            stable[order_slice] = target
    return stable


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


def oof_validation(weight, train_orders):
    slices = build_slices(train_orders)
    labels = np.asarray(
        [
            int(node["@rid"] in order["roots"])
            for order in train_orders
            for node in order["alarms"]
        ],
        dtype=np.int8,
    )
    v11_scores = 0.25 * np.load(v16.V11_DIR / "v11_oof_context.npy") + 0.75 * np.load(
        v16.V11_DIR / "v11_oof_meta.npy"
    )
    graph_time_scores = np.load(v16.OUTPUT_DIR / "v16_graph_time_oof.npy")
    target_count = round(v16.TARGET_TEST_COUNT / 546 * len(train_orders))
    champion_mask = v16.exact_count_mask(v11_scores, slices, target_count)
    blend_scores = (1.0 - weight) * v11_scores + weight * graph_time_scores
    candidate_mask = v16.fixed_k_mask(blend_scores, slices, champion_mask)
    folds, _, _ = v16.grouped_folds(train_orders)
    fold_deltas = []
    changed_orders = []
    for fold in range(v16.N_FOLDS):
        delta = 0
        changed = 0
        for order_index, order_slice in enumerate(slices):
            if folds[order_index] != fold:
                continue
            if not np.array_equal(candidate_mask[order_slice], champion_mask[order_slice]):
                changed += 1
            delta += int(
                np.sum(candidate_mask[order_slice] & (labels[order_slice] == 1))
                - np.sum(champion_mask[order_slice] & (labels[order_slice] == 1))
            )
        fold_deltas.append(delta)
        changed_orders.append(changed)
    return {
        "base": v16.metrics(champion_mask, labels),
        "candidate": v16.metrics(candidate_mask, labels),
        "fold_tp_deltas": fold_deltas,
        "fold_changed_orders": changed_orders,
        "net_tp_delta": int(sum(fold_deltas)),
    }


def change_metadata(
    path,
    weight,
    orders,
    slices,
    mask,
    champion_mask,
    v11_scores,
    graph_time_scores,
    graph_time_per_seed,
    excluded_orders,
    oof,
    variant,
):
    changes = []
    blend_scores = (1.0 - weight) * v11_scores + weight * graph_time_scores
    blend_per_seed = [
        (1.0 - weight) * v11_scores + weight * seed_scores
        for seed_scores in graph_time_per_seed
    ]
    for order_index, order in enumerate(orders):
        order_slice = slices[order_index]
        before_local = set(np.flatnonzero(champion_mask[order_slice]).tolist())
        after_local = set(np.flatnonzero(mask[order_slice]).tolist())
        if before_local == after_local:
            continue
        removed_local = sorted(before_local - after_local)
        added_local = sorted(after_local - before_local)
        assert len(removed_local) == len(added_local)
        assert order["id"] not in excluded_orders
        removed_rows = [order_slice.start + index for index in removed_local]
        added_rows = [order_slice.start + index for index in added_local]
        changes.append(
            {
                "order_id": order["id"],
                "k": len(before_local),
                "removed": [order["alarms"][index]["@rid"] for index in removed_local],
                "added": [order["alarms"][index]["@rid"] for index in added_local],
                "delta_p": 0,
                "v11_boundary_margin": float(
                    min(v11_scores[added_rows]) - max(v11_scores[removed_rows])
                ),
                "graph_time_boundary_margin": float(
                    min(graph_time_scores[added_rows])
                    - max(graph_time_scores[removed_rows])
                ),
                "blend_boundary_margin": float(
                    min(blend_scores[added_rows]) - max(blend_scores[removed_rows])
                ),
                "blend_seed_boundary_margins": [
                    float(min(scores[added_rows]) - max(scores[removed_rows]))
                    for scores in blend_per_seed
                ],
            }
        )

    for order_index, order_slice in enumerate(slices):
        assert int(np.sum(mask[order_slice])) == int(np.sum(champion_mask[order_slice]))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_champion": str(v16.CHAMPION),
        "variant": variant,
        "weight_graph_time": weight,
        "weight_v11": 1.0 - weight,
        "strict_equal_k": True,
        "excluded_order_count": len(excluded_orders),
        "changed_order_count": len(changes),
        "removed_count": sum(len(item["removed"]) for item in changes),
        "added_count": sum(len(item["added"]) for item in changes),
        "delta_p": 0,
        "changes": changes,
        "oof_fixed_k_validation": oof,
        "sha256": sha,
    }


def main():
    train_orders = v16.load_orders(v16.TRAIN_DIR, True)
    test_orders = v16.load_orders(v16.TEST_DIR, False)
    test_slices = build_slices(test_orders)
    champion_rows = v16.read_submission(v16.CHAMPION)
    champion_mask = submission_mask(test_orders, test_slices, champion_rows)

    exclusions = json.loads(v16.EXCLUSIONS.read_text(encoding="utf-8"))
    excluded_orders = set(exclusions["total"])
    locks = json.loads(v16.LOCK_REPORT.read_text(encoding="utf-8"))
    excluded_orders.update(
        item["order_id"] for item in locks["forced_in"] + locks["forced_out"]
    )

    v11_scores = np.load(v16.V11_DIR / "v11_test_scores.npy")
    graph_time_scores = np.load(v16.OUTPUT_DIR / "v16_graph_time_test_scores.npy")
    graph_time_per_seed = [
        np.load(v16.OUTPUT_DIR / f"v16_graph_time_test_seed_{seed}.npy")
        for seed in v16.SEEDS
    ]

    outputs = []
    for weight in WEIGHTS:
        blend_scores = (1.0 - weight) * v11_scores + weight * graph_time_scores
        mean_mask = fixed_k_protected_mask(
            blend_scores, test_orders, test_slices, champion_mask, excluded_orders
        )
        seed_blends = [
            (1.0 - weight) * v11_scores + weight * seed_scores
            for seed_scores in graph_time_per_seed
        ]
        stable_mask = all_seed_stable_mask(
            mean_mask,
            seed_blends,
            test_orders,
            test_slices,
            champion_mask,
            excluded_orders,
        )
        oof = oof_validation(weight, train_orders)
        tag = f"{int(round(weight * 1000)):03d}"
        for variant, mask in (("all", mean_mask), ("stable", stable_mask)):
            path = (
                v16.OUTPUT_DIR
                / f"v16_graph_time_equal_k_blend{tag}_{variant}_p1059.csv"
            )
            write_submission(path, test_orders, test_slices, mask)
            metadata = change_metadata(
                path,
                weight,
                test_orders,
                test_slices,
                mask,
                champion_mask,
                v11_scores,
                graph_time_scores,
                graph_time_per_seed,
                excluded_orders,
                oof,
                variant,
            )
            metadata["validation"] = v16.validate_submission(path, test_orders)
            path.with_suffix(".diff.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            path.with_suffix(".sha256").write_text(
                f"{metadata['sha256']}  {path.name}\n", encoding="ascii"
            )
            outputs.append(
                {
                    "path": str(path),
                    "weight": weight,
                    "variant": variant,
                    "changed_orders": metadata["changed_order_count"],
                    "removed": metadata["removed_count"],
                    "added": metadata["added_count"],
                    "oof_net_tp_delta": oof["net_tp_delta"],
                    "oof_fold_tp_deltas": oof["fold_tp_deltas"],
                    "sha256": metadata["sha256"],
                }
            )

    summary_path = v16.OUTPUT_DIR / "v16_graph_time_equal_k_generation.json"
    summary_path.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
