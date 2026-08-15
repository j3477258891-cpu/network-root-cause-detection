"""Train graph+time V16 and build protected low-weight blend candidates."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, str(Path(__file__).parent))
import v16_structure_signal as v16


WEIGHTS = (0.195, 0.010)


def write_mask_submission(path, orders, slices, mask):
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


def protected_mask(raw_mask, champion_mask, scores, orders, slices, excluded_orders):
    mask = raw_mask.copy()
    excluded_indices = {
        index for index, order in enumerate(orders) if order["id"] in excluded_orders
    }
    for order_index in excluded_indices:
        order_slice = slices[order_index]
        mask[order_slice] = champion_mask[order_slice]

    current = int(np.sum(mask))
    if current > v16.TARGET_TEST_COUNT:
        removable = []
        for order_index, order_slice in enumerate(slices):
            if order_index in excluded_indices or np.sum(mask[order_slice]) <= 1:
                continue
            for local_index in np.flatnonzero(mask[order_slice]):
                row_index = order_slice.start + local_index
                removable.append((float(scores[row_index]), row_index))
        for _, row_index in sorted(removable):
            if current <= v16.TARGET_TEST_COUNT:
                break
            order_index = next(
                index
                for index, order_slice in enumerate(slices)
                if order_slice.start <= row_index < order_slice.stop
            )
            if np.sum(mask[slices[order_index]]) <= 1:
                continue
            mask[row_index] = False
            current -= 1
    elif current < v16.TARGET_TEST_COUNT:
        additions = []
        for order_index, order_slice in enumerate(slices):
            if order_index in excluded_indices or np.sum(mask[order_slice]) >= v16.MAX_ROOTCAUSES:
                continue
            for local_index in np.flatnonzero(~mask[order_slice]):
                row_index = order_slice.start + local_index
                additions.append((float(scores[row_index]), row_index))
        for _, row_index in sorted(additions, reverse=True):
            if current >= v16.TARGET_TEST_COUNT:
                break
            order_index = next(
                index
                for index, order_slice in enumerate(slices)
                if order_slice.start <= row_index < order_slice.stop
            )
            if np.sum(mask[slices[order_index]]) >= v16.MAX_ROOTCAUSES:
                continue
            mask[row_index] = True
            current += 1
    assert current == v16.TARGET_TEST_COUNT
    for order_index in excluded_indices:
        assert np.array_equal(mask[slices[order_index]], champion_mask[slices[order_index]])
    return mask


def diff_metadata(path, orders, slices, mask, champion_mask, weight, excluded_orders):
    changes = []
    for order_index, order in enumerate(orders):
        order_slice = slices[order_index]
        before = {
            order["alarms"][int(index)]["@rid"]
            for index in np.flatnonzero(champion_mask[order_slice])
        }
        after = {
            order["alarms"][int(index)]["@rid"]
            for index in np.flatnonzero(mask[order_slice])
        }
        if before != after:
            changes.append(
                {
                    "order_id": order["id"],
                    "removed": sorted(before - after),
                    "added": sorted(after - before),
                    "delta_p": len(after) - len(before),
                }
            )
    assert not ({item["order_id"] for item in changes} & excluded_orders)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_champion": str(v16.CHAMPION),
        "weight_graph_time": weight,
        "weight_v11": 1.0 - weight,
        "excluded_order_count": len(excluded_orders),
        "changed_order_count": len(changes),
        "removed_count": sum(len(item["removed"]) for item in changes),
        "added_count": sum(len(item["added"]) for item in changes),
        "delta_p": sum(item["delta_p"] for item in changes),
        "changes": changes,
        "sha256": sha,
    }


def main():
    train_orders = v16.load_orders(v16.TRAIN_DIR, True)
    test_orders = v16.load_orders(v16.TEST_DIR, False)
    train_x = np.load(v16.OUTPUT_DIR / "v16_train_features.npy")[:, :97]
    test_x = np.load(v16.OUTPUT_DIR / "v16_test_features.npy")[:, :97]
    labels = np.asarray(
        [
            int(node["@rid"] in order["roots"])
            for order in train_orders
            for node in order["alarms"]
        ],
        dtype=np.int8,
    )
    test_slices = []
    cursor = 0
    for order in test_orders:
        test_slices.append(slice(cursor, cursor + len(order["alarms"])))
        cursor += len(order["alarms"])

    per_seed = []
    for seed in v16.SEEDS:
        model = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=16,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train_x, labels)
        per_seed.append(model.predict_proba(test_x)[:, 1])
    graph_time_scores = np.mean(per_seed, axis=0)
    np.save(v16.OUTPUT_DIR / "v16_graph_time_test_scores.npy", graph_time_scores)
    for seed, scores in zip(v16.SEEDS, per_seed):
        np.save(v16.OUTPUT_DIR / f"v16_graph_time_test_seed_{seed}.npy", scores)

    v11_scores = np.load(v16.V11_DIR / "v11_test_scores.npy")
    champion_rows = v16.read_submission(v16.CHAMPION)
    champion_mask = np.zeros(len(graph_time_scores), dtype=bool)
    for order_index, order in enumerate(test_orders):
        selected = {item["@rid"] for item in champion_rows[order["id"]]}
        for local_index, node in enumerate(order["alarms"]):
            champion_mask[test_slices[order_index].start + local_index] = (
                node["@rid"] in selected
            )
    exclusions = json.loads(v16.EXCLUSIONS.read_text(encoding="utf-8"))
    excluded_orders = set(exclusions["total"])
    locks = json.loads(v16.LOCK_REPORT.read_text(encoding="utf-8"))
    excluded_orders.update(
        item["order_id"] for item in locks["forced_in"] + locks["forced_out"]
    )

    outputs = []
    for weight in WEIGHTS:
        scores = (1.0 - weight) * v11_scores + weight * graph_time_scores
        raw_mask = v16.exact_count_mask(scores, test_slices, v16.TARGET_TEST_COUNT)
        mask = protected_mask(
            raw_mask,
            champion_mask,
            scores,
            test_orders,
            test_slices,
            excluded_orders,
        )
        tag = f"{int(round(weight * 1000)):03d}"
        path = v16.OUTPUT_DIR / f"v16_graph_time_blend{tag}_protected_p1059.csv"
        write_mask_submission(path, test_orders, test_slices, mask)
        validation = v16.validate_submission(path, test_orders)
        metadata = diff_metadata(
            path,
            test_orders,
            test_slices,
            mask,
            champion_mask,
            weight,
            excluded_orders,
        )
        metadata["validation"] = validation
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
                "changed_orders": metadata["changed_order_count"],
                "removed": metadata["removed_count"],
                "added": metadata["added_count"],
                "sha256": metadata["sha256"],
            }
        )

    (v16.OUTPUT_DIR / "v16_graph_time_generation.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
