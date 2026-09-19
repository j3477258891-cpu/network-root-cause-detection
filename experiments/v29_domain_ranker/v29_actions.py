"""Topology-aware action models and leaderboard probe generator."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(r"D:\zgyidong")
DEPS = ROOT / ".deps"
if str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


DATASET = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
CHAMPION = ROOT / "experiments/v28_ten_day_campaign/submissions/day04_verified_checkpoint.csv"
HISTORY = (
    ROOT / "experiments/v28_ten_day_campaign/state.json",
    ROOT / "experiments/v28_ten_day_campaign/reports/phase1_precision_state.json",
)
OUTPUT = ROOT / "experiments/v29_domain_ranker"

KNOWN_PROBES = (
    "v29_delete_top10",
    "v29_add_top10",
    "v29_swap_top8",
    "v29_delete_add_checkpoint_UNVERIFIED",
)

TRUE_ROOTS = 1044
BASELINE_TP = 955
BASELINE_P = 1045
TARGET_TRAIN_P = 3169
MAX_ROOTS = 8
SEED = 20260817


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def remove_stale_probes(output_dir: Path) -> None:
    for name in KNOWN_PROBES:
        for path in (
            output_dir / "submissions" / f"{name}.csv",
            output_dir / "manifests" / f"{name}.json",
        ):
            if path.exists():
                path.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(tp: int, predictions: int) -> float:
    return 2.0 * tp / (TRUE_ROOTS + predictions)


def exact_count_mask(scores: np.ndarray, ptr: np.ndarray, target: int) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    optional: list[int] = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:MAX_ROOTS]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    optional_array = np.asarray(optional, dtype=np.int64)
    optional_array = optional_array[np.argsort(-scores[optional_array], kind="stable")]
    remaining = target - int(selected.sum())
    if not 0 <= remaining <= len(optional_array):
        raise ValueError((target, int(selected.sum()), len(optional_array)))
    selected[optional_array[:remaining]] = True
    return selected


def load_champion(path: Path) -> tuple[list[str], dict[str, list[dict]]]:
    order_ids: list[str] = []
    roots: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            order_id = row["order_id"]
            order_ids.append(order_id)
            roots[order_id] = json.loads(row["output"])["rootcause"]
    return order_ids, roots


def load_touched_orders(paths: tuple[Path, ...]) -> set[str]:
    touched = set()
    for path in paths:
        if not path.exists():
            continue
        state = read_json(path)
        for batch in state.get("batches", []):
            for action in batch.get("actions", []):
                touched.add(action["order_id"])
    return touched


def base_masks(arrays, records, champion_roots):
    train_ptr = arrays["train_alarm_ptr"]
    train_mask = exact_count_mask(arrays["train_v11"], train_ptr, TARGET_TRAIN_P)
    test_ptr = arrays["test_alarm_ptr"]
    test_mask = np.zeros(len(arrays["test_v11"]), dtype=bool)
    for order_index, (start, stop) in enumerate(zip(test_ptr[:-1], test_ptr[1:])):
        order = records["test"][order_index]
        selected = {node["@rid"] for node in champion_roots[order["order_id"]]}
        for local_index, alarm in enumerate(order["alarms"]):
            test_mask[int(start) + local_index] = alarm["rid"] in selected
        if int(test_mask[int(start):int(stop)].sum()) != len(selected):
            raise ValueError(f"champion alignment failed: {order['order_id']}")
    if int(test_mask.sum()) != BASELINE_P:
        raise ValueError(("unexpected champion count", int(test_mask.sum()), BASELINE_P))
    return train_mask, test_mask


def alarm_features(arrays, split: str) -> np.ndarray:
    score_columns = np.column_stack(
        [arrays[f"{split}_v11"], arrays[f"{split}_v13"], arrays[f"{split}_v19"]]
    ).astype(np.float32)
    missing = ~np.isfinite(score_columns)
    score_columns = np.where(missing, score_columns[:, [0]], score_columns)
    topology = np.column_stack(
        [
            arrays[f"{split}_path_node_type"],
            arrays[f"{split}_path_edge_type"],
            arrays[f"{split}_path_length"],
        ]
    ).astype(np.float32)
    static = np.nan_to_num(
        arrays[f"{split}_alarm_x"], nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    return np.column_stack([static, score_columns, missing.astype(np.float32), topology])


def order_features(
    alarm_x: np.ndarray, scores: np.ndarray, ptr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    output = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        local_x = alarm_x[start:stop]
        ranked = np.sort(scores[start:stop])[::-1]
        top_scores = np.full(8, -1.0, dtype=np.float32)
        top_scores[: min(8, len(ranked))] = ranked[:8]
        summary = np.array(
            [
                stop - start,
                int(mask[start:stop].sum()),
                float(scores[start:stop].mean()),
                float(scores[start:stop].std()),
            ],
            dtype=np.float32,
        )
        output.append(np.concatenate([local_x.mean(0), local_x.max(0), summary, top_scores]))
    return np.asarray(output, dtype=np.float32)


def boundary_candidates(
    ptr,
    v11,
    alarm_x,
    order_x,
    mask,
    labels=None,
    kind="delete",
):
    rows, orders, targets, alarm_rows = [], [], [], []
    for order_index, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        if kind == "delete":
            candidates = np.flatnonzero(mask[start:stop])
            candidates = candidates[
                np.argsort(v11[start:stop][candidates], kind="stable")[:3]
            ]
        elif kind == "add":
            candidates = np.flatnonzero(~mask[start:stop])
            candidates = candidates[
                np.argsort(-v11[start:stop][candidates], kind="stable")[:3]
            ]
        else:
            raise ValueError(kind)
        for rank, local_index in enumerate(candidates):
            row = start + int(local_index)
            extra = np.array(
                [rank, v11[row], stop - start, int(mask[start:stop].sum())],
                dtype=np.float32,
            )
            rows.append(np.concatenate([alarm_x[row], order_x[order_index], extra]))
            orders.append(order_index)
            alarm_rows.append(row)
            if labels is not None:
                targets.append(int(1 - labels[row]) if kind == "delete" else int(labels[row]))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(orders, dtype=np.int32),
        np.asarray(targets, dtype=np.int8),
        np.asarray(alarm_rows, dtype=np.int32),
    )


def pair_candidates(ptr, v11, alarm_x, mask, labels=None):
    rows, orders, targets, add_rows, remove_rows = [], [], [], [], []
    for order_index, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        start, stop = int(start), int(stop)
        selected = np.flatnonzero(mask[start:stop])
        unselected = np.flatnonzero(~mask[start:stop])
        selected = selected[
            np.argsort(v11[start:stop][selected], kind="stable")[: min(3, len(selected))]
        ]
        unselected = unselected[
            np.argsort(-v11[start:stop][unselected], kind="stable")[: min(3, len(unselected))]
        ]
        for remove_local in selected:
            for add_local in unselected:
                add_row = start + int(add_local)
                remove_row = start + int(remove_local)
                extra = np.array(
                    [v11[add_row] - v11[remove_row], len(selected), stop - start],
                    dtype=np.float32,
                )
                rows.append(
                    np.concatenate(
                        [
                            alarm_x[add_row],
                            alarm_x[remove_row],
                            alarm_x[add_row] - alarm_x[remove_row],
                            extra,
                        ]
                    )
                )
                orders.append(order_index)
                add_rows.append(add_row)
                remove_rows.append(remove_row)
                if labels is not None:
                    targets.append(int(labels[add_row]) - int(labels[remove_row]))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(orders, dtype=np.int32),
        np.asarray(targets, dtype=np.int8),
        np.asarray(add_rows, dtype=np.int32),
        np.asarray(remove_rows, dtype=np.int32),
    )


def make_model(pair: bool, quick: bool):
    return HistGradientBoostingClassifier(
        max_iter=60 if quick else (250 if pair else 220),
        learning_rate=0.045,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.5,
        class_weight="balanced",
        random_state=SEED,
    )


def positive_probability(model, features, positive_class=1):
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if positive_class not in classes:
        return np.zeros(len(features), dtype=np.float32)
    return probabilities[:, classes.index(positive_class)].astype(np.float32)


def crossfit_binary(
    train_x,
    train_orders,
    targets,
    test_x,
    fold_sets,
    quick,
):
    oof_by_split, test_by_split = {}, {}
    for split_name, folds in fold_sets.items():
        oof = np.zeros(len(targets), dtype=np.float32)
        test_predictions = []
        for fold in range(5):
            train_rows = folds[train_orders] != fold
            validation_rows = ~train_rows
            model = make_model(pair=False, quick=quick)
            model.fit(train_x[train_rows], targets[train_rows])
            oof[validation_rows] = positive_probability(model, train_x[validation_rows])
            test_predictions.append(positive_probability(model, test_x))
        oof_by_split[split_name] = oof
        test_by_split[split_name] = np.mean(test_predictions, axis=0)
    return (
        np.mean(list(oof_by_split.values()), axis=0),
        np.mean(list(test_by_split.values()), axis=0),
        oof_by_split,
        test_by_split,
    )


def crossfit_pair(train_x, train_orders, targets, test_x, folds, quick):
    oof = np.zeros(len(targets), dtype=np.float32)
    test_predictions = []
    for fold in range(5):
        train_rows = folds[train_orders] != fold
        validation_rows = ~train_rows
        model = make_model(pair=True, quick=quick)
        model.fit(train_x[train_rows], targets[train_rows])
        validation_probability = model.predict_proba(train_x[validation_rows])
        classes = list(model.classes_)
        oof[validation_rows] = (
            validation_probability[:, classes.index(1)]
            - validation_probability[:, classes.index(-1)]
        )
        test_probability = model.predict_proba(test_x)
        test_predictions.append(
            test_probability[:, classes.index(1)] - test_probability[:, classes.index(-1)]
        )
    return oof, np.mean(test_predictions, axis=0)


def best_per_order(scores, candidate_orders, order_count):
    chosen = []
    for order_index in range(order_count):
        indices = np.flatnonzero(candidate_orders == order_index)
        if len(indices):
            chosen.append(int(indices[np.argmax(scores[indices])]))
    chosen_array = np.asarray(chosen, dtype=np.int32)
    return chosen_array[np.argsort(-scores[chosen_array], kind="stable")]


def binary_curve(scores, orders, targets, order_count):
    ranked = best_per_order(scores, orders, order_count)
    curve = {}
    for n in (5, 8, 10, 12, 15, 20, 30, 40, 50, 75, 100):
        subset = ranked[:n]
        beneficial = int(targets[subset].sum())
        curve[str(n)] = {
            "beneficial": beneficial,
            "harmful": int(len(subset) - beneficial),
            "minimum_score": float(scores[subset[-1]]),
        }
    return ranked, curve


def pair_curve(scores, orders, targets, order_count):
    ranked = best_per_order(scores, orders, order_count)
    curve = {}
    for n in (5, 8, 10, 12, 15, 20, 30, 40, 50):
        subset = ranked[:n]
        values = targets[subset]
        curve[str(n)] = {
            "delta_tp": int(values.sum()),
            "positive": int((values == 1).sum()),
            "neutral": int((values == 0).sum()),
            "negative": int((values == -1).sum()),
            "minimum_utility": float(scores[subset[-1]]),
        }
    return ranked, curve


def alarm_identity(records, ptr, order_index, row):
    order = records["test"][order_index]
    local_index = int(row - int(ptr[order_index]))
    alarm = order["alarms"][local_index]
    return order, alarm


def boundary_actions(
    kind,
    ranked,
    scores,
    candidate_orders,
    candidate_rows,
    records,
    test_ptr,
    touched_orders,
    excluded_orders,
    champion_roots,
    limit,
    split_scores,
):
    actions = []
    for candidate_index in ranked:
        order_index = int(candidate_orders[candidate_index])
        order, alarm = alarm_identity(records, test_ptr, order_index, candidate_rows[candidate_index])
        order_id = order["order_id"]
        if order_id in touched_orders or order_id in excluded_orders:
            continue
        current_count = len(champion_roots[order_id])
        if kind == "delete" and current_count <= 1:
            continue
        if kind == "add" and current_count >= MAX_ROOTS:
            continue
        action = {
            "action_id": f"v29_{kind}_{len(actions) + 1:02d}_{order_id[:8]}",
            "order_id": order_id,
            "remove_rids": [alarm["rid"]] if kind == "delete" else [],
            "add_rids": [alarm["rid"]] if kind == "add" else [],
            "source": "v29_topology_boundary",
            "expected_gain": float(scores[candidate_index]),
            "evidence": {
                "consensus_probability": float(scores[candidate_index]),
                **{
                    f"{name}_probability": float(values[candidate_index])
                    for name, values in split_scores.items()
                },
                "title": alarm.get("title", ""),
            },
        }
        actions.append(action)
        excluded_orders.add(order_id)
        if len(actions) == limit:
            break
    return actions


def swap_actions(
    ranked,
    scores,
    candidate_orders,
    add_rows,
    remove_rows,
    records,
    test_ptr,
    touched_orders,
    excluded_orders,
    limit,
):
    actions = []
    for candidate_index in ranked:
        order_index = int(candidate_orders[candidate_index])
        order, add_alarm = alarm_identity(records, test_ptr, order_index, add_rows[candidate_index])
        _, remove_alarm = alarm_identity(records, test_ptr, order_index, remove_rows[candidate_index])
        order_id = order["order_id"]
        if order_id in touched_orders or order_id in excluded_orders:
            continue
        actions.append(
            {
                "action_id": f"v29_swap_{len(actions) + 1:02d}_{order_id[:8]}",
                "order_id": order_id,
                "remove_rids": [remove_alarm["rid"]],
                "add_rids": [add_alarm["rid"]],
                "source": "v29_topology_pair",
                "expected_gain": float(scores[candidate_index]),
                "evidence": {
                    "expected_tp_utility": float(scores[candidate_index]),
                    "remove_title": remove_alarm.get("title", ""),
                    "add_title": add_alarm.get("title", ""),
                },
            }
        )
        excluded_orders.add(order_id)
        if len(actions) == limit:
            break
    return actions


def node_for_add(records_by_order, order_id, rid):
    alarm = next(item for item in records_by_order[order_id]["alarms"] if item["rid"] == rid)
    source = alarm["source"]
    return {
        "@rid": rid,
        "title": source.get("title", ""),
        "location": source.get("location", ""),
        "reason": source.get("reason", ""),
    }


def apply_actions(champion_roots, records_by_order, actions):
    output = {order_id: [dict(node) for node in nodes] for order_id, nodes in champion_roots.items()}
    seen_orders = set()
    for action in actions:
        order_id = action["order_id"]
        if order_id in seen_orders:
            raise ValueError(f"multiple actions for order {order_id}")
        seen_orders.add(order_id)
        remove = set(action["remove_rids"])
        existing = {node["@rid"] for node in output[order_id]}
        if not remove <= existing:
            raise ValueError(("remove missing", action))
        output[order_id] = [node for node in output[order_id] if node["@rid"] not in remove]
        for rid in action["add_rids"]:
            if rid in {node["@rid"] for node in output[order_id]}:
                raise ValueError(("duplicate add", action))
            output[order_id].append(node_for_add(records_by_order, order_id, rid))
        if not 1 <= len(output[order_id]) <= MAX_ROOTS:
            raise ValueError(("invalid root count", order_id, len(output[order_id])))
    return output


def write_submission(path, order_ids, roots):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id in order_ids:
            writer.writerow(
                [order_id, json.dumps({"rootcause": roots[order_id]}, ensure_ascii=False)]
            )


def score_possibilities(kind, action_count, predictions):
    if kind == "delete":
        deltas = range(-action_count, 1)
    elif kind == "add":
        deltas = range(0, action_count + 1)
    else:
        deltas = range(-action_count, action_count + 1)
    return [
        {
            "delta_tp": delta,
            "tp": BASELINE_TP + delta,
            "score": round(f1(BASELINE_TP + delta, predictions), 9),
        }
        for delta in deltas
    ]


def emit_probe(name, kind, actions, order_ids, champion_roots, records_by_order, output_dir):
    roots = apply_actions(champion_roots, records_by_order, actions)
    predictions = sum(map(len, roots.values()))
    path = output_dir / "submissions" / f"{name}.csv"
    write_submission(path, order_ids, roots)
    prediction_delta = sum(len(action["add_rids"]) - len(action["remove_rids"]) for action in actions)
    if predictions != BASELINE_P + prediction_delta:
        raise ValueError((predictions, prediction_delta))
    manifest = {
        "probe_id": name,
        "path": str(path),
        "baseline": str(CHAMPION),
        "kind": kind,
        "actions": actions,
        "predictions": predictions,
        "prediction_delta": prediction_delta,
        "sha256": file_sha256(path),
        "score_possibilities": score_possibilities(kind, len(actions), predictions),
    }
    write_json(output_dir / "manifests" / f"{name}.json", manifest)
    return {"path": str(path), **manifest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    remove_stale_probes(args.output)

    with np.load(DATASET, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    order_ids, champion_roots = load_champion(CHAMPION)
    records_by_order = {order["order_id"]: order for order in records["test"]}
    touched_orders = load_touched_orders(HISTORY)
    train_mask, test_mask = base_masks(arrays, records, champion_roots)

    train_alarm_x = alarm_features(arrays, "train")
    test_alarm_x = alarm_features(arrays, "test")
    train_order_x = order_features(
        train_alarm_x, arrays["train_v11"], arrays["train_alarm_ptr"], train_mask
    )
    test_order_x = order_features(
        test_alarm_x, arrays["test_v11"], arrays["test_alarm_ptr"], test_mask
    )
    labels = arrays["train_labels"].astype(np.int8)
    fold_sets = {
        "station": arrays["train_station_folds"].astype(np.int8),
        "template": arrays["train_folds"].astype(np.int8),
    }

    reports, ranked_test, test_parts, candidate_data = {}, {}, {}, {}
    for kind in ("delete", "add"):
        train_data = boundary_candidates(
            arrays["train_alarm_ptr"], arrays["train_v11"], train_alarm_x,
            train_order_x, train_mask, labels, kind,
        )
        test_data = boundary_candidates(
            arrays["test_alarm_ptr"], arrays["test_v11"], test_alarm_x,
            test_order_x, test_mask, None, kind,
        )
        oof, test_score, _, split_test = crossfit_binary(
            train_data[0], train_data[1], train_data[2], test_data[0], fold_sets, args.quick
        )
        _, curve = binary_curve(oof, train_data[1], train_data[2], len(records["train"]))
        reports[kind] = {
            "candidate_rows": len(train_data[2]),
            "positive_targets": int(train_data[2].sum()),
            "curve": curve,
            "test_split_correlation": float(
                np.corrcoef(split_test["station"], split_test["template"])[0, 1]
            ),
        }
        ranked_test[kind] = best_per_order(test_score, test_data[1], len(records["test"]))
        test_parts[kind] = {"consensus": test_score, **split_test}
        candidate_data[kind] = test_data

    train_pair = pair_candidates(
        arrays["train_alarm_ptr"], arrays["train_v11"], train_alarm_x, train_mask, labels
    )
    test_pair = pair_candidates(
        arrays["test_alarm_ptr"], arrays["test_v11"], test_alarm_x, test_mask, None
    )
    pair_oof, pair_test_score = crossfit_pair(
        train_pair[0], train_pair[1], train_pair[2], test_pair[0],
        arrays["train_station_folds"].astype(np.int8), args.quick,
    )
    _, reports["swap"] = pair_curve(
        pair_oof, train_pair[1], train_pair[2], len(records["train"])
    )
    ranked_test["swap"] = best_per_order(
        pair_test_score, test_pair[1], len(records["test"])
    )

    gate = {
        "delete": reports["delete"]["curve"]["10"]["beneficial"] >= 8
        and reports["delete"]["curve"]["10"]["harmful"] <= 2,
        "add": reports["add"]["curve"]["10"]["beneficial"] >= 7,
        "swap": reports["swap"]["8"]["delta_tp"] >= 4
        and reports["swap"]["8"]["negative"] == 0,
    }

    used_orders: set[str] = set()
    actions = {}
    if gate["delete"]:
        data = candidate_data["delete"]
        actions["delete"] = boundary_actions(
            "delete", ranked_test["delete"], test_parts["delete"]["consensus"],
            data[1], data[3], records, arrays["test_alarm_ptr"], touched_orders,
            used_orders, champion_roots, 10,
            {k: v for k, v in test_parts["delete"].items() if k != "consensus"},
        )
    if gate["add"]:
        data = candidate_data["add"]
        actions["add"] = boundary_actions(
            "add", ranked_test["add"], test_parts["add"]["consensus"],
            data[1], data[3], records, arrays["test_alarm_ptr"], touched_orders,
            used_orders, champion_roots, 10,
            {k: v for k, v in test_parts["add"].items() if k != "consensus"},
        )
    if gate["swap"]:
        actions["swap"] = swap_actions(
            ranked_test["swap"], pair_test_score, test_pair[1], test_pair[3], test_pair[4],
            records, arrays["test_alarm_ptr"], touched_orders, used_orders, 8,
        )

    if any(
        len(actions.get(kind, [])) != count
        for kind, count in (("delete", 10), ("add", 10), ("swap", 8))
        if gate[kind]
    ):
        raise ValueError("insufficient conflict-free actions")

    probes = {}
    if gate["delete"]:
        probes["delete"] = emit_probe(
            "v29_delete_top10", "delete", actions["delete"], order_ids,
            champion_roots, records_by_order, args.output,
        )
    if gate["add"]:
        probes["add"] = emit_probe(
            "v29_add_top10", "add", actions["add"], order_ids,
            champion_roots, records_by_order, args.output,
        )
    if gate["swap"]:
        probes["swap"] = emit_probe(
            "v29_swap_top8", "swap", actions["swap"], order_ids,
            champion_roots, records_by_order, args.output,
        )
    if gate["delete"] and gate["add"]:
        probes["combined"] = emit_probe(
            "v29_delete_add_checkpoint_UNVERIFIED", "swap",
            actions["delete"] + actions["add"], order_ids,
            champion_roots, records_by_order, args.output,
        )

    report = {
        "version": "v29-topology-actions-1",
        "baseline": {
            "path": str(CHAMPION),
            "tp": BASELINE_TP,
            "predictions": BASELINE_P,
            "score": f1(BASELINE_TP, BASELINE_P),
            "sha256": file_sha256(CHAMPION),
        },
        "touched_orders_excluded": len(touched_orders),
        "features": int(train_alarm_x.shape[1]),
        "quick": args.quick,
        "oof": reports,
        "gate": gate,
        "actions": actions,
        "probes": probes,
    }
    write_json(args.output / "reports/v29_report.json", report)
    print(json.dumps({"gate": gate, "probes": probes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
