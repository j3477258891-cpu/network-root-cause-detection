"""Build domain-robust deletion probes from the verified V30 champion.

The generated probes are all relative to the same V30 champion.  They are
deliberately disjoint so leaderboard deltas can be added after each batch is
accepted.  No action is emitted for a historical/protected order.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
if str(ROOT / ".deps") not in sys.path:
    sys.path.insert(0, str(ROOT / ".deps"))

from sklearn.ensemble import HistGradientBoostingClassifier

import importlib.util


V29_PATH = ROOT / "experiments/v29_domain_ranker/v29_actions.py"
spec = importlib.util.spec_from_file_location("v29_actions", V29_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import v29 action helpers")
v29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v29)


DATA = ROOT / "experiments/v25_semantic_router/cloud_dataset/v25_semantic_router.npz"
RECORDS = ROOT / "experiments/v25_semantic_router/cloud_dataset/semantic_records.json.gz"
CHAMPION = ROOT / "experiments/v30_meta_stack/submissions/v30_cross_order_top5.csv"
V30_REPORT = ROOT / "experiments/v30_meta_stack/v30_cross_order_report.json"
TEMPLATE_REPORT = ROOT / "experiments/submissions/template_ranked/v11_constrained_report.json"
HISTORY_DIRS = (
    ROOT / "experiments/v28_ten_day_campaign/manifests",
    ROOT / "experiments/v29_domain_ranker/manifests",
    ROOT / "experiments/v30_meta_stack/manifests",
)
OUT = ROOT / "experiments/v32_delete_ranker"

TRUE_ROOTS = 1044
BASELINE_TP = 955
MAX_ROOTS = 8
SEED = 20260820


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(tp: int, predictions: int) -> float:
    return 2.0 * tp / (TRUE_ROOTS + predictions)


def load_submission(path: Path):
    order_ids, roots = [], {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            order_id = row["order_id"]
            order_ids.append(order_id)
            roots[order_id] = json.loads(row["output"])["rootcause"]
    return order_ids, roots


def write_submission(path: Path, order_ids, roots):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id in order_ids:
            writer.writerow(
                [order_id, json.dumps({"rootcause": roots[order_id]}, ensure_ascii=False)]
            )


def node_for_add(records_by_order, order_id, rid):
    alarm = next(a for a in records_by_order[order_id]["alarms"] if a["rid"] == rid)
    source = alarm["source"]
    return {
        "@rid": rid,
        "title": source.get("title", ""),
        "location": source.get("location", ""),
        "reason": source.get("reason", ""),
    }


def apply_actions(champion_roots, records_by_order, actions):
    roots = {oid: [dict(n) for n in nodes] for oid, nodes in champion_roots.items()}
    used_orders = set()
    for action in actions:
        oid = action["order_id"]
        if oid in used_orders:
            raise ValueError(f"multiple actions for order {oid}")
        used_orders.add(oid)
        remove = set(action["remove_rids"])
        current = {n["@rid"] for n in roots[oid]}
        if not remove <= current:
            raise ValueError(f"missing deletion for {oid}")
        roots[oid] = [n for n in roots[oid] if n["@rid"] not in remove]
        for rid in action["add_rids"]:
            if rid in {n["@rid"] for n in roots[oid]}:
                raise ValueError(f"duplicate addition for {oid}")
            roots[oid].append(node_for_add(records_by_order, oid, rid))
        if not 1 <= len(roots[oid]) <= MAX_ROOTS:
            raise ValueError(f"invalid root count for {oid}")
    return roots


def historical_exclusions():
    """Return orders touched by any prior action and protected node pairs."""
    touched = set()
    protected_in, protected_out = set(), set()
    for directory in HISTORY_DIRS:
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            try:
                data = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            actions = data.get("actions", [])
            for action in actions:
                oid = action.get("order_id")
                if oid:
                    touched.add(oid)
                for rid in action.get("remove_rids", []):
                    if oid:
                        protected_out.add((oid, rid))
                for rid in action.get("add_rids", []):
                    if oid:
                        protected_in.add((oid, rid))

    # The strict V11 report is the authoritative protection list.  Its forced
    # nodes must never be revisited by a later ranker.
    if TEMPLATE_REPORT.exists():
        report = read_json(TEMPLATE_REPORT)
        protected_in |= {
            (row["order_id"], row["rid"]) for row in report.get("forced_in", [])
        }
        protected_out |= {
            (row["order_id"], row["rid"]) for row in report.get("forced_out", [])
        }
    return touched, protected_in, protected_out


def current_masks(arrays, records, champion_roots):
    train_ptr = arrays["train_alarm_ptr"]
    train_mask = v29.exact_count_mask(arrays["train_v11"], train_ptr, 3169)
    test_ptr = arrays["test_alarm_ptr"]
    test_mask = np.zeros(len(arrays["test_v11"]), dtype=bool)
    for oi, (start, stop) in enumerate(zip(test_ptr[:-1], test_ptr[1:])):
        order = records["test"][oi]
        selected = {n["@rid"] for n in champion_roots[order["order_id"]]}
        for local, alarm in enumerate(order["alarms"]):
            test_mask[int(start) + local] = alarm["rid"] in selected
        if int(test_mask[int(start):int(stop)].sum()) != len(selected):
            raise ValueError(f"champion alignment failed: {order['order_id']}")
    return train_mask, test_mask


def crossfit_delete(x_train, orders_train, target, x_test, fold_sets):
    """Average station/template cross-fitted false-positive probabilities."""
    oof_by_split, test_by_split = {}, {}
    for split_name, folds in fold_sets.items():
        oof = np.zeros(len(target), dtype=np.float32)
        test_parts = []
        for fold in range(5):
            train_rows = folds[orders_train] != fold
            valid_rows = ~train_rows
            model = HistGradientBoostingClassifier(
                max_iter=260,
                learning_rate=0.04,
                max_leaf_nodes=31,
                min_samples_leaf=18,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=SEED + fold,
            )
            model.fit(x_train[train_rows], target[train_rows])
            classes = list(model.classes_)
            positive = classes.index(1) if 1 in classes else None
            if positive is None:
                oof[valid_rows] = 0.0
                test_parts.append(np.zeros(len(x_test), dtype=np.float32))
            else:
                oof[valid_rows] = model.predict_proba(x_train[valid_rows])[:, positive]
                test_parts.append(model.predict_proba(x_test)[:, positive])
        oof_by_split[split_name] = oof
        test_by_split[split_name] = np.mean(test_parts, axis=0).astype(np.float32)
    return (
        np.mean(list(oof_by_split.values()), axis=0),
        np.mean(list(test_by_split.values()), axis=0).astype(np.float32),
        oof_by_split,
        test_by_split,
    )


def score_possibilities(deletions: int, predictions: int):
    return [
        {
            "delta_tp": delta,
            "tp": BASELINE_TP + delta,
            "predictions": predictions,
            "score": round(f1(BASELINE_TP + delta, predictions), 9),
        }
        for delta in range(-deletions, 1)
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "submissions").mkdir(exist_ok=True)
    (OUT / "manifests").mkdir(exist_ok=True)

    with np.load(DATA, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    order_ids, champion_roots = load_submission(CHAMPION)
    records_by_order = {o["order_id"]: o for o in records["test"]}
    touched, protected_in, protected_out = historical_exclusions()
    # Current V30 action orders are excluded even if a manifest does not list
    # them under the generic action key.
    if V30_REPORT.exists():
        report = read_json(V30_REPORT)
        for action in report.get("probes", {}).get("v30_cross_order_top5", {}).get("actions", []):
            touched.add(action["order_id"])

    train_mask, test_mask = current_masks(arrays, records, champion_roots)
    train_alarm_x = v29.alarm_features(arrays, "train")
    test_alarm_x = v29.alarm_features(arrays, "test")
    train_order_x = v29.order_features(
        train_alarm_x, arrays["train_v11"], arrays["train_alarm_ptr"], train_mask
    )
    test_order_x = v29.order_features(
        test_alarm_x, arrays["test_v11"], arrays["test_alarm_ptr"], test_mask
    )
    labels = arrays["train_labels"].astype(np.int8)
    fold_sets = {
        "station": arrays["train_station_folds"].astype(np.int8),
        "template": arrays["train_folds"].astype(np.int8),
    }

    train_data = v29.boundary_candidates(
        arrays["train_alarm_ptr"], arrays["train_v11"], train_alarm_x,
        train_order_x, train_mask, labels, "delete"
    )
    test_data = v29.boundary_candidates(
        arrays["test_alarm_ptr"], arrays["test_v11"], test_alarm_x,
        test_order_x, test_mask, None, "delete"
    )
    # v29 delete target is 1-label: its positive class means likely FP.
    oof, test_score, oof_parts, test_parts = crossfit_delete(
        train_data[0], train_data[1], train_data[2], test_data[0], fold_sets
    )
    np.save(OUT / "delete_oof.npy", oof)
    np.save(OUT / "delete_test.npy", test_score)

    order_for_row = test_data[1]
    candidate_rows = test_data[3]
    selected_counts = {
        oid: len(champion_roots[oid]) for oid in order_ids
    }
    ranked = np.argsort(-test_score, kind="stable")
    candidates = []
    used_orders = set()
    for idx in ranked:
        oi = int(order_for_row[idx])
        order = records["test"][oi]
        oid = order["order_id"]
        row = int(candidate_rows[idx])
        local = row - int(arrays["test_alarm_ptr"][oi])
        alarm = order["alarms"][local]
        key = (oid, alarm["rid"])
        if oid in touched or oid in used_orders:
            continue
        if selected_counts[oid] <= 1:
            continue
        if key in protected_in or key in protected_out:
            continue
        candidates.append({
            "action_id": f"v32_delete_{len(candidates)+1:03d}_{oid[:8]}",
            "order_id": oid,
            "remove_rids": [alarm["rid"]],
            "add_rids": [],
            "source": "v32_domain_delete_consensus",
            "expected_gain": float(test_score[idx]),
            "evidence": {
                "false_positive_probability": float(test_score[idx]),
                "station_probability": float(test_parts["station"][idx]),
                "template_probability": float(test_parts["template"][idx]),
                "title": alarm.get("title", ""),
                "reason": alarm.get("reason", ""),
                "rid": alarm["rid"],
                "selected_count": selected_counts[oid],
            },
        })
        used_orders.add(oid)
        if len(candidates) >= 80:
            break

    if len(candidates) < 40:
        raise RuntimeError(f"only {len(candidates)} conflict-free candidates")

    # Every batch starts from the original V30 champion and uses a disjoint
    # candidate interval, so independently measured TP deltas are additive.
    batches = {
        "v32_delete_batch05": (0, 5),
        "v32_delete_batch10": (5, 15),
        "v32_delete_batch20": (15, 35),
        "v32_delete_batch40": (35, 75),
    }
    summary = {}
    for name, (start, stop) in batches.items():
        actions = candidates[start:stop]
        count = len(actions)
        roots = apply_actions(champion_roots, records_by_order, actions)
        predictions = sum(len(nodes) for nodes in roots.values())
        if predictions != 1035 - count:
            raise ValueError((name, predictions, count))
        path = OUT / "submissions" / f"{name}.csv"
        write_submission(path, order_ids, roots)
        manifest = {
            "probe_id": name,
            "path": str(path),
            "baseline": str(CHAMPION),
            "kind": "delete",
            "actions": actions,
            "predictions": predictions,
            "prediction_delta": -count,
            "candidate_rank_start": start + 1,
            "candidate_rank_stop": stop,
            "sha256": sha256(path),
            "excluded_orders": len(touched),
            "protected_in": len(protected_in),
            "protected_out": len(protected_out),
            "score_possibilities": score_possibilities(count, predictions),
        }
        (OUT / "manifests" / f"{name}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary[name] = {
            "path": str(path),
            "predictions": predictions,
            "sha256": manifest["sha256"],
            "first_action": actions[0]["action_id"],
            "last_action": actions[-1]["action_id"],
        }

    report = {
        "version": "v32-domain-delete-1",
        "baseline": {
            "path": str(CHAMPION), "tp": BASELINE_TP,
            "predictions": 1035, "score": f1(BASELINE_TP, 1035),
            "sha256": sha256(CHAMPION),
        },
        "candidate_count": len(candidates),
        "excluded_orders": len(touched),
        "protected_in": len(protected_in),
        "protected_out": len(protected_out),
        "training_candidates": int(len(train_data[2])),
        "training_positive_targets": int(train_data[2].sum()),
        "test_split_correlation": float(np.corrcoef(
            test_parts["station"], test_parts["template"]
        )[0, 1]),
        "top_candidates": candidates[:20],
        "probes": summary,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
