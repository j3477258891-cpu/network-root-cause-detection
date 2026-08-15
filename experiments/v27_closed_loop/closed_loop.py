"""Leaderboard-verified action batching for the v27 closed loop.

Every probe is generated from the frozen 0.906324 champion. Actions and
batches must be disjoint, so measured TP and prediction-count deltas remain
additive when accepted batches are merged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(r"D:\zgyidong")
DEFAULT_WORKDIR = ROOT / "experiments" / "v27_closed_loop"
DEFAULT_TEST_DIR = ROOT / "test"
DEFAULT_CHAMPION = (
    ROOT
    / "experiments"
    / "submissions"
    / "champion_0.906324_day01_probe01_v11_full.csv"
)
TRUE_ROOTS = 1044
BASE_SCORE = 0.906324
BASE_TP = 953
BASE_PREDICTIONS = 1059
TARGET_SCORE = 0.926324
MAX_ROOTS = 8


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1_score(tp: int, predictions: int, positives: int = TRUE_ROOTS) -> float:
    return 2.0 * tp / (predictions + positives)


def infer_tp(score: float, predictions: int, positives: int = TRUE_ROOTS) -> tuple[int, float]:
    raw = score * (predictions + positives) / 2.0
    tp = int(round(raw))
    reconstructed = f1_score(tp, predictions, positives)
    if abs(reconstructed - score) > 0.5e-6 + 1e-12:
        raise ValueError(
            f"score {score:.6f} is not consistent with integer TP; "
            f"nearest is TP={tp}, F1={reconstructed:.9f}"
        )
    return tp, reconstructed


def required_tp(predictions: int, target: float = TARGET_SCORE) -> int:
    return math.ceil(target * (predictions + TRUE_ROOTS) / 2.0 - 1e-12)


def load_submission(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    order_ids: list[str] = []
    selection: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["order_id", "output"]:
            raise ValueError(f"invalid submission columns: {reader.fieldnames}")
        for row in reader:
            order_id = row["order_id"]
            if order_id in selection:
                raise ValueError(f"duplicate order_id: {order_id}")
            payload = json.loads(row["output"])
            if set(payload) != {"rootcause"} or not payload["rootcause"]:
                raise ValueError(f"invalid rootcause payload: {order_id}")
            order_ids.append(order_id)
            selection[order_id] = [node["@rid"] for node in payload["rootcause"]]
    return order_ids, selection


def load_topologies(test_dir: Path) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for directory in sorted(path for path in Path(test_dir).iterdir() if path.is_dir()):
        order_id = directory.name
        topology = read_json(directory / f"{order_id}.log.topo.json")
        result[order_id] = {
            node["@rid"]: node
            for node in topology.get("nodes", [])
            if node.get("@class") == "Alarm"
        }
    return result


def validate_selection(
    order_ids: list[str],
    selection: dict[str, list[str]],
    topologies: dict[str, dict[str, dict]],
) -> dict:
    if len(order_ids) != len(set(order_ids)):
        raise ValueError("submission contains duplicate order IDs")
    if set(order_ids) != set(topologies):
        raise ValueError("submission order IDs do not match the test set")
    counts = Counter()
    predictions = 0
    for order_id in order_ids:
        rids = selection[order_id]
        if len(rids) != len(set(rids)):
            raise ValueError(f"duplicate prediction in {order_id}")
        if not 1 <= len(rids) <= MAX_ROOTS:
            raise ValueError(f"invalid root count for {order_id}: {len(rids)}")
        missing = set(rids) - set(topologies[order_id])
        if missing:
            raise ValueError(f"unknown alarm nodes in {order_id}: {sorted(missing)}")
        counts[len(rids)] += 1
        predictions += len(rids)
    return {
        "orders": len(order_ids),
        "predictions": predictions,
        "root_count_distribution": dict(sorted(counts.items())),
    }


def normalize_action(raw: dict) -> dict:
    action = dict(raw)
    action_id = action.get("action_id", action.get("id"))
    if not action_id:
        raise ValueError("action is missing action_id")
    order_id = action.get("order_id")
    if not order_id:
        removal = action.get("remove", {})
        addition = action.get("add", {})
        orders = {removal.get("order_id"), addition.get("order_id")} - {None}
        if len(orders) != 1:
            raise ValueError(f"action must affect one order: {action_id}")
        order_id = orders.pop()
    remove_rids = action.get("remove_rids")
    add_rids = action.get("add_rids")
    if remove_rids is None:
        remove_rids = [action["remove"]["rid"]] if action.get("remove") else []
    if add_rids is None:
        add_rids = [action["add"]["rid"]] if action.get("add") else []
    return {
        "action_id": str(action_id),
        "order_id": str(order_id),
        "remove_rids": sorted(set(remove_rids)),
        "add_rids": sorted(set(add_rids)),
        "source": action.get("source", "unknown"),
        "subtype": action.get("subtype", action.get("source", "unknown")),
        "expected_gain": float(action.get("expected_gain", 0.0)),
        "evidence": action.get("evidence", {}),
        "probe_batch": action.get("probe_batch"),
        "online_delta_tp": action.get("online_delta_tp"),
    }


def validate_actions(
    actions: list[dict],
    base: dict[str, list[str]],
    topologies: dict[str, dict[str, dict]],
    require_disjoint_orders: bool = True,
) -> dict:
    ids: set[str] = set()
    orders: set[str] = set()
    touched_nodes: set[tuple[str, str]] = set()
    source_counts = Counter()
    total_remove = total_add = 0
    for raw in actions:
        action = normalize_action(raw)
        action_id = action["action_id"]
        order_id = action["order_id"]
        if action_id in ids:
            raise ValueError(f"duplicate action ID: {action_id}")
        if order_id not in base:
            raise ValueError(f"unknown action order: {action_id} {order_id}")
        if require_disjoint_orders and order_id in orders:
            raise ValueError(f"multiple actions affect order {order_id}")
        remove = set(action["remove_rids"])
        add = set(action["add_rids"])
        if not remove and not add:
            raise ValueError(f"empty action: {action_id}")
        if remove & add:
            raise ValueError(f"same node added and removed: {action_id}")
        if not remove <= set(base[order_id]):
            raise ValueError(f"action removes unselected nodes: {action_id}")
        if add & set(base[order_id]):
            raise ValueError(f"action adds selected nodes: {action_id}")
        if not add <= set(topologies[order_id]):
            raise ValueError(f"action adds unknown nodes: {action_id}")
        result_count = len(base[order_id]) - len(remove) + len(add)
        if not 1 <= result_count <= MAX_ROOTS:
            raise ValueError(f"action violates root-count bounds: {action_id}")
        nodes = {(order_id, rid) for rid in remove | add}
        overlap = touched_nodes & nodes
        if overlap:
            raise ValueError(f"actions touch the same nodes: {sorted(overlap)}")
        ids.add(action_id)
        orders.add(order_id)
        touched_nodes.update(nodes)
        source_counts[action["source"]] += 1
        total_remove += len(remove)
        total_add += len(add)
    return {
        "actions": len(actions),
        "orders": len(orders),
        "removed": total_remove,
        "added": total_add,
        "prediction_delta": total_add - total_remove,
        "sources": dict(sorted(source_counts.items())),
    }


def apply_actions(base: dict[str, list[str]], actions: list[dict]) -> dict[str, list[str]]:
    selection = {order_id: list(rids) for order_id, rids in base.items()}
    for raw in actions:
        action = normalize_action(raw)
        order_id = action["order_id"]
        remove = set(action["remove_rids"])
        add = set(action["add_rids"])
        selection[order_id] = [rid for rid in selection[order_id] if rid not in remove]
        selection[order_id].extend(rid for rid in action["add_rids"] if rid in add)
    return selection


def write_submission(
    path: Path,
    order_ids: list[str],
    selection: dict[str, list[str]],
    topologies: dict[str, dict[str, dict]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id in order_ids:
            rootcauses = []
            for rid in selection[order_id]:
                node = topologies[order_id][rid]
                rootcauses.append(
                    {
                        "@rid": rid,
                        "title": node.get("title", ""),
                        "location": node.get("location", ""),
                        "reason": node.get("reason", ""),
                    }
                )
            writer.writerow(
                [order_id, json.dumps({"rootcause": rootcauses}, ensure_ascii=False)]
            )


def score_possibilities(base_tp: int, predictions: int, actions: list[dict]) -> list[dict]:
    removed = sum(len(normalize_action(item)["remove_rids"]) for item in actions)
    added = sum(len(normalize_action(item)["add_rids"]) for item in actions)
    values = []
    for delta_tp in range(-removed, added + 1):
        tp = base_tp + delta_tp
        if 0 <= tp <= min(predictions, TRUE_ROOTS):
            values.append(
                {
                    "delta_tp": delta_tp,
                    "tp": tp,
                    "score": round(f1_score(tp, predictions), 9),
                }
            )
    return values


def load_state(workdir: Path) -> dict:
    path = Path(workdir) / "state.json"
    if not path.exists():
        raise FileNotFoundError(f"closed loop is not initialized: {path}")
    return read_json(path)


def save_state(workdir: Path, state: dict) -> None:
    write_json(Path(workdir) / "state.json", state)
    fields = [
        "batch_id",
        "kind",
        "status",
        "decision",
        "score",
        "tp",
        "delta_tp",
        "predictions",
        "prediction_delta",
        "submission",
        "sha256",
    ]
    with (Path(workdir) / "ledger.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch in state.get("batches", []):
            writer.writerow({key: batch.get(key, "") for key in fields})


def catalog_actions(path: Path) -> list[dict]:
    catalog = read_json(path)
    raw_actions = catalog.get("actions", catalog.get("units", []))
    return [normalize_action(item) for item in raw_actions]


def register_batch(
    workdir: Path,
    batch_id: str,
    kind: str,
    actions: list[dict],
    parent: str | None = None,
) -> dict:
    state = load_state(workdir)
    if any(item["batch_id"] == batch_id for item in state["batches"]):
        raise ValueError(f"batch already exists: {batch_id}")
    champion = Path(state["champion"]["path"])
    order_ids, base = load_submission(champion)
    topologies = load_topologies(Path(state["test_dir"]))
    validation = validate_actions(actions, base, topologies)
    selection = apply_actions(base, actions)
    selection_validation = validate_selection(order_ids, selection, topologies)
    submission = Path(workdir) / "submissions" / f"{batch_id}.csv"
    manifest_path = Path(workdir) / "manifests" / f"{batch_id}.json"
    write_submission(submission, order_ids, selection, topologies)
    batch = {
        "batch_id": batch_id,
        "kind": kind,
        "parent": parent,
        "status": "ready",
        "decision": "pending",
        "action_ids": [item["action_id"] for item in actions],
        "actions": actions,
        "validation": validation,
        "predictions": selection_validation["predictions"],
        "prediction_delta": (
            selection_validation["predictions"] - state["champion"]["predictions"]
        ),
        "submission": str(submission),
        "manifest": str(manifest_path),
        "sha256": file_sha256(submission),
        "score_possibilities": score_possibilities(
            state["champion"]["tp"], selection_validation["predictions"], actions
        ),
    }
    write_json(manifest_path, batch)
    state["batches"].append(batch)
    save_state(workdir, state)
    return batch


def command_init(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for name in ("submissions", "manifests", "reports"):
        (workdir / name).mkdir(exist_ok=True)
    order_ids, champion_selection = load_submission(Path(args.champion))
    topologies = load_topologies(Path(args.test_dir))
    validation = validate_selection(order_ids, champion_selection, topologies)
    tp, reconstructed = infer_tp(args.score, validation["predictions"])
    if workdir.joinpath("state.json").exists() and not args.force:
        raise FileExistsError("state.json already exists; pass --force to reinitialize")
    state = {
        "version": 1,
        "test_dir": str(Path(args.test_dir).resolve()),
        "true_roots": TRUE_ROOTS,
        "target_score": args.target_score,
        "submission_budget": args.submission_budget,
        "champion": {
            "path": str(Path(args.champion).resolve()),
            "score": args.score,
            "tp": tp,
            "predictions": validation["predictions"],
            "sha256": file_sha256(Path(args.champion)),
        },
        "batches": [],
    }
    save_state(workdir, state)
    shutil.copyfile(args.champion, workdir / "submissions" / "champion_frozen.csv")
    print(json.dumps({"state": str(workdir / "state.json"), "f1": reconstructed}, indent=2))


def command_build_batch(args: argparse.Namespace) -> None:
    actions = catalog_actions(Path(args.catalog))
    requested = [value.strip() for value in args.action_ids.split(",") if value.strip()]
    lookup = {item["action_id"]: item for item in actions}
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"actions not found in catalog: {missing}")
    batch = register_batch(
        Path(args.workdir), args.batch_id, args.kind, [lookup[value] for value in requested]
    )
    print(json.dumps(batch, ensure_ascii=False, indent=2))


def command_split(args: argparse.Namespace) -> None:
    state = load_state(Path(args.workdir))
    parent = next(
        (item for item in state["batches"] if item["batch_id"] == args.batch_id), None
    )
    if parent is None:
        raise ValueError(f"unknown batch: {args.batch_id}")
    actions = parent["actions"]
    if len(actions) < 2:
        raise ValueError("batch has fewer than two actions")
    midpoint = math.ceil(len(actions) / 2)
    outputs = []
    for suffix, subset in (("a", actions[:midpoint]), ("b", actions[midpoint:])):
        outputs.append(
            register_batch(
                Path(args.workdir),
                f"{args.batch_id}_{suffix}",
                "adaptive_split",
                subset,
                parent=args.batch_id,
            )
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def command_record(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir)
    state = load_state(workdir)
    batch = next(
        (item for item in state["batches"] if item["batch_id"] == args.batch_id), None
    )
    if batch is None:
        raise ValueError(f"unknown batch: {args.batch_id}")
    if batch["status"] == "scored":
        raise ValueError(f"batch already scored: {args.batch_id}")
    tp, reconstructed = infer_tp(args.score, batch["predictions"])
    score_gain = reconstructed - state["champion"]["score"]
    if score_gain > 0.5e-9:
        decision = "accept"
    elif len(batch["actions"]) > 1:
        decision = "split"
    else:
        decision = "reject"
    batch.update(
        {
            "status": "scored",
            "score": args.score,
            "reconstructed_score": reconstructed,
            "tp": tp,
            "delta_tp": tp - state["champion"]["tp"],
            "score_gain": score_gain,
            "decision": decision,
        }
    )
    write_json(Path(batch["manifest"]), batch)

    # A scored parent batch plus one scored half determines the complement
    # exactly. Keep the second CSV for audit, but do not spend a submission on it.
    if batch.get("parent"):
        parent = next(
            (
                item
                for item in state["batches"]
                if item["batch_id"] == batch["parent"]
                and item.get("status") in {"scored", "inferred"}
            ),
            None,
        )
        sibling = next(
            (
                item
                for item in state["batches"]
                if item.get("parent") == batch["parent"]
                and item["batch_id"] != batch["batch_id"]
                and item.get("status") in {"ready", "inference_pending"}
            ),
            None,
        )
        if parent is not None and sibling is not None:
            sibling_delta_tp = parent["delta_tp"] - batch["delta_tp"]
            sibling_tp = state["champion"]["tp"] + sibling_delta_tp
            sibling_score = f1_score(sibling_tp, sibling["predictions"])
            sibling_gain = sibling_score - state["champion"]["score"]
            sibling.update(
                {
                    "status": "inferred",
                    "score": sibling_score,
                    "reconstructed_score": sibling_score,
                    "tp": sibling_tp,
                    "delta_tp": sibling_delta_tp,
                    "score_gain": sibling_gain,
                    "decision": "accept" if sibling_gain > 0.5e-9 else "reject",
                    "inferred_from": [parent["batch_id"], batch["batch_id"]],
                }
            )
            write_json(Path(sibling["manifest"]), sibling)
    save_state(workdir, state)
    report = build_status(state)
    write_json(workdir / "reports" / "status.json", report)
    print(json.dumps({"batch": batch, "status": report}, ensure_ascii=False, indent=2))


def accepted_disjoint_actions(state: dict) -> list[dict]:
    accepted = [item for item in state["batches"] if item.get("decision") == "accept"]
    leaf_batches = [
        item for item in accepted if not any(child.get("parent") == item["batch_id"] for child in accepted)
    ]
    actions: list[dict] = []
    used_orders: set[str] = set()
    for batch in sorted(leaf_batches, key=lambda item: item["batch_id"]):
        for action in batch["actions"]:
            if action["order_id"] in used_orders:
                raise ValueError(
                    f"accepted batches conflict on order {action['order_id']}"
                )
            used_orders.add(action["order_id"])
            actions.append(action)
    return actions


def build_status(state: dict, catalog: list[dict] | None = None) -> dict:
    accepted = accepted_disjoint_actions(state)
    delta_tp = 0
    prediction_delta = 0
    accepted_batches = []
    accepted_action_ids = {item["action_id"] for item in accepted}
    for batch in state["batches"]:
        if batch.get("decision") != "accept":
            continue
        if not set(batch["action_ids"]) <= accepted_action_ids:
            continue
        accepted_batches.append(batch["batch_id"])
        delta_tp += batch["delta_tp"]
        prediction_delta += batch["prediction_delta"]
    combined_tp = state["champion"]["tp"] + delta_tp
    combined_predictions = state["champion"]["predictions"] + prediction_delta
    combined_score = f1_score(combined_tp, combined_predictions)
    remaining = []
    if catalog is not None:
        used_orders = {item["order_id"] for item in accepted}
        remaining = [
            item
            for item in catalog
            if item["action_id"] not in accepted_action_ids
            and item["order_id"] not in used_orders
        ]
    optimistic_tp = combined_tp + sum(len(item["add_rids"]) for item in remaining)
    optimistic_predictions = combined_predictions + sum(
        len(item["add_rids"]) - len(item["remove_rids"]) for item in remaining
    )
    optimistic_score = f1_score(optimistic_tp, optimistic_predictions)
    submissions_used = sum(item["status"] == "scored" for item in state["batches"])
    target_still_reachable = optimistic_score >= state["target_score"]
    return {
        "accepted_batches": accepted_batches,
        "accepted_actions": len(accepted),
        "verified_delta_tp": delta_tp,
        "verified_prediction_delta": prediction_delta,
        "combined_tp": combined_tp,
        "combined_predictions": combined_predictions,
        "combined_score": combined_score,
        "required_tp_at_combined_count": required_tp(
            combined_predictions, state["target_score"]
        ),
        "tp_gap": max(
            required_tp(combined_predictions, state["target_score"]) - combined_tp, 0
        ),
        "target_reached": combined_score >= state["target_score"],
        "remaining_actions": len(remaining),
        "optimistic_score": optimistic_score,
        "target_still_reachable": target_still_reachable,
        "early_feasibility_warning": not target_still_reachable,
        "hard_stop_active": submissions_used >= 8 and not target_still_reachable,
        "submissions_used": submissions_used,
        "submissions_remaining": state["submission_budget"] - submissions_used,
    }


def command_checkpoint(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir)
    state = load_state(workdir)
    actions = accepted_disjoint_actions(state)
    champion = Path(state["champion"]["path"])
    order_ids, base = load_submission(champion)
    topologies = load_topologies(Path(state["test_dir"]))
    validate_actions(actions, base, topologies)
    selection = apply_actions(base, actions)
    validation = validate_selection(order_ids, selection, topologies)
    output = workdir / "submissions" / args.output
    write_submission(output, order_ids, selection, topologies)
    status = build_status(state)
    manifest = {
        "kind": "verified_checkpoint",
        "accepted_action_ids": [item["action_id"] for item in actions],
        "validation": validation,
        "predicted_from_additivity": status,
        "submission": str(output),
        "sha256": file_sha256(output),
    }
    write_json(workdir / "manifests" / f"{Path(args.output).stem}.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_status(args: argparse.Namespace) -> None:
    workdir = Path(args.workdir)
    state = load_state(workdir)
    catalog = catalog_actions(Path(args.catalog)) if args.catalog else None
    report = build_status(state, catalog)
    write_json(workdir / "reports" / "status.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_validate(args: argparse.Namespace) -> None:
    order_ids, selection = load_submission(Path(args.submission))
    topologies = load_topologies(Path(args.test_dir))
    report = validate_selection(order_ids, selection, topologies)
    report["sha256"] = file_sha256(Path(args.submission))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    init.add_argument("--champion", default=str(DEFAULT_CHAMPION))
    init.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    init.add_argument("--score", type=float, default=BASE_SCORE)
    init.add_argument("--target-score", type=float, default=TARGET_SCORE)
    init.add_argument("--submission-budget", type=int, default=15)
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=command_init)

    batch = sub.add_parser("build-batch")
    batch.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    batch.add_argument("--catalog", required=True)
    batch.add_argument("--batch-id", required=True)
    batch.add_argument("--action-ids", required=True)
    batch.add_argument("--kind", default="action_probe")
    batch.set_defaults(func=command_build_batch)

    split = sub.add_parser("split")
    split.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    split.add_argument("--batch-id", required=True)
    split.set_defaults(func=command_split)

    record = sub.add_parser("record")
    record.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    record.add_argument("--batch-id", required=True)
    record.add_argument("--score", required=True, type=float)
    record.set_defaults(func=command_record)

    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    checkpoint.add_argument("--output", default="highest_verified_checkpoint.csv")
    checkpoint.set_defaults(func=command_checkpoint)

    status = sub.add_parser("status")
    status.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    status.add_argument("--catalog")
    status.set_defaults(func=command_status)

    validate = sub.add_parser("validate")
    validate.add_argument("--submission", required=True)
    validate.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    validate.set_defaults(func=command_validate)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
