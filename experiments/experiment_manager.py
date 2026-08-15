import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path


GROUND_TRUTH_POSITIVES = 1044
EXPECTED_ORDERS = 546
EXPECTED_PREDICTIONS = 1059


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_submission(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["order_id", "output"]:
            raise ValueError(f"invalid columns in {path}: {reader.fieldnames}")
        for row in reader:
            if row["order_id"] in result:
                raise ValueError(f"duplicate order_id: {row['order_id']}")
            result[row["order_id"]] = {
                node["@rid"] for node in json.loads(row["output"])["rootcause"]
            }
    return result


def load_topologies(test_dir):
    result = {}
    for directory in sorted(path for path in Path(test_dir).iterdir() if path.is_dir()):
        order_id = directory.name
        topo = json.loads(
            (directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8")
        )
        result[order_id] = {
            node["@rid"]: node
            for node in topo.get("nodes", [])
            if node.get("@class") == "Alarm"
        }
    return result


def validate_selection(selection, topologies, expected_predictions=EXPECTED_PREDICTIONS):
    if len(selection) != EXPECTED_ORDERS or set(selection) != set(topologies):
        raise ValueError("submission order IDs do not match the test set")
    distribution = Counter()
    total = 0
    for order_id, selected in selection.items():
        if not 1 <= len(selected) <= 8:
            raise ValueError(f"invalid rootcause count for {order_id}: {len(selected)}")
        missing = selected - set(topologies[order_id])
        if missing:
            raise ValueError(f"nodes missing from topology for {order_id}: {missing}")
        distribution[len(selected)] += 1
        total += len(selected)
    if expected_predictions is not None and total != expected_predictions:
        raise ValueError(f"expected {expected_predictions} predictions, found {total}")
    return {"orders": len(selection), "predictions": total, "distribution": dict(sorted(distribution.items()))}


def write_submission(path, selection, topologies):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id in sorted(topologies):
            nodes = topologies[order_id]
            rootcauses = []
            for rid, node in nodes.items():
                if rid not in selection[order_id]:
                    continue
                rootcauses.append(
                    {
                        "@rid": rid,
                        "title": node.get("title", ""),
                        "location": node.get("location", ""),
                        "reason": node.get("reason", ""),
                    }
                )
            writer.writerow([order_id, json.dumps({"rootcause": rootcauses}, ensure_ascii=False)])


def infer_tp(score, predictions, positives=GROUND_TRUTH_POSITIVES):
    raw = score * (predictions + positives) / 2.0
    tp = int(round(raw))
    reconstructed = 2.0 * tp / (predictions + positives)
    if abs(reconstructed - score) > 0.5e-6 + 1e-12:
        raise ValueError(
            f"score {score:.6f} is inconsistent with an integer TP; "
            f"nearest is TP={tp}, F1={reconstructed:.9f}"
        )
    return tp, reconstructed


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ledger_path(root):
    return Path(root) / "ledger.json"


def load_ledger(root):
    path = ledger_path(root)
    if not path.exists():
        raise FileNotFoundError(f"ledger not initialized: {path}")
    ledger = load_json(path)
    if "scores" not in ledger:
        champion = ledger["champion"]
        ledger["scores"] = {
            champion["sha256"]: {
                "path": champion["path"],
                "score": champion["score"],
                "tp": champion["tp"],
                "predictions": champion["predictions"],
            }
        }
    return ledger


def save_ledger(root, ledger):
    save_json(ledger_path(root), ledger)
    csv_path = Path(root) / "ledger.csv"
    fields = [
        "id", "kind", "status", "decision", "base_score", "score",
        "base_tp", "tp", "delta_tp", "predictions", "candidate_path", "sha256",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in ledger["experiments"]:
            writer.writerow({name: item.get(name, "") for name in fields})


def find_experiment(ledger, experiment_id):
    matches = [item for item in ledger["experiments"] if item["id"] == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"experiment not found or duplicated: {experiment_id}")
    return matches[0]


def apply_units(base_selection, units):
    selected = {order_id: set(rids) for order_id, rids in base_selection.items()}
    used_removals = set()
    used_additions = set()
    for unit in units:
        removal = (unit["remove"]["order_id"], unit["remove"]["rid"])
        addition = (unit["add"]["order_id"], unit["add"]["rid"])
        if removal in used_removals or addition in used_additions:
            raise ValueError(f"conflicting unit: {unit['id']}")
        if removal[1] not in selected[removal[0]]:
            raise ValueError(f"removal is not selected for {unit['id']}: {removal}")
        if addition[1] in selected[addition[0]]:
            raise ValueError(f"addition is already selected for {unit['id']}: {addition}")
        selected[removal[0]].remove(removal[1])
        selected[addition[0]].add(addition[1])
        used_removals.add(removal)
        used_additions.add(addition)
    return selected


def register_probe(root, experiment_id, base_path, units, kind, test_dir, parent=None):
    root = Path(root)
    ledger = load_ledger(root)
    if any(item["id"] == experiment_id for item in ledger["experiments"]):
        raise ValueError(f"experiment already exists: {experiment_id}")
    topologies = load_topologies(test_dir)
    base = load_submission(base_path)
    selection = apply_units(base, units)
    validation = validate_selection(selection, topologies)
    candidate_path = root / "submissions" / f"{experiment_id}.csv"
    manifest_path = root / "reports" / f"{experiment_id}.json"
    write_submission(candidate_path, selection, topologies)
    candidate_hash = sha256(candidate_path)
    manifest = {
        "id": experiment_id,
        "kind": kind,
        "parent": parent,
        "base_path": str(Path(base_path)),
        "base_sha256": sha256(base_path),
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "unit_ids": [unit["id"] for unit in units],
        "units": units,
        "validation": validation,
    }
    save_json(manifest_path, manifest)
    base_record = ledger.get("scores", {}).get(sha256(base_path))
    base_score = base_record["score"] if base_record is not None else None
    ledger["experiments"].append(
        {
            "id": experiment_id,
            "kind": kind,
            "status": "ready",
            "decision": "pending",
            "base_path": str(Path(base_path)),
            "base_sha256": sha256(base_path),
            "base_score": base_score,
            "base_tp": base_record["tp"] if base_record is not None else None,
            "candidate_path": str(candidate_path),
            "manifest_path": str(manifest_path),
            "predictions": validation["predictions"],
            "sha256": candidate_hash,
            "unit_ids": manifest["unit_ids"],
        }
    )
    save_ledger(root, ledger)
    return manifest


def command_init(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "submissions").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    topologies = load_topologies(args.test_dir)
    champion = load_submission(args.champion)
    validation = validate_selection(champion, topologies)
    tp, reconstructed = infer_tp(args.score, validation["predictions"])
    champion_hash = sha256(args.champion)
    ledger = {
        "version": 1,
        "ground_truth_positives": GROUND_TRUTH_POSITIVES,
        "score_precision": 6,
        "champion": {
            "path": str(Path(args.champion)),
            "sha256": champion_hash,
            "score": args.score,
            "tp": tp,
            "predictions": validation["predictions"],
        },
        "scores": {
            champion_hash: {
                "path": str(Path(args.champion)),
                "score": args.score,
                "tp": tp,
                "predictions": validation["predictions"],
            }
        },
        "experiments": [],
    }
    save_ledger(root, ledger)
    print(json.dumps({"champion": ledger["champion"], "reconstructed_f1": reconstructed}, indent=2))


def command_register_file(args):
    root = Path(args.root)
    ledger = load_ledger(root)
    topologies = load_topologies(args.test_dir)
    candidate = load_submission(args.candidate)
    validation = validate_selection(candidate, topologies)
    if any(item["id"] == args.id for item in ledger["experiments"]):
        raise ValueError(f"experiment already exists: {args.id}")
    destination = root / "submissions" / f"{args.id}.csv"
    shutil.copyfile(args.candidate, destination)
    base_hash = sha256(args.base)
    base_record = ledger.get("scores", {}).get(base_hash)
    base_score = base_record["score"] if base_record is not None else None
    base_tp = base_record["tp"] if base_record is not None else None
    base = load_submission(args.base)
    changed = []
    for order_id in sorted(base):
        removed = sorted(base[order_id] - candidate[order_id])
        added = sorted(candidate[order_id] - base[order_id])
        if removed or added:
            changed.append({"order_id": order_id, "removed": removed, "added": added})
    manifest_path = root / "reports" / f"{args.id}.json"
    manifest = {
        "id": args.id,
        "kind": args.kind,
        "base_path": str(Path(args.base)),
        "base_sha256": base_hash,
        "candidate_path": str(destination),
        "candidate_sha256": sha256(destination),
        "changed_orders": changed,
        "validation": validation,
    }
    save_json(manifest_path, manifest)
    ledger["experiments"].append(
        {
            "id": args.id,
            "kind": args.kind,
            "status": "ready",
            "decision": "pending",
            "base_path": str(Path(args.base)),
            "base_sha256": base_hash,
            "base_score": base_score,
            "base_tp": base_tp,
            "candidate_path": str(destination),
            "manifest_path": str(manifest_path),
            "predictions": validation["predictions"],
            "sha256": sha256(destination),
        }
    )
    save_ledger(root, ledger)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_probe(args):
    catalog = load_json(args.catalog)
    lookup = {unit["id"]: unit for unit in catalog["units"]}
    unit_ids = [value.strip() for value in args.units.split(",") if value.strip()]
    missing = [value for value in unit_ids if value not in lookup]
    if missing:
        raise ValueError(f"units not in catalog: {missing}")
    manifest = register_probe(
        args.root,
        args.id,
        args.base,
        [lookup[value] for value in unit_ids],
        args.kind,
        args.test_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_split(args):
    ledger = load_ledger(args.root)
    parent = find_experiment(ledger, args.experiment)
    manifest = load_json(parent["manifest_path"])
    units = manifest.get("units", [])
    if len(units) < 2:
        raise ValueError("experiment has fewer than two atomic units")
    midpoint = math.ceil(len(units) / 2)
    outputs = []
    for suffix, subset in (("a", units[:midpoint]), ("b", units[midpoint:])):
        experiment_id = f"{args.experiment}_{suffix}"
        outputs.append(
            register_probe(
                args.root,
                experiment_id,
                parent["base_path"],
                subset,
                "binary_split",
                args.test_dir,
                parent=args.experiment,
            )
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def command_merge(args):
    ledger = load_ledger(args.root)
    experiment_ids = [value.strip() for value in args.experiments.split(",") if value.strip()]
    parents = [find_experiment(ledger, value) for value in experiment_ids]
    if any(item.get("decision") != "accept" for item in parents):
        states = {item["id"]: item.get("decision") for item in parents}
        raise ValueError(f"only accepted experiments can be merged: {states}")
    base_hashes = {item["base_sha256"] for item in parents}
    if len(base_hashes) != 1:
        raise ValueError("merged experiments must share the same base submission")
    units = []
    seen = set()
    for item in parents:
        manifest = load_json(item["manifest_path"])
        if not manifest.get("units"):
            raise ValueError(f"experiment has no atomic units: {item['id']}")
        for unit in manifest["units"]:
            if unit["id"] not in seen:
                units.append(unit)
                seen.add(unit["id"])
    manifest = register_probe(
        args.root,
        args.id,
        parents[0]["base_path"],
        units,
        "cumulative_merge",
        args.test_dir,
        parent=experiment_ids,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_record(args):
    ledger = load_ledger(args.root)
    item = find_experiment(ledger, args.experiment)
    if item["status"] == "scored":
        raise ValueError(f"experiment already scored: {args.experiment}")
    tp, reconstructed = infer_tp(args.score, item["predictions"], ledger["ground_truth_positives"])
    base_tp = item.get("base_tp")
    if base_tp is None:
        base_record = ledger.get("scores", {}).get(item["base_sha256"])
        if base_record is None:
            raise ValueError("base submission has no recorded score")
        base_tp = base_record["tp"]
        item["base_tp"] = base_tp
        item["base_score"] = base_record["score"]
    delta_tp = tp - base_tp
    decision = "accept" if delta_tp > 0 else ("split" if delta_tp == 0 else "reject")
    item.update(
        {
            "status": "scored",
            "score": args.score,
            "tp": tp,
            "delta_tp": delta_tp,
            "decision": decision,
            "reconstructed_f1": reconstructed,
        }
    )
    ledger.setdefault("scores", {})[item["sha256"]] = {
        "path": item["candidate_path"],
        "score": args.score,
        "tp": tp,
        "predictions": item["predictions"],
    }
    if decision == "accept" and args.promote:
        frozen = Path(args.root) / "submissions" / f"champion_{args.score:.6f}_{args.experiment}.csv"
        shutil.copyfile(item["candidate_path"], frozen)
        ledger["champion"] = {
            "path": str(frozen),
            "sha256": sha256(frozen),
            "score": args.score,
            "tp": tp,
            "predictions": item["predictions"],
        }
    save_ledger(args.root, ledger)
    print(
        json.dumps(
            {
                "experiment": args.experiment,
                "score": args.score,
                "tp": tp,
                "delta_tp": delta_tp,
                "decision": decision,
                "champion": ledger["champion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_sync(args):
    ledger = load_ledger(args.root)
    save_ledger(args.root, ledger)
    print(
        json.dumps(
            {
                "champion": ledger["champion"],
                "scored_submissions": len(ledger.get("scores", {})),
                "experiments": len(ledger["experiments"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Manage fixed-count leaderboard experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--root", required=True)
    init.add_argument("--champion", required=True)
    init.add_argument("--score", required=True, type=float)
    init.add_argument("--test-dir", required=True)
    init.set_defaults(func=command_init)

    register_file = subparsers.add_parser("register-file")
    register_file.add_argument("--root", required=True)
    register_file.add_argument("--id", required=True)
    register_file.add_argument("--base", required=True)
    register_file.add_argument("--candidate", required=True)
    register_file.add_argument("--kind", default="full_model")
    register_file.add_argument("--test-dir", required=True)
    register_file.set_defaults(func=command_register_file)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--root", required=True)
    probe.add_argument("--id", required=True)
    probe.add_argument("--base", required=True)
    probe.add_argument("--catalog", required=True)
    probe.add_argument("--units", required=True)
    probe.add_argument("--kind", default="differential_probe")
    probe.add_argument("--test-dir", required=True)
    probe.set_defaults(func=command_probe)

    split = subparsers.add_parser("split")
    split.add_argument("--root", required=True)
    split.add_argument("--experiment", required=True)
    split.add_argument("--test-dir", required=True)
    split.set_defaults(func=command_split)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--root", required=True)
    merge.add_argument("--id", required=True)
    merge.add_argument("--experiments", required=True)
    merge.add_argument("--test-dir", required=True)
    merge.set_defaults(func=command_merge)

    record = subparsers.add_parser("record")
    record.add_argument("--root", required=True)
    record.add_argument("--experiment", required=True)
    record.add_argument("--score", required=True, type=float)
    record.add_argument("--promote", action="store_true")
    record.set_defaults(func=command_record)

    sync = subparsers.add_parser("sync")
    sync.add_argument("--root", required=True)
    sync.set_defaults(func=command_sync)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
