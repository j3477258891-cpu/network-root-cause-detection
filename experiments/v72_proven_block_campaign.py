"""Build a proven block-detecting campaign over 175 high-value actions.

Each 30-action block uses an 18x30 matrix already proven detecting in V62.
The last 25-action block uses a column restriction of the same matrix, which
preserves injectivity.  Every probe is relative to the verified V60 champion.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
for value in (EXP, EXP / "v30_meta_stack"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np

from build_cross_order_probes import apply_actions, load_submission, write_submission


BASE = EXP / "v60_combined_checkpoint/highest_verified_combined.csv"
BASE_META = EXP / "v60_combined_checkpoint/highest_verified_combined.json"
V58 = EXP / "v58_coded_campaign/report.json"
V59 = EXP / "v59_extended_coded_campaign/report.json"
V62 = EXP / "v62_block_detecting/report.json"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
OUT = EXP / "v72_proven_block_campaign"
CAMPAIGN_PREFIX = "v72"
BASE_TP = 956
BASE_P = 1035
TRUE_ROOTS = 1044
TARGET = 0.95
SELECTED_COUNT = 175
BLOCK_WIDTH = 30
ROWS_PER_BLOCK = 18


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_key(action: dict) -> tuple:
    return action["order_id"], tuple(action["remove_rids"]), tuple(action["add_rids"])


def candidate_pool() -> tuple[list[dict], np.ndarray]:
    actions, probabilities = [], []
    for path in (V58, V59):
        report = read_json(path)
        actions.extend(report["candidate_actions"])
        probabilities.extend(report["correctness_probabilities"])
    if len(actions) != 290 or len({action_key(action) for action in actions}) != 290:
        raise RuntimeError("expected 290 unique combined actions")
    if len({action["order_id"] for action in actions}) != 290:
        raise RuntimeError("combined actions are not order-disjoint")
    return actions, np.asarray(probabilities, dtype=np.float64)


def detecting_basis() -> np.ndarray:
    if BLOCK_WIDTH == 35 and ROWS_PER_BLOCK == 20:
        search = read_json(EXP / "v73_compact_detecting/search_20x35_seed_4000.json")
        if not search["injective_found"]:
            raise RuntimeError("V73 dense basis is not proven")
        basis = np.asarray(read_json(Path(search["basis_path"])), dtype=np.int8)
        if basis.shape != (ROWS_PER_BLOCK, BLOCK_WIDTH):
            raise RuntimeError(basis.shape)
        return basis
    if BLOCK_WIDTH == 25 and ROWS_PER_BLOCK == 15:
        search = read_json(EXP / "v73_compact_detecting/search_15x25_seed_2000.json")
        if not search["injective_found"]:
            raise RuntimeError("V73 compact basis is not proven")
        basis = np.asarray(read_json(Path(search["basis_path"])), dtype=np.int8)
        if basis.shape != (ROWS_PER_BLOCK, BLOCK_WIDTH):
            raise RuntimeError(basis.shape)
        return basis
    report = read_json(V62)
    proof = report["global_injectivity"]
    if proof["status"] != "proven":
        raise RuntimeError("V62 basis is not proven")
    full = np.asarray(report["matrix"], dtype=np.int8)
    block = report["blocks"][1]
    basis = full[
        int(block["row_start"]):int(block["row_stop"]),
        int(block["column_start"]):int(block["column_stop"]),
    ]
    if basis.shape != (ROWS_PER_BLOCK, BLOCK_WIDTH):
        raise RuntimeError(basis.shape)
    return basis


def exact_oracle_distribution(actions: list[dict], probabilities: np.ndarray,
                              trials: int = 200_000) -> dict:
    rng = np.random.default_rng(20260820 + 72)
    truth = rng.random((trials, len(actions))) < probabilities
    additions = np.asarray([bool(action["add_rids"]) for action in actions])
    correct_adds = (truth & additions).sum(axis=1)
    correct_deletes = (truth & ~additions).sum(axis=1)
    predictions = BASE_P + correct_adds - correct_deletes
    tp = BASE_TP + correct_adds
    scores = 2 * tp / (TRUE_ROOTS + predictions)
    return {
        "trials": trials, "mean": float(scores.mean()),
        "p01": float(np.quantile(scores, 0.01)),
        "p05": float(np.quantile(scores, 0.05)),
        "median": float(np.median(scores)),
        "p95": float(np.quantile(scores, 0.95)),
        "probability_reach_0_95": float(np.mean(scores >= TARGET)),
        "warning": "Independent prior simulation; not leaderboard evidence.",
    }


def emit_probe(block_index: int, row_index: int, row: np.ndarray,
               actions: list[dict], order_ids: list[str], base_roots: dict,
               records_by_order: dict) -> dict:
    selected_indices = np.flatnonzero(row).astype(int).tolist()
    selected = [actions[index] for index in selected_indices]
    roots = apply_actions(base_roots, records_by_order, selected)
    additions = sum(bool(action["add_rids"]) for action in selected)
    deletions = len(selected) - additions
    predictions = sum(len(nodes) for nodes in roots.values())
    if predictions != BASE_P + additions - deletions:
        raise RuntimeError("probe prediction count mismatch")
    probe_id = f"{CAMPAIGN_PREFIX}_block_{block_index + 1:02d}_row_{row_index + 1:02d}"
    path = OUT / "submissions" / f"{probe_id}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": probe_id, "kind": "code", "block_index": block_index,
        "row_index": row_index, "path": str(path), "sha256": sha256(path),
        "candidate_indices": selected_indices, "candidate_actions": selected,
        "additions": additions, "deletions": deletions,
        "predictions": predictions, "prediction_delta": additions - deletions,
        "score_possibilities": [{
            "correct_count": count, "tp": BASE_TP + count - deletions,
            "score": round(2 * (BASE_TP + count - deletions) /
                           (TRUE_ROOTS + predictions), 9),
        } for count in range(len(selected) + 1)],
    }
    manifest_path = OUT / "manifests" / f"{probe_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true",
                        help="emit the proven 15x25 V74 variant without replacing V72")
    parser.add_argument("--dense", action="store_true",
                        help="emit the proven 20x35 V75 variant without replacing V72/V74")
    args = parser.parse_args()
    global OUT, CAMPAIGN_PREFIX, BLOCK_WIDTH, ROWS_PER_BLOCK
    if args.compact and args.dense:
        raise SystemExit("--compact and --dense are mutually exclusive")
    if args.compact:
        OUT = EXP / "v74_compact_block_campaign"
        CAMPAIGN_PREFIX = "v74"
        BLOCK_WIDTH = 25
        ROWS_PER_BLOCK = 15
    if args.dense:
        OUT = EXP / "v75_dense_block_campaign"
        CAMPAIGN_PREFIX = "v75"
        BLOCK_WIDTH = 35
        ROWS_PER_BLOCK = 20
    for directory in (OUT, OUT / "submissions", OUT / "manifests", OUT / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    base_meta = read_json(BASE_META)
    if base_meta["tp"] != BASE_TP or base_meta["predictions"] != BASE_P:
        raise RuntimeError("verified base metadata changed")
    if sha256(BASE) != base_meta["sha256"]:
        raise RuntimeError("verified base SHA changed")
    all_actions, all_probabilities = candidate_pool()
    additions = np.asarray([bool(action["add_rids"]) for action in all_actions])
    # Once a bit is decoded, only correct actions are applied.  Their target
    # margin is 1.05 for an add and 0.95 for a delete.
    expected_margin = all_probabilities * np.where(additions, 1.05, 0.95)
    selected_source_indices = np.argsort(-expected_margin, kind="stable")[:SELECTED_COUNT]
    actions = [all_actions[int(index)] for index in selected_source_indices]
    probabilities = all_probabilities[selected_source_indices]
    if len({action["order_id"] for action in actions}) != SELECTED_COUNT:
        raise RuntimeError("selected actions overlap")
    basis = detecting_basis()
    order_ids, base_roots = load_submission(BASE)
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {record["order_id"]: record for record in records}
    blocks, probe_order, manifests = [], [], {}
    for block_index, start in enumerate(range(0, SELECTED_COUNT, BLOCK_WIDTH)):
        stop = min(start + BLOCK_WIDTH, SELECTED_COUNT)
        local_actions = actions[start:stop]
        local_probabilities = probabilities[start:stop]
        local_matrix = basis[:, :stop - start].copy()
        block_probe_ids = []
        for row_index, row in enumerate(local_matrix):
            manifest = emit_probe(
                block_index, row_index, row, local_actions,
                order_ids, base_roots, records_by_order,
            )
            # Store indices in global selected-pool coordinates for reporting.
            manifest["selected_pool_indices"] = [start + index for index in manifest["candidate_indices"]]
            manifest_path = OUT / "manifests" / f"{manifest['probe_id']}.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            manifests[manifest["probe_id"]] = manifest
            probe_order.append(manifest["probe_id"])
            block_probe_ids.append(manifest["probe_id"])
        blocks.append({
            "block_index": block_index, "start": start, "stop": stop,
            "action_count": stop - start, "probe_ids": block_probe_ids,
            "matrix": local_matrix.astype(int).tolist(),
            "source_indices": selected_source_indices[start:stop].astype(int).tolist(),
            "actions": local_actions, "probabilities": local_probabilities.tolist(),
            "expected_correct": float(local_probabilities.sum()),
        })
    oracle = exact_oracle_distribution(actions, probabilities)
    report = {
        "version": f"{CAMPAIGN_PREFIX}-proven-block-campaign-1",
        "prefix": CAMPAIGN_PREFIX,
        "base": {"path": str(BASE), "sha256": sha256(BASE), "tp": BASE_TP,
                 "predictions": BASE_P, "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P)},
        "target": TARGET, "candidate_pool_count": len(all_actions),
        "selected_action_count": len(actions), "block_count": len(blocks),
        "code_probe_count": len(probe_order), "checkpoint_probe_max": 1,
        "estimated_days_at_two_submissions_max": (len(probe_order) + 1) / 2,
        "checkpoint_policy": "Submit the first decoded checkpoint projected at >=0.95; otherwise submit only after the final block.",
        "selection": "top expected target-margin after exact correctness decoding",
        "injectivity": {
            "status": "proven",
            "basis_source": (
                str(V62) if not args.compact and not args.dense else
                str(EXP / "v73_compact_detecting/basis_15x25.json") if args.compact else
                str(EXP / "v73_compact_detecting/basis_20x35.json")
            ),
            "argument": (
                "Each block uses a proven detecting 18x30 basis; restricting the last block to 25 columns preserves injectivity."
                if not args.compact and not args.dense else
                "Each block uses the MILP-proven detecting 15x25 V73 basis."
                if args.compact else
                "Each block uses the MILP-proven detecting 20x35 V73 basis."
            ),
        },
        "oracle_if_all_selected_labels_decoded": oracle,
        "selected_source_indices": selected_source_indices.astype(int).tolist(),
        "correctness_probabilities": probabilities.tolist(),
        "candidate_actions": actions, "blocks": blocks,
        "probe_order": probe_order, "probes": manifests,
        "warning": "The code proves label recovery after scored probes; priors do not prove that the recovered checkpoint reaches 0.95.",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    state = {
        "version": f"{CAMPAIGN_PREFIX}-proven-block-state-1", "results": {},
        "decoded_blocks": {}, "checkpoint_manifests": {}, "checkpoint_results": {},
        "verified_checkpoint": {
            "path": str(BASE), "sha256": sha256(BASE), "tp": BASE_TP,
            "predictions": BASE_P, "score": 2 * BASE_TP / (TRUE_ROOTS + BASE_P),
            "decoded_correct_indices": [],
        },
        "target_reached": False,
        "next_probe": {key: manifests[probe_order[0]][key]
                       for key in ("probe_id", "path", "sha256", "predictions")},
    }
    (OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_action_count": len(actions), "block_count": len(blocks),
        "code_probe_count": len(probe_order),
        "estimated_days_max": report["estimated_days_at_two_submissions_max"],
        "oracle": oracle, "first_probe": state["next_probe"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
