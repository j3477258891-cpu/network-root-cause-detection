"""Record one V72 public score and advance the proven block campaign."""

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


OUT = EXP / "v72_proven_block_campaign"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
TRUE_ROOTS = 1044
TARGET = 0.95


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset_bits(size: int) -> np.ndarray:
    values = np.arange(1 << size, dtype=np.uint32)[:, None]
    return ((values >> np.arange(size, dtype=np.uint32)) & 1).astype(np.int8)


def decode_unique(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Exact meet-in-the-middle decoder for a detecting matrix block."""
    width = matrix.shape[1]
    split = width // 2
    left_bits, right_bits = subset_bits(split), subset_bits(width - split)
    left_sums = left_bits @ matrix[:, :split].T
    right_sums = right_bits @ matrix[:, split:].T
    lookup = {}
    for index, value in enumerate(left_sums):
        key = value.tobytes()
        if key in lookup:
            raise RuntimeError("left half is not injective")
        lookup[key] = index
    solutions = []
    for right_index, value in enumerate(right_sums):
        need = rhs - value
        if np.any(need < 0):
            continue
        left_index = lookup.get(need.astype(left_sums.dtype).tobytes())
        if left_index is not None:
            solutions.append(np.r_[left_bits[left_index], right_bits[right_index]])
    if len(solutions) != 1:
        raise RuntimeError(f"detecting block yielded {len(solutions)} solutions")
    solution = solutions[0].astype(np.int8)
    if not np.array_equal(matrix @ solution, rhs):
        raise RuntimeError("decoded solution does not satisfy equations")
    return solution


def score_to_result(score: float, manifest: dict, base_tp: int) -> dict:
    predictions = int(manifest["predictions"])
    denominator = TRUE_ROOTS + predictions
    tp = round(score * denominator / 2)
    reconstructed = 2 * tp / (TRUE_ROOTS + predictions)
    # Public scores are displayed to six decimals.  Allow the full rounding
    # interval plus a tiny binary-float tolerance, then explicitly prove that
    # adjacent integer TP values cannot map to the same displayed score.
    tolerance = 0.5005e-6
    if abs(reconstructed - score) > tolerance:
        raise SystemExit("score does not map uniquely to integer TP")
    adjacent = [2 * (tp + delta) / denominator for delta in (-1, 1)]
    if any(abs(value - score) <= tolerance for value in adjacent):
        raise SystemExit("displayed score is ambiguous between integer TP values")
    output = {"score": score, "tp": tp, "predictions": predictions,
              "reconstructed_score": reconstructed}
    if manifest.get("kind") == "code":
        correct_count = tp - base_tp + int(manifest["deletions"])
        if not 0 <= correct_count <= len(manifest["candidate_indices"]):
            raise SystemExit("score implies impossible block count")
        output["correct_count"] = correct_count
    return output


def all_confirmed_correct(report: dict, state: dict) -> list[int]:
    indices = []
    for value in state["decoded_blocks"].values():
        indices.extend(value["correct_selected_pool_indices"])
    return sorted(set(indices))


def emit_checkpoint(report: dict, state: dict, block_index: int) -> dict:
    base = report["base"]
    order_ids, base_roots = load_submission(Path(base["path"]))
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        records = json.load(handle)["test"]
    records_by_order = {record["order_id"]: record for record in records}
    correct_indices = all_confirmed_correct(report, state)
    actions = [report["candidate_actions"][index] for index in correct_indices]
    roots = apply_actions(base_roots, records_by_order, actions)
    additions = sum(bool(action["add_rids"]) for action in actions)
    deletions = len(actions) - additions
    predictions = sum(len(nodes) for nodes in roots.values())
    tp = int(base["tp"]) + additions
    if predictions != int(base["predictions"]) + additions - deletions:
        raise RuntimeError("checkpoint prediction count mismatch")
    score = 2 * tp / (TRUE_ROOTS + predictions)
    probe_id = f"{report.get('prefix', 'v72')}_checkpoint_{block_index + 1:02d}"
    path = OUT / "checkpoints" / f"{probe_id}.csv"
    write_submission(path, order_ids, roots)
    manifest = {
        "probe_id": probe_id, "kind": "checkpoint", "block_index": block_index,
        "path": str(path), "sha256": sha256(path), "predictions": predictions,
        "expected_tp": tp, "expected_score": score,
        "decoded_correct_indices": correct_indices,
        "additions": additions, "deletions": deletions,
    }
    state["checkpoint_manifests"][probe_id] = manifest
    return manifest


def next_code_probe(report: dict, state: dict) -> dict | None:
    for block in report["blocks"]:
        if str(block["block_index"]) in state["decoded_blocks"]:
            continue
        for probe_id in block["probe_ids"]:
            if probe_id not in state["results"]:
                probe = report["probes"][probe_id]
                return {key: probe[key] for key in ("probe_id", "path", "sha256", "predictions")}
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-id", required=True)
    parser.add_argument("--score", required=True, type=float)
    parser.add_argument("--campaign-dir", default="v72_proven_block_campaign")
    args = parser.parse_args()
    global OUT
    OUT = EXP / args.campaign_dir
    report = read_json(OUT / "report.json")
    state = read_json(OUT / "state.json")
    base_tp = int(report["base"]["tp"])
    if args.probe_id in report["probes"]:
        manifest = report["probes"][args.probe_id]
        if args.probe_id in state["results"]:
            raise SystemExit("probe already recorded")
        state["results"][args.probe_id] = score_to_result(args.score, manifest, base_tp)
        block_index = int(manifest["block_index"])
        block = report["blocks"][block_index]
        if all(probe_id in state["results"] for probe_id in block["probe_ids"]):
            matrix = np.asarray(block["matrix"], dtype=np.int16)
            rhs = np.asarray([state["results"][probe_id]["correct_count"]
                              for probe_id in block["probe_ids"]], dtype=np.int16)
            labels = decode_unique(matrix, rhs)
            start = int(block["start"])
            correct = [start + int(index) for index in np.flatnonzero(labels)]
            state["decoded_blocks"][str(block_index)] = {
                "block_index": block_index, "labels": labels.astype(int).tolist(),
                "correct_count": int(labels.sum()),
                "correct_selected_pool_indices": correct,
                "proof": "unique solution of proven detecting block matrix",
            }
            checkpoint = emit_checkpoint(report, state, block_index)
            final_block = len(state["decoded_blocks"]) == len(report["blocks"])
            if checkpoint["expected_score"] >= TARGET or final_block:
                state["next_probe"] = {key: checkpoint[key]
                                       for key in ("probe_id", "path", "sha256", "predictions")}
            else:
                state["next_probe"] = next_code_probe(report, state)
        else:
            state["next_probe"] = next_code_probe(report, state)
    elif args.probe_id in state["checkpoint_manifests"]:
        manifest = state["checkpoint_manifests"][args.probe_id]
        if args.probe_id in state["checkpoint_results"]:
            raise SystemExit("checkpoint already recorded")
        result = score_to_result(args.score, manifest, base_tp)
        if result["tp"] != int(manifest["expected_tp"]):
            raise SystemExit("checkpoint score contradicts decoded equations")
        state["checkpoint_results"][args.probe_id] = result
        state["verified_checkpoint"] = {
            **manifest, "score": result["reconstructed_score"], "tp": result["tp"],
        }
        state["target_reached"] = result["reconstructed_score"] >= TARGET
        state["next_probe"] = None if state["target_reached"] else next_code_probe(report, state)
    else:
        raise SystemExit("unknown probe id")
    (OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "recorded": args.probe_id, "decoded_blocks": len(state["decoded_blocks"]),
        "verified_checkpoint": state["verified_checkpoint"],
        "target_reached": state["target_reached"], "next_probe": state["next_probe"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
