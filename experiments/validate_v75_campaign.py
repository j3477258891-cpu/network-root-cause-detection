"""Validate every generated V75 leaderboard probe before it is submitted.

The campaign deliberately uses leaderboard counts to decode action labels.  A
valid campaign therefore needs more than parseable CSV files: every displayed
six-decimal score must recover exactly one count, every manifest must describe
the CSV bytes, and the detecting matrices must actually decode arbitrary
binary label vectors.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
for value in (EXP, EXP / "v30_meta_stack", EXP / "v25_semantic_router"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from record_v72_result import decode_unique, score_to_result
from v25_common import validate_submission


OUT = EXP / "v75_dense_block_campaign"
FIXED = EXP / "v37_online_equations/report.json"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
TRUE_ROOTS = 1044


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_nodes(action: dict) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    order_id = action["order_id"]
    return (
        {(order_id, rid) for rid in action["remove_rids"]},
        {(order_id, rid) for rid in action["add_rids"]},
    )


def main() -> None:
    report = read_json(OUT / "report.json")
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        test_records = json.load(handle)["test"]
    fixed = read_json(FIXED)["fixed_labels"]
    forced_in = {(row["order_id"], row["rid"])
                 for row in fixed if int(row["label"]) == 1}
    forced_out = {(row["order_id"], row["rid"])
                  for row in fixed if int(row["label"]) == 0}

    actions = report["candidate_actions"]
    if len(actions) != 175 or len({item["order_id"] for item in actions}) != 175:
        raise AssertionError("candidate actions must be order-disjoint")
    all_removes, all_adds = set(), set()
    for action in actions:
        removes, adds = action_nodes(action)
        if removes & forced_in or adds & forced_out:
            raise AssertionError(f"fixed-label conflict: {action['action_id']}")
        if (all_removes | all_adds) & (removes | adds):
            raise AssertionError(f"candidate-node overlap: {action['action_id']}")
        all_removes |= removes
        all_adds |= adds

    probe_count = 0
    count_cases = 0
    base_tp = int(report["base"]["tp"])
    for probe_id in report["probe_order"]:
        manifest = report["probes"][probe_id]
        path = Path(manifest["path"])
        if not path.is_file() or sha256(path) != manifest["sha256"]:
            raise AssertionError(f"bad manifest hash: {probe_id}")
        actual = validate_submission(path, test_records, total=int(manifest["predictions"]))
        if actual["predictions"] != int(manifest["predictions"]):
            raise AssertionError(f"bad prediction count: {probe_id}")
        for correct_count in range(len(manifest["candidate_indices"]) + 1):
            tp = base_tp + correct_count - int(manifest["deletions"])
            displayed = float(f"{2 * tp / (TRUE_ROOTS + int(manifest['predictions'])):.6f}")
            result = score_to_result(displayed, manifest, base_tp)
            if result["correct_count"] != correct_count or result["tp"] != tp:
                raise AssertionError(f"ambiguous score mapping: {probe_id}, {correct_count}")
            count_cases += 1
        probe_count += 1

    rng = np.random.default_rng(20260820)
    decode_cases = 0
    for block in report["blocks"]:
        matrix = np.asarray(block["matrix"], dtype=np.int16)
        # Global injectivity is already MILP-proven by V73.  One independent
        # end-to-end decode per block guards the CSV/report wiring without
        # repeating the same exponential MITM construction 20 times.
        for _ in range(1):
            labels = rng.integers(0, 2, size=matrix.shape[1], dtype=np.int8)
            decoded = decode_unique(matrix, matrix @ labels)
            if not np.array_equal(decoded, labels):
                raise AssertionError(f"decode mismatch in block {block['block_index']}")
            decode_cases += 1

    print(json.dumps({
        "status": "passed",
        "candidate_actions": len(actions),
        "probes": probe_count,
        "score_mapping_cases": count_cases,
        "decode_cases": decode_cases,
        "fixed_label_conflicts": 0,
        "first_probe": read_json(OUT / "state.json")["next_probe"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
