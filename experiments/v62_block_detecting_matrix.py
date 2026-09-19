"""Construct a globally detecting 175x290 fallback matrix.

The fast V58/V59 matrices are locally unique in sampled audits, but proving
global injectivity in one MILP times out.  This fallback is block diagonal:
each small basis matrix is independently proven to contain no non-zero
{-1,0,1} null vector, which proves the full matrix is detecting.
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from v57_detecting_matrix_search import collision
from v58_coded_campaign import sidon_matrix


OUT = EXP / "v62_block_detecting"
V58 = EXP / "v58_coded_campaign/report.json"
V59 = EXP / "v59_extended_coded_campaign/report.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(action):
    return {
        "action_id": action["action_id"], "order_id": action["order_id"],
        "remove_rids": action["remove_rids"], "add_rids": action["add_rids"],
        "source": action["source"],
    }


def proven_basis(rows, columns, first_count, preferred_seed, search_limit=200):
    attempts = []
    for offset in range(search_limit):
        seed = preferred_seed + offset
        matrix = sidon_matrix(rows, columns, first_count, seed)
        result = collision(matrix, 15.0)
        if result is None:
            return matrix, {"seed": seed, "attempts": attempts + [{"seed": seed, "status": "injective"}]}
        status = result if isinstance(result, str) else f"collision_support_{int(np.count_nonzero(result))}"
        attempts.append({"seed": seed, "status": status})
    raise RuntimeError((rows, columns, first_count, attempts[-10:]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_actions = (
        read_json(V58)["candidate_actions"] + read_json(V59)["candidate_actions"]
    )
    if len(source_actions) != 290:
        raise RuntimeError(len(source_actions))
    # The first basis has V49's four actions in row zero.  The remaining
    # 30-column blocks can reuse one independently proven detecting basis.
    first, first_proof = proven_basis(18, 30, 4, 1000)
    regular, regular_proof = proven_basis(18, 30, 15, 400)
    tail, tail_proof = proven_basis(13, 20, 10, 300)
    block_specs = [(first, 0, 30, "v49_compatible")]
    for index in range(1, 9):
        block_specs.append((regular, index * 30, 30, f"regular_{index:02d}"))
    block_specs.append((tail, 270, 20, "tail"))
    total_rows = sum(matrix.shape[0] for matrix, _, _, _ in block_specs)
    full = np.zeros((total_rows, 290), dtype=np.int8)
    blocks, start = [], 0
    for matrix, column_start, width, name in block_specs:
        stop = start + matrix.shape[0]
        full[start:stop, column_start:column_start + width] = matrix
        blocks.append({
            "block_id": name, "row_start": start, "row_stop": stop,
            "column_start": column_start, "column_stop": column_start + width,
            "rows": matrix.shape[0], "columns": width,
        })
        start = stop
    if full.shape != (175, 290) or not np.array_equal(full[0, :4], np.ones(4, dtype=np.int8)):
        raise RuntimeError(full.shape)
    if np.any(full[0, 4:]):
        raise RuntimeError("first row must contain only the four V49 actions")
    report = {
        "version": "v62-block-detecting-matrix-1",
        "shape": list(full.shape), "candidate_count": 290,
        "queries": 175, "final_checkpoint_queries": 1,
        "days_at_two_submissions": 88,
        "global_injectivity": {
            "status": "proven",
            "argument": "The matrix is block diagonal and every basis block has no non-zero {-1,0,1} null vector.",
            "first_basis": first_proof,
            "regular_basis": regular_proof,
            "tail_basis": tail_proof,
        },
        "first_row": {
            "candidate_indices": [0, 1, 2, 3],
            "compatible_probe": str(EXP / "v58_coded_campaign/submissions/v58_code_01.csv"),
        },
        "column_mapping": [signature(action) for action in source_actions],
        "source_reports": {
            "v58": {"path": str(V58), "sha256": sha256(V58)},
            "v59": {"path": str(V59), "sha256": sha256(V59)},
        },
        "blocks": blocks, "matrix": full.astype(int).tolist(),
        "tradeoff": "Global decoding proof costs 40 more coded queries than V58+V59's 135-query local-uniqueness route.",
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "shape", "candidate_count", "queries", "days_at_two_submissions",
        "global_injectivity", "first_row", "tradeoff",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
