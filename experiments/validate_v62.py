"""Independently verify V62's block-diagonal global injectivity proof."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from v57_detecting_matrix_search import collision


REPORT = EXP / "v62_block_detecting/report.json"


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    matrix = np.asarray(report["matrix"], dtype=np.int8)
    assert matrix.shape == (175, 290)
    assert report["global_injectivity"]["status"] == "proven"
    assert len(report["column_mapping"]) == 290
    assert len({row["order_id"] for row in report["column_mapping"]}) == 290
    v58 = json.loads((EXP / "v58_coded_campaign/report.json").read_text(encoding="utf-8"))
    v59 = json.loads((EXP / "v59_extended_coded_campaign/report.json").read_text(encoding="utf-8"))
    expected_ids = [row["action_id"] for row in v58["candidate_actions"] + v59["candidate_actions"]]
    assert [row["action_id"] for row in report["column_mapping"]] == expected_ids
    assert np.array_equal(np.flatnonzero(matrix[0]), np.asarray([0, 1, 2, 3]))
    covered_rows, covered_columns = set(), set()
    for block in report["blocks"]:
        row_range = range(block["row_start"], block["row_stop"])
        column_range = range(block["column_start"], block["column_stop"])
        assert not covered_rows & set(row_range)
        assert not covered_columns & set(column_range)
        covered_rows |= set(row_range)
        covered_columns |= set(column_range)
        outside = matrix[block["row_start"]:block["row_stop"]].copy()
        outside[:, block["column_start"]:block["column_stop"]] = 0
        assert not np.any(outside)
    assert covered_rows == set(range(175))
    assert covered_columns == set(range(290))
    first = matrix[0:18, 0:30]
    regular = matrix[18:36, 30:60]
    tail = matrix[162:175, 270:290]
    proofs = {}
    for name, basis in (("first", first), ("regular", regular), ("tail", tail)):
        result = collision(basis, 30.0)
        assert result is None, (name, result)
        proofs[name] = {"shape": list(basis.shape), "injective": True}
    print(json.dumps({
        "status": "ok", "full_shape": list(matrix.shape),
        "block_count": len(report["blocks"]), "basis_proofs": proofs,
        "first_row_candidate_indices": np.flatnonzero(matrix[0]).astype(int).tolist(),
        "days_at_two_submissions": report["days_at_two_submissions"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
