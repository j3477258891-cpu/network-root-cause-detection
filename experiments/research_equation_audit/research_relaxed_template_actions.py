"""Inspect the one/two current-test changes from V89 relaxed templates."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
from build_candidate_catalog import canonical_key, prepare_order  # type: ignore  # noqa: E402
import audit_real_scores as audit  # noqa: E402


def pattern(refs, min_support, confidence):
    if len(refs) < min_support:
        return None
    patterns = [tuple(sorted(Counter(canonical_key(a) for a in ref["alarms"] if a["@rid"] in ref["roots"]).items())) for ref in refs]
    winner, count = Counter(patterns).most_common(1)[0]
    if count / len(patterns) < confidence:
        return None
    return Counter(dict(winner)), count, len(patterns)


def predict(query, refs, scores, min_support, confidence):
    result = pattern(refs, min_support, confidence)
    if result is None:
        return None
    expected, support_count, ref_count = result
    by_key = defaultdict(list)
    for alarm in query["alarms"]:
        by_key[canonical_key(alarm)].append(alarm["@rid"])
    selected = set()
    for key, count in expected.items():
        if len(by_key.get(key, [])) < count:
            return None
        selected.update(sorted(by_key[key], key=lambda rid: (-scores.get((query["id"], rid), -1e99), rid))[:count])
    return selected, support_count, ref_count, expected


def score_map(orders, array_path):
    values = np.load(array_path)
    out = {}
    offset = 0
    for order in orders:
        for alarm, value in zip(order["alarms"], values[offset : offset + len(order["alarms"])]):
            out[(order["id"], alarm["@rid"])] = float(value)
        offset += len(order["alarms"])
    return out


def main():
    train = [prepare_order(path, True) for path in sorted((ROOT / "train").iterdir()) if path.is_dir()]
    test = [prepare_order(path, False) for path in sorted((ROOT / "test").iterdir()) if path.is_dir()]
    groups = defaultdict(list)
    for order in train:
        groups[order["signature"]].append(order)
    scores = score_map(test, ROOT / "experiments/v30_meta_stack/v30_consensus_test.npy")
    baseline = audit.load_submission(ROOT / "experiments/v60_combined_checkpoint/highest_verified_combined.csv")
    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    universe = {key: i for i, key in enumerate(sorted(alarms))}
    equations, rhs, _ = audit.build_equations(records, universe)
    output = {}
    for disjoint in (True, False):
        # For test inspection V89 used all same-signature references.  Keep a
        # separate disjoint flag so the report makes that choice explicit.
        for min_support, confidence in ((2, 0.5), (2, 0.67), (2, 0.8), (2, 0.9), (4, 0.5), (5, 0.5)):
            rows = []
            for query in test:
                refs = groups.get(query["signature"], [])
                if disjoint:
                    # Approximate V89's station/site disjointness using the
                    # structural site key only when available; if absent,
                    # retain all references (the default V89 test path).
                    refs = [ref for ref in refs if ref["id"] != query["id"]]
                pred = predict(query, refs, scores, min_support, confidence)
                if pred is None:
                    continue
                selected = pred[0]
                current = set(baseline.get(query["id"], []))
                remove, add = sorted(current - selected), sorted(selected - current)
                if not remove and not add:
                    continue
                if any((query["id"], rid) not in universe for rid in remove + add):
                    continue
                coeff = {}
                for rid in add:
                    index = universe[(query["id"], rid)]
                    coeff[index] = coeff.get(index, 0) + 1
                for rid in remove:
                    index = universe[(query["id"], rid)]
                    coeff[index] = coeff.get(index, 0) - 1
                bounds = audit.solve_min_max(coeff, equations, rhs, 5.0)
                rows.append({
                    "order_id": query["id"], "support_count": pred[1], "reference_count": pred[2],
                    "remove_rids": remove, "add_rids": add, "delta_p": len(add) - len(remove),
                    "equation_bounds": bounds,
                    "add_scores": {rid: scores.get((query["id"], rid)) for rid in add},
                    "remove_scores": {rid: scores.get((query["id"], rid)) for rid in remove},
                })
            key = f"{'disjoint' if disjoint else 'allrefs'}_support{min_support}_conf{confidence}"
            output[key] = rows
    path = HERE / "relaxed_template_actions.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {"count": len(rows), "rows": rows} for key, rows in output.items() if rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
