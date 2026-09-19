"""Read-only robust search with older V119/V123 actions added for context.

The official V120/V121/V122 result is intentionally unchanged.  This optional
diagnostic asks whether the older nested V119 addition/deletion pools or V123
pair catalog create a *balanced* robust TP gain when combined.  It never
creates a submission file.
"""

from __future__ import annotations

import json
from pathlib import Path

import audit_real_scores as audit
import robust_combo_bounds as robust

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def nested_v119(path, universe, baseline):
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, row in enumerate(payload.get("additions", [])):
        oid, rid = row.get("order_id"), row.get("rid")
        if not oid or not rid or (oid, rid) not in universe or rid in set(baseline.get(oid, [])):
            continue
        out.append({
            "candidate_id": f"v119_add_{i:03d}", "catalog_sources": ["v119"],
            "catalog_ids": [f"v119_add_{i:03d}"], "kind": "addition",
            "order_id": oid, "add_rid": rid,
            "v119_prior": row.get("prior"),
            "nominal_p_net_gain": row.get("prior"),
        })
    for i, row in enumerate(payload.get("deletions", [])):
        oid, rid = row.get("order_id"), row.get("rid")
        if not oid or not rid or (oid, rid) not in universe or rid not in set(baseline.get(oid, [])):
            continue
        out.append({
            "candidate_id": f"v119_del_{i:03d}", "catalog_sources": ["v119"],
            "catalog_ids": [f"v119_del_{i:03d}"], "kind": "deletion",
            "order_id": oid, "remove_rid": rid,
            "v119_false_prior": row.get("false_prior"),
            "nominal_p_net_gain": row.get("false_prior"),
        })
    return out


def load_v123(path, universe, baseline):
    out = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    for i, row in enumerate(payload if isinstance(payload, list) else payload.get("candidates", [])):
        oid, rem, add = row.get("order_id"), row.get("remove_rid"), row.get("add_rid")
        if not oid or not rem or not add or (oid, rem) not in universe or (oid, add) not in universe:
            continue
        cur = set(baseline.get(oid, []))
        if rem not in cur or add in cur:
            continue
        item = dict(row)
        item.update({
            "candidate_id": item.get("candidate_id", f"v123_{i:03d}"),
            "catalog_sources": ["v123"], "catalog_ids": [item.get("candidate_id", f"v123_{i:03d}")],
            "kind": "swap",
        })
        if "nominal_p_net_gain" not in item:
            item["nominal_p_net_gain"] = item.get("p_pair_gain")
        out.append(item)
    return out


def load_v16(path, universe, baseline):
    import csv
    out = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for i, row in enumerate(rows):
        oid, rem, add = row.get("order_id"), row.get("removed_rid"), row.get("added_rid")
        if not oid or not rem or not add or (oid, rem) not in universe or (oid, add) not in universe:
            continue
        cur = set(baseline.get(oid, []))
        if rem not in cur or add in cur:
            continue
        out.append({
            "candidate_id": f"v16_{i:04d}", "catalog_sources": ["v16"],
            "catalog_ids": [f"v16_{i:04d}"], "kind": "swap",
            "order_id": oid, "remove_rid": rem, "add_rid": add,
            "nominal_p_net_gain": float(row.get("seed_min_margin") or 0.0),
            "seed_min_margin": float(row.get("seed_min_margin") or 0.0),
        })
    return out


def load_v30_manifests(folder, universe, baseline):
    out = []
    seen = set()
    for path in sorted(folder.glob("manifests/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for i, row in enumerate(payload.get("actions", [])):
            oid = row.get("order_id")
            rems, adds = row.get("remove_rids", []), row.get("add_rids", [])
            if not oid or len(rems) > 1 or len(adds) > 1 or (not rems and not adds):
                continue
            rem = rems[0] if rems else None
            add = adds[0] if adds else None
            if rem and (oid, rem) not in universe or add and (oid, add) not in universe:
                continue
            cur = set(baseline.get(oid, []))
            kind = "swap" if rem and add else "deletion" if rem else "addition"
            if kind == "addition" and add in cur:
                continue
            if kind == "deletion" and rem not in cur:
                continue
            if kind == "swap" and (rem not in cur or add in cur):
                continue
            key = (oid, rem or "", add or "")
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "candidate_id": f"v30_{len(out):04d}", "catalog_sources": ["v30"],
                "catalog_ids": [str(row.get("action_id", f"v30_{i:04d}"))], "kind": kind,
                "order_id": oid, "remove_rid": rem, "add_rid": add,
                "nominal_p_net_gain": row.get("expected_gain"),
                "v30_expected_gain": row.get("expected_gain"),
            })
    return out


def merge(base, extras):
    merged = {}
    for item in base + extras:
        key = (item["order_id"], item.get("remove_rid", ""), item.get("add_rid", ""))
        if key not in merged:
            merged[key] = dict(item)
        else:
            merged[key].setdefault("catalog_sources", []).extend(item.get("catalog_sources", []))
            merged[key].setdefault("catalog_ids", []).extend(item.get("catalog_ids", []))
    out = list(merged.values())
    out.sort(key=lambda c: (-float(c.get("nominal_p_net_gain") or 0.0), c["order_id"], c.get("remove_rid", ""), c.get("add_rid", "")))
    return out


def run(name, candidates, equations, rhs, universe):
    vectors = robust.coefficient_vectors(candidates, universe, equations.shape[1])
    groups = robust.build_selection_groups(candidates, False)
    result = robust.robust_search(candidates, vectors, equations, rhs, groups, 15.0, 15.0, 80)
    best = result.get("best", {}) if isinstance(result.get("best"), dict) else {}
    selected = best.get("selected", [])
    dp = sum(1 if candidates[j].get("kind") == "addition" else -1 if candidates[j].get("kind") == "deletion" else 0 for j in selected)
    worst = best.get("verified_worst_case_delta")
    output = {"name": name, "candidate_count": len(candidates), "status": result.get("status"), "iterations": result.get("iterations"), "worst_case_delta": worst, "selected_count": len(selected), "prediction_delta": dp}
    if worst is not None:
        denominator = 1044 + 1035 + dp
        output["worst_case_f1"] = 2 * (956 + int(worst)) / denominator
    output["selected_ids"] = [candidates[j].get("candidate_id") for j in selected]
    return output


def main():
    records, _ = audit.collect_records()
    alarms, _ = audit.load_test_universe()
    keys = sorted(alarms)
    universe = {key: i for i, key in enumerate(keys)}
    equations, rhs, _ = audit.build_equations(records, universe)
    baseline = audit.load_submission(audit.BASE_PATH)
    official = robust.load_candidates(universe, baseline)
    old = nested_v119(ROOT / "experiments/v119_joint_campaign/candidate_catalog.json", universe, baseline)
    v123 = load_v123(ROOT / "experiments/v123_pair_count_campaign/candidate_catalog.json", universe, baseline)
    v16 = load_v16(ROOT / "experiments/v16/v16_test_swap_catalog.csv", universe, baseline)
    v30 = load_v30_manifests(ROOT / "experiments/v30_meta_stack", universe, baseline)
    sets = {
        "official_v120_v121_v122": official,
        "official_plus_v119": merge(official, old),
        "official_plus_v123": merge(official, v123),
        "official_plus_v119_v123": merge(official, old + v123),
        "official_plus_v16": merge(official, v16),
        "official_plus_v16_v119_v123": merge(official, v16 + old + v123),
        "official_plus_v30": merge(official, v30),
        "official_plus_v30_v119_v123": merge(official, v30 + old + v123),
    }
    output = {
        name: run(name, candidates, equations, rhs, universe)
        for name, candidates in sets.items()
    }
    path = HERE / "robust_broad_catalogs.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
