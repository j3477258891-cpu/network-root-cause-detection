"""
V27 Online Probe Framework — 动作目录构建
从 champion 出发，生成可独立验证的原子动作池
"""

import json, csv, hashlib
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

CHAMPION_PATH = "D:/zgyidong/experiments/submissions/champion_0.906324_day01_probe01_v11_full.csv"
V11_OOF_CONTEXT = "D:/zgyidong/codexgz/v11/v11_oof_context.npy"
V11_OOF_META = "D:/zgyidong/codexgz/v11/v11_oof_meta.npy"
V11_TEST = "D:/zgyidong/codexgz/v11/v11_test_scores.npy"
OUTPUT_DIR = Path("D:/zgyidong/experiments/v27_online_probes")
MAX_ROOT = 8


def load_champion():
    """Load champion CSV, return (order_ids, predictions, all_rids)."""
    with open(CHAMPION_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    orders = {}
    for r in rows:
        pred = json.loads(r["output"])
        orders[r["order_id"]] = {
            "rids": [rc["@rid"] for rc in pred["rootcause"]],
            "raw": pred["rootcause"],
        }
    return orders


def load_v11_scores():
    """Load V11 test scores and compute per-order rankings."""
    scores = np.load(V11_TEST).astype(np.float32)
    return scores


def build_catalog():
    """
    Build the master action catalog from champion.
    For each test order, identify:
    - Selected nodes (V11 predictions) with their confidence
    - Unselected boundary candidates (near the selection threshold)
    """
    import sys
    sys.path.insert(0, "D:/zgyidong/codexgz/work")
    import v10_grouped_ensemble as v10

    champion = load_champion()
    v11_scores = load_v11_scores()

    # Load test orders for alarm metadata
    test_orders = v10.load_orders(Path("D:/zgyidong/test"), False)

    # Build slices manually (avoid v10.prepare with empty train)
    test_slices = []
    offset = 0
    for order in test_orders:
        n = len(order["alarms"])
        test_slices.append(slice(offset, offset + n))
        offset += n

    catalog = {
        "champion": {
            "f1": 0.906324,
            "tp": 953,
            "predictions": 1059,
            "hidden_total": 1044,
        },
        "orders": {},
        "actions": [],
    }

    for oi, order in enumerate(test_orders):
        sl = test_slices[oi]
        ov = v11_scores[sl]
        alarms = order["alarms"]
        n = len(alarms)

        # Champion selections for this order
        champ_rids = set(champion.get(order["id"], {}).get("rids", []))
        if not champ_rids:
            continue

        # Build selection mask
        selected = np.array([
            alarms[i]["@rid"] in champ_rids for i in range(n)
        ])
        n_selected = int(np.sum(selected))

        # Rank all alarms by V11 score
        ranked = np.argsort(-ov, kind="stable")
        ranks = np.zeros(n, dtype=int)
        for r, idx in enumerate(ranked):
            ranks[idx] = r + 1

        # Identify boundary: last selected, first unselected
        sel_indices = np.where(selected)[0]
        unsel_indices = np.where(~selected)[0]

        boundary_info = {}
        if len(sel_indices) > 0 and len(unsel_indices) > 0:
            weakest_sel = sel_indices[np.argmin(ov[sel_indices])]
            strongest_unsel = unsel_indices[np.argmax(ov[unsel_indices])]

            boundary_info = {
                "weakest_selected": {
                    "local_idx": int(weakest_sel),
                    "rid": alarms[weakest_sel]["@rid"],
                    "score": float(ov[weakest_sel]),
                    "rank": int(ranks[weakest_sel]),
                    "title": str(alarms[weakest_sel].get("title", "")),
                },
                "strongest_unselected": {
                    "local_idx": int(strongest_unsel),
                    "rid": alarms[strongest_unsel]["@rid"],
                    "score": float(ov[strongest_unsel]),
                    "rank": int(ranks[strongest_unsel]),
                    "title": str(alarms[strongest_unsel].get("title", "")),
                },
                "gap": float(ov[weakest_sel] - ov[strongest_unsel]),
            }

        # Build candidate lists for precision probes
        selected_sorted = sel_indices[np.argsort(ov[sel_indices])]  # lowest score first
        unselected_sorted = unsel_indices[np.argsort(-ov[unsel_indices])]  # highest score first

        catalog["orders"][order["id"]] = {
            "order_index": oi,
            "n_alarms": n,
            "n_selected": n_selected,
            "boundary": boundary_info,
            "selected": [
                {
                    "local_idx": int(i), "rid": alarms[i]["@rid"],
                    "score": float(ov[i]), "rank": int(ranks[i]),
                    "title": str(alarms[i].get("title", "")),
                }
                for i in selected_sorted
            ],
            "unselected_top": [
                {
                    "local_idx": int(i), "rid": alarms[i]["@rid"],
                    "score": float(ov[i]), "rank": int(ranks[i]),
                    "title": str(alarms[i].get("title", "")),
                }
                for i in unselected_sorted[:8]
            ],
        }

    # Statistics
    orders_with_boundary = sum(
        1 for o in catalog["orders"].values() if o["boundary"]
    )
    small_gaps = sum(
        1 for o in catalog["orders"].values()
        if o["boundary"] and o["boundary"]["gap"] < 0.05
    )
    tiny_gaps = sum(
        1 for o in catalog["orders"].values()
        if o["boundary"] and o["boundary"]["gap"] < 0.02
    )

    catalog["stats"] = {
        "total_orders": len(catalog["orders"]),
        "orders_with_boundary": orders_with_boundary,
        "boundary_gap_under_0.05": small_gaps,
        "boundary_gap_under_0.02": tiny_gaps,
    }

    return catalog


def build_precision_probes(catalog, batch_size=5):
    """
    Build precision probe batches.
    Each batch removes 3-5 boundary candidates to measure TP loss.

    Batch A: Remove lowest-confidence selected nodes (tightest boundary first)
    Batch B: Remove next batch of low-confidence selected nodes
    """
    champion_orders = load_champion()

    # Collect all candidate removes: weakly selected nodes
    # Skip orders with only 1 prediction (can't remove the only one)
    candidates = []
    for oid, info in catalog["orders"].items():
        if not info["boundary"]:
            continue
        if info["n_selected"] <= 1:
            continue
        ws = info["boundary"]["weakest_selected"]
        candidates.append({
            "order_id": oid,
            "rid": ws["rid"],
            "score": ws["score"],
            "rank": ws["rank"],
            "title": ws["title"],
            "n_selected": info["n_selected"],
        })

    # Sort by score ascending (weakest first)
    candidates.sort(key=lambda x: x["score"])

    # Build non-overlapping batches
    probes = []
    used_orders = set()

    for batch_name, indices in [
        ("A", range(0, min(4, len(candidates)))),
        ("B", range(4, min(8, len(candidates)))),
    ]:
        batch_cands = []
        for i in indices:
            c = candidates[i]
            if c["order_id"] in used_orders:
                continue
            batch_cands.append(c)
            used_orders.add(c["order_id"])

        if batch_cands:
            probes.append({
                "name": f"precision_probe_{batch_name}",
                "description": f"Remove {len(batch_cands)} low-confidence selected nodes",
                "actions": [
                    {
                        "action_id": f"pp_{batch_name}_{j:02d}",
                        "order_id": c["order_id"],
                        "remove_rids": [c["rid"]],
                        "add_rids": [],
                        "source": "precision_probe",
                        "expected_gain": -1.0,
                        "evidence": {"score": c["score"], "rank": c["rank"]},
                    }
                    for j, c in enumerate(batch_cands)
                ],
            })

    return probes


def apply_probe(champion_orders, probe, test_orders):
    """Apply a probe's actions to champion, generating a new CSV."""
    out_dir = OUTPUT_DIR / "submissions"
    out_dir.mkdir(exist_ok=True)

    new_orders = {}
    for oid, champ in champion_orders.items():
        new_orders[oid] = set(champ["rids"])

    for action in probe["actions"]:
        oid = action["order_id"]
        if oid in new_orders:
            for rid in action["remove_rids"]:
                new_orders[oid].discard(rid)
            for rid in action["add_rids"]:
                new_orders[oid].add(rid)

    out_path = out_dir / f"{probe['name']}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for order in test_orders:
            oid = order["id"]
            rids = new_orders.get(oid, set())
            rcs = []
            for alarm in order["alarms"]:
                if alarm["@rid"] in rids:
                    rcs.append({
                        "@rid": alarm["@rid"],
                        "title": alarm.get("title", ""),
                        "location": alarm.get("location", ""),
                        "reason": alarm.get("reason", ""),
                    })
            writer.writerow([oid, json.dumps({"rootcause": rcs}, ensure_ascii=False)])

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    total_preds = sum(len(new_orders.get(order["id"], set())) for order in test_orders)

    return {
        "path": str(out_path),
        "sha256": sha,
        "total_predictions": total_preds,
        "n_actions": len(probe["actions"]),
    }


def main():
    print("=" * 60)
    print("V27 Online Probe Framework — Action Catalog")
    print("=" * 60)

    print("\n[1/3] Building action catalog...")
    catalog = build_catalog()

    print(f"  Orders: {catalog['stats']['total_orders']}")
    print(f"  With boundary: {catalog['stats']['orders_with_boundary']}")
    print(f"  Gap < 0.05: {catalog['stats']['boundary_gap_under_0.05']}")
    print(f"  Gap < 0.02: {catalog['stats']['boundary_gap_under_0.02']}")

    # Show top boundary orders
    print(f"\n  Top 10 tightest boundary orders:")
    boundary_orders = [
        (oid, info["boundary"])
        for oid, info in catalog["orders"].items()
        if info["boundary"]
    ]
    boundary_orders.sort(key=lambda x: x[1]["gap"])
    for oid, b in boundary_orders[:10]:
        print(f"    gap={b['gap']:.4f}  weakest_sel={b['weakest_selected']['title'][:30]} "
              f"score={b['weakest_selected']['score']:.4f}")

    # Save catalog
    catalog_path = OUTPUT_DIR / "catalog" / "action_catalog.json"
    catalog_path.parent.mkdir(exist_ok=True)
    # Convert to serializable format
    with open(catalog_path, "w") as f:
        json.dump({k: v for k, v in catalog.items() if k != "orders"},
                  f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / "catalog" / "order_boundaries.json", "w") as f:
        boundaries = {
            oid: info["boundary"]
            for oid, info in catalog["orders"].items()
            if info["boundary"]
        }
        json.dump(boundaries, f, ensure_ascii=False, indent=2)
    print(f"\n  Catalog saved: {catalog_path}")

    print("\n[2/3] Building precision probes...")
    probes = build_precision_probes(catalog)

    champion_orders = load_champion()

    # Load test orders for CSV generation
    import sys
    sys.path.insert(0, "D:/zgyidong/codexgz/work")
    import v10_grouped_ensemble as v10
    test_orders = v10.load_orders(Path("D:/zgyidong/test"), False)

    for probe in probes:
        result = apply_probe(champion_orders, probe, test_orders)
        probe["submission"] = result
        print(f"  {probe['name']}: {len(probe['actions'])} actions, "
              f"P={result['total_predictions']}, SHA256={result['sha256'][:16]}")

    # Save probes
    probes_path = OUTPUT_DIR / "probes" / "precision_probes.json"
    probes_path.parent.mkdir(exist_ok=True)
    with open(probes_path, "w") as f:
        json.dump(probes, f, ensure_ascii=False, indent=2)
    print(f"\n  Probes saved: {probes_path}")

    print("\n[3/3] Tracking setup...")
    tracker = {
        "champion": catalog["champion"],
        "submissions": [],
        "verified_gains": {"total_delta_tp": 0, "total_delta_p": 0},
    }
    tracker_path = OUTPUT_DIR / "tracking" / "submission_tracker.json"
    tracker_path.parent.mkdir(exist_ok=True)
    with open(tracker_path, "w") as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2)
    print(f"  Tracker initialized: {tracker_path}")

    print(f"\n{'='*60}")
    print("Next step: Submit precision_probe_A.csv to leaderboard")
    print("Then record the returned score in submission_tracker.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
