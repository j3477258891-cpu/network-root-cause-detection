"""analyze_probes.py - 深度分析 probe 单元并生成优化建议"""
import json, csv, sys
from pathlib import Path
from collections import defaultdict

TEST_DIR = Path(r"D:\zgyidong\test")
CATALOG_PATH = Path(r"D:\zgyidong\experiments\candidate_catalog.json")
CHAMP_PATH = Path(r"D:\zgyidong\codexgz\result_record_probe_swap12_score_0.905373.csv")
OUTPUT = Path(r"D:\zgyidong\experiments\reports\probe_analysis.json")

def main():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    # Build node lookup from test topologies
    node_lookup = {}
    for wo_dir in sorted(TEST_DIR.iterdir()):
        if not wo_dir.is_dir():
            continue
        topo = json.loads((wo_dir / f"{wo_dir.name}.log.topo.json").read_text(encoding="utf-8"))
        for nd in topo["nodes"]:
            if nd.get("@class") == "Alarm":
                node_lookup[nd["@rid"]] = nd

    # Read champion
    champ = {}
    with open(CHAMP_PATH, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            champ[row["order_id"]] = {n["@rid"] for n in json.loads(row["output"])["rootcause"]}

    results = []
    for u in catalog["units"]:
        rid_rm, rid_add = u["remove"]["rid"], u["add"]["rid"]
        order = u["remove"]["order_id"]
        e = u.get("evidence", {})

        rm_node = node_lookup.get(rid_rm, {})
        add_node = node_lookup.get(rid_add, {})

        rm_title = rm_node.get("title", "?")
        add_title = add_node.get("title", "?")
        rm_device = rm_node.get("device", "?")
        add_device = add_node.get("device", "?")
        rm_reason = rm_node.get("reason", "")[:40]
        add_reason = add_node.get("reason", "")[:40]

        has_scores = "add_score" in e

        if has_scores:
            margin = e["add_score"] - e["remove_score"]
            rm_seeds = e.get("remove_seed_scores", [])
            ad_seeds = e.get("add_seed_scores", [])
            rm_spread = max(rm_seeds) - min(rm_seeds) if rm_seeds else 0
            ad_spread = max(ad_seeds) - min(ad_seeds) if ad_seeds else 0
            rm_stability = 1.0 - rm_spread
            ad_stability = 1.0 - ad_spread
            stability = (rm_stability + ad_stability) / 2
            quality = margin * stability * e["add_score"]
        else:
            margin = 0
            rm_spread = 0
            ad_spread = 0
            stability = 0
            quality = 0

        title_preserve = rm_title == add_title
        device_preserve = rm_device == add_device

        if has_scores and margin > 0.10 and stability > 0.95:
            risk = "LOW"
        elif has_scores and margin > 0.05 and stability > 0.93:
            risk = "MEDIUM"
        elif has_scores and margin > 0:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        results.append({
            "id": u["id"],
            "order": order,
            "order_short": order[:8],
            "rm_title": rm_title,
            "add_title": add_title,
            "rm_device": rm_device,
            "add_device": add_device,
            "rm_reason": rm_reason,
            "add_reason": add_reason,
            "title_preserve": title_preserve,
            "device_preserve": device_preserve,
            "margin": round(margin, 6),
            "rm_spread": round(rm_spread, 6),
            "ad_spread": round(ad_spread, 6),
            "stability": round(stability, 6),
            "quality": round(quality, 6),
            "risk": risk,
            "has_scores": has_scores,
        })

    # Sort by quality
    results.sort(key=lambda x: (-x["quality"], -x["margin"]))

    print("=" * 120)
    print("  UNIT QUALITY RANKING (sorted by composite quality)")
    print("=" * 120)
    header = f"  {'RANK':<5} {'UNIT':<20} {'MARGIN':<9} {'STABIL':<8} {'RISK':<9} {'TYPE':<22} {'SWAP'}"
    print(header)
    print("-" * 120)
    for i, r in enumerate(results):
        if r["title_preserve"]:
            ptype = "title preserve"
        elif r["device_preserve"]:
            ptype = "device preserve"
        else:
            ptype = "full type+dev change"
        swap_desc = f"[{r['rm_title'][:16]}] -> [{r['add_title'][:16]}]"
        print(f"  {i+1:<5} {r['id']:<20} {r['margin']:+.6f}  {r['stability']:.4f}  {r['risk']:<9} {ptype:<22} {swap_desc}")

    # Category summary
    low_risk = [r for r in results if r["risk"] == "LOW"]
    med_risk = [r for r in results if r["risk"] == "MEDIUM"]
    high_risk = [r for r in results if r["risk"] == "HIGH"]
    critical = [r for r in results if r["risk"] == "CRITICAL"]

    print()
    print("=" * 120)
    print("  RISK CATEGORY SUMMARY")
    print("=" * 120)
    for label, group in [("LOW", low_risk), ("MEDIUM", med_risk), ("HIGH", high_risk), ("CRITICAL", critical)]:
        print(f"\n  {label} risk ({len(group)} units):")
        for r in group:
            print(f"    {r['id']:<20} | {r['rm_title'][:20]:<20} -> {r['add_title'][:20]:<20} "
                  f"| same_device={r['device_preserve']} | margin={r['margin']:+.4f}")

    # Recommendation
    print()
    print("=" * 120)
    print("  RECOMMENDATIONS")
    print("=" * 120)

    print(f"""
  1. CONSERVATIVE (lowest risk): Submit only LOW-risk units
     Batch: {', '.join(r['id'] for r in low_risk)}  

  2. BALANCED (recommended): Submit LOW + MEDIUM risk units together
     This gives the model its highest-confidence changes first.
     Batch size: {len(low_risk) + len(med_risk)}

  3. DROP CRITICAL units: {len(critical)} units have no model scoring evidence.
     These are structural-only changes (removing x to add y elsewhere).
     They should NOT be submitted without evidence generation.
     
  4. CURRENT vs OPTIMAL comparison:
     Current probe01: ALL 26 units (includes {len(critical)} CRITICAL)
     Optimal probe01: {len(low_risk) + len(med_risk)} units (LOW+MEDIUM only)
     → Removes {len(high_risk) + len(critical)} high-risk units from first submission
""")

    # Save for later use
    output = {
        "analysis_time": "2026-08-01",
        "total_units": len(results),
        "risk_distribution": {
            "LOW": len(low_risk),
            "MEDIUM": len(med_risk),
            "HIGH": len(high_risk),
            "CRITICAL": len(critical),
        },
        "ranking": results,
        "recommendations": {
            "conservative_batch": [r["id"] for r in low_risk],
            "balanced_batch": [r["id"] for r in low_risk + med_risk],
            "drop_critical": [r["id"] for r in critical],
        },
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Analysis saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
