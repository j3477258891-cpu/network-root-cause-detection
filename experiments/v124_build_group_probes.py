"""V124 high-risk group-decode campaign file builder.

Base = probe_13_single.csv (current champion, TP=957, P=1035).
Group probes: champion + 3 (or 2) same-order swap actions; P stays 1035.
Add20 calibration: champion + top-20 v122 additions; P=1055 (for q estimation).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(r"D:\zgyidong")
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))
import v121_equation_safe_campaign as v121  # noqa: E402

OUT = ROOT / "experiments/v124_campaign"
CHAMPION = ROOT / "experiments/v120_swap_campaign/probe_13_single.csv"
CATALOG = ROOT / "experiments/v120_swap_campaign/candidate_catalog.json"
V122_CATALOG = ROOT / "experiments/v122_multimodel_addition_campaign/candidate_catalog.json"

GROUPS = {
    1: ["v120_pair_008", "v120_pair_014", "v120_pair_044"],
    2: ["v120_pair_049", "v120_pair_084", "v120_pair_015"],
    3: ["v120_pair_011", "v120_pair_018", "v120_pair_045"],
    4: ["v120_pair_030", "v120_pair_054", "v120_pair_033"],
    5: ["v120_pair_075", "v120_pair_076", "v120_pair_017"],
    6: ["v120_pair_016", "v120_pair_052", "v120_pair_042"],
    7: ["v120_pair_083", "v120_pair_073", "v120_pair_046"],
    8: ["v120_pair_036", "v120_pair_004", "v120_pair_041"],
    9: ["v120_pair_013", "v120_pair_029", "v120_pair_056"],
    10: ["v120_pair_087", "v120_pair_077", "v120_pair_065"],
    11: ["v120_pair_026", "v120_pair_001", "v120_pair_021"],
    12: ["v120_pair_022", "v120_pair_055"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_champion_rows() -> list[dict[str, Any]]:
    out = []
    with CHAMPION.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            out.append({"order_id": r["order_id"], "roots": json.loads(r["output"])["rootcause"]})
    return out


def apply_actions(base_rows: list[dict[str, Any]], actions: list[dict[str, Any]],
                  alarms: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    by_oid = {a["order_id"]: a for a in actions}
    out = []
    for row in base_rows:
        roots = [dict(node) for node in row["roots"]]
        act = by_oid.get(row["order_id"])
        if act is not None:
            present = {node["@rid"] for node in roots}
            if act["remove_rid"] not in present or act["add_rid"] in present:
                raise ValueError(f"invalid action {row['order_id']}")
            roots = [node for node in roots if node["@rid"] != act["remove_rid"]]
            roots.append(v121.alarm_node(alarms[(row["order_id"], act["add_rid"])]))
        out.append({"order_id": row["order_id"], "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["order_id", "output"])
        writer.writeheader()
        writer.writerows(rows)


def validate(rows: list[dict[str, Any]], expected_p: int) -> tuple[bool, int, str]:
    total = sum(len(json.loads(x["output"])["rootcause"]) for x in rows)
    n546 = len(rows) == 546
    roots_ok = all(0 < len(json.loads(x["output"])["rootcause"]) <= 8 for x in rows)
    return (total == expected_p and n546 and roots_ok), total, f"P={total} n546={n546} roots_ok={roots_ok}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cat = json.loads(CATALOG.read_text(encoding="utf-8"))
    by_id = {c["candidate_id"]: c for c in cat["candidates"]}
    alarms, _ = v121.load_alarm_records()
    base_rows = load_champion_rows()
    base_by = {r["order_id"]: {x["@rid"] for x in r["roots"]} for r in base_rows}

    probe_meta = []
    for gid in sorted(GROUPS):
        actions = []
        for cid in GROUPS[gid]:
            c = by_id[cid]
            assert c["remove_rid"] in base_by[c["order_id"]], (cid, "remove not in champion")
            assert c["add_rid"] not in base_by[c["order_id"]], (cid, "add already in champion")
            actions.append({"order_id": c["order_id"], "remove_rid": c["remove_rid"], "add_rid": c["add_rid"]})
        rows = apply_actions(base_rows, actions, alarms)
        ok, total, msg = validate(rows, 1035)
        if not ok:
            raise RuntimeError(f"group {gid} invalid: {msg}")
        path = OUT / f"v124_group_{gid:02d}.csv"
        write_csv(path, rows)
        probe_meta.append({
            "probe_id": gid, "file": str(path), "predictions": 1035,
            "sha256": sha256(path), "candidates": GROUPS[gid],
            "orders": [by_id[c]["order_id"] for c in GROUPS[gid]],
        })
        print(f"group {gid:02d}: P=1035 ok sha256={sha256(path)[:16]}")

    # add20 calibration from v122 catalog (top by support, then consensus).
    v122 = json.loads(V122_CATALOG.read_text(encoding="utf-8"))["candidates"]
    # The V124 plan explicitly ranks additions by consensus score.  Support
    # channels remain recorded metadata, but must not override that ranking.
    v122_sorted = sorted(v122, key=lambda x: (-x.get("consensus_score", float("-inf")), x.get("order_id", ""), x.get("add_rid", "")))
    # Exclude RIDs that conflict with the retained champion replacement (v120_pair_038).
    retained = by_id["v120_pair_038"]
    conflict_rids = {retained["order_id"]}
    chosen = []
    for c in v122_sorted:
        oid = c["order_id"]
        if oid in conflict_rids:
            continue
        n_roots = len(base_by.get(oid, set()))
        if n_roots >= 8:
            continue
        if c["add_rid"] in base_by.get(oid, set()):
            continue
        chosen.append(c)
        conflict_rids.add(oid)
        if len(chosen) >= 20:
            break
    if len(chosen) < 20:
        print(f"WARN: only {len(chosen)} add candidates found")
    add_rows = []
    for row in base_rows:
        roots = [dict(node) for node in row["roots"]]
        for c in chosen:
            if c["order_id"] == row["order_id"]:
                if len(roots) >= 8 or any(n["@rid"] == c["add_rid"] for n in roots):
                    raise ValueError("add invalid")
                roots.append(v121.alarm_node(alarms[(c["order_id"], c["add_rid"])]))
        add_rows.append({"order_id": row["order_id"], "output": json.dumps({"rootcause": roots}, ensure_ascii=False)})
    ok, total, msg = validate(add_rows, 1055)
    if not ok:
        raise RuntimeError(f"add20 invalid: {msg}")
    path = OUT / "v124_add20_calibration.csv"
    write_csv(path, add_rows)
    print(f"add20: P={total} ok sha256={sha256(path)[:16]} n_add={len(chosen)}")
    write_json(OUT / "v124_group_matrix.json", {
        "version": "v124-group-decode", "base": str(CHAMPION), "base_tp": 957,
        "base_sha256": sha256(CHAMPION),
        "groups": probe_meta,
        "add20": {"file": str(path), "predictions": total, "sha256": sha256(path),
                  "n_add": len(chosen),
                  "chosen": [{"order_id": c["order_id"], "add_rid": c["add_rid"],
                              "consensus": c.get("consensus_score"), "support": c.get("evidence_support_channels")} for c in chosen]},
    })


if __name__ == "__main__":
    main()
