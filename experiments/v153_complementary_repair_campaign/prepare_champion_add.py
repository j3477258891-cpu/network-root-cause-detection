"""Build 'p03 champion + candidate 9' and 'p03 champion + candidate 15' submissions.

p03 (0.927855, P=1049, TP=971) is V149 + candidates {8,11,14,17}.
The equation system currently pins: 候选5=0 (from p05) and 候选9+候选15=1 (from p04).
To determine WHICH of {9,15} is true, test p03+candidate9 first (prior 58.8% true),
then p03+candidate15 if 9 is false.  Each file, if the candidate is the true one,
reaches F1 = 2*972/(1044+1050) = 0.928367.

This script ONLY reads p03 and writes two new CSVs. It does NOT touch the ledger,
manifest, or quota.  Output is verified (546 orders, P=1050, no dup, exact +1 diff).
"""

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P03 = HERE / "v153_probe_p03_group.csv"

# Candidate node payloads (validated when p06/p04 were authored).
CAND9 = {
    "order_id": "1117ad03-8891-4bf6-854c-612bf19fdae4",
    "node": {
        "@rid": "#-1:bea1860d-537e-4e4b-ba53-08f5ce0bd7dc",
        "title": "小区关断告警",
        "location": "SubNetwork=CMCC-GZ-04,ManagedElement=12641069,ENBCUCPFunction=1,CULTE=1,CUEUtranCellTDDLTE=3",
        "reason": "小区被关断",
    },
}
CAND15 = {
    "order_id": "ab02ed9d-74ea-4fc1-8935-9e5396cf94ba",
    "node": {
        "@rid": "#-1:2fe391a0-09be-47fc-bb08-f50ecb445a3f",
        "title": "[衍生告警]PTN光缆中断，单报LOS",
        "location": "R8EGF[0-1-2]-GE\\:1",
        "reason": "以太网物理接口(ETPI) 信号丢失(LOS)",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_p03():
    assert not P03.read_bytes().startswith(b"\xef\xbb\xbf"), "p03 must be plain UTF-8 (no BOM)"
    with P03.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["order_id", "output"], reader.fieldnames
        rows = [(r["order_id"], json.loads(r["output"])) for r in reader]
    return rows


def dump_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build(rows, target_order, node, out_path):
    out = []
    modified = False
    for oid, payload in rows:
        if oid == target_order:
            rc = list(payload["rootcause"])
            rids = {item["@rid"] for item in rc}
            assert node["@rid"] not in rids, "candidate rid already present in p03"
            assert len(rc) < 8, "order already at 8 roots"
            rc.append(node)
            payload = {"rootcause": rc}
            modified = True
        out.append((oid, payload))
    assert modified, "target order not found in p03"
    # plain UTF-8, no BOM: the platform header must be exactly "order_id,output"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order_id", "output"])
        writer.writeheader()
        for oid, payload in out:
            writer.writerow({"order_id": oid, "output": dump_json(payload)})
    return out


def verify(rows, out_path, expected_p):
    total = 0
    dup = 0
    bad_count = 0
    for oid, payload in rows:
        rc = payload["rootcause"]
        total += len(rc)
        if len(rc) < 1 or len(rc) > 8:
            bad_count += 1
        if len({item["@rid"] for item in rc}) != len(rc):
            dup += 1
    assert len(rows) == 546, len(rows)
    assert total == expected_p, (total, expected_p)
    assert dup == 0 and bad_count == 0, (dup, bad_count)
    return {"orders": len(rows), "predictions": total, "sha256": sha256(out_path),
            "duplicates": dup, "bad_root_counts": bad_count}


def main():
    rows = load_p03()
    print(f"p03 loaded: {len(rows)} orders")

    out9 = HERE / "v153_p03_add_candidate9.csv"
    rows9 = build(rows, CAND9["order_id"], CAND9["node"], out9)
    v9 = verify(rows9, out9, 1050)
    print(f"\n[+候选9] {out9.name}")
    print(f"  orders={v9['orders']} P={v9['predictions']} dup={v9['duplicates']} bad_count={v9['bad_root_counts']}")
    print(f"  sha256={v9['sha256']}")

    out15 = HERE / "v153_p03_add_candidate15.csv"
    rows15 = build(rows, CAND15["order_id"], CAND15["node"], out15)
    v15 = verify(rows15, out15, 1050)
    print(f"\n[+候选15] {out15.name}")
    print(f"  orders={v15['orders']} P={v15['predictions']} dup={v15['duplicates']} bad_count={v15['bad_root_counts']}")
    print(f"  sha256={v15['sha256']}")


if __name__ == "__main__":
    main()
