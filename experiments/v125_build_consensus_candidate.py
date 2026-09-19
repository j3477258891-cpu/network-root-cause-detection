"""Build a single cross-model-consensus swap on the verified pair015 champion."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
OUT = EXP / "v125_attack_campaign"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
import v124_build_group_probes as v124  # noqa: E402

BASE = ROOT / "experiments/v124_campaign/v124_single_pair_015_from_probe13.csv"
ORDER = "06d84a90-670f-4ccb-b482-9afdb874d164"
REMOVE = "#-1:d1c9980d-85c4-40db-9036-f831f95a0aa7"
ADD = "#-1:c90741fc-ee68-47d7-87d0-59265f5ff600"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    alarms, _ = v124.v121.load_alarm_records()
    rows = []
    with BASE.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"order_id": r["order_id"],
                         "roots": json.loads(r["output"])["rootcause"]})
    present = {r["order_id"]: {n["@rid"] for n in r["roots"]} for r in rows}
    if REMOVE not in present[ORDER] or ADD in present[ORDER]:
        raise RuntimeError("consensus action is not valid on current champion")
    out = v124.apply_actions(rows, [{"order_id": ORDER, "remove_rid": REMOVE, "add_rid": ADD}], alarms)
    valid, predictions, message = v124.validate(out, 1035)
    if not valid:
        raise RuntimeError(message)
    path = OUT / "v125_consensus_06d84_from_pair015.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order_id", "output"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    meta = {
        "version": "v125-four-model-consensus-single",
        "base": str(BASE), "base_sha256": sha256(BASE),
        "base_public_f1": 0.921597, "base_tp": 958,
        "order_id": ORDER, "remove_rid": REMOVE, "add_rid": ADD,
        "support": 4,
        "source": "research_pairwise_error_model/pair_delta_report.json",
        "model_scores": {
            "template_extra": 0.8815262913703918,
            "template_hist": 0.6503706576061249,
            "site_extra": 0.8677830100059509,
            "site_hist": 0.7202482223510742,
        },
        "predictions": predictions, "orders": len(out),
        "sha256": sha256(path), "status": "exploratory_not_scored",
        "warning": "OOF/model consensus is not leaderboard evidence.",
    }
    (OUT / "v125_consensus_06d84_from_pair015.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
