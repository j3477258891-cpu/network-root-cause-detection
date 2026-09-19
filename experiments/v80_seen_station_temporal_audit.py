"""Audit transductive seen-station priors under a chronological split."""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\zgyidong")
EXP = ROOT / "experiments"
RECORDS = EXP / "v25_semantic_router/cloud_dataset/semantic_records.json.gz"
V30 = EXP / "v30_meta_stack"
OUT = EXP / "v80_seen_station_temporal"
TARGET_TRAIN = 3169
TARGET_TEST = 1059


def exact(scores, ptr, target):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        order = np.argsort(-scores[start:stop], kind="stable")
        selected[start + order[0]] = True
        optional.extend((start + order[1:min(8, stop - start)]).tolist())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[: target - int(selected.sum())]] = True
    return selected


def title_key(alarm: dict) -> tuple:
    return (alarm.get("title", ""), alarm.get("reason", ""), alarm.get("timeline", ""))


def evaluate(labels, scores, ptr, target, truth, mask_orders=None):
    selected = exact(scores, ptr, target)
    if mask_orders is not None:
        keep = np.repeat(mask_orders, np.diff(ptr))
        selected &= keep
        # metrics on selected orders only
        truth = truth & keep
    tp = int(np.sum(selected & truth)); fp = int(np.sum(selected & ~truth)); fn = int(np.sum(~selected & truth))
    return {"tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "selected": selected}


def main():
    with gzip.open(RECORDS, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    tr = sorted(data["train"], key=lambda row: row["fault_time"])
    te = data["test"]
    oof_raw = np.load(V30 / "v30_consensus_oof.npy")
    test_scores = np.load(V30 / "v30_consensus_test.npy")
    original_train = data["train"]
    raw_keys = [(o["order_id"], a["rid"]) for o in original_train for a in o["alarms"]]
    score_by_key = dict(zip(raw_keys, oof_raw))
    oof_scores = np.asarray([score_by_key[(a["order_id"], alarm["rid"])]
                             for a in tr for alarm in a["alarms"]], dtype=np.float32)
    tr_ptr = np.cumsum([0] + [len(row["alarms"]) for row in tr])
    te_ptr = np.cumsum([0] + [len(row["alarms"]) for row in te])
    truth = np.asarray([alarm["is_root"] for row in tr for alarm in row["alarms"]], dtype=bool)

    rows = []
    for holdout in (0.2, 0.3):
        cut = int(len(tr) * (1 - holdout))
        history = tr[:cut]; query = tr[cut:]
        counts = defaultdict(Counter)
        totals = Counter()
        for order in history:
            for alarm in order["alarms"]:
                for station in order["station_ids"]:
                    key = (station, title_key(alarm))
                    counts[key][int(alarm["is_root"])] += 1
                    totals[(station, title_key(alarm))] += 1
        q_scores = []
        q_truth = []
        for order in query:
            stations = order["station_ids"]
            for alarm in order["alarms"]:
                vals = []
                for station in stations:
                    key = (station, title_key(alarm))
                    if totals[key] >= 2:
                        vals.append((counts[key][1] + 1) / (totals[key] + 2))
                prior = max(vals) if vals else 0.5
                q_scores.append(float(prior))
                q_truth.append(bool(alarm["is_root"]))
        q_ptr = np.cumsum([0] + [len(row["alarms"]) for row in query])
        q_truth = np.asarray(q_truth, dtype=bool)
        q_scores = np.asarray(q_scores)
        for alpha in (0.25, 0.5, 0.75, 1.0):
            # Normalize station prior against each order, then blend with V30
            # scores from the same chronological query rows.
            start = tr_ptr[cut]
            end = tr_ptr[-1]
            base = oof_scores[start:end]
            z = np.zeros_like(q_scores)
            for a, b in zip(q_ptr[:-1], q_ptr[1:]):
                local = q_scores[a:b]
                z[a:b] = (local - local.mean()) / (local.std() + 1e-6)
            b0 = np.zeros_like(base)
            for a, b in zip(q_ptr[:-1], q_ptr[1:]):
                local = base[a:b]
                b0[a:b] = (local - local.mean()) / (local.std() + 1e-6)
            score = b0 + alpha * z
            sel = exact(score, q_ptr, int(np.sum(q_truth)))
            tp = int(np.sum(sel & q_truth)); fp = int(np.sum(sel & ~q_truth)); fn = int(np.sum(~sel & q_truth))
            rows.append({"holdout": holdout, "alpha": alpha, "orders": len(query),
                         "seen_order_rate": float(np.mean([bool(set(o["station_ids"]) & {s for x in history for s in x["station_ids"]}) for o in query])),
                         "f1": 2 * tp / max(2 * tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn})

    # Test seen-station coverage and a candidate score file for inspection.
    history = tr
    counts = defaultdict(Counter); totals = Counter()
    for order in history:
        for alarm in order["alarms"]:
            for station in order["station_ids"]:
                key = (station, title_key(alarm)); counts[key][int(alarm["is_root"])] += 1; totals[key] += 1
    test_prior=[]
    train_stations={s for o in tr for s in o["station_ids"]}
    for order in te:
        for alarm in order["alarms"]:
            vals=[]
            for station in order["station_ids"]:
                key=(station,title_key(alarm))
                if totals[key]>=2: vals.append((counts[key][1]+1)/(totals[key]+2))
            test_prior.append(max(vals) if vals else 0.5)
    test_prior=np.asarray(test_prior); z=np.zeros_like(test_prior); b=np.zeros_like(test_scores)
    for a,bb in zip(te_ptr[:-1],te_ptr[1:]):
        z[a:bb]=(test_prior[a:bb]-test_prior[a:bb].mean())/(test_prior[a:bb].std()+1e-6)
        b[a:bb]=(test_scores[a:bb]-test_scores[a:bb].mean())/(test_scores[a:bb].std()+1e-6)
    candidate_scores=b+0.5*z
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "v80_test_scores.npy", candidate_scores)
    out={"version":"v80-seen-station-temporal-audit-1","rows":rows,
         "test_seen_order_rate":float(np.mean([bool(set(o["station_ids"])&train_stations) for o in te])),
         "test_candidate_scores":str(OUT/"v80_test_scores.npy"),
         "warning":"Chronological OOF proxy; candidate not public-score verified."}
    (OUT/"report.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
