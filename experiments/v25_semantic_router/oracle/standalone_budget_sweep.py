"""Standalone oracle budget sweep (numpy-only, reads the V25 bundle directly).

Re-derives the theoretical ceiling WITHOUT sklearn / v10 / v25_data so it can run
on the bare managed numpy.  Reproduces base_tp=2826 / oracle budget2=+165 from
the existing v25_oracle_report.json as a correctness check, then sweeps budget.

Key fact being verified: this "oracle" ranks each order's false-negatives by
V11's OWN logits and adds the top-K — i.e. it is a *label oracle on V11 logits*,
not a ceiling for the semantic model.
"""

import json
from pathlib import Path

import numpy as np

BUNDLE = Path("cloud_dataset/v25_semantic_router.npz")
TARGET_TRAIN = 3169
MAX_ROOT = 8


def exact_count_mask(scores, ptr, target_count, max_roots=MAX_ROOT):
    selected = np.zeros(len(scores), dtype=bool)
    optional = []
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        ranked = np.argsort(-scores[start:stop], kind="stable")[:max_roots]
        selected[start + ranked[0]] = True
        optional.extend((start + ranked[1:]).tolist())
    remaining = target_count - int(selected.sum())
    optional = np.asarray(optional, dtype=np.int64)
    optional = optional[np.argsort(-scores[optional], kind="stable")]
    selected[optional[:remaining]] = True
    return selected


def oracle_sweep(v11, labels, ptr, max_action):
    base = exact_count_mask(v11, ptr, TARGET_TRAIN)
    total_base = 0
    total_oracle = 0
    total_fn = 0
    total_covered = 0
    for start, stop in zip(ptr[:-1], ptr[1:]):
        start, stop = int(start), int(stop)
        base_sl = base[start:stop]
        lab_sl = labels[start:stop]
        v11_sl = v11[start:stop]
        base_tp = int((base_sl & (lab_sl == 1)).sum())
        base_count = int(base_sl.sum())
        fn_idx = np.flatnonzero((base_sl == 0) & (lab_sl == 1))
        fp_idx = np.flatnonzero(base_sl & (lab_sl == 0))
        n_fn, n_fp = len(fn_idx), len(fp_idx)
        total_fn += n_fn
        if n_fn == 0 and n_fp == 0:
            total_base += base_tp
            total_oracle += base_tp
            continue
        fn_sorted = fn_idx[np.argsort(-v11_sl[fn_idx], kind="stable")]
        fp_sorted = fp_idx[np.argsort(v11_sl[fp_idx], kind="stable")]
        best = base_tp
        best_add = 0
        for na in range(0, min(n_fn, max_action) + 1):
            for nr in range(0, min(n_fp, max_action) + 1):
                if na + nr > max_action:
                    continue
                if not (1 <= base_count + na - nr <= MAX_ROOT):
                    continue
                new_tp = base_tp + na  # adds are true FNs; removes are FPs (no TP change)
                if new_tp > best:
                    best = new_tp
                    best_add = na
        total_base += base_tp
        total_oracle += best
        total_covered += best_add
    return int(total_base), int(total_oracle), int(total_fn), int(total_covered)


def main():
    with np.load(BUNDLE, allow_pickle=False) as archive:
        v11 = archive["train_v11"].astype(np.float32)
        labels = archive["train_labels"].astype(np.int8)
        ptr = archive["train_alarm_ptr"].astype(np.int64)

    rows = []
    for budget in (2, 3, 4, 5):
        base_tp, oracle_tp, fn_total, fn_covered = oracle_sweep(v11, labels, ptr, budget)
        gain = oracle_tp - base_tp
        coverage = fn_covered / max(fn_total, 1)
        passed = gain >= 170 and coverage >= 0.80
        rows.append(
            {
                "budget": budget,
                "base_tp": base_tp,
                "oracle_tp": oracle_tp,
                "gain": gain,
                "fn_total": fn_total,
                "fn_covered": fn_covered,
                "fn_coverage": round(coverage, 6),
                "gate_passed": passed,
            }
        )
        print(
            f"budget={budget}: base={base_tp} oracle={oracle_tp} gain={gain:+d} "
            f"fn={fn_total} covered={fn_covered} ({coverage:.1%}) "
            f"gate={'PASS' if passed else 'FAIL'}"
        )

    out = Path("outputs/v25_oracle_budget_sweep.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
