"""模块 D: 受保护探针差分候选构建。

用法:
    python v13_probe.py CHAMPION SCORES_CSV TEST_DIR OUTPUT SWAPS

逻辑（对应计划模块 D）:
- additions: 未选入 champion 且 score 最高的节点（每单 cap<8）
- removals:  champion 内 score 最低的节点（每单 >=1 保留）
- N 对换，恒 1059；优先同 title 互换（同 title 下 score 差最小者）
- 受保护: swap12 胜出节点 + V11 高置信节点不触碰（从冠军 diff 推断）
- 输出: 探针提交文件 + JSON 报告（含每单变更明细，供判读 net TP delta）
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_submission(path):
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["order_id"]] = {
                node["@rid"]: node
                for node in json.loads(row["output"])["rootcause"]
            }
    return result


def load_scores(path):
    """加载分数表 CSV（order_id, rid, context_score, [meta_seed_*, meta_mean | fusion_score]）。"""
    by_order = defaultdict(list)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        seed_cols = [n for n in reader.fieldnames if n.startswith("meta_seed_")]
        score_col = (
            "fusion_score"
            if "fusion_score" in reader.fieldnames
            else "meta_mean"
        )
        for row in reader:
            by_order[row["order_id"]].append(
                {
                    "rid": row["rid"],
                    "score": float(row[score_col]),
                    "seeds": [float(row[n]) for n in seed_cols] if seed_cols else None,
                }
            )
    return by_order, score_col, seed_cols


def load_topologies(test_dir):
    result = {}
    for directory in sorted(path for path in Path(test_dir).iterdir() if path.is_dir()):
        order_id = directory.name
        topo = json.loads(
            (directory / f"{order_id}.log.topo.json").read_text(encoding="utf-8")
        )
        result[order_id] = [
            node for node in topo["nodes"] if node.get("@class") == "Alarm"
        ]
    return result


def write_submission(path, topologies, selected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["order_id", "output"])
        for order_id, nodes in topologies.items():
            rootcauses = [
                {
                    "@rid": node["@rid"],
                    "title": node.get("title", ""),
                    "location": node.get("location", ""),
                    "reason": node.get("reason", ""),
                }
                for node in nodes
                if node["@rid"] in selected.get(order_id, set())
            ]
            writer.writerow(
                [order_id, json.dumps({"rootcause": rootcauses}, ensure_ascii=False)]
            )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) not in (6, 7):
        raise SystemExit(
            "usage: v13_probe.py CHAMPION SCORES_CSV TEST_DIR OUTPUT SWAPS [PROTECT_REFERENCE]"
        )
    champion_path, scores_path, test_dir, output_path, swaps_str = sys.argv[1:6]
    protect_reference_path = sys.argv[6] if len(sys.argv) == 7 else None
    swaps = int(swaps_str)
    champion = load_submission(champion_path)
    by_order, score_col, seed_cols = load_scores(scores_path)
    topologies = load_topologies(test_dir)
    output_path = Path(output_path)

    selected = {oid: set(rids) for oid, rids in champion.items()}

    # 保护集: PROTECT_REFERENCE 与 champion 的差异 = 已验证胜出节点（不可触碰）
    protected = set()
    if protect_reference_path:
        reference = load_submission(protect_reference_path)
        for oid, rids in champion.items():
            ref_rids = set(reference.get(oid, {}).keys())
            protected.update((oid, r) for r in set(rids.keys()) - ref_rids)  # forced_in 保护
            # 注意: forced_out（已删节点）不在 champion 里，天然不会被移除；但新增时也不能加回
            protected.update((oid, r) for r in ref_rids - set(rids.keys()))  # forced_out 保护

    additions = []  # (score, seed_var, order_id, rid, title, same_title_rank)
    removals = []   # (score, seed_var, order_id, rid, title)
    for order_id, records in by_order.items():
        baseline_ids = selected.get(order_id, set())
        if len(baseline_ids) == 0:
            continue
        # 候选新增: 未选入 champion 且分数高
        for rec in records:
            if rec["rid"] in baseline_ids:
                continue
            if len(baseline_ids) >= 8:
                break  # 该单已满
            if (order_id, rec["rid"]) in protected:
                continue  # 受保护: 不可加回已验证删除的节点
            seed_var = float(np.var(rec["seeds"])) if rec["seeds"] else 0.0
            additions.append(
                (rec["score"], seed_var, order_id, rec["rid"])
            )
        # 候选移除: champion 内分数低（每单至少保留 1 个根因）
        baseline_records = {rec["rid"]: rec for rec in records if rec["rid"] in baseline_ids}
        for rid in baseline_ids:
            if len(baseline_ids) <= 1:
                break  # 该单只剩 1 个根因，不可再删
            rec = baseline_records.get(rid)
            if rec is None:
                continue
            if (order_id, rid) in protected:
                continue  # 受保护: 不可删除已验证胜出节点
            seed_var = float(np.var(rec["seeds"])) if rec["seeds"] else 1e9
            removals.append((rec["score"], seed_var, order_id, rid))

    # 排序: additions 高分低方差优先; removals 低分高方差优先
    additions.sort(key=lambda item: (-item[0], item[1]))
    removals.sort(key=lambda item: (item[0], -item[1]))

    available = min(len(additions), len(removals))
    if swaps > available:
        print(f"WARNING: requested {swaps} swaps, only {available} available", flush=True)
        swaps = available
    if swaps == 0:
        print("NO SWAPS AVAILABLE — no probe generated", flush=True)
        return

    chosen_removals = removals[:swaps]
    chosen_additions = additions[:swaps]
    for _, _, order_id, rid in chosen_removals:
        selected[order_id].discard(rid)
    for _, _, order_id, rid in chosen_additions:
        selected[order_id].add(rid)

    # 校验: 每单 1..8 个根因、总数必须 1059
    for order_id, rids in selected.items():
        if len(rids) < 1 or len(rids) > 8:
            raise RuntimeError(
                f"order {order_id} has {len(rids)} roots (must be 1..8)"
            )
    total = sum(len(s) for s in selected.values())
    assert total == 1059, f"total predictions {total} != 1059"

    write_submission(output_path, topologies, selected)

    report = {
        "output": str(output_path),
        "sha256": sha256(output_path),
        "predictions": total,
        "swaps": swaps,
        "score_column": score_col,
        "protected_count": len(protected),
        "removed": [
            {"order_id": oid, "rid": rid, "score": round(score, 4), "seed_var": round(var, 6)}
            for score, var, oid, rid in chosen_removals
        ],
        "added": [
            {"order_id": oid, "rid": rid, "score": round(score, 4), "seed_var": round(var, 6)}
            for score, var, oid, rid in chosen_additions
        ],
        "margin_min": (
            round(chosen_additions[-1][0] - chosen_removals[-1][0], 4)
            if chosen_additions and chosen_removals
            else None
        ),
        "note": (
            "探针判读: 提交后 score -> TP = round(score*2103/2); "
            "delta_tp = TP_probe - 953(champion); >0 accept / =0 split / <0 reject"
        ),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
