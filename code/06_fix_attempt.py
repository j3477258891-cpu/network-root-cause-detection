"""
06_fix_attempt.py - F1=0 急救修复
=============================
可能的修复方案：
A. 去掉 @rid 里的 "#-1:" 前缀（只留 UUID）
B. 用 "id" 替代 "@rid"
C. output 用 {"rootcause": [...]} 包裹

运行：python 06_fix_attempt.py
"""

import json, csv
from pathlib import Path

SUBMIT_DIR = Path("D:/zgyidong/code/submit")

with open(SUBMIT_DIR / "predictions_v3.json", "r", encoding="utf-8") as f:
    preds = json.load(f)


def make_csv(name, transform):
    """生成指定格式的 CSV"""
    out_path = SUBMIT_DIR / name
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for wo_id, v in preds.items():
            rc_list = transform(v["rootcause"])
            writer.writerow([wo_id, json.dumps(rc_list, ensure_ascii=False)])
    return out_path


# ====== 方案 A: 去掉 # 前缀 ======
def strip_hash(rc):
    new = []
    for r in rc:
        r2 = dict(r)
        rid = r2.get("@rid", "")
        if rid.startswith("#-1:"):
            r2["@rid"] = rid[4:]  # 去掉 # 前缀（保留 :1:UUID）
        elif rid.startswith("#"):
            r2["@rid"] = rid[1:]  # 去掉 #
        new.append(r2)
    return new

# ====== 方案 B: 字段名 @rid -> id ======
def rename_id(rc):
    return [{**{("id" if k=="@rid" else k): v for k,v in r.items()}} for r in rc]

# ====== 方案 C: 完整改字段名（更接近中文描述"节点id"）======
def chinese_names(rc):
    return [{
        "节点id": r.get("@rid", ""),
        "告警标题": r.get("title", ""),
        "告警位置": r.get("location", ""),
        "告警原因": r.get("reason", ""),
    } for r in rc]

# ====== 方案 D: 双字段都加（兼容） ======
def both_ids(rc):
    new = []
    for r in rc:
        r2 = dict(r)
        r2["id"] = r2.get("@rid", "")
        new.append(r2)
    return new


variants = [
    ("result_record_v2.csv", strip_hash, "去掉 # 前缀"),
    ("result_record_v3.csv", rename_id, "@rid → id"),
    ("result_record_v4.csv", chinese_names, "中文字段名"),
    ("result_record_v5.csv", both_ids, "@rid + id 双字段"),
]

for fname, fn, desc in variants:
    p = make_csv(fname, fn)
    print(f"✅ {fname} ({desc}): {p}")

print("\n生成完成，请挑一个试一次（还有 1 次提交机会）")