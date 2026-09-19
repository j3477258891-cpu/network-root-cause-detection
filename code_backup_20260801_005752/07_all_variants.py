"""
07_all_variants.py - 生成所有可能的字段名组合
=============================================
根据中文描述推导可能的字段名：
  节点id → @rid / id / node_id / rid
  告警标题 → title
  告警位置 → location  
  告警原因 → reason

输出所有组合的 CSV，明天各试一次
"""

import json, csv
from pathlib import Path

SUBMIT_DIR = Path("D:/zgyidong/code/submit")
with open(SUBMIT_DIR / "predictions_v3.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# 需要试的字段名映射
variants = {
    "v6_original_rootcause": {  # 跟训练集完全一致
        "fmt": "rootcause_wrap",
        "fields": {"@rid": "@rid", "title": "title", "location": "location", "reason": "reason"}
    },
    "v7_bare_array": {  # 裸数组
        "fmt": "bare",
        "fields": {"@rid": "@rid", "title": "title", "location": "location", "reason": "reason"}
    },
    "v8_id_field": {  # @rid → id
        "fmt": "rootcause_wrap",
        "fields": {"id": "@rid", "title": "title", "location": "location", "reason": "reason"}
    },
    "v9_node_id_field": {  # @rid → node_id
        "fmt": "rootcause_wrap",
        "fields": {"node_id": "@rid", "title": "title", "location": "location", "reason": "reason"}
    },
    "v10_rid_field": {  # @rid → rid
        "fmt": "rootcause_wrap",
        "fields": {"rid": "@rid", "title": "title", "location": "location", "reason": "reason"}
    },
    "v11_chinese_keys": {  # 中文字段名
        "fmt": "rootcause_wrap",
        "fields": {"节点id": "@rid", "告警标题": "title", "告警位置": "location", "告警原因": "reason"}
    },
    "v12_remove_hash_prefix": {  # 去掉 #-1: 前缀
        "fmt": "rootcause_wrap",
        "fields": {"@rid": "@rid", "title": "title", "location": "location", "reason": "reason"},
        "transform_rid": lambda r: r[4:] if r.startswith("#-1:") else r
    },
}

for name, config in variants.items():
    fmt = config["fmt"]
    fields = config["fields"]
    transform_rid = config.get("transform_rid", lambda r: r)

    csv_path = SUBMIT_DIR / f"result_record_{name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "output"])
        for wo_id, v in preds.items():
            rc_list = []
            for rc in v["rootcause"]:
                new_rc = {}
                for new_key, old_key in fields.items():
                    val = rc.get(old_key, "")
                    if old_key == "@rid" and new_key in ("@rid", "id", "rid", "node_id", "节点id"):
                        val = transform_rid(val)
                    new_rc[new_key] = val
                rc_list.append(new_rc)
            
            if fmt == "rootcause_wrap":
                output = {"rootcause": rc_list}
            else:
                output = rc_list
            writer.writerow([wo_id, json.dumps(output, ensure_ascii=False)])

    # 验证
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader)
        parsed = json.loads(first["output"])
    
    print(f"✅ {name}.csv")
    if isinstance(parsed, dict):
        print(f"   格式: {list(parsed.keys())}")
        sample_rc = parsed.get("rootcause", [parsed])[0] if "rootcause" in parsed else parsed[0] if isinstance(parsed, list) else list(parsed.values())[0]
    elif isinstance(parsed, list):
        print(f"   格式: [array]")
        sample_rc = parsed[0]
    else:
        sample_rc = {}
    print(f"   字段: {list(sample_rc.keys())}")
    print(f"   样本: {json.dumps(sample_rc, ensure_ascii=False)[:120]}")
    print()

print("完成！7 个变体就绪，明天逐个提交。")
