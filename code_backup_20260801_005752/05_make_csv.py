"""
05_make_csv.py - 转换为官方提交格式
=====================================
官方要求：
- CSV 文件名：result_record.csv
- 编码：utf-8
- 字段：order_id (工单ID), output (根因列表的JSON字符串)
- 每天可提交 2 次

运行：python 05_make_csv.py
"""

import json, csv
from pathlib import Path

SUBMIT_DIR = Path("D:/zgyidong/code/submit")

with open(SUBMIT_DIR / "predictions_v3.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

csv_path = SUBMIT_DIR / "result_record.csv"
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["order_id", "output"])
    for wo_id, v in preds.items():
        # output 列：JSON 字符串（每行一个工单的根因列表）
        output_json = json.dumps(v["rootcause"], ensure_ascii=False)
        writer.writerow([wo_id, output_json])

# 验证
with open(csv_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
print(f"✅ result_record.csv 已生成")
print(f"   总行数: {len(lines)} (含表头)")
print(f"   前3行预览:")
for line in lines[:3]:
    print(f"   {line.strip()[:200]}")

# 校验
print(f"\n=== 校验 ===")
import csv as csvmod
with open(csv_path, "r", encoding="utf-8", newline="") as f:
    reader = csvmod.DictReader(f)
    rows = list(reader)
print(f"工单数: {len(rows)}")
print(f"字段: {list(rows[0].keys())}")
# 解析第一条 check
parsed = json.loads(rows[0]["output"])
print(f"首个工单根因数: {len(parsed)}")
print(f"首个根因字段: {list(parsed[0].keys())}")
