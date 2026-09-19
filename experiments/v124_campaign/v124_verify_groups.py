import json
from collections import Counter

cat = json.load(open(r"D:/zgyidong/experiments/v120_swap_campaign/candidate_catalog.json", encoding="utf-8"))
cands = cat["candidates"]
by_id = {c["candidate_id"]: c for c in cands}
cb = json.load(open(r"D:/zgyidong/experiments/research_equation_audit/candidate_bounds.json", encoding="utf-8"))
bounds_by_id = {c["candidate_id"]: c.get("equation", {}) for c in cb["candidates"]}

groups = {
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
all_ids = [i for g in groups.values() for i in g]
print("total:", len(all_ids))
print("missing in v120 catalog:", [i for i in all_ids if i not in by_id])
print("missing in bounds:", [i for i in all_ids if i not in bounds_by_id])

orders = [(gid, cid, by_id[cid]["order_id"]) for gid, g in groups.items() for cid in g if cid in by_id]
oc = Counter(o[2] for o in orders)
dup = {o: n for o, n in oc.items() if n > 1}
print("order dups:", dup if dup else "NONE ok")
print("unique orders:", len(oc))

for gid, g in groups.items():
    row = []
    for cid in g:
        c = by_id.get(cid)
        if not c:
            row.append(cid + ":MISSING")
            continue
        eq = bounds_by_id.get(cid, {})
        row.append("%s(%s,n=%.2f,c=%.2f,eq=[%s,%s])" % (
            cid, c["order_id"][:8], c.get("nominal_p_net_gain", 0),
            c.get("conservative_p_net_gain", 0), eq.get("min"), eq.get("max")))
    print("group %02d: %s" % (gid, " | ".join(row)))

probe13 = ("8df78ab7-ca49-44f4-9fc2-1427cf850964", "920eca60a16b")
print("\nprobe_13 order match:")
for c in cands:
    if c["order_id"] == probe13[0]:
        mark = "  <== probe13?" if c["remove_rid"].endswith(probe13[1]) else ""
        print("  %s: rem=%s add=%s n=%.3f%s" % (c["candidate_id"], c["remove_rid"][-12:], c["add_rid"][-12:], c.get("nominal_p_net_gain", 0), mark))
probe14 = ("aa321b94-f3d9-45cf-b1ee-f3a1fe38da68", "4e271b8b1117")
print("\nprobe_14 order match:")
for c in cands:
    if c["order_id"] == probe14[0]:
        mark = "  <== probe14?" if c["remove_rid"].endswith(probe14[1]) else ""
        print("  %s: rem=%s add=%s n=%.3f%s" % (c["candidate_id"], c["remove_rid"][-12:], c["add_rid"][-12:], c.get("nominal_p_net_gain", 0), mark))

# equation max>=1 筛选统计
eq_ok = [i for i in all_ids if bounds_by_id.get(i, {}).get("max", -9) >= 1]
print("\neq max>=1 的候选:", len(eq_ok), "| eq max<1:", [i for i in all_ids if i not in eq_ok])
# 被 probe13/14 占用的订单是否在分组里
used_orders = {probe13[0], probe14[0]}
group_orders = set(o[2] for o in orders)
print("分组与 probe13/14 订单冲突:", group_orders & used_orders if group_orders & used_orders else "NONE ok")
