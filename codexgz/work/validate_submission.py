import csv
import json
import sys
from collections import Counter
from pathlib import Path


submission = Path(sys.argv[1])
test_dir = Path(sys.argv[2])
test_ids = sorted(path.name for path in test_dir.iterdir() if path.is_dir())

with submission.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames == ["order_id", "output"], reader.fieldnames
    rows = list(reader)

row_ids = [row["order_id"] for row in rows]
assert len(rows) == len(test_ids), (len(rows), len(test_ids))
assert len(set(row_ids)) == len(row_ids), "duplicate order_id"
assert sorted(row_ids) == test_ids, "submission/test order IDs differ"

counts = Counter()
predicted_nodes = 0
for row in rows:
    order_id = row["order_id"]
    payload = json.loads(row["output"])
    assert set(payload) == {"rootcause"}, (order_id, payload.keys())
    rootcauses = payload["rootcause"]
    assert isinstance(rootcauses, list) and rootcauses, order_id

    topo_path = test_dir / order_id / f"{order_id}.log.topo.json"
    with topo_path.open("r", encoding="utf-8") as handle:
        topo = json.load(handle)
    nodes = {node["@rid"]: node for node in topo["nodes"]}

    seen = set()
    for prediction in rootcauses:
        assert list(prediction) == ["@rid", "title", "location", "reason"], (
            order_id,
            list(prediction),
        )
        rid = prediction["@rid"]
        assert rid not in seen, (order_id, rid, "duplicate prediction")
        seen.add(rid)
        assert rid in nodes, (order_id, rid, "missing from topology")
        source = nodes[rid]
        for field in ("title", "location", "reason"):
            assert prediction[field] == source.get(field, ""), (order_id, rid, field)

    counts[len(rootcauses)] += 1
    predicted_nodes += len(rootcauses)

print(f"rows={len(rows)}")
print(f"predicted_nodes={predicted_nodes}")
print(f"rootcause_count_distribution={dict(sorted(counts.items()))}")
print("validation=PASS")
