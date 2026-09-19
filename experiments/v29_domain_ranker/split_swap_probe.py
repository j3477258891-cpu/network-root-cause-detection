"""Emit disjoint second-stage probes from the verified V29 swap manifest."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from v29_actions import CHAMPION, OUTPUT, emit_probe, load_champion


def main() -> None:
    manifest_path = OUTPUT / "manifests/v29_swap_top8.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actions = manifest["actions"]
    if len(actions) != 8:
        raise ValueError(("expected eight source actions", len(actions)))
    order_ids, champion_roots = load_champion(CHAMPION)
    with gzip.open(
        Path(r"D:\zgyidong\experiments\v25_semantic_router\cloud_dataset\semantic_records.json.gz"),
        "rt",
        encoding="utf-8",
    ) as handle:
        records = json.load(handle)
    records_by_order = {record["order_id"]: record for record in records["test"]}
    for name, subset in (("v29_swap_top5", actions[:5]), ("v29_swap_tail3", actions[5:])):
        emit_probe(name, "swap", subset, order_ids, champion_roots, records_by_order, OUTPUT)


if __name__ == "__main__":
    main()
