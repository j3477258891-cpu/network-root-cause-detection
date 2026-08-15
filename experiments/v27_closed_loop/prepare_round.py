"""Materialize planned v27 probes and the three requested handoff submissions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from closed_loop import (
    DEFAULT_WORKDIR,
    apply_actions,
    catalog_actions,
    file_sha256,
    load_state,
    load_submission,
    load_topologies,
    register_batch,
    validate_actions,
    validate_selection,
    write_json,
    write_submission,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--catalog")
    parser.add_argument("--batch-plan")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    catalog_path = Path(args.catalog) if args.catalog else workdir / "action_catalog.json"
    plan_path = Path(args.batch_plan) if args.batch_plan else workdir / "batch_plan.json"
    state = load_state(workdir)
    actions = catalog_actions(catalog_path)
    lookup = {item["action_id"]: item for item in actions}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))["batches"]
    existing = {item["batch_id"] for item in state["batches"]}
    generated = []
    for spec in plan:
        if spec["status"] == "empty" or spec["batch_id"] in existing:
            continue
        batch_actions = [lookup[action_id] for action_id in spec["action_ids"]]
        generated.append(
            register_batch(workdir, spec["batch_id"], "planned_probe", batch_actions)
        )

    state = load_state(workdir)
    champion_path = Path(state["champion"]["path"])
    order_ids, champion = load_submission(champion_path)
    topologies = load_topologies(Path(state["test_dir"]))

    # Initially the frozen champion is also the highest verified checkpoint.
    checkpoint = workdir / "submissions" / "highest_verified_checkpoint.csv"
    shutil.copyfile(champion_path, checkpoint)

    # This file is deliberately marked unverified. It is an optimistic ceiling
    # artifact and must not replace the checkpoint without leaderboard evidence.
    validate_actions(actions, champion, topologies)
    enhanced = apply_actions(champion, actions)
    enhanced_validation = validate_selection(order_ids, enhanced, topologies)
    enhanced_path = workdir / "submissions" / "target_enhanced_UNVERIFIED.csv"
    write_submission(enhanced_path, order_ids, enhanced, topologies)

    handoff = {
        "champion": {
            "path": str(champion_path),
            "sha256": file_sha256(champion_path),
            "verified_score": state["champion"]["score"],
        },
        "highest_verified_checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "verified_score": state["champion"]["score"],
        },
        "target_enhanced": {
            "path": str(enhanced_path),
            "sha256": file_sha256(enhanced_path),
            "verified": False,
            "validation": enhanced_validation,
            "warning": "Do not submit before its component batches are scored and accepted.",
        },
        "generated_probe_batches": [item["batch_id"] for item in generated],
    }
    write_json(workdir / "reports" / "handoff.json", handoff)
    print(json.dumps(handoff, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
