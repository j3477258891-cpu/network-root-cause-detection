"""Report the single next action for the two-phase V58/V59 campaign."""

from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from pathlib import Path

from combine_v58_v59_checkpoints import main as rebuild_combined_checkpoint


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
CAMPAIGNS = (
    ("v58", EXP / "v58_coded_campaign", "record_v58_result.py"),
    ("v59", EXP / "v59_extended_coded_campaign", "record_v59_result.py"),
)
COMBINED = EXP / "v60_combined_checkpoint/highest_verified_combined.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_unscored(report, state):
    return next((probe_id for probe_id in report["probe_order"]
                 if probe_id not in state.get("results", {})), None)


def first_unscored_adaptive(state):
    return next((probe for probe_id, probe in state.get("adaptive_checkpoints", {}).items()
                 if probe_id not in state.get("adaptive_results", {})), None)


def main():
    # Keep the recommendation authoritative even if the caller recorded a
    # score but forgot to run the standalone combine command.
    with redirect_stdout(io.StringIO()):
        rebuild_combined_checkpoint()
    combined = read_json(COMBINED)
    if combined["target_reached"]:
        output = {
            "status": "target_reached", "score": combined["score"],
            "submission": combined["path"], "sha256": combined["sha256"],
            "message": "Submit/retain the combined checkpoint; do not spend more probes.",
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    for name, directory, recorder in CAMPAIGNS:
        report = read_json(directory / "report.json")
        state = read_json(directory / "state.json")
        phase_complete = bool(state.get("phase_complete"))
        adaptive_probe = first_unscored_adaptive(state)
        if adaptive_probe is not None and not state.get("target_reached"):
            output = {
                "status": "submit_adaptive_checkpoint", "phase": name,
                "scored": len(state.get("results", {})),
                "probe_id": adaptive_probe["probe_id"],
                "submission": adaptive_probe["path"],
                "sha256": adaptive_probe["sha256"],
                "predictions": adaptive_probe["predictions"],
                "record_command": f"python experiments\\{recorder} --probe-id {adaptive_probe['probe_id']} --score <score>",
                "message": "This posterior checkpoint is only a candidate; record its public score before accepting it.",
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return
        probe_id = first_unscored(report, state)
        if phase_complete or probe_id is None:
            continue
        probe = report["probes"][probe_id]
        output = {
            "status": "submit_probe", "phase": name,
            "scored": len(state.get("results", {})), "total": len(report["probe_order"]),
            "probe_id": probe_id, "submission": probe["path"],
            "sha256": probe["sha256"], "predictions": probe["predictions"],
            "record_command": f"python experiments\\{recorder} --probe-id {probe_id} --score <score>",
            "combine_command": "python experiments\\combine_v58_v59_checkpoints.py",
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    output = {
        "status": "candidate_pool_exhausted", "score": combined["score"],
        "submission": combined["path"],
        "message": "All coded phases are complete but the verified target is not reached.",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
