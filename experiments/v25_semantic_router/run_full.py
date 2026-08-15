"""Run remaining V25 seeds, final gate, and guarded submission generation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--locks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(__file__).parent
    adapter = args.output / "dapt" / "adapter"
    if not adapter.exists():
        raise FileNotFoundError("run the one-seed probe first; DAPT adapter is missing")
    probe = json.loads((args.output / "v25_probe_report.json").read_text(encoding="utf-8"))
    if probe.get("status") != "gate_passed":
        raise SystemExit("probe gate failed; refusing to train remaining seeds")
    for seed in config["seeds"]:
        for fold in range(config["folds"]):
            result = args.output / f"semantic_seed_{seed}_fold_{fold}.npz"
            if not result.exists():
                run([sys.executable, root / "train_semantic_fold.py", "--data-root", args.data_root, "--model", args.model, "--dapt-adapter", adapter, "--output", args.output, "--config", args.config, "--fold", fold, "--seed", seed, "--device", args.device])
        retrieval = args.output / f"retrieval_seed_{seed}.npz"
        if not retrieval.exists():
            run([sys.executable, root / "retrieval_expert.py", "--data-root", args.data_root, "--semantic-dir", args.output, "--output", args.output, "--config", args.config, "--seed", seed])
        run([sys.executable, root / "oracle_audit.py", "--data-root", args.data_root, "--semantic-dir", args.output, "--retrieval-dir", args.output, "--output", args.output / f"oracle_seed_{seed}.json", "--config", args.config, "--seed", seed])
        run([sys.executable, root / "train_router.py", "--data-root", args.data_root, "--semantic-dir", args.output, "--retrieval-dir", args.output, "--locks", args.locks, "--output", args.output, "--config", args.config, "--seed", seed])
    run([sys.executable, root / "evaluate_gate.py", "--data-root", args.data_root, "--semantic-dir", args.output, "--retrieval-dir", args.output, "--router-dir", args.output, "--output", args.output, "--config", args.config, "--mode", "final", "--seeds", *config["seeds"]])
    run([sys.executable, root / "generate_submission.py", "--data-root", args.data_root, "--results", args.output, "--output", args.submissions])


if __name__ == "__main__":
    main()
