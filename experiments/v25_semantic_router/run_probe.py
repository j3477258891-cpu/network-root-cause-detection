"""Run DAPT, one semantic seed, oracle audit, router, and the probe gate."""

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
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-dapt", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = config["seeds"][0]
    root = Path(__file__).parent
    args.output.mkdir(parents=True, exist_ok=True)
    dapt = args.output / "dapt"
    run([sys.executable, root / "cloud_smoke.py", "--data-root", args.data_root, "--model", args.model, "--output", args.output / "cloud_smoke_report.json", "--config", args.config, "--device", args.device])
    if not args.skip_dapt:
        run([sys.executable, root / "domain_adapt.py", "--data-root", args.data_root, "--model", args.model, "--output", dapt, "--config", args.config, "--device", args.device])
    adapter = dapt / "adapter"
    for fold in range(config["folds"]):
        run([sys.executable, root / "train_semantic_fold.py", "--data-root", args.data_root, "--model", args.model, "--dapt-adapter", adapter, "--output", args.output, "--config", args.config, "--fold", fold, "--seed", seed, "--device", args.device])
    run([sys.executable, root / "retrieval_expert.py", "--data-root", args.data_root, "--semantic-dir", args.output, "--output", args.output, "--config", args.config, "--seed", seed])
    run([sys.executable, root / "oracle_audit.py", "--data-root", args.data_root, "--semantic-dir", args.output, "--retrieval-dir", args.output, "--output", args.output / "oracle_report.json", "--config", args.config, "--seed", seed])
    run([sys.executable, root / "train_router.py", "--data-root", args.data_root, "--semantic-dir", args.output, "--retrieval-dir", args.output, "--locks", args.locks, "--output", args.output, "--config", args.config, "--seed", seed])
    run([sys.executable, root / "evaluate_gate.py", "--data-root", args.data_root, "--semantic-dir", args.output, "--retrieval-dir", args.output, "--router-dir", args.output, "--output", args.output, "--config", args.config, "--mode", "probe", "--seeds", seed])


if __name__ == "__main__":
    main()
