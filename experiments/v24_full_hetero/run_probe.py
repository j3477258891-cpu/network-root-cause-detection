"""Run V24 smoke plus the first honest five-fold seed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config_probe.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = config["seeds"][0]
    root = Path(__file__).parent
    run(
        [
            sys.executable, root / "cloud_smoke.py", "--data-root", args.data_root,
            "--config", args.config, "--output", args.output / "smoke_report.json",
            "--device", args.device,
        ]
    )
    for fold in range(config["folds"]):
        run(
            [
                sys.executable, root / "train_fold.py", "--data-root", args.data_root,
                "--output", args.output, "--config", args.config, "--fold", fold,
                "--seed", seed, "--device", args.device,
            ]
        )
    run(
        [
            sys.executable, root / "evaluate_probe.py", "--data-root", args.data_root,
            "--predictions", args.output, "--output", args.output / "probe_gate_report.json",
            "--config", args.config, "--seed", seed,
        ]
    )


if __name__ == "__main__":
    main()
