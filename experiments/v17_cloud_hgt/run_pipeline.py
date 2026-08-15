"""Run all fixed V17 folds/seeds and aggregate the gated outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).parent
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    predictions = args.output / "fold_predictions"
    candidates = args.output / "candidates"
    predictions.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)
    for seed in config["seeds"]:
        for fold in range(config["folds"]):
            expected = predictions / f"seed_{seed}_fold_{fold}.npz"
            if args.resume and expected.exists():
                print(f"skip {expected}", flush=True)
                continue
            command = [
                sys.executable,
                str(root / "train_fold.py"),
                "--data-root",
                str(args.data_root),
                "--output",
                str(predictions),
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "aggregate.py"),
            "--data-root",
            str(args.data_root),
            "--predictions",
            str(predictions),
            "--output",
            str(candidates),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
