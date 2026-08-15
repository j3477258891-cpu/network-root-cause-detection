"""Retrain and persist the three graph+time OOF seed predictions."""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


SEEDS = (20260803, 20260817, 20260831)
N_FOLDS = 5


def main():
    root = Path(__file__).resolve().parent
    payload = np.load(root / "v19_graph_time_payload.npz")
    features = payload["features"]
    labels = payload["labels"]
    row_folds = payload["row_folds"]
    results = []
    per_seed = []
    for seed in SEEDS:
        scores = np.zeros(len(labels), dtype=np.float64)
        for heldout in range(N_FOLDS):
            train_rows = row_folds != heldout
            validation_rows = row_folds == heldout
            model = ExtraTreesClassifier(
                n_estimators=350,
                max_depth=16,
                min_samples_leaf=3,
                max_features=0.75,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed + heldout,
            )
            model.fit(features[train_rows], labels[train_rows])
            scores[validation_rows] = model.predict_proba(features[validation_rows])[:, 1]
            print(f"seed={seed} fold={heldout} complete", flush=True)
        path = root / f"v19_graph_time_oof_seed_{seed}.npy"
        np.save(path, scores)
        per_seed.append(scores)
        results.append({"seed": seed, "path": path.name})
    mean_scores = np.mean(per_seed, axis=0)
    np.save(root / "v19_graph_time_oof_mean.npy", mean_scores)
    report = {
        "seeds": results,
        "mean_path": "v19_graph_time_oof_mean.npy",
        "mean_min": float(mean_scores.min()),
        "mean_max": float(mean_scores.max()),
    }
    (root / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
