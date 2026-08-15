"""Build the minimal V19 cloud retraining payload."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\zgyidong")
sys.path.insert(0, str(ROOT / "experiments"))
import v19_nested_count_audit as audit  # noqa: E402


def main():
    orders = audit.load_orders(audit.TRAIN_DIR, True)
    slices = audit.build_slices(orders)
    labels = np.asarray(
        [int(node["@rid"] in order["roots"]) for order in orders for node in order["alarms"]],
        dtype=np.int8,
    )
    folds, group_count, fold_sizes = audit.grouped_folds(orders)
    row_folds = np.concatenate(
        [np.full(len(order["alarms"]), folds[index], dtype=np.int8) for index, order in enumerate(orders)]
    )
    features = np.asarray(
        np.load(ROOT / "experiments" / "v16" / "v16_train_features.npy", mmap_mode="r")[:, :97],
        dtype=np.float32,
    )
    v11 = 0.25 * np.load(ROOT / "codexgz" / "v11" / "v11_oof_context.npy") + 0.75 * np.load(
        ROOT / "codexgz" / "v11" / "v11_oof_meta.npy"
    )
    starts = np.asarray([item.start for item in slices], dtype=np.int32)
    stops = np.asarray([item.stop for item in slices], dtype=np.int32)
    output = ROOT / "experiments" / "v19_cloud" / "v19_graph_time_payload.npz"
    np.savez_compressed(
        output,
        features=features,
        labels=labels,
        folds=folds,
        row_folds=row_folds,
        starts=starts,
        stops=stops,
        v11_scores=v11,
    )
    metadata = {
        "rows": len(labels),
        "orders": len(orders),
        "features": int(features.shape[1]),
        "positive_labels": int(labels.sum()),
        "group_count": group_count,
        "fold_sizes": fold_sizes,
    }
    (output.parent / "payload_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
