"""Validate the compact cloud dataset before upload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.data_root / "metadata.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = args.data_root / name
        assert path.stat().st_size == expected["bytes"]
        assert sha256(path) == expected["sha256"]
    arrays = np.load(args.data_root / "rootcause_v17_graphs.npz")
    assert len(arrays["train_order_ptr"]) == 1635
    assert len(arrays["test_order_ptr"]) == 547
    assert arrays["train_x"].shape == (9334, 252)
    assert arrays["test_x"].shape[1] == 252
    assert int(arrays["train_labels"].sum()) == 3041
    assert int(arrays["test_champion_mask"].sum()) == 1059
    assert set(np.unique(arrays["train_folds"])) == {0, 1, 2, 3, 4}
    assert len(metadata["test"]) == 546
    assert metadata["champion_sha256"] == "6b59ad62b5d1c3d5153a91309f514532228f72a5498a591bf56c86913c703e93"
    report = {
        "valid": True,
        "train_orders": 1634,
        "test_orders": 546,
        "train_nodes": int(arrays["train_x"].shape[0]),
        "test_nodes": int(arrays["test_x"].shape[0]),
        "train_edges": int(len(arrays["train_edge_src"])),
        "test_edges": int(len(arrays["test_edge_src"])),
        "predictions": int(arrays["test_champion_mask"].sum()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
