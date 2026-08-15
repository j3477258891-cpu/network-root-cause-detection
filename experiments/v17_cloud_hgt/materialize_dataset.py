"""Restore the binary graph tensor from the platform-compatible text package."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_data_root(data_root: Path) -> Path:
    data_root = Path(data_root).resolve()
    binary = data_root / "rootcause_v17_graphs.npz"
    if binary.exists():
        return data_root

    encoded = data_root / "rootcause_v17_graphs.npz.b64.txt"
    manifest_path = data_root / "manifest.json"
    metadata_path = data_root / "metadata.json"
    if not (encoded.exists() and manifest_path.exists() and metadata_path.exists()):
        raise FileNotFoundError(
            "data root must contain the NPZ or its Base64 text package: " f"{data_root}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["files"]["rootcause_v17_graphs.npz"]
    target = Path(tempfile.gettempdir()) / "rootcause-v17-v1-materialized"
    target.mkdir(parents=True, exist_ok=True)
    output = target / "rootcause_v17_graphs.npz"
    if not (
        output.exists()
        and output.stat().st_size == expected["bytes"]
        and sha256(output) == expected["sha256"]
    ):
        output.write_bytes(base64.b64decode(encoded.read_text(encoding="ascii")))
    if output.stat().st_size != expected["bytes"] or sha256(output) != expected["sha256"]:
        raise ValueError("materialized graph tensor failed SHA256 validation")
    for source in (manifest_path, metadata_path):
        destination = target / source.name
        destination.write_bytes(source.read_bytes())
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    print(materialize_data_root(args.data_root))
