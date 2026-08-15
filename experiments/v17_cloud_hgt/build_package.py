"""Create deterministic V17 upload archives and their checksum manifest."""

from __future__ import annotations

import hashlib
import json
import base64
import zipfile
from pathlib import Path


ROOT = Path(__file__).parent
DIST = ROOT / "dist_v4"
DATASET = ROOT / "cloud_dataset"
CODE_FILES = (
    "config.json",
    "README.md",
    "v17_data.py",
    "v17_model.py",
    "train_fold.py",
    "aggregate.py",
    "run_pipeline.py",
    "run_cloud.sh",
    "cloud_smoke.py",
    "validate_package.py",
    "materialize_dataset.py",
    "analyze_rank_only.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip(path, files, prefix=""):
    timestamp = (2026, 8, 3, 0, 0, 0)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in sorted(files, key=lambda item: item[1]):
            info = zipfile.ZipInfo((prefix + name).replace("\\", "/"), timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())


def main():
    DIST.mkdir(parents=True, exist_ok=True)
    dataset_zip = DIST / "rootcause-v17-original-v1.zip"
    registered_dataset_zip = DIST / "rootcause-v17-v1-text.zip"
    code_zip = DIST / "v17-cloud-hgt-code-v4.zip"
    write_zip(
        dataset_zip,
        [(path, path.name) for path in DATASET.iterdir() if path.is_file()],
        prefix="rootcause-v17-v1/",
    )
    graph_path = DATASET / "rootcause_v17_graphs.npz"
    graph_text = DIST / "rootcause_v17_graphs.npz.b64.txt"
    graph_text.write_text(
        base64.b64encode(graph_path.read_bytes()).decode("ascii"), encoding="ascii"
    )
    write_zip(
        registered_dataset_zip,
        [
            (DATASET / "manifest.json", "manifest.json"),
            (DATASET / "metadata.json", "metadata.json"),
            (graph_text, graph_text.name),
        ],
    )
    write_zip(code_zip, [(ROOT / name, name) for name in CODE_FILES], prefix="v17_cloud_hgt/")
    manifest = {
        "version": "v17-upload-1",
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (dataset_zip, registered_dataset_zip, code_zip)
        },
        "champion_sha256": "6b59ad62b5d1c3d5153a91309f514532228f72a5498a591bf56c86913c703e93",
    }
    (DIST / "upload_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
