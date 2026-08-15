"""Build uploadable V25 code and dataset archives."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from v25_common import sha256, write_json


def zip_files(path, files, base):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in files:
            archive.write(source, source.relative_to(base))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    args.output.mkdir(parents=True, exist_ok=True)
    code = args.output / "v25_semantic_router_code.zip"
    dataset = args.output / "v25_semantic_router_dataset.zip"
    code_files = [
        path for path in root.iterdir()
        if path.is_file() and path.suffix in {".py", ".json", ".txt", ".md"}
    ]
    data_files = [
        args.data_root / name
        for name in (
            "v25_semantic_router.npz",
            "metadata.json",
            "semantic_records.json.gz",
            "validation_report.json",
        )
    ]
    zip_files(code, code_files, root)
    zip_files(dataset, data_files, args.data_root)
    manifest = {
        "code": {"path": str(code), "sha256": sha256(code), "bytes": code.stat().st_size},
        "dataset": {"path": str(dataset), "sha256": sha256(dataset), "bytes": dataset.stat().st_size},
        "model_weights_included": False,
        "required_external_model": "Qwen2.5-7B-Instruct BF16",
    }
    write_json(args.output / "package_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
