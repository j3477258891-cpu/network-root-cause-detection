"""Create deterministic V24 code and dataset ZIP packages with SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


CODE_FILES = (
    "README.md",
    "config_probe.json",
    "v24_data.py",
    "v24_model.py",
    "train_fold.py",
    "evaluate_probe.py",
    "cloud_smoke.py",
    "run_probe.py",
)
DATA_FILES = (
    "rootcause_v24_full_hetero.npz",
    "metadata.json",
    "manifest.json",
    "validation_report.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip(path: Path, sources: list[tuple[Path, str]]):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in sources:
            info = zipfile.ZipInfo(name)
            info.date_time = (2026, 8, 4, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    code_sources = [(args.root / name, name) for name in CODE_FILES]
    data_root = args.root / "cloud_dataset"
    data_sources = [(data_root / name, name) for name in DATA_FILES]
    missing = [str(path) for path, _ in code_sources + data_sources if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    code_zip = args.output / "v24_full_hetero_probe_code.zip"
    data_zip = args.output / "v24_full_hetero_dataset.zip"
    write_zip(code_zip, code_sources)
    write_zip(data_zip, data_sources)
    report = {
        "version": "v24-full-hetero-probe-1",
        "packages": {
            code_zip.name: {"bytes": code_zip.stat().st_size, "sha256": sha256(code_zip)},
            data_zip.name: {"bytes": data_zip.stat().st_size, "sha256": sha256(data_zip)},
        },
        "code_files": list(CODE_FILES),
        "data_files": list(DATA_FILES),
    }
    report_path = args.output / "package_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
