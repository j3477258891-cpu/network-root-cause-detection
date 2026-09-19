"""Public CLI for the V149 online-calibrated addition campaign.

Spreadsheet authoring is delegated to the bundled artifact-tool JavaScript
builder so generated CSVs are created, re-imported, inspected, and rendered
through the workspace spreadsheet runtime.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\zgyidong")
CAMPAIGN_DIR = ROOT / "experiments" / "v149_online_calibrated_addition_campaign"
BUILDER = CAMPAIGN_DIR / "v149_artifact_builder.mjs"
NODE = Path(
    r"C:\Users\86158\.cache\codex-runtimes\codex-primary-runtime"
    r"\dependencies\node\bin\node.exe"
)


def main() -> int:
    if not NODE.is_file():
        raise SystemExit(f"Bundled Node.js runtime not found: {NODE}")
    if not BUILDER.is_file():
        raise SystemExit(f"V149 artifact builder not found: {BUILDER}")
    completed = subprocess.run(
        [str(NODE), str(BUILDER), *sys.argv[1:]],
        cwd=str(CAMPAIGN_DIR),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
