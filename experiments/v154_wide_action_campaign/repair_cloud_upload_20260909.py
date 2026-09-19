"""Repair one verified interrupted upload, preserving the original partial file."""
from pathlib import Path
import hashlib
import sys
import subprocess

SIZE = 16768986
CUT = 15728640
FULL_SHA = 'd3ca1a256f12232398b5ccb217930ef9c82ca184cae0c520da709e0c7122e564'
PREFIX_SHA = '72db85b46a0854dd1ef36ef288e1e9957f9a4c2d5ede70b5fa25e692a2b2a51d'


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--prepare-local-tail':
        root = Path(__file__).resolve().parent
        data = (root / 'v154_compact_training.zip').read_bytes()
        assert len(data) == SIZE and hashlib.sha256(data).hexdigest() == FULL_SHA
        assert hashlib.sha256(data[:CUT]).hexdigest() == PREFIX_SHA
        (root / 'v154_upload_tail.bin').write_bytes(data[CUT:])
        print('Tail bytes:', len(data) - CUT)
        return
    root = Path('/root/work').resolve(strict=True)
    archive = root / 'v154_compact_training.zip'
    partial = archive.read_bytes()
    if len(partial) == SIZE and hashlib.sha256(partial).hexdigest() == FULL_SHA:
        print('Archive already complete; no repair needed')
    else:
        assert len(partial) == CUT and hashlib.sha256(partial).hexdigest() == PREFIX_SHA
        tail = (root / 'v154_upload_tail.bin').read_bytes()
        full = partial + tail
        assert len(full) == SIZE and hashlib.sha256(full).hexdigest() == FULL_SHA
        backup = root / 'v154_compact_training.partial_15MiB.zip'
        rebuilt = root / 'v154_compact_training.verified.tmp'
        assert not backup.exists() and not rebuilt.exists()
        with rebuilt.open('xb') as f:
            f.write(full)
        archive.rename(backup)
        rebuilt.rename(archive)
        print('Full archive verified; original partial preserved', flush=True)
    subprocess.run([sys.executable, str(root / 'resume_cloud_20260909.py')], check=True)


if __name__ == '__main__':
    main()
