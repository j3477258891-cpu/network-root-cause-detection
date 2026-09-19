"""Verified original bundle -> isolated, newly authorized six-hour cloud run.

Never submits competition files; refuses to overwrite an existing run directory.
"""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import subprocess
import sys
import zipfile

ARCHIVE_SHA = 'd3ca1a256f12232398b5ccb217930ef9c82ca184cae0c520da709e0c7122e564'
ARCHIVE_SIZE = 16768986


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    work = Path('/root/work').resolve(strict=True)
    archive = work / 'v154_compact_training.zip'
    if archive.stat().st_size != ARCHIVE_SIZE or sha(archive) != ARCHIVE_SHA:
        raise RuntimeError('Incomplete or changed training archive; no extraction or training')
    dest = work / 'v154_run_20260909'
    if dest.exists():
        raise RuntimeError('Run directory already exists; inspect its logs before any repeat')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError('Duplicate ZIP members')
        for member in z.infolist():
            p = PurePosixPath(member.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in member.filename:
                raise RuntimeError('Unsafe ZIP member')
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError('ZIP symlink rejected')
        if z.testzip() is not None:
            raise RuntimeError('ZIP CRC failure')
        dest.mkdir()
        z.extractall(dest)
    manifests = list(dest.rglob('manifest.json'))
    if len(manifests) != 1:
        raise RuntimeError('Expected exactly one training manifest')
    root = manifests[0].parent
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['files'].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or sha(target) != digest:
            raise RuntimeError('Original manifest failed: ' + name)
    old_manifest = manifest_path.read_bytes()
    old_window = (root / 'training_window.json').read_bytes()
    (root / 'manifest.original.json').write_bytes(old_manifest)
    (root / 'training_window.original.json').write_bytes(old_window)
    now = datetime.now(timezone.utc)
    window = dict(manifest['window'])
    window.update(
        started_at=now.isoformat(),
        deadline_at=(now + timedelta(hours=6)).isoformat(),
        authorization='2026-09-09 user: continue unfinished task; prior explicit cloud-training authorization. New bounded run disclosed in commentary.',
        previous_window=json.loads(old_window),
        competition_upload_authorized=False,
        competition_budget_remaining_at_start=6,
    )
    (root / 'training_window.json').write_text(json.dumps(window, indent=2))
    manifest['window'] = window
    manifest['files']['training_window.json'] = sha(root / 'training_window.json')
    manifest['resume_provenance'] = {
        'original_archive_sha256': ARCHIVE_SHA,
        'original_manifest_sha256': hashlib.sha256(old_manifest).hexdigest(),
        'only_original_file_changed': 'training_window.json',
        'bootstrap_sha256': sha(Path(__file__).resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with (root / 'launcher.log').open('xb') as log:
        process = subprocess.Popen(
            [sys.executable, '-u', str(root / 'launch.py')],
            cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    started = {'pid': process.pid, 'directory': str(root), 'window': window,
               'status': 'launcher_spawned_not_yet_training_verified', 'competition_submitted': False}
    (dest / 'bootstrap_state.json').write_text(json.dumps(started, indent=2))
    print(json.dumps(started, indent=2), flush=True)


if __name__ == '__main__':
    main()
