"""One guarded repair for platform PIP_USER leaking into the isolated venv."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def main():
    root = Path('/root/work/v154_run_20260909/v154_wide_action').resolve(strict=True)
    state_path = root / 'launch_state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'failed_or_timed_out'
    try:
        os.kill(state['launcher_pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Previous launcher still exists; do not duplicate')
    assert datetime.fromisoformat(state['deadline_at']) > datetime.now(timezone.utc)
    log = (root / 'launcher.log').read_text()
    assert "Can not perform a '--user' install" in log
    assert not (root / 'results').exists() and not (root / 'smoke_results').exists()
    launcher = root / 'launch.py'
    original = launcher.read_bytes()
    assert hashlib.sha256(original).hexdigest() == '3bd5775f80808201a9572d39e8db64ab0e1632d01819ec75ae5259fd784b7a7d'
    source = original.decode()
    before = "'-m','pip','install','--index-url'"
    assert source.count(before) == 1
    fixed = source.replace(before, "'-m','pip','--isolated','install','--no-user','--index-url'")
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for name in ('launch.pre_dependency_repair.py', 'launch_state.dependency_failure.json',
                 'launcher.dependency_failure.log', 'manifest.pre_dependency_repair.json'):
        assert not (root / name).exists()
    (root / 'launch.pre_dependency_repair.py').write_bytes(original)
    (root / 'manifest.pre_dependency_repair.json').write_bytes(manifest_path.read_bytes())
    launcher.write_text(fixed)
    manifest['files']['launch.py'] = hashlib.sha256(launcher.read_bytes()).hexdigest()
    manifest['dependency_repair'] = {'reason': 'platform forced user install into venv',
                                    'change': 'pip --isolated install --no-user',
                                    'deadline_unchanged': state['deadline_at']}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    state_path.rename(root / 'launch_state.dependency_failure.json')
    (root / 'launcher.log').rename(root / 'launcher.dependency_failure.log')
    with (root / 'launcher.log').open('xb') as output:
        process = subprocess.Popen([sys.executable, '-u', str(launcher)], cwd=root,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    print('RETRY_PID', process.pid, 'DEADLINE', state['deadline_at'], flush=True)


if __name__ == '__main__':
    main()
