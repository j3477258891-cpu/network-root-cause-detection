"""Run frozen V154 paired five-fold comparison locally without changing old runs."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import os
from datetime import datetime, timezone, timedelta

HERE = Path(__file__).resolve().parent


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    source = HERE / 'compact_bundle'
    target = HERE / 'local_run_20260909'
    if target.exists():
        raise RuntimeError('Local run exists; inspect its state before restarting')
    original = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    for name, digest in original['files'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen input drift: ' + name)
    shutil.copytree(source, target)
    shutil.copy2(target / 'manifest.json', target / 'manifest.original.json')
    shutil.copy2(target / 'training_window.json', target / 'training_window.original.json')
    now = datetime.now(timezone.utc)
    window = dict(original['window'])
    window.update(started_at=now.isoformat(), deadline_at=(now + timedelta(hours=6)).isoformat(),
                  authorization='User explicitly requested local V154 five-fold comparison on 2026-09-09; new local six-hour cap, old cloud deadline preserved.',
                  competition_upload_authorized=False)
    save(target / 'training_window.json', window)
    manifest = dict(original)
    manifest['window'] = window
    manifest['files']['training_window.json'] = hashlib.sha256((target / 'training_window.json').read_bytes()).hexdigest()
    manifest['local_provenance'] = {'original_manifest_sha256': hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
                                   'same_models_and_folds': True, 'public_score': False}
    save(target / 'manifest.json', manifest)
    state = {'status': 'starting', 'window': window, 'python': sys.executable, 'controller_pid': os.getpid(),
             'competition_submitted': False, 'directory': str(target)}
    save(target / 'local_state.json', state)
    with (target / 'training.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(target / 'worker.py')], cwd=target,
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        state.update(status='running', worker_pid=process.pid)
        save(target / 'local_state.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        try:
            code = process.wait(timeout=6 * 3600)
            state.update(status='completed' if code == 0 else 'failed', exit_code=code)
        except subprocess.TimeoutExpired:
            # Exact owned worker and its descendants only; no broad process-name kill.
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.terminate()
            process.wait(timeout=30)
            state['status'] = 'timed_out'
        finally:
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            save(target / 'local_state.json', state)
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
