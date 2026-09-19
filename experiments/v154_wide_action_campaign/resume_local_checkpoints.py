"""Resume verified V154 checkpoints within the original local deadline."""
from pathlib import Path
import json, hashlib, subprocess, sys, os
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent / 'local_run_20260909'

def main():
    state_path = ROOT / 'local_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    remaining = (datetime.fromisoformat(state['window']['deadline_at']) - datetime.now(timezone.utc)).total_seconds()
    assert remaining > 0, 'Original deadline expired; do not reset'
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for name, sha in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
    checkpoints = list((ROOT/'results').glob('fold*.json'))
    previous = ROOT/'local_state.before_resume.json'
    assert not previous.exists(), 'Inspect previous resume before retrying'
    previous.write_bytes(state_path.read_bytes())
    state['resume'] = {'at': datetime.now(timezone.utc).isoformat(), 'checkpoints': len(checkpoints),
                       'reason': 'Previous owned process IDs absent; stale running state', 'deadline_extended': False}
    state['controller_pid'] = os.getpid()
    with (ROOT/'resume.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(ROOT/'worker.py')], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        state.update(status='running', worker_pid=proc.pid)
        state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
        try:
            code = proc.wait(timeout=remaining)
            state.update(status='completed' if code == 0 else 'failed', exit_code=code)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], creationflags=subprocess.CREATE_NO_WINDOW)
            proc.wait(timeout=30)
            state.update(status='timed_out')
        finally:
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(state))

if __name__ == '__main__':
    main()
