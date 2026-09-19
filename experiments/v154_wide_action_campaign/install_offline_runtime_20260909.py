"""Cloud installation only, no model-comparison run or leaderboard access."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys
import zipfile


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path('/root/work').resolve(strict=True)
    assert platform.system() == 'Linux' and platform.machine() == 'aarch64'
    assert sys.version_info[:2] == (3, 10)
    manifest = json.loads((source / 'v154_offline_manifest.json').read_text())
    for entry in manifest['parts']:
        assert Path(entry['file']).name == entry['file']
        path = source / entry['file']
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Incomplete part: ' + path.name
    root = source / 'v154_offline_runtime_20260909'
    if root.exists(): raise RuntimeError('Runtime directory exists; inspect state before retry')
    root.mkdir()
    state = {'status': 'reconstructing', 'started_at': datetime.now(timezone.utc).isoformat(),
             'competition_submitted': False, 'source_manifest_sha256': sha(source / 'v154_offline_manifest.json')}
    def save(): (root / 'install_state.json').write_text(json.dumps(state, indent=2))
    save()
    try:
        archive = root / 'runtime.zip'
        with archive.open('xb') as output:
            for entry in manifest['parts']:
                output.write((source / entry['file']).read_bytes())
        assert archive.stat().st_size == manifest['archive_bytes'] and sha(archive) == manifest['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == len(set(z.namelist()))
            for entry in z.infolist():
                name = PurePosixPath(entry.filename)
                assert not name.is_absolute() and '..' not in name.parts and '\\' not in entry.filename
                assert ((entry.external_attr >> 16) & 0o170000) != 0o120000
            assert z.testzip() is None
            z.extractall(root)
        for entry in manifest['wheels']:
            assert sha(root / 'wheels' / entry['file']) == entry['sha256']
        state['status'] = 'installing_offline'; save()
        subprocess.run([sys.executable, '-m', 'venv', str(root / '.venv')], check=True, timeout=120)
        python = root / '.venv/bin/python'
        subprocess.run([str(python), '-m', 'pip', '--isolated', 'install', '--no-user', '--no-index',
                        '--find-links', str(root / 'wheels'), '--require-hashes', '-r', str(root / 'requirements.lock')],
                       check=True, timeout=600)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True, timeout=60)
        smoke = """import json,numpy as np,catboost,sklearn
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
x=np.arange(72,dtype=float).reshape(24,3); y=np.arange(24)%3
for m in [CatBoostClassifier(iterations=3,depth=2,verbose=False,thread_count=2,allow_writing_files=False),ExtraTreesClassifier(n_estimators=3,n_jobs=2),HistGradientBoostingClassifier(max_iter=3,min_samples_leaf=2)]:
 m.fit(x,y); assert m.predict_proba(x).shape==(24,3)
print(json.dumps({'numpy':np.__version__,'sklearn':sklearn.__version__,'catboost':catboost.__version__,'three_model_smoke':'passed_not_performance_validation'}))
"""
        smoke_result = subprocess.run([str(python), '-c', smoke], check=True, timeout=180, capture_output=True, text=True)
        state.update(status='ready', smoke=smoke_result.stdout.strip(), python=str(python))
        print(smoke_result.stdout, flush=True)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); raise
    finally:
        state['updated_at'] = datetime.now(timezone.utc).isoformat(); save()
    print('OFFLINE_RUNTIME_READY', str(root), flush=True)


if __name__ == '__main__': main()
