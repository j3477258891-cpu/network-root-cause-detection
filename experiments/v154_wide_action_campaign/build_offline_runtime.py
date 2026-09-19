"""Verify downloaded ARM64 wheels against PyPI, lock all dependencies, split payload."""
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from pathlib import Path
import hashlib
import json
import zipfile
from pip._vendor import requests

HERE = Path(__file__).resolve().parent


def verify(path):
    with zipfile.ZipFile(path) as z:
        metadata = [n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
        assert len(metadata) == 1
        fields = BytesParser().parsebytes(z.read(metadata[0]))
    name, version = fields['Name'], fields['Version']
    url = f'https://pypi.org/pypi/{name}/{version}/json'
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    published = response.json()
    entries = [r for r in published['urls'] if r['filename'] == path.name]
    assert len(entries) == 1, 'Not found in official PyPI: ' + path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entries[0]['digests']['sha256'], 'Official digest mismatch: ' + path.name
    return {'file': path.name, 'name': name, 'version': version, 'sha256': digest,
            'bytes': path.stat().st_size, 'official_metadata_url': url}


def main():
    wheels = sorted((HERE / 'offline_wheels_aarch64').glob('*.whl'))
    assert len(wheels) >= 6, 'Dependencies have not finished downloading'
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(verify, wheels))
    required = {'numpy','scipy','scikit-learn','catboost','threadpoolctl','joblib'}
    assert required <= {r['name'].lower().replace('_','-') for r in records}
    names = [r['name'].lower().replace('_','-') for r in records]
    assert len(names) == len(set(names)), 'Ambiguous dependency versions'
    out = HERE / 'offline_payload'
    out.mkdir(exist_ok=False)
    lock = '\n'.join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}" for r in records) + '\n'
    archive = out / 'v154_offline_runtime.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        z.writestr('requirements.lock', lock)
        for path in wheels:
            z.write(path, 'wheels/' + path.name)
    parts = []
    with archive.open('rb') as stream:
        while data := stream.read(8 * 1024 * 1024):
            name = f'v154_offline_part_{len(parts)+1:03d}.bin'
            (out / name).write_bytes(data)
            parts.append({'file': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'archive_bytes': archive.stat().st_size, 'parts': parts, 'wheels': records,
                'target': 'CPython 3.10 Linux aarch64; offline install, no public submissions'}
    (out / 'v154_offline_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'wheel_count': len(records), 'part_count': len(parts), 'bytes': manifest['archive_bytes']}))


if __name__ == '__main__': main()
