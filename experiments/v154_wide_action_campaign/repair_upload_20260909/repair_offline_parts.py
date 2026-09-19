"""Reconstruct verified parts in a NEW directory; originals are never modified."""
from pathlib import Path
import hashlib, json, shutil, argparse

def sha(data): return hashlib.sha256(data).hexdigest()
def require(ok, message):
    if not ok: raise RuntimeError(message)

def reconstruct(source, config_dir):
    source = source.resolve(strict=True)
    config_dir = config_dir.resolve(strict=True)
    config = json.loads((config_dir/'v154_repair_config.json').read_text())
    target = source/'v154_repaired_parts_20260909'
    require(not target.exists(), 'Repaired directory exists; inspect it before retrying')
    manifest = (source/'v154_offline_manifest.json').read_bytes()
    installer = (source/'install_offline_runtime_20260909.py').read_bytes()
    require(sha(manifest) == config['manifest_sha256'], 'Original manifest changed')
    require(sha(installer) == config['installer_sha256'], 'Original installer changed')
    entries = config['parts']
    require(len({p['file'] for p in entries}) == len(entries), 'Duplicate part names')
    repaired, archive = [], hashlib.sha256()
    total = 0
    for entry in entries:
        name = entry['file']
        require(Path(name).name == name and '/' not in name and '\\' not in name, 'Unsafe filename')
        prefix = (source/name).read_bytes()
        # Accept already-complete source parts, but do not append the tail twice.
        if len(prefix) == entry['full_bytes'] and sha(prefix) == entry['full_sha256']:
            data = prefix
        else:
            require(len(prefix) == entry['prefix_bytes'] and sha(prefix) == entry['prefix_sha256'], 'Prefix mismatch: '+name)
            tail_name = entry.get('tail_file')
            require(tail_name is not None, 'Unexpected incomplete part: '+name)
            require(Path(tail_name).name == tail_name and '/' not in tail_name and '\\' not in tail_name, 'Unsafe tail name')
            tail = (config_dir/tail_name).read_bytes()
            require(len(tail) == entry['tail_bytes'] and sha(tail) == entry['tail_sha256'], 'Tail incomplete or corrupt: '+tail_name)
            data = prefix + tail
        require(len(data) == entry['full_bytes'] and sha(data) == entry['full_sha256'], 'Reconstruction mismatch: '+name)
        archive.update(data); total += len(data); repaired.append((name, data))
    require(total == config['archive_bytes'] and archive.hexdigest() == config['archive_sha256'], 'Full archive mismatch')
    # Prepare a separate installer bound to the new directory, not to the originals.
    original = "source = Path('/root/work').resolve(strict=True)"
    text = installer.decode('utf-8')
    require(text.count(original) == 1, 'Unexpected installer source binding')
    text = text.replace(original, 'source = Path(__file__).resolve().parent')
    # All verification above completes BEFORE any destination is created.
    target.mkdir()
    for name, data in repaired: (target/name).write_bytes(data)
    (target/'v154_offline_manifest.json').write_bytes(manifest)
    (target/'install_offline_runtime_20260909.py').write_text(text, encoding='utf-8')
    result = {'status':'repaired_verified_not_installed', 'parts':len(repaired), 'bytes':total,
              'archive_sha256':archive.hexdigest(), 'original_files_preserved':True,
              'competition_submitted':False, 'directory':str(target)}
    (target/'repair_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('/root/work'))
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.source, args.config_dir), indent=2))
