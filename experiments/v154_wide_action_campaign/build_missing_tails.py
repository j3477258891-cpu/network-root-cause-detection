"""Verify user-reported remote prefixes; emit only missing bytes and repair tool."""
from pathlib import Path
import hashlib, json, shutil

HERE = Path(__file__).resolve().parent
HASHES = '''30393fdccb8430c0b53c8ce62b7757e98d5bd4d8d462cefb78e9198270f58087
f9cc8388a6edec4075894db7c2e9e34a6875c942f78912a738a53a917c5d3847
caecfd05b179d2ac902d2d5c820016a9f6fa8445f0614a236a5c46572b5d6c33
dc37ad0a686077d05afa760ccf78babdf690c805fd21ae3ccbf5a791f99b2837
eff15d7150bdc46c825dbdeeb930152fa8e3f74a8f621c9fb1f139e4804d157d
3eb5dcd30ccfe89c608ba3f103ce14967df5c6249f013f8ee1f3e8329d473358
f77c3fdeeb19d85d24abd0d743b5973fcd47efbe2420e6862c3ad7fdee260177
384e0d6fde566a70dfb61b90468f919582b4ee2b33face641e14b084818c5983
63c351500f4d6a3c97376a7015aca8ec0bd406a6919c28233ecbffb7b6ef6cfc
2c91fe93b93bb1ce7b6693cf7497bf00455c521151b0312adff6a75ac3db5e2c
d3dd0c7f8558cadf8a44c5deca3880fdaa8c6b0038980cd2778e3f9c2b452904
60116f7b688872d3953c88dff8ddd08aa991bbe0eb9f79d61a663f506ff7cfe9
edb536f97d6424ec5dc3d6bfa8ccc8be8a5f5069fb20e3dfebd4cf3ed5d4850b
31f04442e5a93c0139ae5c07d64a3107e67640bf6b7b2e68b900e8d055e06937
98748ca84914fea8232a71a7b7fd48c94e264524994d75815beaa1209686849e
7060c08b2826e971d7a4147cc6b6ec9e643f4e484dda4f33997e17f6ace4e68b
450aff6e360bac252ee4feae4a9a0efdc20b9ce9d4d037a0075b277d7f7e45df
d54243c955b1f2800de679b34b7b415f923f8986f8c9c0930d46d9e42400645d
28a9062b3b783b01fbff4b294069773362fd6cab8eba63217c7d11a88e547507
dca5d9ea3ca911aec5503c80d1d8f41e3def21c972da8e7f69c247b38d2a3af4
db29cf9bf0098ad0287a31cdf92796ece48271945dc43c81c3f3544c45f0e11f
044164a3f475e8a8042ded04c7fabb9c185028e60467cc96928411db262e89c4
2f1ce53b0c8047ca6b90e16d1f14f579ea10ce3b1faed831c7ad69377c0c83b7
38041785aa641ccacf80a8a54fa10845e86128001b025c716b189f4264962b00
7b3907a234d9f78464511837fd326db33b8d4bc1037ff248360578661076eb8c'''.splitlines()

def digest(data): return hashlib.sha256(data).hexdigest()

def main():
    source = HERE / 'offline_payload'
    manifest = json.loads((source/'v154_offline_manifest.json').read_text())
    sizes = [v*1024*1024 for v in [6,7,6,6,7,6,6,6,6,7,7,6,6,6,7,7,6,5,7,6,6,6,6,6]] + [2852974]
    assert len(manifest['parts']) == len(HASHES) == len(sizes) == 25
    tails, observations = {}, []
    for part, size, remote_hash in zip(manifest['parts'], sizes, HASHES):
        data = (source/part['file']).read_bytes()
        assert len(data) == part['bytes'] and digest(data) == part['sha256'], 'Local full part changed'
        # Screenshot sizes are a transcription aid; the user-supplied SHA is decisive.
        # Jupyter uploaded in 1 MiB chunks. Resolve any transcription discrepancy
        # against exact local prefix hashes, never concatenate an unverified prefix.
        matches = [n for n in sorted(set([len(data)] + list(range(1024*1024, len(data), 1024*1024))))
                   if digest(data[:n]) == remote_hash]
        assert len(matches) == 1, 'Remote prefix mismatch: ' + part['file']
        if size != matches[0]:
            print('Screenshot size transcription corrected from SHA:', part['file'], size, matches[0])
        size = matches[0]
        item = {'file': part['file'], 'prefix_bytes': size, 'prefix_sha256': remote_hash,
                'full_bytes': len(data), 'full_sha256': part['sha256']}
        if size < len(data):
            name = part['file'].replace('offline_part_', 'repair_tail_')
            tails[name] = data[size:]
            item.update(tail_file=name, tail_bytes=len(data)-size, tail_sha256=digest(data[size:]))
        observations.append(item)
    installer = HERE/'install_offline_runtime_20260909.py'
    config = {'parts': observations, 'archive_sha256': manifest['archive_sha256'],
              'archive_bytes': manifest['archive_bytes'],
              'manifest_sha256': digest((source/'v154_offline_manifest.json').read_bytes()),
              'installer_sha256': digest(installer.read_bytes()),
              'prefixes_verified': 25, 'missing_bytes': sum(map(len, tails.values()))}
    out = HERE/'repair_upload_20260909'
    out.mkdir(exist_ok=False)
    for name, data in tails.items(): (out/name).write_bytes(data)
    (out/'v154_repair_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    shutil.copy2(HERE/'repair_offline_parts.py', out/'repair_offline_parts.py')
    print(json.dumps({'prefixes_verified':25, 'tail_count':len(tails), 'missing_bytes':config['missing_bytes'],
                      'upload_files':len(tails)+2,'output':str(out)}, indent=2))

if __name__ == '__main__': main()
