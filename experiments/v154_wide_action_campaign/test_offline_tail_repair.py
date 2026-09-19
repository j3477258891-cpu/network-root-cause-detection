import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from repair_offline_parts import reconstruct

def digest(data): return hashlib.sha256(data).hexdigest()

class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v154_tail_test_')
        self.root = Path(self.temp.name)
        self.source = self.root/'source'; self.source.mkdir()
        self.config = self.root/'upload'; self.config.mkdir()
        self.full = b'known prefix and missing tail'
        self.prefix = self.full[:12]; self.tail = self.full[12:]
        self.name = 'v154_offline_part_001.bin'
        (self.source/self.name).write_bytes(self.prefix)
        (self.config/'tail.bin').write_bytes(self.tail)
        manifest = b'{}'
        installer = b"source = Path('/root/work').resolve(strict=True)\n"
        (self.source/'v154_offline_manifest.json').write_bytes(manifest)
        (self.source/'install_offline_runtime_20260909.py').write_bytes(installer)
        entry = dict(file=self.name, prefix_bytes=len(self.prefix), prefix_sha256=digest(self.prefix),
                     tail_file='tail.bin', tail_bytes=len(self.tail), tail_sha256=digest(self.tail),
                     full_bytes=len(self.full), full_sha256=digest(self.full))
        self.meta = dict(parts=[entry], archive_bytes=len(self.full), archive_sha256=digest(self.full),
                         manifest_sha256=digest(manifest), installer_sha256=digest(installer))
        self.save()
    def save(self):
        (self.config/'v154_repair_config.json').write_text(json.dumps(self.meta))
    def tearDown(self): self.temp.cleanup()
    def test_success_preserves_original(self):
        r = reconstruct(self.source, self.config)
        self.assertEqual((Path(r['directory'])/self.name).read_bytes(), self.full)
        self.assertEqual((self.source/self.name).read_bytes(), self.prefix)
        self.assertFalse(r['competition_submitted'])
    def test_incomplete_tail_rejected_before_any_output(self):
        (self.config/'tail.bin').write_bytes(self.tail[:-1])
        with self.assertRaisesRegex(RuntimeError, 'Tail incomplete'): reconstruct(self.source,self.config)
        self.assertFalse((self.source/'v154_repaired_parts_20260909').exists())
    def test_corrupt_prefix_rejected(self):
        (self.source/self.name).write_bytes(b'x'*len(self.prefix))
        with self.assertRaisesRegex(RuntimeError, 'Prefix mismatch'): reconstruct(self.source,self.config)
    def test_already_complete_not_double_appended(self):
        (self.source/self.name).write_bytes(self.full)
        r = reconstruct(self.source,self.config)
        self.assertEqual((Path(r['directory'])/self.name).read_bytes(), self.full)
    def test_no_overwrite_on_repeat(self):
        reconstruct(self.source,self.config)
        with self.assertRaisesRegex(RuntimeError, 'directory exists'): reconstruct(self.source,self.config)
    def test_wrong_archive_hash_rejected(self):
        self.meta['archive_sha256'] = '0'*64; self.save()
        with self.assertRaisesRegex(RuntimeError, 'Full archive mismatch'): reconstruct(self.source,self.config)
        self.assertFalse((self.source/'v154_repaired_parts_20260909').exists())

class FullPayloadIntegration(unittest.TestCase):
    def test_actual_25_prefixes_and_missing_tails(self):
        here = Path(__file__).resolve().parent
        upload = here/'repair_upload_20260909'
        config = json.loads((upload/'v154_repair_config.json').read_text())
        with tempfile.TemporaryDirectory(prefix='v154_real_repair_') as tmp:
            source = Path(tmp)
            for p in config['parts']:
                full = (here/'offline_payload'/p['file']).read_bytes()
                (source/p['file']).write_bytes(full[:p['prefix_bytes']])
            (source/'v154_offline_manifest.json').write_bytes((here/'offline_payload/v154_offline_manifest.json').read_bytes())
            (source/'install_offline_runtime_20260909.py').write_bytes((here/'install_offline_runtime_20260909.py').read_bytes())
            result = reconstruct(source, upload)
            self.assertEqual(result['parts'], 25)
            self.assertEqual(result['bytes'], 204179566)
            self.assertEqual(result['archive_sha256'], '4c73bf79f287fc22d4f5ee23f847a6da66b0d3dc87bfcc1cc64f7f36de1fb9db')
            for p in config['parts']:
                self.assertEqual(digest((source/p['file']).read_bytes()), p['prefix_sha256'])

if __name__ == '__main__': unittest.main()
