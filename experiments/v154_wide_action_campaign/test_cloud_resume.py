"""Small isolated tests for recovery safety; these do not train or contact cloud."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import zipfile

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ResumeTests(unittest.TestCase):
    def fixture(self, work, unsafe=False):
        path = work / 'v154_compact_training.zip'
        window = {'deadline_at': '2026-09-09T04:40:00+08:00'}
        files = {'training_window.json': json.dumps(window).encode(), 'launch.py': b'pass\n'}
        manifest = {'window': window, 'files': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
        with zipfile.ZipFile(path, 'w') as z:
            for k, v in files.items():
                z.writestr('v154_wide_action/' + k, v)
            z.writestr('v154_wide_action/manifest.json', json.dumps(manifest))
            if unsafe:
                z.writestr('../escape.txt', 'blocked')
        return path

    def run_case(self, unsafe=False, altered=False, existing=False):
        mod = module('resume_cloud_20260909')
        with tempfile.TemporaryDirectory(prefix='v154_resume_test_') as tmp:
            work = Path(tmp)
            archive = self.fixture(work, unsafe)
            mod.ARCHIVE_SHA = hashlib.sha256(archive.read_bytes()).hexdigest()
            mod.ARCHIVE_SIZE = archive.stat().st_size
            if altered:
                archive.write_bytes(archive.read_bytes() + b'changed')
            if existing:
                (work / 'v154_run_20260909').mkdir()
            real_path = Path
            def routed(value):
                return work if value == '/root/work' else real_path(value)
            with patch.object(mod, 'Path', side_effect=routed), patch.object(mod.subprocess, 'Popen', return_value=Mock(pid=123)) as launch:
                if unsafe or altered or existing:
                    with self.assertRaises(RuntimeError):
                        mod.main()
                    launch.assert_not_called()
                else:
                    mod.main()
                    launch.assert_called_once()
                    root = work / 'v154_run_20260909/v154_wide_action'
                    new = json.loads((root / 'manifest.json').read_text())
                    old = json.loads((root / 'manifest.original.json').read_text())
                    self.assertEqual(old['window']['deadline_at'], '2026-09-09T04:40:00+08:00')
                    self.assertFalse(new['window']['competition_upload_authorized'])
                    self.assertEqual(new['window']['competition_budget_remaining_at_start'], 6)
                    for key, digest in new['files'].items():
                        self.assertEqual(hashlib.sha256((root / key).read_bytes()).hexdigest(), digest)

    def test_verified_isolated_launch(self): self.run_case()
    def test_changed_archive_blocked(self): self.run_case(altered=True)
    def test_zip_traversal_blocked(self): self.run_case(unsafe=True)
    def test_existing_run_not_overwritten(self): self.run_case(existing=True)


if __name__ == '__main__':
    unittest.main()
