import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(r"D:\zgyidong\experiments\experiment_manager.py")
SPEC = importlib.util.spec_from_file_location("experiment_manager", MODULE_PATH)
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


class ExperimentManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(r"D:\zgyidong\test")
        cls.champion_path = Path(
            r"D:\zgyidong\codexgz\result_record_probe_swap12_score_0.905373.csv"
        )
        cls.catalog_path = Path(r"D:\zgyidong\experiments\candidate_catalog.json")
        cls.topologies = manager.load_topologies(cls.test_dir)
        cls.champion = manager.load_submission(cls.champion_path)

    def test_known_score_reconstructs_integer_tp(self):
        tp, reconstructed = manager.infer_tp(0.905373, 1059)
        self.assertEqual(tp, 952)
        self.assertAlmostEqual(reconstructed, 1904 / 2103)

    def test_invalid_score_is_rejected(self):
        with self.assertRaises(ValueError):
            manager.infer_tp(0.905000, 1059)

    def test_atomic_probe_preserves_submission_invariants(self):
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        selection = manager.apply_units(self.champion, catalog["units"][:8])
        validation = manager.validate_selection(selection, self.topologies)
        self.assertEqual(validation["orders"], 546)
        self.assertEqual(validation["predictions"], 1059)

    def test_conflicting_unit_is_rejected(self):
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        unit = catalog["units"][0]
        with self.assertRaises(ValueError):
            manager.apply_units(self.champion, [unit, unit])

    def test_record_infers_positive_delta_and_persists_score(self):
        candidate = Path(
            r"D:\zgyidong\experiments\submissions\day01_probe02_v11_local_a.csv"
        )
        baseline_hash = manager.sha256(self.champion_path)
        candidate_hash = manager.sha256(candidate)
        ledger = {
            "version": 1,
            "ground_truth_positives": 1044,
            "score_precision": 6,
            "champion": {
                "path": str(self.champion_path),
                "sha256": baseline_hash,
                "score": 0.905373,
                "tp": 952,
                "predictions": 1059,
            },
            "scores": {
                baseline_hash: {
                    "path": str(self.champion_path),
                    "score": 0.905373,
                    "tp": 952,
                    "predictions": 1059,
                }
            },
            "experiments": [
                {
                    "id": "probe",
                    "kind": "test",
                    "status": "ready",
                    "decision": "pending",
                    "base_sha256": baseline_hash,
                    "base_score": 0.905373,
                    "base_tp": 952,
                    "candidate_path": str(candidate),
                    "predictions": 1059,
                    "sha256": candidate_hash,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            manager.save_json(Path(directory) / "ledger.json", ledger)
            manager.command_record(
                SimpleNamespace(
                    root=directory,
                    experiment="probe",
                    score=0.906324,
                    promote=False,
                )
            )
            saved = manager.load_ledger(directory)
            experiment = saved["experiments"][0]
            self.assertEqual(experiment["tp"], 953)
            self.assertEqual(experiment["delta_tp"], 1)
            self.assertEqual(experiment["decision"], "accept")
            self.assertIn(candidate_hash, saved["scores"])


if __name__ == "__main__":
    unittest.main()
