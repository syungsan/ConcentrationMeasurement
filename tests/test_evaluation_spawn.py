"""Exercise real spawn workers, including a parent with an open SQLite connection."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/tools"))
sys.path.insert(0, str(ROOT / "tests"))
import evaluate_labeled_dataset as evaluation
from test_training_audit import dataset


class EvaluationSpawnTests(unittest.TestCase):
    def test_spawn_workers_match_single_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = dataset(Path(tmp), is_train=False)
            indexed = evaluation.IndexedDataset(ds)
            # Open the connection in the parent before serializing.
            expected = next(iter(DataLoader(indexed, batch_size=1,
                                            collate_fn=evaluation.make_indexed_collate(None))))
            restored = pickle.loads(pickle.dumps(indexed))
            self.assertIsNone(restored.dataset)
            loader = DataLoader(restored, batch_size=1, num_workers=2,
                                multiprocessing_context="spawn",
                                collate_fn=evaluation.make_indexed_collate(None))
            actual = list(loader)
            self.assertEqual(len(actual), 1)
            self.assertEqual(actual[0][0], expected[0])
            for result, target in zip(actual[0][1:], expected[1:]):
                torch.testing.assert_close(result, target)
            for conn in ds._conns.values():
                conn.close()


if __name__ == "__main__":
    unittest.main()
