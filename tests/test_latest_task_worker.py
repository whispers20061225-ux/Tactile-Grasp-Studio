import importlib.util
from pathlib import Path
import sys
import time
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "core" / "latest_task_worker.py"
spec = importlib.util.spec_from_file_location("latest_task_worker_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)
LatestTaskWorker = module.LatestTaskWorker


class LatestTaskWorkerTests(unittest.TestCase):
    def test_latest_only_submission_overwrites_pending_task(self):
        seen = []

        def worker_fn(task):
            time.sleep(0.05)
            seen.append(task["value"])
            return task["value"]

        worker = LatestTaskWorker("test-worker", worker_fn)
        worker.start()
        try:
            worker.submit({"value": 1})
            time.sleep(0.01)
            worker.submit({"value": 2})
            worker.submit({"value": 3})
            time.sleep(0.25)

            result = worker.take_result()
            self.assertIsNotNone(result)
            self.assertEqual(result.value, 3)
            self.assertGreaterEqual(worker.take_overwrite_count(), 1)
            self.assertIn(3, seen)
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
