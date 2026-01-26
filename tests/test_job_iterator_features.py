import unittest
from unittest.mock import patch
import os
import sys

import torch
import torch.multiprocessing as mp

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from modelscape.backend.job_iterator import main as run_job_iterator


def random_bfn(bsz, gen=None, X=None, y=None, dim=4, **kwargs):
    def batch_fn(step):
        device = gen.device if gen is not None else torch.device("cpu")
        Xb = torch.randn(bsz, dim, device=device)
        yb = torch.randn(bsz, device=device)
        return Xb, yb
    return batch_fn


def _base_global_config(online=True):
    return {
        "LR": 1e-2,
        "MAX_ITER": 2,
        "LOSS_CHECKPOINTS": [1e9],
        "GAMMA": 1.0,
        "EMA_SMOOTHER": 0.0,
        "ONLYTHRESHOLDS": True,
        "VERBOSE": False,
        "ONLINE": online,
        "N_TEST": 8,
        "SEED": 0,
        "DIM": 4,
        "width": 8,
        "depth": 2,
    }


def _config2outcome(trace_obj):
    if isinstance(trace_obj, dict) and "config2outcome" in trace_obj:
        return trace_obj["config2outcome"]
    return {}


class TestJobIteratorFeatures(unittest.TestCase):
    def test_dynamic_iterables(self):
        iterators = [(n for n in [4, 8]), [0]]
        iterator_names = ["ntrain", "trial"]
        global_config = _base_global_config(online=True)
        bfn_config = {"base_bfn": random_bfn, "dim": 4}
        result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config, use_mp=False)
        self.assertEqual(len(result["jobs"]), 2)
        self.assertTrue(_config2outcome(result["losses"]))

    def test_online_offline_paths(self):
        iterators = [[4], [0]]
        iterator_names = ["ntrain", "trial"]
        bfn_config = {"base_bfn": random_bfn, "dim": 4}

        result_online = run_job_iterator(
            iterators, iterator_names, _base_global_config(online=True), bfn_config=bfn_config, use_mp=False
        )
        result_offline = run_job_iterator(
            iterators, iterator_names, _base_global_config(online=False), bfn_config=bfn_config, use_mp=False
        )

        self.assertTrue(_config2outcome(result_online["losses"]))
        self.assertTrue(_config2outcome(result_offline["losses"]))

    def test_dynamic_grabs(self):
        iterators = [[4], [0], [1.5, 2.5]]
        iterator_names = ["ntrain", "trial", "alpha"]

        global_config = _base_global_config(online=True)
        global_config.update({"otherreturns": {"alpha_val": lambda model, opt, alpha, **kwargs: float(alpha)}})
        bfn_config = {"base_bfn": random_bfn, "dim": 4}

        result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config, use_mp=False)
        extras = result["extras"]["alpha_val"]
        vals = list(_config2outcome(extras).values())
        self.assertEqual(set(vals), {1.5, 2.5})

    def test_multiprocessing_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        iterators = [[4], [0]]
        iterator_names = ["ntrain", "trial"]
        global_config = _base_global_config(online=True)
        bfn_config = {"base_bfn": random_bfn, "dim": 4}

        with patch("torch.cuda.device_count", return_value=1):
            result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config, use_mp=True)

        self.assertEqual(len(result["jobs"]), 1)
        self.assertTrue(_config2outcome(result["losses"]))


if __name__ == "__main__":
    unittest.main()
