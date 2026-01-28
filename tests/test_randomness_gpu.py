import unittest
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from modelscape.backend.job_iterator import main as run_job_iterator


def random_bfn(bsz, gen=None, X=None, y=None, dim=4, **kwargs):
    def batch_fn(step):
        device = gen.device if gen is not None else torch.device("cpu")
        Xb = torch.randn(bsz, dim, generator=gen, device=device)
        yb = torch.randn(bsz, generator=gen, device=device)
        return Xb, yb
    return batch_fn


def xte_sum_grab(X_te, **kwargs):
    return float(X_te.sum().item())


def xte_head_grab(X_te, **kwargs):
    return X_te.flatten()[:10].cpu().numpy()


def _config2outcome(trace_obj):
    if isinstance(trace_obj, dict) and "config2outcome" in trace_obj:
        return trace_obj["config2outcome"]
    return {}


class TestRandomnessGPU(unittest.TestCase):
    def test_trials_generate_different_data_gpu(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if torch.cuda.device_count() < 1:
            self.skipTest("No CUDA devices detected")

        iterators = [[8], [0, 1]]
        iterator_names = ["ntrain", "trial"]
        global_config = {
            "LR": 1e-2,
            "MAX_ITER": 1,
            "LOSS_CHECKPOINTS": [1e9],
            "GAMMA": 1.0,
            "EMA_SMOOTHER": 0.0,
            "ONLYTHRESHOLDS": False,
            "VERBOSE": False,
            "ONLINE": True,
            "N_TEST": 8,
            "SEED": 0,
            "DIM": 4,
            "width": 8,
            "depth": 2,
            "otherreturns": {"xte_sum": xte_sum_grab},
        }
        bfn_config = {"base_bfn": random_bfn, "dim": 4}

        result = run_job_iterator(
            iterators, iterator_names, global_config, bfn_config=bfn_config, use_mp=False
        )

        extras = result["extras"]["xte_sum"]
        vals = list(_config2outcome(extras).values())
        self.assertEqual(len(vals), 2)
        self.assertFalse(torch.allclose(torch.tensor(vals[0]), torch.tensor(vals[1])))


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        print("CUDA not available; skipping GPU randomness demo.")
    else:
        iterators = [[8], [0, 1]]
        iterator_names = ["ntrain", "trial"]
        global_config = {
            "LR": 1e-2,
            "MAX_ITER": 1,
            "LOSS_CHECKPOINTS": [0.1],
            "GAMMA": 1.0,
            "EMA_SMOOTHER": 0.0,
            "ONLYTHRESHOLDS": False,
            "VERBOSE": False,
            "ONLINE": True,
            "N_TEST": 8,
            "SEED": 0,
            "DIM": 4,
            "width": 8,
            "depth": 2,
            "otherreturns": {"xte_head": xte_head_grab},
        }
        bfn_config = {"base_bfn": random_bfn, "dim": 4}

        result = run_job_iterator(
            iterators, iterator_names, global_config, bfn_config=bfn_config, use_mp=False
        )

        extras = result["extras"]["xte_head"]
        vals = _config2outcome(extras).values()
        for idx, val in enumerate(vals):
            print(f"trial[{idx}] X[:10]:", val.tolist())

    unittest.main()
