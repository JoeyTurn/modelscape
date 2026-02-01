import unittest
from unittest.mock import patch
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from modelscape.backend.job import run_job


def simple_bfn(bsz, gen=None, X=None, y=None, **kwargs):
    def batch_fn(step):
        Xb = torch.randn(bsz, 4, device="cpu")
        yb = torch.randn(bsz, device="cpu")
        return Xb, yb
    return batch_fn


def device_grab(model, **kwargs):
    return 1 if str(next(model.parameters()).device) == "cpu" else 0


class TestCpuTraining(unittest.TestCase):
    def test_cpu_training_runs(self):
        global_config = {
            "LR": 1e-2,
            "MAX_ITER": 2,
            "LOSS_CHECKPOINTS": [1e-1],
            "GAMMA": 1.0,
            "EMA_SMOOTHER": 0.0,
            "ONLYTHRESHOLDS": True,
            "VERBOSE": False,
            "ONLINE": True,
            "N_TEST": 8,
            "SEED": 0,
            "DIM": 4,
            "otherreturns": {"device": device_grab},
        }
        bfn_config = {"base_bfn": simple_bfn}
        job = (8, 0)
        iterator_names = ["ntrain", "trial"]

        torch.set_default_device("cpu")
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.device_count", return_value=0):
            payload = run_job(
                0,
                job,
                global_config,
                bfn_config,
                iterator_names,
            )

        _, _, _, _, device_flag = payload
        self.assertEqual(float(device_flag), 1.0)


if __name__ == "__main__":
    unittest.main()
