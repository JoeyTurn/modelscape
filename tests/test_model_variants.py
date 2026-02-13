import unittest
from unittest.mock import patch
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from modelscape.backend.job import run_job
from modelscape.model import MLP, SimpleCNN, SimpleTransformer


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simple_cnn_bfn(bsz, gen=None, X=None, y=None, **kwargs):
    def batch_fn(step):
        device = _default_device()
        Xb = torch.randn(bsz, 1, 8, 8, device=device)
        yb = torch.randn(bsz, device=device)
        return Xb, yb
    return batch_fn


def simple_transformer_bfn(bsz, gen=None, X=None, y=None, **kwargs):
    def batch_fn(step):
        device = _default_device()
        Xb = torch.randn(bsz, 4, 8, device=device)
        yb = torch.randn(bsz, device=device)
        return Xb, yb
    return batch_fn


class TestModelVariants(unittest.TestCase):
    def _run_job(self, global_config, bfn_config):
        job = (8, 0)
        iterator_names = ["ntrain", "trial"]
        return run_job(0, job, global_config, bfn_config, iterator_names)

    def test_simple_cnn_model_class(self):
        class ShouldNotInstantiateMLP(MLP):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Default MLP should not be instantiated")

        global_config = {
            "LR": 1e-2,
            "MAX_ITER": 2,
            "LOSS_CHECKPOINTS": [1e-1],
            "EMA_SMOOTHER": 0.0,
            "ONLYTHRESHOLDS": True,
            "VERBOSE": False,
            "ONLINE": True,
            "N_TEST": 8,
            "SEED": 0,
            "DIM": 4,
            "in_channels": 1,
            "img_size": 8,
            "width": 8,
            "depth": 2,
            "d_out": 1,
            "MODEL_CLASS": SimpleCNN,
        }
        bfn_config = {"base_bfn": simple_cnn_bfn}
        with patch("modelscape.backend.job.MLP", ShouldNotInstantiateMLP):
            payload = self._run_job(global_config, bfn_config)
        self.assertEqual(len(payload), 4)

    def test_simple_transformer_model_class(self):
        class ShouldNotInstantiateMLP(MLP):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Default MLP should not be instantiated")

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
            "DIM": 8,
            "d_in": 8,
            "seq_len": 4,
            "width": 16,
            "depth": 2,
            "d_out": 1,
            "MODEL_CLASS": SimpleTransformer,
        }
        bfn_config = {"base_bfn": simple_transformer_bfn}
        with patch("modelscape.backend.job.MLP", ShouldNotInstantiateMLP):
            payload = self._run_job(global_config, bfn_config)
        self.assertEqual(len(payload), 4)


if __name__ == "__main__":
    unittest.main()
