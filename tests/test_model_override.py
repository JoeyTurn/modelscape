import unittest
from unittest.mock import patch
import os
import sys
import tempfile

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from modelscape.backend.job import run_job
from modelscape.model import MLP


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simple_bfn(bsz, gen=None, X=None, y=None, **kwargs):
    def batch_fn(step):
        device = _default_device()
        Xb = torch.randn(bsz, 4, device=device)
        yb = torch.randn(bsz, device=device)
        return Xb, yb
    return batch_fn


class CustomMLP(MLP):
    pass


class TestModelOverride(unittest.TestCase):
    def _run_job(self, global_config, bfn_config):
        job = (8, 0)
        iterator_names = ["ntrain", "trial"]
        return run_job(0, job, global_config, bfn_config, iterator_names)

    def test_model_class_override(self):
        class ShouldNotInstantiateMLP(MLP):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Default MLP should not be instantiated")

        global_config = {
            "LR": 1e-2,
            "MAX_ITER": 2,
            "LOSS_CHECKPOINTS": [1e9],
            "GAMMA": 1.0,
            "EMA_SMOOTHER": 0.0,
            "ONLYTHRESHOLDS": True,
            "VERBOSE": False,
            "ONLINE": True,
            "N_TEST": 8,
            "SEED": 0,
            "DIM": 4,
            "width": 8,
            "depth": 2,
            "MODEL_CLASS": CustomMLP,
        }
        bfn_config = {"base_bfn": simple_bfn}
        with patch("modelscape.backend.job.MLP", ShouldNotInstantiateMLP):
            payload = self._run_job(global_config, bfn_config)
        self.assertEqual(len(payload), 4)

    def test_model_file_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "custom_model.py")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(
                    "import torch\n"
                    "import torch.nn as nn\n"
                    "class CustomMLP(nn.Module):\n"
                    "    def __init__(self, d_in=1, width=8, depth=2, d_out=1, **kwargs):\n"
                    "        super().__init__()\n"
                    "        self.input_layer = nn.Linear(d_in, width)\n"
                    "        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])\n"
                    "        self.output_layer = nn.Linear(width, d_out)\n"
                    "        self.nonlin = nn.ReLU()\n"
                    "    def forward(self, x):\n"
                    "        h = self.nonlin(self.input_layer(x))\n"
                    "        for layer in self.hidden_layers:\n"
                    "            h = self.nonlin(layer(h))\n"
                    "        return self.output_layer(h)\n"
                )

            global_config = {
                "LR": 1e-2,
                "MAX_ITER": 2,
                "LOSS_CHECKPOINTS": [1e9],
                "GAMMA": 1.0,
                "EMA_SMOOTHER": 0.0,
                "ONLYTHRESHOLDS": True,
                "VERBOSE": False,
                "ONLINE": True,
                "N_TEST": 8,
                "SEED": 0,
                "DIM": 4,
                "width": 8,
                "depth": 2,
                "MODEL_FILE": model_path,
                "MODEL_NAME": "CustomMLP",
            }
            bfn_config = {"base_bfn": simple_bfn}
            class ShouldNotInstantiateMLP(MLP):
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Default MLP should not be instantiated")

            with patch("modelscape.backend.job.MLP", ShouldNotInstantiateMLP):
                payload = self._run_job(global_config, bfn_config)
            self.assertEqual(len(payload), 4)


if __name__ == "__main__":
    unittest.main()
