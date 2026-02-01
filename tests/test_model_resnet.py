import unittest
from unittest.mock import patch
import os
import sys

import torch
import torch.nn as nn

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from modelscape.backend.job import run_job
from modelscape.model import MLP


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resnet_bfn(bsz, gen=None, X=None, y=None, **kwargs):
    def batch_fn(step):
        device = _default_device()
        Xb = torch.randn(bsz, 1, 8, 8, device=device)
        yb = torch.randn(bsz, device=device)
        return Xb, yb
    return batch_fn


class BasicBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.act(out + x)


class SimpleResNet(nn.Module):
    def __init__(self, in_channels=1, width=8, num_blocks=2, d_out=1):
        super().__init__()
        self.input_layer = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False)
        self.hidden_layers = nn.ModuleList([BasicBlock(width) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(width, d_out)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.input_layer(x))
        for block in self.hidden_layers:
            h = block(h)
        h = h.mean(dim=(2, 3))
        return self.output_layer(h)


class TestResNetOverride(unittest.TestCase):
    def test_resnet_defined_in_test(self):
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
            "in_channels": 1,
            "width": 8,
            "num_blocks": 2,
            "d_out": 1,
            "MODEL_CLASS": SimpleResNet,
        }
        bfn_config = {"base_bfn": resnet_bfn}
        with patch("modelscape.backend.job.MLP", ShouldNotInstantiateMLP):
            payload = run_job(0, (8, 0), global_config, bfn_config, ["ntrain", "trial"])
        self.assertEqual(len(payload), 4)


if __name__ == "__main__":
    unittest.main()
