import unittest
from unittest.mock import patch
import os
import sys

import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from modelscape.backend.job import run_job


def ce_bfn(bsz, gen=None, X=None, y=None, dim=4, num_classes=3, **kwargs):
    def batch_fn(step):
        device = gen.device if gen is not None else torch.device("cpu")
        Xb = torch.randn(bsz, dim, device=device)
        yb = torch.randint(0, num_classes, (bsz,), device=device, dtype=torch.long)
        return Xb, yb
    return batch_fn


def optimizer_name_grab(opt, **kwargs):
    return opt.__class__.__name__


class TestOptimizerLoss(unittest.TestCase):
    def test_adam_cross_entropy_runs(self):
        num_classes = 3
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
            "width": 8,
            "depth": 2,
            "d_out": num_classes,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "otherreturns": {"optimizer_name": optimizer_name_grab},
        }
        bfn_config = {"base_bfn": ce_bfn, "dim": 4, "num_classes": num_classes}
        job = (8, 0)
        iterator_names = ["ntrain", "trial"]

        torch.set_default_device("cpu")
        with patch("torch.cuda.is_available", return_value=False):
            payload = run_job(0, job, global_config, bfn_config, iterator_names)

        _, timekeys, train_loss, test_loss, optimizer_name = payload
        self.assertEqual(optimizer_name, "Adam")
        self.assertTrue(torch.isfinite(torch.tensor(train_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(test_loss)))
        self.assertGreaterEqual(len(timekeys), 1)


if __name__ == "__main__":
    unittest.main()
    print('Optimizer and loss function can be dynamically defined.')