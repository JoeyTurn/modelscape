import unittest
from unittest.mock import patch
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


def first_layer_weight_head(model, **kwargs):
    if hasattr(model, "input_layer") and hasattr(model.input_layer, "weight"):
        w = model.input_layer.weight
    else:
        w = next((p for p in model.parameters() if p.ndim >= 2), None)
        if w is None:
            raise ValueError("Could not find a matrix-like layer weight in model")
    return w.detach().flatten()[:10].cpu().numpy()


def _config2outcome(trace_obj):
    if isinstance(trace_obj, dict) and "config2outcome" in trace_obj:
        return trace_obj["config2outcome"]
    return {}


class TestRandomnessModel(unittest.TestCase):
    def test_trials_generate_different_first_layer_weights(self):
        iterators = [[8], [0, 1]]
        iterator_names = ["ntrain", "trial"]
        global_config = {
            "LR": 1e-2,
            "MAX_ITER": 1,
            "LOSS_CHECKPOINTS": [1e-1],
            "EMA_SMOOTHER": 0.0,
            "ONLYTHRESHOLDS": False,
            "VERBOSE": False,
            "ONLINE": True,
            "N_TEST": 8,
            "SEED": 0,
            "DIM": 4,
            "width": 8,
            "depth": 2,
            "otherreturns": {"w1_head": first_layer_weight_head},
        }
        bfn_config = {"base_bfn": random_bfn, "dim": 4}

        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.cuda.device_count", return_value=0):
            result = run_job_iterator(
                iterators, iterator_names, global_config, bfn_config=bfn_config, use_mp=False
            )

        extras = result["extras"]["w1_head"]
        vals = [torch.as_tensor(v).reshape(-1)[:10] for v in _config2outcome(extras).values()]
        self.assertEqual(len(vals), 2)
        self.assertFalse(torch.allclose(vals[0], vals[1]))


if __name__ == "__main__":
    iterators = [[8], [0, 1]]
    iterator_names = ["ntrain", "trial"]
    global_config = {
        "LR": 1e-2,
        "MAX_ITER": 1,
        "LOSS_CHECKPOINTS": [0.1],
        "EMA_SMOOTHER": 0.0,
        "ONLYTHRESHOLDS": False,
        "VERBOSE": False,
        "ONLINE": True,
        "N_TEST": 8,
        "SEED": 0,
        "DIM": 4,
        "width": 8,
        "depth": 2,
        "otherreturns": {"w1_head": first_layer_weight_head},
    }
    bfn_config = {"base_bfn": random_bfn, "dim": 4}

    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.cuda.device_count", return_value=0):
        result = run_job_iterator(
            iterators, iterator_names, global_config, bfn_config=bfn_config, use_mp=False
        )

    extras = result["extras"]["w1_head"]
    vals = _config2outcome(extras).values()
    for idx, val in enumerate(vals):
        head = torch.as_tensor(val).reshape(-1)[:10].tolist()
        print(f"run[{idx}] W1[:10]:", head)

    unittest.main()
