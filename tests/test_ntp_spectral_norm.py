import unittest
import os
import sys
import math

import numpy as np
import torch

TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from modelscape.model import MLP
from modelscape.backend.trainloop import train_model


def _to_torch(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _load_cifar10_subset(n_train=256, n_test=64, classes=None):
    try:
        from ImageData import ImageData, preprocess
    except Exception as e:
        raise unittest.SkipTest(f"ImageData not available: {e}")

    data_root = (
        os.getenv("DATASETPATH")
        or os.getenv("DATASET_PATH")
        or os.getenv("CIFAR10_PATH")
        or os.path.join(TEST_ROOT, "data")
    )
    if not os.path.exists(data_root):
        raise unittest.SkipTest(f"CIFAR10 data not found under {data_root}")

    pixel_normalized = False
    if classes is not None:
        imdata = ImageData("cifar10", data_root, classes=classes, onehot=len(classes) != 2)
    else:
        imdata = ImageData("cifar10", data_root, classes=classes, onehot=False)

    X_train, y_train = imdata.get_dataset(n_train, get="train")
    X_train = preprocess(X_train, center=True, greyscale=True, normalize=pixel_normalized)
    X_test, y_test = imdata.get_dataset(n_test, get="test")
    X_test = preprocess(X_test, center=True, greyscale=True, normalize=pixel_normalized)

    X_train, y_train, X_test, y_test = map(_to_torch, (X_train, y_train, X_test, y_test))
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    X = torch.cat((X_train, X_test), dim=0)
    y = torch.cat((y_train, y_test), dim=0).float()

    return X, y


def _make_batch_fn(X, y, bsz, gen):
    def batch_fn(step):
        idx = torch.randint(0, X.shape[0], (bsz,), generator=gen)
        return X[idx], y[idx]
    return batch_fn


def _spectral_norm(tensor):
    return float(torch.linalg.matrix_norm(tensor, ord=2).item())


def _layer_multiplier(layer):
    return float(getattr(layer, "_multiplier", 1.0))


def _scaling_exponent(widths, values):
    x = np.log(np.asarray(widths, dtype=float))
    y = np.log(np.asarray(values, dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def _run_width(width, X, y, steps=10, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bfn = _make_batch_fn(X, y, bsz=64, gen=gen)

    model = MLP(d_in=X.shape[1], width=width, depth=2, d_out=1)

    out = train_model(
        model=model,
        batch_function=bfn,
        lr=1e-2,
        max_iter=steps,
        loss_checkpoints=[0.0],
        gamma=1.0,
        ema_smoother=0.0,
        only_thresholds=True,
        verbose=False,
        mup_param="ntp",
    )

    wrapper = out["model"]
    baseline = wrapper.baseline
    trained = wrapper.model

    w0 = baseline.hidden_layers[0].weight.detach().clone()
    w1 = trained.hidden_layers[0].weight.detach().clone()

    g_hidden = _layer_multiplier(baseline.hidden_layers[0])
    # g_readout = _layer_multiplier(baseline.output_layer)

    init_eff = g_hidden * w0
    # Account for feedforward + gradient prefactors from mupify.
    delta_eff = g_hidden * (w1 - w0)

    return _spectral_norm(init_eff), _spectral_norm(delta_eff)


class TestNTPSpectralNorms(unittest.TestCase):
    def test_ntp_spectral_norms_scale_with_width(self):
        np.random.seed(0)

        X, y = _load_cifar10_subset(n_train=256, n_test=64, classes=[[0], [6]])
        widths = [64, 128, 256]

        init_norms = []
        delta_norms = []
        for w in widths:
            init_n, delta_n = _run_width(w, X, y, steps=10, seed=0)
            init_norms.append(init_n)
            delta_norms.append(delta_n)

        slope_init = _scaling_exponent(widths, init_norms)
        slope_delta = _scaling_exponent(widths, delta_norms)

        self.assertTrue(math.isfinite(slope_init) and math.isfinite(slope_delta))
        # NTP: init stays O(1); delta shrinks with width once prefactors are applied.
        self.assertLess(abs(slope_init), 0.2)
        self.assertLess(slope_delta, -0.1)
        self.assertGreater(slope_delta, -1.0)


if __name__ == "__main__":
    try:
        X, y = _load_cifar10_subset(n_train=256, n_test=64, classes=[[0], [6]])
    except unittest.SkipTest as e:
        print(f"[SKIP] {e}")
        unittest.main()
        raise SystemExit(0)

    widths = [64, 128, 256, 512]
    print("width, init_spec_norm, delta_spec_norm")
    for w in widths:
        init_n, delta_n = _run_width(w, X, y, steps=10, seed=0)
        print(f"{w}, {init_n:.6f}, {delta_n:.6f}")

    unittest.main()
