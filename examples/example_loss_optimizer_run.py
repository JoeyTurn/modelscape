import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import torch.multiprocessing as mp
from modelscape.backend.cli import parse_args
from modelscape.backend.job_iterator import main as run_job_iterator
from modelscape.backend.utils import ensure_torch

import os, sys
from FileManager import FileManager

try:
    from mupify import mupify, rescale
    _MUPIFY_AVAILABLE = True
except Exception as _mupify_err:
    mupify = None
    rescale = None
    _MUPIFY_AVAILABLE = False
    _MUPIFY_IMPORT_ERROR = _mupify_err


class Muon(optim.Optimizer):
    """
    Minimal momentum SGD-style optimizer for example usage.
    """
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.add_(d_p, alpha=-lr)
        return loss


def _post_init_mupify(model, opt, gamma=1.0, mup_param="mup", **_):
    mupify(model, opt, param=mup_param)
    rescale(model, gamma)
    return model, opt


def _format_labels(y, loss_name, d_out):
    if loss_name == "MSELoss":
        if d_out == 1:
            return y.float().unsqueeze(-1)
        return F.one_hot(y.long(), num_classes=d_out).float()
    return y.long()


def general_batch_fn(X_total, y_total, loss_name, d_out, X=None, y=None, bsz=128, gen=None, **kwargs):
    def batch_fn(step: int, X=X, y=y):
        if (X is not None) and (y is not None):
            X_fixed = ensure_torch(X)
            y_fixed = _format_labels(ensure_torch(y), loss_name, d_out)
            return X_fixed, y_fixed
        with torch.no_grad():
            N_total = X_total.shape[0]
            indices = torch.randint(0, N_total, (bsz,), generator=gen, device=gen.device)
            X_batch = ensure_torch(X_total.to(gen.device)[indices])
            y_batch = _format_labels(ensure_torch(y_total.to(gen.device)[indices]), loss_name, d_out)
            return X_batch, y_batch
    return batch_fn


if __name__ == "__main__":

    args = parse_args() #default args

    # Set any args that we want to differ
    args.ONLINE = False
    args.N_TRAIN = 4000
    args.N_TEST = 1000
    args.N_TOT = args.N_TEST + args.N_TRAIN
    args.NUM_TRIALS = 1
    args.N_SAMPLES = [1024]
    args.GAMMA = 1.0
    args.MAX_ITER = 5e1
    args.LR = 1e3
    args.LOSS_CHECKPOINTS = 0.01#1.0
    args.VERBOSE = True
    args.EMA_SMOOTHER = 0.0

    args.loss = "CrossEntropyLoss"
    args.optimizer_class = Muon
    args.momentum = 0.9
    args.weight_decay = 0.0

    args.CLASSES = None
    num_classes = 10
    args.d_out = num_classes

    if _MUPIFY_AVAILABLE:
        args.post_init_fn = _post_init_mupify
        args.mup_param = "mup"

    iterators = [args.N_SAMPLES, range(args.NUM_TRIALS)]
    iterator_names = ["ntrain", "trial"]

    datapath = os.getenv("DATASETPATH")
    exptpath = os.getenv("EXPTPATH")
    if datapath is None:
        raise ValueError("must set $DATASETPATH environment variable")
    if exptpath is None:
        raise ValueError("must set $EXPTPATH environment variable")
    expt_dir = os.path.join(exptpath, "example_folder", "example_mlp_run", "loss_optimizer")

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    expt_fm = FileManager(expt_dir)
    print(f"Working in directory {expt_dir}.")

    from ImageData import ImageData, preprocess
    PIXEL_NORMALIZED = False
    classes = args.CLASSES

    if classes is not None:
        imdata = ImageData("cifar10", "../data", classes=classes, onehot=False)
    else:
        imdata = ImageData("cifar10", "../data", classes=classes, onehot=False)

    X_train, y_train = imdata.get_dataset(args.N_TRAIN, get="train")
    X_train = preprocess(X_train, center=True, greyscale=True, normalize=PIXEL_NORMALIZED)
    X_test, y_test = imdata.get_dataset(args.N_TEST, get="test")
    X_test = preprocess(X_test, center=True, greyscale=True, normalize=PIXEL_NORMALIZED)
    X_train, y_train, X_test, y_test = map(ensure_torch, (X_train, y_train, X_test, y_test))
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    normalized = True
    if normalized:
        X_train = X_train / torch.linalg.norm(X_train)
        X_test = X_test / torch.linalg.norm(X_test)

    X_full = torch.cat((X_train, X_test), dim=0)
    y_full = torch.cat((y_train, y_test), dim=0)

    bfn_config = dict(
        X_total=X_full,
        y_total=y_full,
        loss_name=args.loss,
        d_out=args.d_out,
        base_bfn=general_batch_fn,
    )
    del X_full, y_full

    global_config = args.__dict__.copy()

    grabs = {}
    global_config.update({"otherreturns": grabs})

    mp.set_start_method("spawn", force=True)

    result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config)
    print(f"Results saved to {expt_dir}")
    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()
