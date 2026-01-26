import numpy as np
import torch

import torch.multiprocessing as mp
from modelscape.backend.cli import parse_args
from modelscape.backend.job_iterator import main as run_job_iterator
from modelscape.backend.utils import ensure_torch

import os, sys
from FileManager import FileManager
import torch.nn as nn

def general_batch_fn(X_total, y_total, X=None, y=None, bsz=128,
                     gen=None, **kwargs):
    def batch_fn(step: int, X=X, y=y):
        if (X is not None) and (y is not None):
            X = ensure_torch(X)
            y = ensure_torch(y)
            return X, y
        with torch.no_grad():
            N_total = X_total.shape[0]
            indices = torch.randint(0, N_total, (bsz,), generator=gen, device=gen.device)
            X_batch = ensure_torch(X_total.to(gen.device)[indices])
            y_batch = ensure_torch(y_total.to(gen.device)[indices])
            return X_batch, y_batch
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

if __name__ == "__main__":

    args = parse_args() #default args

    # Set any args that we want to differ
    args.ONLINE = False
    args.N_TRAIN=4000
    args.N_TEST=1000
    args.N_TOT = args.N_TEST+args.N_TRAIN
    args.CLASSES = [[0], [6]]
    args.NORMALIZED = True
    args.NUM_TRIALS = 2
    args.N_SAMPLES = [1024]
    args.GAMMA = [0.1, 1, 10]
    args.MAX_ITER = 1e2
    args.LR = 2e-1
    args.LOSS_CHECKPOINTS = 0.1
    args.MODEL_CLASS = SimpleResNet
    args.OPTIMIZER_CLASS = torch.optim.Adam
    args.LOSS_CLASS = nn.CrossEntropyLoss
    args.width=64
    args.num_blocks=3

    iterators = [args.N_SAMPLES, range(args.NUM_TRIALS), args.GAMMA]
    iterator_names = ["ntrain", "trial", "GAMMA"]
    
    datapath = os.getenv("DATASETPATH") #datapath = os.path.join(os.getenv(...))
    exptpath = os.getenv("EXPTPATH") #same here
    if datapath is None:
        raise ValueError("must set $DATASETPATH environment variable")
    if exptpath is None:
        raise ValueError("must set $EXPTPATH environment variable")
    expt_dir = os.path.join(exptpath, "example_folder", "example_resnet_run", "synthetic")

    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    expt_fm = FileManager(expt_dir)
    print(f"Working in directory {expt_dir}.")


    from ImageData import ImageData, preprocess
    PIXEL_NORMALIZED =  False # Don't normalize pixels, normalize samples
    classes = args.CLASSES
    normalized = args.NORMALIZED
    use_grayscale = True

    if classes is not None:
        imdata = ImageData('cifar10', "../data", classes=classes, onehot=len(classes)!=2)
    else:
        imdata = ImageData('cifar10', "../data", classes=classes, onehot=False)
    X_train, y_train = imdata.get_dataset(args.N_TRAIN, get='train')
    X_test, y_test = imdata.get_dataset(args.N_TEST, get='test')

    X_train = ensure_torch(X_train)
    X_test = ensure_torch(X_test)
    if X_train.ndim == 4 and X_train.shape[1] not in (1, 3) and X_train.shape[-1] in (1, 3):
        X_train = X_train.permute(0, 3, 1, 2)
        X_test = X_test.permute(0, 3, 1, 2)

    _, c, h, w = X_train.shape
    X_train = preprocess(X_train, center=True, grayscale=use_grayscale, normalize=PIXEL_NORMALIZED)
    X_test = preprocess(X_test, center=True, grayscale=use_grayscale, normalize=PIXEL_NORMALIZED)
    if use_grayscale:
        X_train = X_train.view(-1, 1, h, w)
        X_test = X_test.view(-1, 1, h, w)
    else:
        X_train = X_train.view(-1, c, h, w)
        X_test = X_test.view(-1, c, h, w)

    X_train, y_train, X_test, y_test = map(ensure_torch, (X_train, y_train, X_test, y_test))
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    X_train, y_train, X_test, y_test = [t/torch.linalg.norm(t) for t in (X_train, y_train, X_test, y_test)] if normalized else (X_train, y_train, X_test, y_test)
    if normalized:
        X_train *= args.N_TRAIN**(0.5); X_test *= args.N_TEST**(0.5)
        y_train *= args.N_TRAIN**(0.5); y_test *= args.N_TEST**(0.5)
    X_full = torch.cat((X_train, X_test), dim=0)
    y_full = torch.cat((y_train, y_test), dim=0)
    X_full_flat = X_full.flatten(start_dim=1)
    data_eigvals = torch.linalg.svdvals(X_full_flat)**2
    data_eigvals /= data_eigvals.sum()

    U, lambdas, Vt = torch.linalg.svd(X_full_flat, full_matrices=False)
    dim = X_full_flat.shape[1]
    args.DIM = dim

    args.in_channels = X_full.shape[1]
    
    bfn_config = dict(X_total = X_full, y_total = y_full, base_bfn=general_batch_fn)
    del X_full, y_full   

    global_config = args.__dict__.copy()

    grabs = {}
    global_config.update({"otherreturns": grabs})
    
    mp.set_start_method("spawn", force=True)
    
    result = run_job_iterator(iterators, iterator_names, global_config, bfn_config=bfn_config)
    print(f"Results saved to {expt_dir}")
    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()