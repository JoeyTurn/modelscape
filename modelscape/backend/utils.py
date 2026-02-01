import numpy as np
import torch
import os
import json
import inspect


def ensure_numpy(x, dtype=np.float64, clone=False):
    """Convert torch.Tensor to numpy array if necessary.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    result = np.asarray(x, dtype=dtype)
    if clone and result is x:
        return result.copy()
    return result


def ensure_torch(x, dtype=torch.float32, device=None, clone=False):
    """
    Convert array-like to torch.Tensor on the desired device/dtype.

    Notes:
      - In multiprocessing, call torch.cuda.set_device(device_id) in each worker.
        Leaving device=None will then use that worker's GPU via 'cuda'.
      - On single-GPU, device=None -> 'cuda' automatically.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")  # current CUDA device (per-process)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)

    if not isinstance(x, torch.Tensor):
        return torch.as_tensor(x, dtype=dtype, device=device)

    # Already a tensor: move/cast only if needed
    needs_move = (x.device != device) or (x.dtype != dtype)
    if needs_move:
        x = x.to(device=device, dtype=dtype, non_blocking=True)

    return x.clone() if clone else x

def tuple_to_numpy(x, dtype=np.float64, clone=False):
    """Recursively convert array-like leaves to NumPy via ensure_numpy.
    Preserves lists/tuples/dicts (namedtuples keep their type).
    Sets become tuples to avoid unhashable ndarrays.
    Leaves scalars/strings/None unchanged.
    """
    # simple leaves
    if x is None or isinstance(x, (str, bytes, bytearray)):
        return x
    if np.isscalar(x):
        return x

    # containers
    if isinstance(x, dict):
        return {k: tuple_to_numpy(v, dtype=dtype, clone=clone) for k, v in x.items()}
    if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
        return type(x)(*(tuple_to_numpy(v, dtype=dtype, clone=clone) for v in x))
    if isinstance(x, tuple):
        return tuple(tuple_to_numpy(v, dtype=dtype, clone=clone) for v in x)
    if isinstance(x, list):
        return [tuple_to_numpy(v, dtype=dtype, clone=clone) for v in x]
    if isinstance(x, set):
        return tuple(tuple_to_numpy(v, dtype=dtype, clone=clone) for v in x)

    try:
        return ensure_numpy(x, dtype=dtype, clone=clone)
    except Exception:
        return x


def _make_trace(var_names):
    if not isinstance(var_names, list):
        var_names = list(var_names)
    if "outcome" in var_names:
        raise ValueError("variable name 'outcome' disallowed")
    return {"var_names": var_names, "config2outcome": {}, "outcome_shape": None}


def _trace_set(trace, config, outcome):
    if not isinstance(trace, dict):
        raise TypeError("trace must be a dict")
    if trace.get("outcome_shape") is None:
        out_array = np.asarray(outcome)
        if not np.issubdtype(out_array.dtype, np.number):
            raise ValueError("measurement outcome must be numeric")
        trace["outcome_shape"] = out_array.shape
    elif np.shape(outcome) != trace["outcome_shape"]:
        raise ValueError(f"outcome shape {np.shape(outcome)} != expected {trace['outcome_shape']}")

    config = (config,) if not isinstance(config, tuple) else config
    var_names = trace.get("var_names")
    if var_names is not None and len(config) != len(var_names):
        raise ValueError(f"len config {len(config)} != num vars {len(var_names)}")
    allowed_types = (int, float, str, tuple, np.integer, np.floating)
    if not all(isinstance(c, allowed_types) for c in config):
        raise ValueError(f"config {config} elements must be one of {allowed_types}")

    config2outcome = trace.setdefault("config2outcome", {})
    if config in config2outcome:
        raise ValueError(f"config {config} already exists. overwriting not supported")
    config2outcome[config] = outcome
    return trace


def trace_deserialize(data):
    if not isinstance(data, dict):
        raise TypeError("trace data must be a dict")
    for key in ("var_names", "config2outcome", "outcome_shape"):
        if key not in data:
            raise ValueError(f"Missing key in serialized data: {key}")
    return {
        "var_names": list(data["var_names"]),
        "config2outcome": dict(data["config2outcome"]),
        "outcome_shape": data["outcome_shape"],
    }


## ---- seeding utils ----

def derive_seed(master_seed, device_id=0):
    """
    Deterministic if master_seed is not None; otherwise uses OS entropy.
    """
    if master_seed is None:
        # non-deterministic (different each run)
        s = (os.getpid() ^ (device_id << 16)) & 0xFFFFFFFF
        return int(s)

    ss = np.random.SeedSequence([int(master_seed), int(device_id)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])

def seed_everything(seed_int, device_id=0, make_generators=True, deterministic=False):
    """
    Set Python/NumPy/Torch (CPU & current CUDA device) RNGs.
    Returns a torch.Generator() seeded the same way if requested.
    """
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_int)  # current device
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if not make_generators:
        return None
    if torch.cuda.is_available():
        gen = torch.Generator(device=f"cuda:{device_id}").manual_seed(seed_int)
    else:
        gen = torch.Generator(device="cpu").manual_seed(seed_int)
    return gen, np.random.default_rng(seed_int)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_config(path):
    """
    Load a configuration file based on its extension.
    Supports .yaml/.yml and .json.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    with open(path, 'r') as f:
        if ext in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(f)
        if ext == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported configuration file extension: {ext}")


def _extract_kwargs_for(target, kwargs):
    """
    `target` can be:
      - a class (e.g. MLP)
      - a function (e.g. polynomial_batch_fn)

    Returns:
      used_kwargs, rest_kwargs
    """
    if inspect.isclass(target):
        # classes
        sig = inspect.signature(target.__init__)
        ignore = {"self"}
    else:
        # functions
        sig = inspect.signature(target)
        ignore = set()

    valid = set(sig.parameters.keys()) - ignore

    used = {}
    rest = {}
    for k, v in kwargs.items():
        if k in valid:
            used[k] = v
        else:
            rest[k] = v
    return used, rest
