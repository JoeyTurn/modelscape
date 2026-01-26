import torch.nn as nn
import torch
import copy
import numpy as np
import math

class CenteredWrapper(nn.Module):
    """
    Wrap any nn.Module so that forward(x) returns model(x) - baseline(x),
    where baseline is a frozen snapshot taken at wrap time.
    """
    def __init__(self, model: nn.Module, baseline_dtype=None):
        super().__init__()
        self.model = model
        self.baseline = copy.deepcopy(model)  # snapshot at wrap time
        for p in self.baseline.parameters():
            p.requires_grad = False
        self.baseline.eval()
        self._baseline_dtype = baseline_dtype
        if baseline_dtype is not None:
            self.baseline.to(dtype=baseline_dtype)

    @torch.no_grad()
    def recenter(self):
        """Reset baseline to current model weights (still frozen)."""
        self.baseline.load_state_dict(self.model.state_dict())
        if self._baseline_dtype is not None:
            self.baseline.to(dtype=self._baseline_dtype)
        self.baseline.eval()

    def forward(self, x):
        y = self.model(x)  # grads here
        with torch.inference_mode():
            y0 = self.baseline(x)  # no grads / low mem
        return y - y0


def centeredmodel(model: nn.Module, baseline_dtype=None) -> nn.Module:
    """
    Usage:
        centerednet = centeredmodel(baseline_net)     # wrap for centering
    """
    return CenteredWrapper(model, baseline_dtype=baseline_dtype)


class MLP(nn.Module):
    def __init__(self, d_in=1, width=4096, depth=2, d_out=1, bias=True, nonlinearity=None, forcezeros=False):
        super().__init__()
        self.d_in, self.width, self.depth, self.d_out = d_in, width, depth, d_out

        self.input_layer = nn.Linear(d_in, width, bias)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width, bias) for _ in range(depth - 1)])
        self.output_layer = nn.Linear(width, d_out, bias)
        if forcezeros:
            with torch.no_grad():
                self.output_layer.weight.zero_()
                if self.output_layer.bias is not None:
                    self.output_layer.bias.zero_()
        self.nonlin = nonlinearity if nonlinearity is not None else nn.ReLU()
        
    def forward(self, x):
        h = self.nonlin(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
        out = self.output_layer(h)
        return out

    def get_activations(self, x):
        h_acts = []
        h = self.nonlin(self.input_layer(x))
        h_acts.append(h)
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
            h_acts.append(h)
        h_out = self.output_layer(h)
        return h_acts, h_out


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, img_size=8, width=16, depth=2, d_out=1, bias=True):
        super().__init__()
        if depth < 2:
            raise ValueError("SimpleCNN depth must be >= 2")
        self.in_channels = in_channels
        self.img_size = img_size
        self.width = width
        self.depth = depth
        self.d_out = d_out

        self.input_layer = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=bias)
        self.hidden_layers = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=3, padding=1, bias=bias) for _ in range(depth - 1)]
        )
        self.output_layer = nn.Linear(width * img_size * img_size, d_out, bias=bias)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        h = self.nonlin(self.input_layer(x))
        for layer in self.hidden_layers:
            h = self.nonlin(layer(h))
        h = h.flatten(start_dim=1)
        return self.output_layer(h)


class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, bias=True):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scale = 1.0 / math.sqrt(q.shape[-1])
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        return self.proj(attn @ v)


class SimpleMLPBlock(nn.Module):
    def __init__(self, d_model, mlp_mult=4, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mlp_mult * d_model, bias=bias)
        self.fc2 = nn.Linear(mlp_mult * d_model, d_model, bias=bias)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.nonlin(self.fc1(x)))


class SimpleTransformer(nn.Module):
    def __init__(self, d_in=8, width=16, depth=2, d_out=1, mlp_mult=4, bias=True, seq_len=None):
        super().__init__()
        if depth < 1:
            raise ValueError("SimpleTransformer depth must be >= 1")
        self.d_in = d_in
        self.width = width
        self.depth = depth
        self.d_out = d_out
        self.seq_len = seq_len

        self.input_layer = nn.Linear(d_in, width, bias=bias)
        self.hidden_layers = nn.ModuleList(
            [SimpleTransformerBlock(width, mlp_mult=mlp_mult, bias=bias) for _ in range(depth)]
        )
        self.output_layer = nn.Linear(width, d_out, bias=bias)
        self.nonlin = nn.ReLU()

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.hidden_layers:
            h = block(h)
        h = h.mean(dim=1)
        return self.output_layer(h)


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model, mlp_mult=4, bias=True):
        super().__init__()
        self.mlp = SimpleMLPBlock(d_model, mlp_mult=mlp_mult, bias=bias)
        self.attn = SimpleSelfAttention(d_model, bias=bias)

    def forward(self, x):
        x = self.mlp(x)
        x = self.attn(x)
        return x
