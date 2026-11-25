"""A small convolutional + fully connected network, implemented in numpy only.

Architecture (input 1x32x32):

    conv 3x3 (1->8)   + ReLU + maxpool 2  ->  8x16x16
    conv 3x3 (8->16)  + ReLU + maxpool 2  -> 16x8x8
    conv 3x3 (16->32) + ReLU + maxpool 2  -> 32x4x4
    flatten 512 -> linear 64 -> ReLU -> linear 8

Every layer that holds a weight matrix is "quantizable": a conv layer is
unrolled into one (out_ch, in_ch*kh*kw) matrix (plus bias) exactly as it is
evaluated, so the same quantizer code handles conv and fully connected layers.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# im2col helpers
# ---------------------------------------------------------------------------


def im2col(x: np.ndarray, kh: int, kw: int, pad: int, stride: int) -> np.ndarray:
    """(N,C,H,W) -> (N*OH*OW, C*kh*kw) patch matrix."""
    n, c, h, w = x.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, c, kh, kw, oh, ow), dtype=x.dtype)
    for i in range(kh):
        i_end = i + stride * oh
        for j in range(kw):
            j_end = j + stride * ow
            cols[:, :, i, j, :, :] = xp[:, :, i:i_end:stride, j:j_end:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, c * kh * kw)


def col2im(
    cols: np.ndarray, x_shape: tuple[int, int, int, int], kh: int, kw: int, pad: int, stride: int
) -> np.ndarray:
    n, c, h, w = x_shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    cols6 = cols.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        i_end = i + stride * oh
        for j in range(kw):
            j_end = j + stride * ow
            xp[:, :, i:i_end:stride, j:j_end:stride] += cols6[:, :, i, j, :, :]
    if pad == 0:
        return xp
    return xp[:, :, pad:-pad, pad:-pad]


def _pool2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, c, h, w = x.shape
    xr = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
    idx = xr.argmax(axis=-1)
    return xr.max(axis=-1), idx


def _unpool2(dout: np.ndarray, idx: np.ndarray, shape: tuple[int, int, int, int]) -> np.ndarray:
    n, c, h, w = shape
    oh, ow = h // 2, w // 2
    flat = np.zeros((n, c, oh, ow, 4), dtype=dout.dtype)
    np.put_along_axis(flat, idx[..., None], dout[..., None], axis=-1)
    return flat.reshape(n, c, oh, ow, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class Conv2d:
    def __init__(self, cin: int, cout: int, k: int = 3, pad: int = 1, rng=None):
        self.cin, self.cout, self.k, self.pad = cin, cout, k, pad
        scale = np.sqrt(2.0 / (cin * k * k))
        self.weight = (rng.normal(0.0, scale, size=(cout, cin * k * k))).astype(np.float64)
        self.bias = np.zeros(cout, dtype=np.float64)
        self.weight_name = None
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        cols = im2col(x, self.k, self.k, self.pad, 1)
        oh = ow = h
        out = cols @ self.weight.T + self.bias
        out = out.reshape(n, oh, ow, self.cout).transpose(0, 3, 1, 2)
        if self.training:
            self._cache = (cols, x.shape, out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        cols, x_shape, out = self._cache
        dflat = dout.transpose(0, 2, 3, 1).reshape(-1, self.cout)
        self.grad_weight = dflat.T @ cols
        self.grad_bias = dflat.sum(axis=0)
        dcols = dflat @ self.weight
        self._cache = None
        return col2im(dcols, x_shape, self.k, self.k, self.pad, 1)

    training = True


class MaxPool2:
    def __init__(self):
        self._cache = None

    def forward(self, x):
        out, idx = _pool2(x)
        if self.training:
            self._cache = (idx, x.shape)
        return out

    def backward(self, dout):
        idx, shape = self._cache
        self._cache = None
        return _unpool2(dout, idx, shape)

    training = True


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        if self.training:
            self._mask = x > 0
        return np.maximum(x, 0.0)

    def backward(self, dout):
        return dout * self._mask

    training = True


class Linear:
    def __init__(self, nin: int, nout: int, rng=None):
        scale = np.sqrt(2.0 / nin)
        self.weight = rng.normal(0.0, scale, size=(nout, nin)).astype(np.float64)
        self.bias = np.zeros(nout, dtype=np.float64)
        self.weight_name = None
        self._cache = None

    def forward(self, x):
        if self.training:
            self._cache = x
        return x @ self.weight.T + self.bias

    def backward(self, dout):
        x = self._cache
        self.grad_weight = dout.T @ x
        self.grad_bias = dout.sum(axis=0)
        self._cache = None
        return dout @ self.weight

    training = True


class Flatten:
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)

    training = True


class Net:
    """Container with named layers; also exposes quantizable weight matrices."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.layers = []
        self._add(Conv2d(1, 8, 3, 1, rng), "conv1")
        self._add(ReLU())
        self._add(MaxPool2())
        self._add(Conv2d(8, 16, 3, 1, rng), "conv2")
        self._add(ReLU())
        self._add(MaxPool2())
        self._add(Conv2d(16, 32, 3, 1, rng), "conv3")
        self._add(ReLU())
        self._add(MaxPool2())
        self._add(Flatten())
        self._add(Linear(32 * 4 * 4, 64, rng), "fc1")
        self._add(ReLU())
        self._add(Linear(64, 8, rng), "fc2")

    def _add(self, layer, weight_name: str | None = None):
        if weight_name is not None:
            layer.weight_name = weight_name
        self.layers.append(layer)

    # -- parameters -------------------------------------------------------
    def weight_layers(self):
        return [l for l in self.layers if getattr(l, "weight_name", None) is not None]

    def params(self):
        for l in self.weight_layers():
            yield l.weight_name + ".weight", l.weight
            yield l.weight_name + ".bias", l.bias

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params()}

    def load_state_dict(self, sd: dict[str, np.ndarray]) -> None:
        for l in self.weight_layers():
            l.weight = np.array(sd[l.weight_name + ".weight"], dtype=np.float64)
            l.bias = np.array(sd[l.weight_name + ".bias"], dtype=np.float64)

    def set_training(self, mode: bool):
        for l in self.layers:
            l.training = mode

    # -- forward ----------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        for l in self.layers:
            x = l.forward(x)
        return x

    def forward_upto(self, x: np.ndarray, weight_name: str) -> np.ndarray:
        """Input seen by the layer ``weight_name`` (its conv/linear input)."""
        for l in self.layers:
            if getattr(l, "weight_name", None) == weight_name:
                return x
            x = l.forward(x)
        raise KeyError(weight_name)

    def collect_inputs(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Return the input matrix of each weight layer for a batch.

        Conv layers are unrolled to (N*OH*OW, C*kh*kw); linear layers keep
        (N, nin).  This is exactly the matrix the quantizer sees.
        """
        out: dict[str, np.ndarray] = {}
        self.set_training(False)
        for l in self.layers:
            name = getattr(l, "weight_name", None)
            if name is not None:
                if isinstance(l, Conv2d):
                    n, c, h, w = x.shape
                    out[name] = im2col(x, l.k, l.k, l.pad, 1)
                else:
                    out[name] = x.copy()
            x = l.forward(x)
        return out

    def logits(self, x: np.ndarray, batch: int = 256) -> np.ndarray:
        self.set_training(False)
        parts = []
        for i in range(0, x.shape[0], batch):
            parts.append(self.forward(x[i : i + batch]))
        self.set_training(True)
        return np.concatenate(parts, axis=0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    p = softmax(logits)
    n = y.shape[0]
    loss = -np.log(np.maximum(p[np.arange(n), y], 1e-12)).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return float(loss), d / n


def accuracy(net: Net, x: np.ndarray, y: np.ndarray) -> float:
    return float((net.logits(x).argmax(axis=1) == y).mean())


# ---------------------------------------------------------------------------
# Tiny Adam implementation
# ---------------------------------------------------------------------------


class Adam:
    def __init__(self, params, lr: float = 3e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads, lr: float | None = None):
        lr = self.lr if lr is None else lr
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            p -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def train(
    net: Net,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 12,
    batch: int = 64,
    lr: float = 3e-3,
    seed: int = 0,
    log=print,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    params, grads = [], []
    for l in net.weight_layers():
        params.append(l.weight)
        params.append(l.bias)
        grads.append(None)
        grads.append(None)
    opt = Adam(params, lr=lr)
    n = x.shape[0]
    steps_per_epoch = n // batch
    total_steps = epochs * steps_per_epoch
    step = 0
    history = []
    net.set_training(True)
    for ep in range(epochs):
        perm = rng.permutation(n)
        run_loss = 0.0
        run_correct = 0
        for it in range(steps_per_epoch):
            idx = perm[it * batch : (it + 1) * batch]
            xb, yb = x[idx], y[idx]
            logits = net.forward(xb)
            loss, dlogits = cross_entropy(logits, yb)
            d = dlogits
            for l in reversed(net.layers):
                d = l.backward(d)
            gl = []
            for l in net.weight_layers():
                gl.append(l.grad_weight)
                gl.append(l.grad_bias)
            # cosine decay
            frac = step / max(1, total_steps - 1)
            cur_lr = lr * 0.5 * (1.0 + np.cos(np.pi * frac))
            opt.step(gl, cur_lr)
            run_loss += loss * len(idx)
            run_correct += int((logits.argmax(axis=1) == yb).sum())
            step += 1
        hist = {"epoch": ep + 1, "loss": run_loss / (steps_per_epoch * batch), "train_acc": run_correct / (steps_per_epoch * batch)}
        history.append(hist)
        log(f"epoch {ep + 1:2d}/{epochs}  loss {hist['loss']:.4f}  train_acc {hist['train_acc']:.4f}")
    net.set_training(True)
    return history
