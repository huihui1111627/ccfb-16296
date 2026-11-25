"""Quantise a whole network layer by layer with sequential calibration.

Layers are converted in execution order.  The calibration inputs of layer
``k`` are taken from the *already quantised* model up to that point, so the
input statistics of deeper layers include the error injected by the shallower
ones (sequential GPTQ-style calibration).

Convolution layers are unrolled to their (out_ch, in_ch*kh*kw) matrix, which
is exactly how they are evaluated, so conv and fully connected layers go
through the same code path.

Nothing here mutates the network passed in: all work happens on copies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .model import Conv2d, im2col
from .quantize import LayerTrace, QParams, quantize_matrix


@dataclass
class QuantConfig:
    bits: int = 4
    group_size: int = 128
    symmetric: bool = False
    order_mode: str = "identity"
    damp: float = 1e-3
    compensation: bool = True
    calib_batch: int = 256

    def tag(self) -> str:
        return (
            f"b{self.bits}_g{self.group_size}_"
            f"{'sym' if self.symmetric else 'asym'}_{self.order_mode}_"
            f"{'gptq' if self.compensation else 'rtn'}"
        )


@dataclass
class LayerResult:
    name: str
    trace: LayerTrace
    qparams: QParams
    weight: np.ndarray          # quantised (dequantised) weight matrix
    bias: np.ndarray
    shape: tuple[int, ...]      # original weight shape
    is_conv: bool
    # output MSE of this layer with the float input (quantised everything
    # before it, this layer float) -> error injected by this layer alone
    out_err_isolated: float = 0.0
    # output MSE with the full quantised stack (input already perturbed)
    out_err_stacked: float = 0.0

    def summary(self) -> dict:
        d = self.trace.summary()
        d["is_conv"] = self.is_conv
        d["weight_shape"] = list(self.shape)
        d["out_err_isolated"] = self.out_err_isolated
        d["out_err_stacked"] = self.out_err_stacked
        return d


@dataclass
class NetResult:
    layers: list[LayerResult]
    config: QuantConfig
    float_accuracy: float = 0.0
    quant_accuracy: float = 0.0
    float_logits_mse: float = 0.0

    def summary(self) -> dict:
        return {
            "config": {
                "bits": self.config.bits,
                "group_size": self.config.group_size,
                "symmetric": self.config.symmetric,
                "order_mode": self.config.order_mode,
                "compensation": self.config.compensation,
            },
            "tag": self.config.tag(),
            "float_accuracy": self.float_accuracy,
            "quant_accuracy": self.quant_accuracy,
            "logits_mse": self.float_logits_mse,
            "layers": [l.summary() for l in self.layers],
        }


def _layer_input_matrix(x: np.ndarray, layer) -> np.ndarray:
    if isinstance(layer, Conv2d):
        return im2col(x, layer.k, layer.k, layer.pad, 1)
    return x.reshape(x.shape[0], -1)


def quantize_net(
    net,
    calib_x: np.ndarray,
    *,
    cfg: QuantConfig,
    keep_hessian: bool = False,
    record_wave: bool = True,
    log=None,
) -> NetResult:
    """Quantise every weight layer of ``net`` in execution order.

    Returns a ``NetResult``; the quantised model can be rebuilt with
    :func:`apply_result` on a fresh copy of ``net``.
    """
    work = copy.deepcopy(net)
    work.set_training(False)
    results: list[LayerResult] = []
    x = np.asarray(calib_x[: cfg.calib_batch], dtype=np.float64)
    for layer in work.layers:
        name = getattr(layer, "weight_name", None)
        if name is None:
            x = layer.forward(x)
            continue
        Xmat = _layer_input_matrix(x, layer)
        W = np.asarray(layer.weight, dtype=np.float64)
        yf = Xmat @ W.T
        Wq, qp, trace = quantize_matrix(
            W,
            Xmat,
            name=name,
            bits=cfg.bits,
            group_size=cfg.group_size,
            symmetric=cfg.symmetric,
            order_mode=cfg.order_mode,
            damp=cfg.damp,
            compensation=cfg.compensation,
            record_wave=record_wave,
            keep_hessian=keep_hessian,
        )
        yq = Xmat @ Wq.T
        results.append(
            LayerResult(
                name=name,
                trace=trace,
                qparams=qp,
                weight=Wq,
                bias=np.array(layer.bias, dtype=np.float64),
                shape=tuple(W.shape),
                is_conv=isinstance(layer, Conv2d),
                out_err_isolated=float(np.mean((yf - yq) ** 2)),
            )
        )
        layer.weight = Wq.copy()
        x = layer.forward(x)
        if log:
            log(
                f"  {name:6s} {trace.n_rows:4d}x{trace.n_cols:<5d} "
                f"obj {trace.loss_plain:.3e} -> {trace.loss_compensated:.3e} "
                f"({trace.loss_plain / max(trace.loss_compensated, 1e-30):.2f}x)"
            )
    work.set_training(True)
    return NetResult(layers=results, config=cfg)


def apply_result(net, result: NetResult) -> None:
    """Copy the quantised weights from ``result`` into ``net`` in place."""
    by_name = {r.name: r for r in result.layers}
    for layer in net.layers:
        name = getattr(layer, "weight_name", None)
        if name is None or name not in by_name:
            continue
        layer.weight = by_name[name].weight.copy()
        layer.bias = by_name[name].bias.copy()


def layer_outputs(net, x: np.ndarray) -> dict[str, np.ndarray]:
    """Output of every weight layer for input batch ``x``."""
    net.set_training(False)
    out: dict[str, np.ndarray] = {}
    for layer in net.layers:
        x = layer.forward(x)
        name = getattr(layer, "weight_name", None)
        if name is not None:
            out[name] = np.array(x, dtype=np.float64)
    return out


def fill_stacked_errors(float_net, quant_net, result: NetResult, x: np.ndarray) -> None:
    """Fill ``out_err_stacked`` by comparing per-layer outputs of both nets."""
    a = layer_outputs(float_net, x)
    b = layer_outputs(quant_net, x)
    for lr in result.layers:
        lr.out_err_stacked = float(np.mean((a[lr.name] - b[lr.name]) ** 2))


def evaluate(net, result: NetResult, x: np.ndarray, y: np.ndarray, float_net) -> dict:
    """Accuracy of the quantised net and its logit MSE against ``float_net``."""
    apply_result(net, result)
    ql = net.logits(x)
    fl = float_net.logits(x)
    qa = float((ql.argmax(axis=1) == y).mean())
    fa = float((fl.argmax(axis=1) == y).mean())
    mse = float(np.mean((ql - fl) ** 2))
    result.float_accuracy = fa
    result.quant_accuracy = qa
    result.float_logits_mse = mse
    return {
        "float_accuracy": fa,
        "quant_accuracy": qa,
        "logits_mse": mse,
    }
