"""Export a quantised model as packed low-bit weights plus numpy inference code.

The bundle is a single ``.npz`` file holding

    manifest          JSON (stored as a uint8 array): op list, quant params
    <name>.codes      bit-packed integer codes, one row per output channel
    <name>.bias       float64 bias
    <name>.scale      float64 per-group scale   (rows, n_groups)
    <name>.zero       uint16 per-group zero point (rows, n_groups)
    <name>.shape      original weight shape
    norm.mean/std     float64 input normalisation

The op list is derived from the network itself (conv / relu / maxpool /
flatten / linear), so the exported ``forward`` mirrors the trained model's
pipeline exactly instead of assuming a fixed architecture.  Dequantisation in
the bundle uses the same expression as the simulator, and everything is
float64, so the bundle's logits equal the simulated quantised model's logits
element for element (verified in the test-suite).

Codes are bit-packed little-endian within each byte, ``bits`` codes per byte
(8 bits: one byte per code).
"""

from __future__ import annotations

import json

import numpy as np

from .model import Conv2d, Flatten, Linear, MaxPool2, ReLU, Net

ARCH_TAG = "wildlife-quant-v1"


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack uint codes (rows, cols) into a uint8 stream, little-endian."""
    codes = np.asarray(codes, dtype=np.uint16)
    if bits == 8:
        return codes.astype(np.uint8)
    per_byte = 8 // bits
    rows, cols = codes.shape
    pad = (-cols) % per_byte
    if pad:
        codes = np.pad(codes, ((0, 0), (0, pad)), constant_values=0)
    out = np.zeros((rows, codes.shape[1] // per_byte), dtype=np.uint8)
    for k in range(per_byte):
        out |= ((codes[:, k::per_byte] & ((1 << bits) - 1)).astype(np.uint8)) << (k * bits)
    return out


def unpack_codes(packed: np.ndarray, bits: int, cols: int) -> np.ndarray:
    """Inverse of :func:`pack_codes`."""
    packed = np.asarray(packed)
    if bits == 8:
        return packed[:, :cols].astype(np.uint16)
    per_byte = 8 // bits
    rows = packed.shape[0]
    out = np.zeros((rows, packed.shape[1] * per_byte), dtype=np.uint16)
    mask = (1 << bits) - 1
    for k in range(per_byte):
        out[:, k::per_byte] = ((packed >> (k * bits)) & mask).astype(np.uint16)
    return out[:, :cols]


def codes_for(weight: np.ndarray, qparams) -> np.ndarray:
    s, z = qparams.expand(weight.shape[1])
    return np.clip(np.round(weight / s) + z, 0, qparams.qmax).astype(np.uint16)


def dequantise(weight_shape, codes, scale, zero, group_size) -> np.ndarray:
    """Rebuild the float weight from codes -- shared by bundle and checks."""
    n_cols = int(weight_shape[1]) if len(weight_shape) > 1 else int(weight_shape[0])
    s = np.repeat(scale, group_size, axis=1)[:, :n_cols].astype(np.float64)
    z = np.repeat(zero, group_size, axis=1)[:, :n_cols].astype(np.float64)
    return ((codes.astype(np.float64) - z) * s).reshape(weight_shape)


# ---------------------------------------------------------------------------
# Op list
# ---------------------------------------------------------------------------


def _op_list(net: Net) -> list[dict]:
    ops: list[dict] = []
    for layer in net.layers:
        name = getattr(layer, "weight_name", None)
        if isinstance(layer, Conv2d):
            ops.append(
                {
                    "op": "conv",
                    "name": name,
                    "cin": layer.cin,
                    "cout": layer.cout,
                    "k": layer.k,
                    "pad": layer.pad,
                    "weight_shape": list(np.asarray(layer.weight).shape),
                }
            )
        elif isinstance(layer, Linear):
            ops.append(
                {
                    "op": "linear",
                    "name": name,
                    "nin": int(np.asarray(layer.weight).shape[1]),
                    "nout": int(np.asarray(layer.weight).shape[0]),
                    "weight_shape": list(np.asarray(layer.weight).shape),
                }
            )
        elif isinstance(layer, ReLU):
            ops.append({"op": "relu"})
        elif isinstance(layer, MaxPool2):
            ops.append({"op": "maxpool", "size": 2})
        elif isinstance(layer, Flatten):
            ops.append({"op": "flatten"})
        else:  # pragma: no cover - unknown layer type
            raise TypeError(f"cannot export layer {type(layer).__name__}")
    return ops


# ---------------------------------------------------------------------------
# Bundle writing
# ---------------------------------------------------------------------------


def export_bundle(
    path: str,
    net: Net,
    result,
    *,
    norm_mean: float,
    norm_std: float,
    class_names: list[str],
    extra: dict | None = None,
) -> dict:
    """Write a self-contained ``.npz`` bundle; return the manifest."""
    by_name = {r.name: r for r in result.layers}
    ops = _op_list(net)
    manifest: dict = {
        "arch": ARCH_TAG,
        "input_shape": [1, 32, 32],
        "norm_mean": float(norm_mean),
        "norm_std": float(norm_std),
        "classes": list(class_names),
        "config": {
            "bits": int(result.config.bits),
            "group_size": int(result.config.group_size),
            "symmetric": bool(result.config.symmetric),
            "order_mode": result.config.order_mode,
            "compensation": bool(result.config.compensation),
        },
        "float_accuracy": float(result.float_accuracy),
        "quant_accuracy": float(result.quant_accuracy),
        "ops": [],
    }
    arrays: dict[str, np.ndarray] = {}
    for op in ops:
        if "name" not in op:
            manifest["ops"].append(op)
            continue
        name = op["name"]
        r = by_name[name]
        qp = r.qparams
        bits = qp.bits
        packed = pack_codes(codes_for(np.asarray(r.weight), qp), bits)
        arrays[f"{name}.codes"] = packed
        arrays[f"{name}.bias"] = np.asarray(r.bias, dtype=np.float64)
        arrays[f"{name}.scale"] = qp.scale.astype(np.float64)
        arrays[f"{name}.zero"] = qp.zero.astype(np.uint16)
        entry = dict(op)
        entry.update(
            {
                "bits": bits,
                "group_size": int(qp.group_size),
                "symmetric": bool(qp.symmetric),
                "n_rows": int(np.asarray(r.weight).shape[0]),
                "n_cols": int(np.asarray(r.weight).shape[1]),
                "n_groups": int(qp.n_groups),
                "packed_bytes": int(packed.shape[1]),
            }
        )
        manifest["ops"].append(entry)
    if extra:
        manifest["extra"] = extra
    arrays["manifest"] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)
    return manifest


# ---------------------------------------------------------------------------
# Inference code shipped with the bundle
# ---------------------------------------------------------------------------

INFER_PY = '''\
"""Reference inference for an exported low-bit bundle.  Requires only numpy.

    import infer, numpy as np
    z, manifest = infer.load("model.npz")
    logits = infer.forward(z, manifest, raw_images)   # (N,1,32,32) in [0,1]
"""
import json
import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    manifest = json.loads(bytes(z["manifest"].tolist()).decode("utf-8"))
    return z, manifest


def unpack_codes(packed, bits, cols):
    packed = np.asarray(packed)
    if bits == 8:
        return packed[:, :cols].astype(np.uint16)
    per_byte = 8 // bits
    rows = packed.shape[0]
    out = np.zeros((rows, packed.shape[1] * per_byte), dtype=np.uint16)
    mask = (1 << bits) - 1
    for k in range(per_byte):
        out[:, k::per_byte] = ((packed >> (k * bits)) & mask).astype(np.uint16)
    return out[:, :cols]


def dequantise(entry, codes, scale, zero):
    n_rows = entry["n_rows"]
    n_cols = entry["n_cols"]
    gs = entry["group_size"]
    s = np.repeat(scale, gs, axis=1)[:, :n_cols]
    z = np.repeat(zero, gs, axis=1)[:, :n_cols]
    return ((codes.astype(np.float64) - z) * s).reshape(entry["weight_shape"])


def load_weights(z, manifest):
    out = {}
    for op in manifest["ops"]:
        name = op.get("name")
        if name is None:
            continue
        codes = unpack_codes(z[name + ".codes"], op["bits"], op["n_cols"])
        out[name] = (
            dequantise(op, codes, z[name + ".scale"], z[name + ".zero"]),
            z[name + ".bias"].astype(np.float64),
        )
    return out


def im2col(x, k, pad, stride=1):
    n, c, h, w = x.shape
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((n, c, k, k, oh, ow), dtype=x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j, :, :] = xp[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, c * k * k)


def maxpool2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def forward(z, manifest, images):
    """images: raw (N, 1, 32, 32) samples; normalisation comes from the bundle."""
    weights = load_weights(z, manifest)
    x = (np.asarray(images, dtype=np.float64) - manifest["norm_mean"]) / manifest["norm_std"]
    for op in manifest["ops"]:
        kind = op["op"]
        if kind == "conv":
            w, b = weights[op["name"]]
            cols = im2col(x, op["k"], op["pad"])
            n = x.shape[0]
            oh = x.shape[2]
            ow = x.shape[3]
            cout = w.shape[0]
            y = cols @ w.reshape(cout, -1).T + b
            x = y.reshape(n, oh, ow, cout).transpose(0, 3, 1, 2)
        elif kind == "linear":
            w, b = weights[op["name"]]
            x = x.reshape(x.shape[0], -1) @ w.T + b
        elif kind == "relu":
            x = np.maximum(x, 0.0)
        elif kind == "maxpool":
            x = maxpool2(x)
        elif kind == "flatten":
            x = x.reshape(x.shape[0], -1)
        else:
            raise ValueError("unknown op " + kind)
    return x
'''


def write_inference_code(path: str) -> None:
    with open(path, "w") as f:
        f.write(INFER_PY)


def bundle_forward_code():
    """Return the bundle's ``forward`` as a live module (float64 reference)."""
    import types

    mod = types.ModuleType("bundle_infer")
    exec(INFER_PY, mod.__dict__)
    return mod
