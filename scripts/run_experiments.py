"""Run the quantisation experiments and write everything the web page shows.

    python scripts/run_experiments.py [--quick]

Outputs
-------
out/model.npz              trained float model (from train_model.py)
out/experiments.json       all figures for the web page
out/export/<tag>/bundle.npz  packed low-bit bundle
out/export/<tag>/infer.py    inference code shipped with it

The run is deterministic: model seed, data seeds, calibration subset and
quantisation order are all fixed, and the GPTQ path has no randomness.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from quantlib import export
from quantlib.data import CLASS_NAMES, make_splits
from quantlib.model import Net
from quantlib.netquant import (
    QuantConfig,
    evaluate,
    fill_stacked_errors,
    quantize_net,
)

from train_model import build


OUT = os.path.join(ROOT, "out")
BITS = [8, 4, 3]
GROUP_SIZE = 32
ORDERS = ["identity", "diag", "random"]
MAX_CALIB = 256


def load_model() -> tuple[Net, dict]:
    res = build(force=False)
    return res["net"], res["info"]


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _compact(obj) -> str:
    """JSON with one record per line so the file stays readable and small."""
    if isinstance(obj, list):
        return "[" + ",\n".join(_compact(v) for v in obj) + "]"
    if isinstance(obj, dict):
        return "{" + ", ".join(json.dumps(k) + ": " + _compact(v) for k, v in obj.items()) + "}"
    return json.dumps(obj)


def _downsample(a: np.ndarray, max_rows: int, max_cols: int) -> list:
    """Block-mean a 2-D array down to at most max_rows x max_cols."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return []
    r, c = a.shape
    rs = max(1, int(np.ceil(r / max_rows)))
    cs = max(1, int(np.ceil(c / max_cols)))
    out = []
    for i in range(0, r, rs):
        row = []
        for j in range(0, c, cs):
            block = a[i : i + rs, j : j + cs]
            row.append(round(float(block.mean()), 6))
        out.append(row)
    return out


def _r(x, nd=6):
    """Round floats so the JSON stays small but keeps every visible digit."""
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return 0.0 if v == 0 else float(f"{v:.{nd}g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def layer_record(results, *, detail: str = "summary") -> list[dict]:
    """Serialisable per-layer record.

    ``detail="summary"`` keeps shapes, objective scalars, per-group scales and
    zero points.  ``detail="trace"`` adds the per-column arrays and the
    down-sampled compensation matrix used by the visualisation.
    """
    recs = []
    for r in results.layers:
        tr = r.trace
        rec = {
            "name": r.name,
            "is_conv": r.is_conv,
            "shape": list(r.shape),
            "n_rows": tr.n_rows,
            "n_cols": tr.n_cols,
            "bits": tr.bits,
            "symmetric": tr.symmetric,
            "group_size": tr.group_size,
            "n_groups": tr.qparams.n_groups,
            "order_mode": tr.order_mode,
            "loss_compensated": _r(tr.loss_compensated),
            "loss_plain": _r(tr.loss_plain),
            "loss_ratio": _r(tr.loss_plain / max(tr.loss_compensated, 1e-300)),
            "out_mse_compensated": _r(tr.out_mse_compensated),
            "out_mse_plain": _r(tr.out_mse_plain),
            "out_err_isolated": _r(r.out_err_isolated),
            "out_err_stacked": _r(r.out_err_stacked),
            # first output row of every group: enough to show the trend
            "scale": [[_r(v) for v in row] for row in tr.qparams.scale[:4]],
            "zero": [[int(v) for v in row] for row in tr.qparams.zero[:4]],
        }
        if detail == "trace":
            rec.update(
                {
                    "order": [int(v) for v in tr.order],
                    "inc_loss": [_r(v) for v in tr.inc_loss],
                    "rounding_err": [_r(v) for v in tr.rounding_err],
                    "correction_norm": [_r(v) for v in tr.correction_norm],
                    "received_correction": [_r(v) for v in tr.received_correction],
                    "final_err_by_col": [_r(v) for v in tr.final_err_by_col],
                    "final_err_weighted": [_r(v) for v in tr.final_err_weighted],
                    "objective_curve": [_r(v) for v in tr.objective_curve],
                    "wave": _downsample(tr.wave, 96, 96) if tr.wave is not None else [],
                    "wave_stride": tr.wave_stride,
                }
            )
        recs.append(rec)
    return recs


def run_bits_sweep(net, d, calib: int) -> dict:
    """Accuracy for every bit width, GPTQ and round-to-nearest, both grids."""
    x_test, y_test = d["test"]
    out = []
    for bits in BITS:
        for symmetric in (False, True):
            row = {"bits": bits, "symmetric": symmetric}
            for comp, tag in ((True, "gptq"), (False, "rtn")):
                cfg = QuantConfig(
                    bits=bits,
                    group_size=GROUP_SIZE,
                    symmetric=symmetric,
                    order_mode="identity",
                    compensation=comp,
                    calib_batch=calib,
                )
                t0 = time.time()
                res = quantize_net(net, d["calib"][0], cfg=cfg, record_wave=False)
                qnet = copy.deepcopy(net)
                ev = evaluate(qnet, res, x_test, y_test, net)
                fill_stacked_errors(net, qnet, res, d["calib"][0][:128])
                row[tag] = {
                    "accuracy": _r(ev["quant_accuracy"]),
                    "logits_mse": _r(ev["logits_mse"]),
                    "seconds": _r(time.time() - t0, 3),
                    "layers": layer_record(res),
                }
            out.append(row)
    return {"sweep": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller calibration set")
    args = ap.parse_args()
    calib = 128 if args.quick else MAX_CALIB

    os.makedirs(os.path.join(OUT, "export"), exist_ok=True)
    net, info = load_model()
    d = make_splits(info["n_train"], info["n_test"], info["n_calib"], seed=info["seed"], noise=info["noise"])
    print("model accuracy:", json.dumps(info["accuracy"]))

    payload: dict = {
        "meta": {
            "classes": CLASS_NAMES,
            "n_train": info["n_train"],
            "n_test": info["n_test"],
            "n_calib": info["n_calib"],
            "calib_used": calib,
            "group_size": GROUP_SIZE,
            "bits_list": BITS,
            "seed": info["seed"],
            "net_seed": info["net_seed"],
            "norm_mean": info["norm_mean"],
            "norm_std": info["norm_std"],
            "accuracy": info["accuracy"],
            "noise": info.get("noise"),
            "history": info["history"],
            "architecture": "conv3x3(1->8)-relu-pool / conv3x3(8->16)-relu-pool / "
                            "conv3x3(16->32)-relu-pool / fc(512->64)-relu / fc(64->8)",
        },
        "generated": None,
    }

    print("bit-width sweep ...")
    payload.update(run_bits_sweep(net, d, calib))

    print("order comparison ...")
    orders = []
    for mode in ORDERS:
        cfg = QuantConfig(bits=4, group_size=GROUP_SIZE, order_mode=mode, compensation=True, calib_batch=calib)
        res = quantize_net(net, d["calib"][0], cfg=cfg, record_wave=False)
        qnet = copy.deepcopy(net)
        ev = evaluate(qnet, res, d["test"][0], d["test"][1], net)
        orders.append({"mode": mode, "accuracy": _r(ev["quant_accuracy"]), "logits_mse": _r(ev["logits_mse"]), "layers": layer_record(res)})
    payload["orders"] = orders

    print("wave / trace for the web page ...")
    cfg = QuantConfig(bits=4, group_size=GROUP_SIZE, order_mode="identity", compensation=True, calib_batch=calib)
    res = quantize_net(net, d["calib"][0], cfg=cfg, record_wave=True)
    qnet = copy.deepcopy(net)
    ev = evaluate(qnet, res, d["test"][0], d["test"][1], net)
    fill_stacked_errors(net, qnet, res, d["calib"][0][:128])
    payload["main"] = {
        "accuracy": _r(ev["quant_accuracy"]),
        "logits_mse": _r(ev["logits_mse"]),
        "config": {"bits": 4, "group_size": GROUP_SIZE, "symmetric": False, "order_mode": "identity"},
        "layers": layer_record(res, detail="trace"),
    }
    for mode in ("diag", "random"):
        cfg = QuantConfig(bits=4, group_size=GROUP_SIZE, order_mode=mode, compensation=True, calib_batch=calib)
        r2 = quantize_net(net, d["calib"][0], cfg=cfg, record_wave=True)
        payload.setdefault("order_traces", {})[mode] = layer_record(r2, detail="trace")

    print("export bundle ...")
    tag = "b4_g32_asym_identity_gptq"
    outdir = os.path.join(OUT, "export", tag)
    os.makedirs(outdir, exist_ok=True)
    bundle = os.path.join(outdir, "bundle.npz")
    manifest = export.export_bundle(
        bundle,
        net,
        res,
        norm_mean=info["norm_mean"],
        norm_std=info["norm_std"],
        class_names=CLASS_NAMES,
        extra={"quant_accuracy": ev["quant_accuracy"]},
    )
    export.write_inference_code(os.path.join(outdir, "infer.py"))
    payload["export"] = {
        "tag": tag,
        "path": os.path.relpath(bundle, ROOT),
        "bytes": os.path.getsize(bundle),
        "manifest": manifest,
        "float_bytes": float_bytes(net),
    }

    text = _compact(_jsonable(payload))
    with open(os.path.join(OUT, "experiments.json"), "w") as f:
        f.write(text)
    # the page is opened from web/, so keep a copy next to it
    os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
    with open(os.path.join(ROOT, "web", "data.json"), "w") as f:
        f.write(text)
    print("wrote out/experiments.json")


def float_bytes(net) -> int:
    n = 0
    for l in net.weight_layers():
        n += np.asarray(l.weight).size + np.asarray(l.bias).size
    return int(n * 4)


if __name__ == "__main__":
    main()
