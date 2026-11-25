"""Verify the exported bundle against the quantised float simulation.

    python scripts/check_export.py [--bundle out/export/<tag>/bundle.npz]

Loads the trained model and the bundle, rebuilds the simulated quantised
network from the bundle's own codes, and compares logits element by element.
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from quantlib.data import make_splits
from quantlib.netquant import QuantConfig, apply_result, quantize_net
from train_model import build


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(ROOT, "out", "export", "b4_g32_asym_identity_gptq", "bundle.npz"))
    ap.add_argument("--samples", type=int, default=512)
    args = ap.parse_args()

    net = build(force=False)["net"]
    z = np.load(os.path.join(ROOT, "out", "model.npz"))
    info = json.loads(z["info"].tobytes().decode("utf-8"))
    d = make_splits(info["n_train"], info["n_test"], info["n_calib"], seed=info["seed"], noise=info["noise"])
    x, y = d["test"]
    x = x[: args.samples]

    # 1. quantise exactly like the experiment that produced the bundle
    cfg = QuantConfig(bits=4, group_size=32, order_mode="identity", compensation=True, calib_batch=256)
    res = quantize_net(net, d["calib"][0], cfg=cfg)
    qnet = copy.deepcopy(net)
    apply_result(qnet, res)
    sim = qnet.logits(x)

    # 2. run the shipped inference code on the raw images
    infer_path = os.path.join(os.path.dirname(args.bundle), "infer.py")
    spec = importlib.util.spec_from_file_location("exported_infer", infer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    zb, manifest = mod.load(args.bundle)
    raw = x * info["norm_std"] + info["norm_mean"]
    got = mod.forward(zb, manifest, raw)

    diff = np.abs(got - sim)
    same_classes = np.array_equal(got.argmax(1), sim.argmax(1))
    print(f"bundle       : {os.path.relpath(args.bundle, ROOT)} ({os.path.getsize(args.bundle)} bytes)")
    print(f"samples      : {len(x)}")
    print(f"max |diff|   : {diff.max():.3e}")
    print(f"mean |diff|  : {diff.mean():.3e}")
    print(f"same argmax  : {same_classes}")
    print(f"accuracy     : sim {float((sim.argmax(1) == y[:len(x)]).mean()):.4f} "
          f"bundle {float((got.argmax(1) == y[:len(x)]).mean()):.4f}")

    # 3. dequantised weights must equal the simulated quantised weights
    weights = mod.load_weights(zb, manifest)
    worst = 0.0
    for layer in res.layers:
        w_bundle = weights[layer.name][0]
        worst = max(worst, float(np.abs(w_bundle - layer.weight).max()))
    print(f"max |W diff| : {worst:.3e}  (bundle vs simulator)")

    ok = diff.max() < 1e-10 and same_classes and worst == 0.0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
