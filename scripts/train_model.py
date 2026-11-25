"""Train the small conv net on synthetic wildlife images and cache it.

    python scripts/train_model.py [--force]

Writes ``out/model.npz`` (weights + normalisation stats + history) so the
experiments start from an identical, fixed-seed checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from quantlib.data import NOISE, make_splits
from quantlib.model import Net, accuracy, train

OUT = os.path.join(ROOT, "out")
CACHE = os.path.join(OUT, "model.npz")

SEED = 20240601
N_TRAIN, N_TEST, N_CALIB = 6000, 1500, 2048
EPOCHS = 20


def build(force: bool = False) -> dict:
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(CACHE) and not force:
        z = np.load(CACHE)
        info = json.loads(z["info"].tobytes().decode("utf-8"))
        net = Net(seed=info["net_seed"])
        net.load_state_dict({k: z[k] for k in z.files if k not in ("info",)})
        return {"net": net, "info": info, "cached": True}

    splits = make_splits(N_TRAIN, N_TEST, N_CALIB, seed=SEED)
    net = Net(seed=0)
    history = train(net, *splits["train"], epochs=EPOCHS, batch=64, lr=3e-3, seed=7)
    mean, std = splits["stats"]
    accs = {
        "train": accuracy(net, *splits["train"]),
        "test": accuracy(net, *splits["test"]),
        "calib": accuracy(net, *splits["calib"]),
    }
    info = {
        "net_seed": 0,
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_calib": N_CALIB,
        "epochs": EPOCHS,
        "noise": NOISE,
        "norm_mean": mean,
        "norm_std": std,
        "accuracy": accs,
        "history": history,
    }
    arrays = {k: v for k, v in net.state_dict().items()}
    arrays["info"] = np.frombuffer(json.dumps(info).encode("utf-8"), dtype=np.uint8)
    np.savez(CACHE, **arrays)
    return {"net": net, "info": info, "cached": False, "splits": splits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res = build(args.force)
    print(("loaded cached" if res["cached"] else "trained") + f" model -> {CACHE}")
    print(json.dumps(res["info"]["accuracy"], indent=2))


if __name__ == "__main__":
    main()
