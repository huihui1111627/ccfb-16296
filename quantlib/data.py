"""Synthetic wildlife-camera style image data.

Images are 32x32 grayscale pictures of one of eight simple shapes with
random position, scale, rotation, brightness, background clutter and noise.
Everything is generated from explicit seeds so every experiment is
reproducible.
"""

from __future__ import annotations

import numpy as np

SIZE = 32
N_CLASSES = 8
CLASS_NAMES = [
    "disk",
    "ring",
    "square",
    "triangle",
    "cross",
    "h-stripes",
    "v-stripes",
    "checker",
]


def _rot(xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, ang: float):
    u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    return u, v


def _shape_mask(kind: int, xx, yy, rng: np.random.Generator) -> np.ndarray:
    """Return a boolean mask for one shape centred near (16, 16)."""
    size = int(rng.integers(0, 4))  # 0..3
    cx = 15.5 + rng.uniform(-3.0, 3.0)
    cy = 15.5 + rng.uniform(-3.0, 3.0)
    ang = rng.uniform(0.0, np.pi)
    if kind == 0:  # disk
        r = 6.0 + size * 0.8 + rng.uniform(0.0, 1.0)
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if kind == 1:  # ring
        r = 6.5 + size * 0.8 + rng.uniform(0.0, 1.0)
        t = 1.8 + 0.4 * size + rng.uniform(0.0, 0.6)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return (d2 <= r * r) & (d2 >= (r - t) ** 2)
    if kind == 2:  # square
        u, v = _rot(xx, yy, cx, cy, ang)
        half = 4.5 + size * 0.8 + rng.uniform(0.0, 1.0)
        return (np.abs(u) <= half) & (np.abs(v) <= half)
    if kind == 3:  # triangle
        u, v = _rot(xx, yy, cx, cy, ang)
        h = 5.0 + size * 0.8 + rng.uniform(0.0, 1.0)
        # equilateral triangle with apex at (0,-h) and base at v=h/2
        inside = v >= -h
        inside &= np.sqrt(3.0) * np.abs(u) <= (v + h)
        inside &= v <= h * 0.55
        return inside
    if kind == 4:  # cross
        u, v = _rot(xx, yy, cx, cy, ang)
        length = 6.0 + size * 0.8 + rng.uniform(0.0, 1.0)
        thick = 1.2 + 0.25 * size + rng.uniform(0.0, 0.5)
        return ((np.abs(u) <= length) & (np.abs(v) <= thick)) | (
            (np.abs(v) <= length) & (np.abs(u) <= thick)
        )
    if kind == 5:  # horizontal stripes
        u, v = xx, yy
        period = 2.8 + 0.5 * size + rng.uniform(0.0, 0.8)
        length = 6.0 + size * 0.7 + rng.uniform(0.0, 1.0)
        width = 7.0 + size * 0.5 + rng.uniform(0.0, 1.0)
        phase = 1.0 + rng.uniform(0.0, period)
        bars = (np.mod(v - cy + phase, period) < period * 0.55).astype(float)
        return (bars > 0.5) & (np.abs(u - cx) <= length) & (np.abs(v - cy) <= width)
    if kind == 6:  # vertical stripes
        u, v = xx, yy
        period = 2.8 + 0.5 * size + rng.uniform(0.0, 0.8)
        length = 6.0 + size * 0.7 + rng.uniform(0.0, 1.0)
        width = 7.0 + size * 0.5 + rng.uniform(0.0, 1.0)
        phase = 1.0 + rng.uniform(0.0, period)
        bars = (np.mod(u - cx + phase, period) < period * 0.55).astype(float)
        return (bars > 0.5) & (np.abs(u - cx) <= width) & (np.abs(v - cy) <= length)
    # checkerboard
    cell = 2.6 + 0.55 * size + rng.uniform(0.0, 0.6)
    half = 6.0 + size * 0.5 + rng.uniform(0.0, 0.8)
    u, v = _rot(xx, yy, cx, cy, 0.0)
    inside = (np.abs(u) <= half) & (np.abs(v) <= half)
    cells = (np.floor((u + half) / cell) + np.floor((v + half) / cell)) % 2 == 0
    return inside & cells


def make_images(
    n: int,
    seed: int,
    size: int = SIZE,
    n_classes: int = N_CLASSES,
    noise: float = 0.11,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` images and labels, balanced across classes."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    labels = rng.permutation(np.arange(n) % n_classes).astype(np.int64)
    imgs = np.zeros((n, 1, size, size), dtype=np.float64)
    for i in range(n):
        k = int(labels[i])
        mask = _shape_mask(k, xx, yy, rng)
        intensity = rng.uniform(0.55, 1.0)
        img = mask.astype(np.float64) * intensity
        # background clutter: a few faint blobs
        n_dots = int(rng.integers(0, 3))
        for _ in range(n_dots):
            dx = rng.uniform(3, size - 4)
            dy = rng.uniform(3, size - 4)
            dr = rng.uniform(0.8, 2.2)
            val = rng.uniform(0.15, 0.45)
            img += ((xx - dx) ** 2 + (yy - dy) ** 2 <= dr * dr) * val
        img += rng.normal(0.0, noise, size=(size, size))
        imgs[i, 0] = np.clip(img, 0.0, 1.0)
    return imgs, labels


def normalize(imgs: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (imgs - mean) / std


def dataset_stats(imgs: np.ndarray) -> tuple[float, float]:
    return float(imgs.mean()), float(imgs.std())


NOISE = 0.3


def make_splits(
    n_train: int = 4000,
    n_test: int = 1000,
    n_calib: int = 2048,
    seed: int = 20240601,
    noise: float = NOISE,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return normalized train/test/calib splits and the normalization stats.

    Keys: train, test, calib.  Stats are stored under ``"stats"``.
    """
    tr_x, tr_y = make_images(n_train, seed + 11, noise=noise)
    te_x, te_y = make_images(n_test, seed + 22, noise=noise)
    ca_x, ca_y = make_images(n_calib, seed + 33, noise=noise)
    mean, std = dataset_stats(tr_x)
    out = {
        "train": (normalize(tr_x, mean, std), tr_y),
        "test": (normalize(te_x, mean, std), te_y),
        "calib": (normalize(ca_x, mean, std), ca_y),
    }
    out["stats"] = (mean, std)
    return out
