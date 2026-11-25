"""Training-free weight quantisation with GPTQ-style error compensation.

Layer objective
---------------
For a weight matrix ``W`` (rows x cols) and the layer's calibration inputs
``X`` (samples x cols) the squared output error of the layer is

    Obj(Wq) = sum_r 1/S ||(Wq_r - W_r) X^T||^2
            = sum_r (Wq_r - W_r) H (Wq_r - W_r)^T ,   H = X^T X / S .

``H`` is the second-moment matrix of the calibration inputs; it separates
over the rows of ``W``, so every row sees the same quadratic form.

Column-by-column compensation
-----------------------------
Columns are fixed one at a time.  Let ``F`` be the columns that are still free
when column ``j`` is rounded.  The rounding error ``e = Wq[:, j] - W[:, j]``
is pushed onto the free columns with the *exact* conditional minimiser

    Wq[:, F] -= e * (H_FF)^{-1} H_Fj        (outer product over rows)

which is the optimal adjustment of the remaining columns given the prefix:
it makes the free columns satisfy ``(Wq - W) H[:, F] = 0``, so no further
error accumulates there.  The objective added at this step is
``||e||^2 / [H_jj - H_jF (H_FF)^{-1} H_Fj]``.

Implementation: the exact update without inverting every suffix block
------------------------------------------------------------------------
Write ``U`` for the upper-triangular factor with ``H^{-1} = U^T U`` (the
"reverse" Cholesky of the inverse).  For every suffix ``F`` starting at
``j + 1`` one has exactly

    (H_FF)^{-1} H_Fj = U[j, F] / U[j, j] ,

and the pivot appearing in the step objective is ``1 / U[j, j]^2``.  So the
whole loop needs one factor ``U`` of the full matrix:

    step objective  = ||e||^2 / U[j, j]^2
    update          = e * U[j, F] / U[j, j]

Using rows of ``H^{-1}`` directly instead is *not* the same thing: it happens
to agree with the update only for the first column (the two blocks coincide
there) and then drifts.  This was confirmed numerically while building the
tool; ``tests/test_quantize.py`` pins the correct behaviour by checking the
per-step increments against the true objective to machine precision.

``U`` is obtained from ``H^{-1}``'s Cholesky factor (``np.linalg.cholesky``
of the inverse, transposed); for the layer sizes here that is cheap.

Numerics / performance
----------------------
* ``H`` is symmetrised and damped by ``damp * mean(diag(H))``; the loading
  keeps it strictly positive definite with collinear calibration inputs and
  bounds the amplification of rounding error.
* Divisors use the diagonal of ``U`` only and are floored at a tiny positive
  value, so the update cannot diverge.
* The per-step update is one rank-1 BLAS operation on the still-free suffix;
  a 2048-column layer quantises in a few seconds.
* ``H`` is built from the calibration inputs once; for very wide layers the
  Gram matrix is the only O(cols^2) object and the Cholesky is O(cols^3/3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------


@dataclass
class QParams:
    """Per-group affine quantisation parameters.

    ``scale`` / ``zero`` / ``lo`` / ``hi`` have shape (rows, n_groups); group
    ``g`` covers original columns ``[g*group_size, (g+1)*group_size)``.
    """

    scale: np.ndarray
    zero: np.ndarray
    bits: int
    symmetric: bool
    group_size: int
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None

    @property
    def n_groups(self) -> int:
        return int(self.scale.shape[1])

    @property
    def qmax(self) -> int:
        return (1 << self.bits) - 1

    def expand(self, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
        """Broadcast per-group params to per-element arrays (rows, n_cols)."""
        s = np.repeat(self.scale, self.group_size, axis=1)[:, :n_cols]
        z = np.repeat(self.zero, self.group_size, axis=1)[:, :n_cols]
        return np.ascontiguousarray(s), np.ascontiguousarray(z)


def group_bounds(n_cols: int, group_size: int) -> list[tuple[int, int]]:
    gs = max(1, min(int(group_size), int(n_cols)))
    return [(j0, min(j0 + gs, n_cols)) for j0 in range(0, n_cols, gs)]


def compute_qparams(W: np.ndarray, bits: int, symmetric: bool, group_size: int) -> QParams:
    """Per-group scale / zero point from the unquantised weights.

    asymmetric: grid spans the group's actual min..max, so all 2^b codes are
    usable.  symmetric: grid is symmetric about 0, zero point 2^(b-1), range
    set by max|w| in the group.  Degenerate (constant) groups get scale 1.
    """
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    qmax = (1 << bits) - 1
    half = 1 << (bits - 1)
    bounds = group_bounds(cols, group_size)
    ng = len(bounds)
    scale = np.ones((rows, ng), dtype=np.float64)
    zero = np.zeros((rows, ng), dtype=np.int64)
    lo_g = np.zeros((rows, ng), dtype=np.float64)
    hi_g = np.zeros((rows, ng), dtype=np.float64)
    for gi, (a, b) in enumerate(bounds):
        seg = W[:, a:b]
        if symmetric:
            amax = np.abs(seg).max(axis=1)
            s = amax / half
            s = np.where(s > 0, s, 1.0)
            z = np.full(rows, half, dtype=np.int64)
            lo, hi = -amax, amax
        else:
            lo = seg.min(axis=1)
            hi = seg.max(axis=1)
            s = (hi - lo) / qmax
            s = np.where(s > 0, s, 1.0)
            z = np.clip(np.round(-lo / s), 0, qmax).astype(np.int64)
        scale[:, gi] = s
        zero[:, gi] = z
        lo_g[:, gi] = lo
        hi_g[:, gi] = hi
    return QParams(scale, zero, bits, symmetric, int(group_size), lo=lo_g, hi=hi_g)


def quantize_values(w: np.ndarray, scale: np.ndarray, zero: np.ndarray, bits: int) -> np.ndarray:
    """Round to nearest on the affine grid (elementwise)."""
    qmax = (1 << bits) - 1
    q = np.clip(np.round(w / scale) + zero, 0, qmax)
    return (q - zero) * scale


def to_codes(w: np.ndarray, scale: np.ndarray, zero: np.ndarray, bits: int) -> np.ndarray:
    qmax = (1 << bits) - 1
    return np.clip(np.round(w / scale) + zero, 0, qmax).astype(np.int64)


# ---------------------------------------------------------------------------
# Column orders
# ---------------------------------------------------------------------------


ORDER_MODES = ("identity", "diag", "diag_reverse", "random")


def make_order(n_cols: int, H: np.ndarray | None, mode: str, perm_seed: int = 12345) -> np.ndarray:
    """Permutation of column indices: ``order[pos]`` = column fixed at step pos."""
    if mode == "identity":
        return np.arange(n_cols)
    if mode in ("diag", "diag_reverse"):
        if H is None:
            raise ValueError(f"{mode} order needs the Hessian")
        d = np.diag(H)
        return np.argsort(-d, kind="stable") if mode == "diag" else np.argsort(d, kind="stable")
    if mode == "random":
        return np.random.default_rng(perm_seed).permutation(n_cols)
    raise ValueError(f"unknown order mode {mode!r}; expected one of {ORDER_MODES}")


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class LayerTrace:
    name: str
    n_rows: int
    n_cols: int
    bits: int
    symmetric: bool
    group_size: int
    order_mode: str
    qparams: QParams
    order: np.ndarray            # order[pos] = original column index
    inc_loss: np.ndarray         # per step: objective added by that step
    rounding_err: np.ndarray     # per step: ||e||_2 of the column at rounding
    correction_norm: np.ndarray  # per step: Frobenius norm of the update
    # per original column: norm of the accumulated corrections it received
    # before being quantised ("error pushed onto this column")
    received_correction: np.ndarray
    final_err_by_col: np.ndarray      # per original column ||Wq - W||^2
    final_err_weighted: np.ndarray    # per original column ||e||^2 * H_jj
    wave: np.ndarray | None           # steps x column-block correction magnitude
    wave_stride: int
    loss_compensated: float
    loss_plain: float
    out_mse_compensated: float
    out_mse_plain: float
    objective_curve: np.ndarray       # sampled objective after each step
    compensated_weight: np.ndarray = field(repr=False)
    plain_weight: np.ndarray = field(repr=False)
    hessian: np.ndarray | None = field(default=None, repr=False)
    objective_via_hessian: float = 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "bits": self.bits,
            "symmetric": self.symmetric,
            "group_size": self.group_size,
            "n_groups": self.qparams.n_groups,
            "order_mode": self.order_mode,
            "loss_compensated": self.loss_compensated,
            "loss_plain": self.loss_plain,
            "loss_ratio": self.loss_plain / max(self.loss_compensated, 1e-300),
            "out_mse_compensated": self.out_mse_compensated,
            "out_mse_plain": self.out_mse_plain,
            "out_mse_ratio": self.out_mse_plain / max(self.out_mse_compensated, 1e-300),
        }


# ---------------------------------------------------------------------------
# Hessian helpers
# ---------------------------------------------------------------------------


def build_hessian(X: np.ndarray, damp: float = 1e-3) -> np.ndarray:
    """Damped, symmetrised second-moment matrix of the calibration inputs."""
    S = X.shape[0]
    H = (X.T @ X) / float(S)
    H = 0.5 * (H + H.T)
    d = float(np.mean(np.diag(H)))
    if not np.isfinite(d) or d <= 0:
        d = 1.0
    return H + damp * d * np.eye(H.shape[0])


def inverse_cholesky(H: np.ndarray) -> np.ndarray:
    """Upper-triangular ``U`` with ``H^{-1} = U^T U``.

    This is the factor the GPTQ row update needs: ``(H_FF)^{-1} H_Fj`` equals
    ``U[j, F] / U[j, j]`` for every suffix ``F``.  It is *not* ``L^{-T}`` of
    the Cholesky factor ``H = L L^T`` (that matrix is upper too, but satisfies
    the reversed product); it is the reverse Cholesky of ``H^{-1}``.

    If the factorisation fails the diagonal is loaded with increasing jitter
    until it succeeds.
    """
    n = H.shape[0]
    scale = float(np.mean(np.diag(H)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    jitter = 0.0
    for _ in range(8):
        try:
            Hj = H + jitter * np.eye(n)
            Hi = np.linalg.inv(Hj)
            Hi = 0.5 * (Hi + Hi.T)
            return np.linalg.cholesky(Hi).T   # upper, H^{-1} = U^T U
        except np.linalg.LinAlgError:
            jitter = scale * 1e-6 if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("Hessian could not be regularised into a positive definite matrix")


# ---------------------------------------------------------------------------
# Core quantiser
# ---------------------------------------------------------------------------


def quantize_matrix(
    W: np.ndarray,
    X: np.ndarray,
    *,
    name: str = "layer",
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    order_mode: str = "identity",
    damp: float = 1e-3,
    compensation: bool = True,
    record_wave: bool = True,
    wave_max_cols: int = 512,
    keep_hessian: bool = False,
) -> tuple[np.ndarray, QParams, LayerTrace]:
    """Quantise one weight matrix with per-column error compensation.

    Parameters
    ----------
    W : (rows, cols) weight matrix; conv layers are unrolled beforehand.
    X : (samples, cols) calibration inputs of this layer.
    bits : 3, 4 or 8 in the experiments; any value in [2, 16] works.
    group_size : columns sharing one scale / zero point.
    symmetric : symmetric (zero point 2^(b-1)) vs asymmetric grid.
    order_mode : ``identity`` | ``diag`` | ``diag_reverse`` | ``random``.
    damp : relative Hessian damping as a fraction of the mean diagonal.
    compensation : False gives the plain round-to-nearest baseline.
    record_wave : store the step x column correction magnitude matrix
        (down-sampled by whole blocks for layers wider than ``wave_max_cols``).
    keep_hessian : attach the damped Hessian to the trace.

    Returns ``(Wq, qparams, trace)``.  ``Wq`` is the dequantised matrix and
    lies exactly on the grid defined by ``qparams``.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be 2-D")
    if X.ndim != 2 or X.shape[1] != W.shape[1]:
        raise ValueError(f"X must be (samples, {W.shape[1]}), got {X.shape}")
    rows, cols = W.shape
    if rows == 0 or cols == 0:
        raise ValueError("empty weight matrix")

    qp = compute_qparams(W, bits, symmetric, group_size)
    s_full, z_full = qp.expand(cols)

    # Reference: round-to-nearest on the unmodified weights, same grid.
    Wrtn = quantize_values(W, s_full, z_full, bits)

    H = None
    U = None
    if compensation or order_mode in ("diag", "diag_reverse"):
        H = build_hessian(X, damp)
    order = make_order(cols, H, order_mode)

    # Work in permuted column order so the free columns are a contiguous
    # suffix.  Group boundaries remain those of the original column order.
    Wp = W[:, order].copy()
    sp = s_full[:, order]
    zp = z_full[:, order]

    Wq_p = Wp.copy()
    inc = np.zeros(cols, dtype=np.float64)
    corr_norm = np.zeros(cols, dtype=np.float64)
    round_err = np.zeros(cols, dtype=np.float64)
    received = np.zeros(cols, dtype=np.float64)
    objective_curve = np.zeros(cols, dtype=np.float64)
    stride = 1
    wave = None
    if record_wave and compensation:
        stride = max(1, int(np.ceil(cols / wave_max_cols)))
        wave = np.zeros((cols, (cols + stride - 1) // stride), dtype=np.float32)
    sample_every = max(1, cols // 64)

    Hp = None
    if compensation:
        Hp = H[np.ix_(order, order)]
        U = inverse_cholesky(Hp)
        dU = np.diag(U).copy()
        # floor the pivots so a numerically singular direction cannot blow up
        floor = max(float(dU.max()) * 1e-9, 1e-30)
        dU = np.maximum(np.abs(dU), floor)

        for pos in range(cols):
            j = pos
            w = Wq_p[:, j]
            q = quantize_values(w, sp[:, j], zp[:, j], bits)
            e = w - q
            Wq_p[:, j] = q
            round_err[pos] = float(np.linalg.norm(e))
            g = 1.0 / (dU[pos] * dU[pos])
            inc[pos] = float(e @ e) * g
            if pos + 1 < cols:
                coef = U[pos, pos + 1 :] / dU[pos]
                corr = np.outer(e, coef)
                Wq_p[:, pos + 1 :] -= corr
                corr_norm[pos] = float(np.linalg.norm(corr))
                col_mag = np.sqrt(np.mean(corr * corr, axis=0))
                received[pos + 1 :] += col_mag
                if wave is not None:
                    # aggregate the free columns (pos+1 .. cols-1) into blocks;
                    # block b covers permuted columns [b*stride, (b+1)*stride)
                    idx = np.arange(pos + 1, cols)
                    block = idx // stride
                    sums = np.bincount(block, weights=col_mag * col_mag, minlength=wave.shape[1])
                    counts = np.bincount(block, minlength=wave.shape[1])
                    wave[pos] = np.sqrt(sums / np.maximum(counts, 1))
        # The step contributions sum to the exact objective (see the test
        # suite), so the curve is the running total with no extra work.
        objective_curve = np.cumsum(inc)
    else:
        for pos in range(cols):
            j = pos
            w = Wq_p[:, j]
            q = quantize_values(w, sp[:, j], zp[:, j], bits)
            Wq_p[:, j] = q
            round_err[pos] = float(np.linalg.norm(w - q))
            if pos % sample_every == 0 or pos == cols - 1:
                objective_curve[pos] = float(np.sum((Wq_p - Wp) ** 2))

    Wq = np.empty_like(W)
    Wq[:, order] = Wq_p
    received_o = np.empty(cols)
    received_o[order] = received
    round_err_o = np.empty(cols)
    round_err_o[order] = round_err

    # ---- metrics ---------------------------------------------------------
    def _obj(Wx: np.ndarray) -> float:
        Y = (Wx - W) @ X.T
        return float(np.sum(np.mean(Y * Y, axis=1)))

    loss_comp = _obj(Wq)
    loss_plain = _obj(Wrtn)
    obj_h = float(np.einsum("rc,cd,rd->", Wq - W, H, Wq - W)) if H is not None else loss_comp

    err_by_col = np.sum((Wq - W) ** 2, axis=0)
    err_w = err_by_col * (np.diag(H)[order] if H is not None else 1.0)

    trace = LayerTrace(
        name=name,
        n_rows=rows,
        n_cols=cols,
        bits=int(bits),
        symmetric=bool(symmetric),
        group_size=int(group_size),
        order_mode=order_mode,
        qparams=qp,
        order=order,
        inc_loss=inc,
        rounding_err=round_err_o,
        correction_norm=corr_norm,
        received_correction=received_o,
        final_err_by_col=err_by_col,
        final_err_weighted=err_w,
        wave=wave,
        wave_stride=int(stride),
        loss_compensated=loss_comp,
        loss_plain=loss_plain,
        out_mse_compensated=loss_comp / rows,
        out_mse_plain=loss_plain / rows,
        objective_curve=objective_curve,
        compensated_weight=Wq,
        plain_weight=Wrtn,
        hessian=(H if keep_hessian else None),
        objective_via_hessian=obj_h,
    )
    return Wq, qp, trace
