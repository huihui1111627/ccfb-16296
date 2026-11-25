"""Correctness of the quantiser core.

* On tiny matrices the tool is compared against an exhaustive search over
  *every* code combination per row: the tool must land near that optimum and
  clearly beat round-to-nearest when the inputs are correlated.
* The per-step objective increments must sum to the final objective (the
  invariant the visualisation relies on).
* Scales / zero points honour the group grid and the bit-width bounds.
* Degenerate calibrations (collinear, zero, constant, S < cols) stay finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantlib.quantize import (
    build_hessian,
    compute_qparams,
    group_bounds,
    inverse_cholesky,
    make_order,
    quantize_matrix,
    quantize_values,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def brute_force_optimum(W, X, bits, symmetric, group_size):
    """Exhaustive minimum of the layer objective over all codes, per row."""
    qp = compute_qparams(W, bits, symmetric, group_size)
    s, z = qp.expand(W.shape[1])
    rows, cols = W.shape
    qmax = 1 << bits
    idx = np.indices([qmax] * cols).reshape(cols, -1).T  # (qmax^cols, cols)
    total = 0.0
    for r in range(rows):
        vals = (idx.astype(np.float64) - z[r]) * s[r]
        E = vals - W[r]
        Y = E @ X.T
        total += float(np.min(np.mean(Y * Y, axis=1)))
    return total


def objective(Wq, W, X):
    E = Wq - W
    return float(np.sum(np.mean((E @ X.T) ** 2, axis=1)))


def correlated_inputs(n_samples, n_cols, rho, seed):
    C = rho ** np.abs(np.subtract.outer(np.arange(n_cols), np.arange(n_cols)))
    L = np.linalg.cholesky(C)
    return np.random.default_rng(seed).normal(size=(n_samples, n_cols)) @ L.T


# ---------------------------------------------------------------------------
# exhaustive optimality checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("rho", [0.6, 0.8])
def test_near_bruteforce_optimum(bits, rho):
    """Tool objective stays close to the exhaustive optimum, clearly below RTN."""
    tool_ratio, rtn_ratio, improvements = [], [], []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        X = correlated_inputs(300, 4, rho, 100 + seed)
        W = rng.normal(size=(3, 4))
        Wq, qp, tr = quantize_matrix(
            W, X, bits=bits, group_size=4, order_mode="identity", record_wave=False
        )
        best = brute_force_optimum(W, X, bits, False, 4)
        tool_ratio.append(objective(Wq, W, X) / best)
        rtn_ratio.append(objective(tr.plain_weight, W, X) / best)
        improvements.append(objective(tr.plain_weight, W, X) / objective(Wq, W, X))

    tool_ratio = np.array(tool_ratio)
    rtn_ratio = np.array(rtn_ratio)
    improvements = np.array(improvements)
    # never below the true optimum (up to float rounding)
    assert tool_ratio.min() > 0.99
    # close to the exhaustive optimum, and the worst case is not wild
    assert tool_ratio.mean() < 1.25, tool_ratio
    assert tool_ratio.max() < 2.1, tool_ratio
    # clearly better than round-to-nearest on this correlated data
    assert rtn_ratio.mean() > tool_ratio.mean() * 1.15, (rtn_ratio, tool_ratio)
    assert rtn_ratio.mean() > 1.4
    assert improvements.mean() > 1.25, improvements
    assert improvements.max() >= 1.4
    # in no case is the tool worse than round-to-nearest
    assert improvements.min() >= 1.0 - 1e-9


def test_exact_optimum_on_uncorrelated_inputs():
    """With independent inputs the order-optimal code is the nearest code."""
    X = np.random.default_rng(3).normal(size=(400, 4))
    W = np.random.default_rng(4).normal(size=(2, 4))
    Wq, qp, tr = quantize_matrix(W, X, bits=3, group_size=4, record_wave=False)
    best = brute_force_optimum(W, X, 3, False, 4)
    assert objective(Wq, W, X) == pytest.approx(best, rel=0.02)


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------


def test_increments_sum_to_objective():
    """The per-step increments are the exact decomposition of the objective."""
    rng = np.random.default_rng(0)
    for cols, samples in ((8, 300), (32, 200), (97, 300)):
        X = rng.normal(size=(samples, cols))
        W = rng.normal(size=(5, cols))
        Wq, qp, tr = quantize_matrix(
            W, X, bits=4, group_size=8, order_mode="diag", record_wave=False
        )
        H = build_hessian(X, 1e-3)
        obj_h = float(np.einsum("rc,cd,rd->", Wq - W, H, Wq - W))
        assert tr.inc_loss.sum() == pytest.approx(obj_h, rel=1e-9)
        assert tr.objective_via_hessian == pytest.approx(obj_h, rel=1e-9)
        assert tr.objective_curve[-1] == pytest.approx(obj_h, rel=1e-9)


def test_compensated_never_worse_than_plain():
    """Every order is the optimum of its own sequential problem, so it can
    only match or beat round-to-nearest; equal only when inputs do not
    correlate the columns."""
    rng = np.random.default_rng(11)
    for mode in ("identity", "diag", "random"):
        X = correlated_inputs(300, 24, 0.7, 5)
        W = rng.normal(size=(6, 24))
        Wq, qp, tr = quantize_matrix(W, X, bits=4, group_size=8, order_mode=mode, record_wave=False)
        assert tr.loss_compensated <= tr.loss_plain * (1 + 1e-12)


def test_codes_roundtrip_and_bit_bounds():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(7, 40)) * 0.3
    X = rng.normal(size=(100, 40))
    for bits in (3, 4, 8):
        for sym in (False, True):
            Wq, qp, tr = quantize_matrix(W, X, bits=bits, group_size=8, symmetric=sym, record_wave=False)
            qmax = (1 << bits) - 1
            s, z = qp.expand(W.shape[1])
            codes = np.round(Wq / s) + z
            assert codes.min() >= 0 and codes.max() <= qmax
            recon = (np.clip(codes, 0, qmax) - z) * s
            np.testing.assert_allclose(recon, Wq, atol=1e-12)
            assert z.min() >= 0 and z.max() <= qmax
            if sym:
                assert np.all(z == (1 << (bits - 1)))


def test_group_scale_from_group_range():
    """Asymmetric scale/zero cover the group's actual min..max."""
    W = np.array([[0.0, 1.0, -2.0, 3.0, 5.0, 0.0, -1.0, 2.0]])
    qp = compute_qparams(W, 4, False, 3)
    assert qp.n_groups == 3
    # group 0 = cols 0..2: min -2, max 1 -> scale 3/15, zero 10
    assert qp.scale[0, 0] == pytest.approx(3.0 / 15.0)
    assert qp.lo[0, 0] == pytest.approx(-2.0) and qp.hi[0, 0] == pytest.approx(1.0)
    assert qp.zero[0, 0] == 10
    # group 1 = cols 3..5: min 0, max 5 -> scale 5/15, zero point 0
    assert qp.scale[0, 1] == pytest.approx(5.0 / 15.0)
    assert qp.lo[0, 1] == pytest.approx(0.0) and qp.zero[0, 1] == 0
    # group 2 = cols 6..7: min -1, max 2
    assert qp.scale[0, 2] == pytest.approx(3.0 / 15.0)
    assert qp.zero[0, 2] == 5
    # every weight maps into the code range and back
    Wq = quantize_values(W, *qp.expand(W.shape[1]), 4)
    np.testing.assert_allclose(Wq, W, atol=qp.scale.max())


def test_group_bounds_cover_all_columns():
    assert group_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert group_bounds(4, 8) == [(0, 4)]


def test_diag_order_puts_large_diagonal_first():
    H = np.diag([0.1, 5.0, 1.0, 9.0])
    order = make_order(4, H, "diag")
    assert list(order[:2]) == [3, 1]
    assert list(make_order(4, H, "diag_reverse")[:2]) == [0, 2]


def test_order_is_permutation():
    for mode in ("identity", "diag", "diag_reverse", "random"):
        H = np.eye(20) * np.arange(1, 21)
        order = make_order(20, H, mode)
        assert sorted(order.tolist()) == list(range(20))


# ---------------------------------------------------------------------------
# numerical robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_X",
    [
        pytest.param(lambda r: np.zeros((100, 8)), id="all-zero"),
        pytest.param(lambda r: np.tile(r.normal(size=(100, 4)), 2), id="duplicate-cols"),
        pytest.param(lambda r: _with_const_col(r, 8, 5, 0.7), id="constant-col"),
        pytest.param(lambda r: _with_zero_col(r, 8, 3), id="zero-col"),
        pytest.param(lambda r: _scaled(r, 8), id="scale-spread-1e6"),
        pytest.param(lambda r: r.normal(size=(3, 32)), id="samples-lt-cols"),
    ],
)
@pytest.mark.parametrize("bits", [3, 4, 8])
def test_degenerate_calibration_stays_finite(make_X, bits):
    rng = np.random.default_rng(7)
    X = make_X(rng)
    W = rng.normal(size=(6, X.shape[1]))
    for order in ("identity", "diag"):
        Wq, qp, tr = quantize_matrix(W, X, bits=bits, group_size=4, order_mode=order, record_wave=True)
        assert np.isfinite(Wq).all()
        assert np.isfinite(tr.inc_loss).all()
        assert np.isfinite(tr.loss_compensated)
        assert np.isfinite(tr.wave).all()
        s, z = qp.expand(W.shape[1])
        codes = np.round(Wq / s) + z
        assert codes.min() >= 0 and codes.max() <= (1 << bits) - 1


def test_inverse_cholesky_is_upper_factor_of_inverse():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 12))
    H = build_hessian(X, 1e-3)
    U = inverse_cholesky(H)
    assert np.allclose(np.tril(U, -1), 0.0)
    np.testing.assert_allclose(U.T @ U, np.linalg.inv(H), atol=1e-10)


def _with_const_col(rng, cols, idx, value):
    X = rng.normal(size=(100, cols))
    X[:, idx] = value
    return X


def _with_zero_col(rng, cols, idx):
    X = rng.normal(size=(100, cols))
    X[:, idx] = 0.0
    return X


def _scaled(rng, cols):
    X = rng.normal(size=(100, cols))
    X[:, 0] *= 1e6
    X[:, 1] *= 1e-6
    return X


# ---------------------------------------------------------------------------
# column counts / performance
# ---------------------------------------------------------------------------


def test_thousand_column_layer_is_fast_and_stable():
    import time

    rng = np.random.default_rng(0)
    rows, cols, samples = 64, 1536, 256
    X = rng.normal(size=(samples, cols))
    W = rng.normal(size=(rows, cols)) * 0.05
    t0 = time.time()
    Wq, qp, tr = quantize_matrix(W, X, bits=4, group_size=128, order_mode="diag", record_wave=True)
    elapsed = time.time() - t0
    assert np.isfinite(Wq).all()
    assert np.abs(Wq).max() < 10 * np.abs(W).max()
    assert tr.loss_compensated < tr.loss_plain
    assert elapsed < 30.0, f"1536-column layer took {elapsed:.1f}s"


def test_wave_matrix_is_lower_triangular():
    """Error is only pushed onto columns that are still free."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 24))
    W = rng.normal(size=(4, 24))
    Wq, qp, tr = quantize_matrix(W, X, bits=4, group_size=8, record_wave=True)
    wave = tr.wave
    assert wave is not None and wave.shape[0] == 24
    # at step i only columns after i receive corrections
    for i in range(24):
        block_start = i // tr.wave_stride
        assert np.all(wave[i, :block_start] == 0)
    assert wave.sum() > 0
