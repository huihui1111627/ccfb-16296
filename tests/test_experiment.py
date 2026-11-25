"""End-to-end check of the numbers shown on the web page.

Runs the same quantisation as ``scripts/run_experiments.py`` and asserts the
claims the page makes:

* accuracy ordering across bit widths is monotone for both methods,
* at 3 bits the compensated setting beats round-to-nearest,
* the exported bundle matches the float simulation elementwise,
* the run is reproducible from the fixed seeds.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

from quantlib.data import CLASS_NAMES, make_splits
from quantlib.export import bundle_forward_code, export_bundle
from quantlib.model import Net
from quantlib.netquant import QuantConfig, apply_result, evaluate, quantize_net

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "out", "model.npz")


@pytest.fixture(scope="module")
def trained():
    if not os.path.exists(CACHE):
        pytest.skip("run scripts/train_model.py first")
    z = np.load(CACHE)
    info = json.loads(z["info"].tobytes().decode("utf-8"))
    net = Net(seed=info["net_seed"])
    net.load_state_dict({k: z[k] for k in z.files if k != "info"})
    return net, info


@pytest.fixture(scope="module")
def data(trained):
    net, info = trained
    return make_splits(info["n_train"], info["n_test"], info["n_calib"], seed=info["seed"], noise=info["noise"])


def run(net, splits, bits, compensation):
    cfg = QuantConfig(
        bits=bits,
        group_size=32,
        symmetric=False,
        order_mode="identity",
        compensation=compensation,
        calib_batch=256,
    )
    res = quantize_net(net, splits["calib"][0], cfg=cfg)
    qnet = copy.deepcopy(net)
    return res, qnet


def test_accuracy_curve_is_monotone(trained, data):
    net, info = trained
    x, y = data["test"]
    acc = {}
    for bits in (8, 4, 3):
        for comp in (True, False):
            res, qnet = run(net, data, bits, comp)
            ev = evaluate(qnet, res, x, y, net)
            acc[(bits, comp)] = ev["quant_accuracy"]
    assert acc[(8, True)] >= acc[(4, True)] - 0.01
    assert acc[(4, True)] >= acc[(3, True)] - 0.01
    assert acc[(8, False)] >= acc[(4, False)] - 0.01
    assert acc[(4, False)] >= acc[(3, False)] - 0.01


def test_compensation_helps_at_low_bits(trained, data):
    net, info = trained
    x, y = data["test"]
    res_q, qnet = run(net, data, 3, True)
    res_r, rnet = run(net, data, 3, False)
    aq = evaluate(qnet, res_q, x, y, net)["quant_accuracy"]
    ar = evaluate(rnet, res_r, x, y, net)["quant_accuracy"]
    assert aq > ar, (aq, ar)
    assert aq - ar > 0.01
    # the whole-net logit MSE improves too
    assert res_q.layers[-1].trace.loss_compensated < res_r.layers[-1].trace.loss_plain


def test_every_layer_improves_with_compensation(trained, data):
    """Holding the calibration input fixed, each layer's objective improves.

    The two whole-network runs cannot be compared layer-wise: sequential
    calibration feeds deeper layers the output of the already-quantised
    prefix, so the compensation run and the round-to-nearest run genuinely
    see different calibration inputs.  The per-layer claim is therefore
    checked with one shared input per layer.
    """
    from quantlib.quantize import quantize_matrix

    net, info = trained
    inputs = net.collect_inputs(data["calib"][0][:128])
    for layer in net.weight_layers():
        W = np.asarray(layer.weight)
        X = inputs[layer.weight_name]
        _, _, tq = quantize_matrix(W, X, bits=4, group_size=32, compensation=True, record_wave=False)
        _, _, tr = quantize_matrix(W, X, bits=4, group_size=32, compensation=False, record_wave=False)
        assert tq.loss_compensated <= tr.loss_plain * 1.0001, layer.weight_name
        assert tq.loss_plain == pytest.approx(tr.loss_plain, rel=1e-12)
        # and the compensation actually does something on the correlated
        # layers, not just tie
        if layer.weight_name in ("conv2", "fc1"):
            assert tq.loss_compensated < tr.loss_plain * 0.9, layer.weight_name


def test_export_matches_float_simulation(trained, data, tmp_path):
    net, info = trained
    x, y = data["test"]
    x = x[:200]
    res, qnet = run(net, data, 4, True)
    apply_result(qnet, res)
    sim = qnet.logits(x)
    path = str(tmp_path / "bundle.npz")
    export_bundle(
        path,
        net,
        res,
        norm_mean=info["norm_mean"],
        norm_std=info["norm_std"],
        class_names=CLASS_NAMES,
    )
    mod = bundle_forward_code()
    z, manifest = mod.load(path)
    raw = x * info["norm_std"] + info["norm_mean"]
    got = mod.forward(z, manifest, raw)
    np.testing.assert_allclose(got, sim, rtol=0, atol=1e-12)
    assert np.array_equal(got.argmax(1), sim.argmax(1))


def test_seeded_rerun_is_identical(trained, data):
    net, info = trained
    a = quantize_net(net, data["calib"][0][:128], cfg=QuantConfig(bits=4, calib_batch=128))
    b = quantize_net(net, data["calib"][0][:128], cfg=QuantConfig(bits=4, calib_batch=128))
    for la, lb in zip(a.layers, b.layers):
        np.testing.assert_array_equal(la.weight, lb.weight)
        assert la.trace.loss_compensated == lb.trace.loss_compensated
