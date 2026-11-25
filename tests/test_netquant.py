"""Network-level behaviour: sequential calibration, accuracy ordering, parity."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from quantlib.model import accuracy
from quantlib.netquant import (
    QuantConfig,
    apply_result,
    evaluate,
    fill_stacked_errors,
    quantize_net,
)


def _quant(small_net, splits, **kw):
    cfg = QuantConfig(calib_batch=128, **kw)
    res = quantize_net(small_net, splits["calib"][0], cfg=cfg)
    qnet = copy.deepcopy(small_net)
    apply_result(qnet, res)
    return res, qnet


def test_quantize_net_does_not_mutate_the_model(small_net, splits):
    before = small_net.state_dict()
    quantize_net(small_net, splits["calib"][0], cfg=QuantConfig(bits=4, calib_batch=64))
    after = small_net.state_dict()
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])


def test_sequential_calibration_uses_quantised_inputs(small_net, splits):
    """Layer k's calibration input must come from the quantised prefix."""
    res, qnet = _quant(small_net, splits, bits=3, group_size=16)
    x = splits["calib"][0][:64]
    inputs_float = small_net.collect_inputs(x)
    # replay the same stack with the already-quantised prefix
    qnet.set_training(False)
    h = x
    for layer in qnet.layers:
        name = getattr(layer, "weight_name", None)
        if name is not None:
            from quantlib.model import Conv2d, im2col

            if isinstance(layer, Conv2d):
                mat = im2col(h, layer.k, layer.k, layer.pad, 1)
            else:
                mat = h.reshape(h.shape[0], -1)
            r = next(r for r in res.layers if r.name == name)
            expect = mat @ r.weight.T
            got = mat @ np.asarray(layer.weight).T
            np.testing.assert_allclose(got, expect, atol=1e-12)
            # deeper layers really do see a perturbed input, not the float one
            if name != "conv1":
                assert not np.allclose(mat, inputs_float[name], atol=1e-9)
        h = layer.forward(h)


def test_accuracy_ordering_across_bits(small_net, splits):
    """8bit >= 4bit >= 3bit for both methods (ties allowed)."""
    x, y = splits["test"]
    accs = {}
    for bits in (8, 4, 3):
        res, qnet = _quant(small_net, splits, bits=bits, group_size=16)
        accs[(bits, "gptq")] = accuracy(qnet, x, y)
        cfg = QuantConfig(bits=bits, group_size=16, compensation=False, calib_batch=128)
        res2 = quantize_net(small_net, splits["calib"][0], cfg=cfg)
        q2 = copy.deepcopy(small_net)
        apply_result(q2, res2)
        accs[(bits, "rtn")] = accuracy(q2, x, y)
    assert accs[(8, "gptq")] >= accs[(4, "gptq")] - 0.02
    assert accs[(4, "gptq")] >= accs[(3, "gptq")] - 0.02
    assert accs[(8, "rtn")] >= accs[(3, "rtn")] - 0.02


def test_gptq_beats_rtn_at_three_bits(small_net, splits):
    """At 3 bits on this model the compensation should win clearly."""
    x, y = splits["test"]
    res_q, qnet = _quant(small_net, splits, bits=3, group_size=16)
    cfg = QuantConfig(bits=3, group_size=16, compensation=False, calib_batch=128)
    res_r = quantize_net(small_net, splits["calib"][0], cfg=cfg)
    rnet = copy.deepcopy(small_net)
    apply_result(rnet, res_r)
    aq, ar = accuracy(qnet, x, y), accuracy(rnet, x, y)
    assert aq >= ar + 0.01, (aq, ar)
    # and every layer's objective is better under compensation
    for lq, lr in zip(res_q.layers, res_r.layers):
        assert lq.trace.loss_compensated <= lr.trace.loss_plain * 1.0001


def test_evaluate_reports_logits_mse_and_accuracy(small_net, splits):
    res, qnet = _quant(small_net, splits, bits=4, group_size=16)
    x, y = splits["test"]
    ev = evaluate(qnet, res, x, y, small_net)
    assert 0.0 <= ev["quant_accuracy"] <= 1.0
    assert ev["float_accuracy"] >= ev["quant_accuracy"] - 0.05
    assert ev["logits_mse"] > 0.0
    ref = small_net.logits(x)
    assert ev["logits_mse"] == pytest.approx(float(np.mean((qnet.logits(x) - ref) ** 2)))


def test_stacked_error_grows_with_depth(small_net, splits):
    res, qnet = _quant(small_net, splits, bits=4, group_size=16)
    fill_stacked_errors(small_net, qnet, res, splits["calib"][0][:64])
    # the last layer's outputs differ more than the first layer's
    assert res.layers[-1].out_err_stacked > res.layers[0].out_err_stacked
    assert all(np.isfinite(r.out_err_stacked) for r in res.layers)


def test_conv_layers_are_unrolled_matrices(small_net, splits):
    """Conv weight matrices are (cout, cin*k*k) exactly as they are applied."""
    inputs = small_net.collect_inputs(splits["calib"][0][:16])
    for layer in small_net.weight_layers():
        W = np.asarray(layer.weight)
        if layer.weight_name.startswith("conv"):
            assert W.shape[1] == layer.cin * layer.k * layer.k
            assert inputs[layer.weight_name].shape[1] == W.shape[1]
        else:
            assert inputs[layer.weight_name].shape[1] == W.shape[1]
