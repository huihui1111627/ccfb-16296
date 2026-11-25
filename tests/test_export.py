"""Export bundle: packed weights, inference parity, and file layout."""

from __future__ import annotations

import types

import numpy as np
import pytest

from quantlib.export import (
    INFER_PY,
    bundle_forward_code,
    export_bundle,
    pack_codes,
    unpack_codes,
)
from quantlib.netquant import QuantConfig, apply_result, quantize_net


def _bundle(tmp_path, net, result, splits, bits=4, group=32):
    path = str(tmp_path / "bundle.npz")
    export_bundle(
        path,
        net,
        result,
        norm_mean=splits["stats"][0],
        norm_std=splits["stats"][1],
        class_names=["a", "b"],
    )
    return path


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_pack_unpack_roundtrip(bits):
    rng = np.random.default_rng(0)
    cols = 37
    codes = rng.integers(0, 1 << bits, size=(5, cols))
    packed = pack_codes(codes, bits)
    if bits < 8:
        # 8/bits codes per byte
        assert packed.shape[1] == int(np.ceil(cols / (8 // bits)))
    back = unpack_codes(packed, bits, cols)
    np.testing.assert_array_equal(back, codes)
    # values never leave the bit range
    assert back.max() < (1 << bits)


def test_bundle_logits_match_simulator(fresh_net, splits, tmp_path):
    """Bundle inference equals the quantised float simulation elementwise."""
    cfg = QuantConfig(bits=4, group_size=32, order_mode="identity", compensation=True, calib_batch=128)
    res = quantize_net(fresh_net, splits["calib"][0], cfg=cfg)
    x, y = splits["test"]
    x = x[:64]
    # simulated quantised network
    qnet = types.SimpleNamespace()
    import copy

    qnet = copy.deepcopy(fresh_net)
    apply_result(qnet, res)
    sim = qnet.logits(x)

    path = _bundle(tmp_path, fresh_net, res, splits)
    mod = bundle_forward_code()
    z, manifest = mod.load(path)
    raw = x * splits["stats"][1] + splits["stats"][0]  # bundle normalises itself
    got = mod.forward(z, manifest, raw)

    assert got.shape == sim.shape
    np.testing.assert_allclose(got, sim, rtol=0, atol=1e-12)
    # accuracy is identical too
    assert np.array_equal(got.argmax(axis=1), sim.argmax(axis=1))


def test_bundle_weights_are_exactly_the_quantised_weights(fresh_net, splits, tmp_path):
    cfg = QuantConfig(bits=4, group_size=32, compensation=True, calib_batch=128)
    res = quantize_net(fresh_net, splits["calib"][0], cfg=cfg)
    path = _bundle(tmp_path, fresh_net, res, splits)
    mod = bundle_forward_code()
    z, manifest = mod.load(path)
    weights = mod.load_weights(z, manifest)
    for layer in res.layers:
        got = weights[layer.name][0]
        np.testing.assert_array_equal(got, layer.weight)
        np.testing.assert_array_equal(weights[layer.name][1], layer.bias)


def test_bundle_reports_manifest_and_size(fresh_net, splits, tmp_path):
    cfg = QuantConfig(bits=4, group_size=32, compensation=True, calib_batch=128)
    res = quantize_net(fresh_net, splits["calib"][0], cfg=cfg)
    res.quant_accuracy = 0.9
    res.float_accuracy = 0.95
    path = _bundle(tmp_path, fresh_net, res, splits)
    z, manifest = mod = None, None
    mod = bundle_forward_code()
    z, manifest = mod.load(path)
    assert manifest["arch"] == "wildlife-quant-v1"
    assert manifest["config"]["bits"] == 4
    names = [op.get("name") for op in manifest["ops"] if op.get("name")]
    assert names == [l.name for l in res.layers]
    # packed code stream is smaller than float32 weights
    total_codes = sum(int(z[f"{n}.codes"].nbytes) for n in names)
    total_float = sum(np.asarray(l.weight).size * 4 for l in res.layers)
    assert total_codes < total_float


def test_exported_infer_py_is_self_contained(fresh_net, splits, tmp_path):
    """The shipped infer.py file loads and reproduces the reference logits."""
    cfg = QuantConfig(bits=4, group_size=32, compensation=True, calib_batch=128)
    res = quantize_net(fresh_net, splits["calib"][0], cfg=cfg)
    path = _bundle(tmp_path, fresh_net, res, splits)
    infer_path = tmp_path / "infer.py"
    infer_path.write_text(INFER_PY)
    import importlib.util

    spec = importlib.util.spec_from_file_location("shipped_infer", infer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import copy

    qnet = copy.deepcopy(fresh_net)
    apply_result(qnet, res)
    x = splits["test"][0][:32]
    sim = qnet.logits(x)
    raw = x * splits["stats"][1] + splits["stats"][0]
    z, manifest = mod.load(path)
    got = mod.forward(z, manifest, raw)
    np.testing.assert_allclose(got, sim, atol=1e-12)
    # no imports beyond numpy/json
    src = infer_path.read_text()
    assert "import numpy" in src
    assert "torch" not in src and "scipy" not in src
