"""Shared fixtures: a small trained model and deterministic data."""

from __future__ import annotations

import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quantlib.data import make_splits
from quantlib.model import Net, train


@pytest.fixture(scope="session")
def splits():
    return make_splits(n_train=1600, n_test=400, n_calib=256, seed=99)


@pytest.fixture(scope="session")
def small_net(splits):
    net = Net(seed=0)
    train(net, *splits["train"], epochs=8, batch=64, lr=3e-3, seed=3, log=lambda *a: None)
    return net


@pytest.fixture()
def fresh_net(small_net):
    return copy.deepcopy(small_net)
