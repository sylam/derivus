"""Suite-wide test hygiene, and the spine home a gate records to.

Pin the global torch default dtype to float32 at the start of EVERY test. Past ~12 collected
modules the process-wide default flips to float64 at COLLECTION time - a threshold effect on the
imported set, not one culprit - and a value-function net built under it multiplies a float32 input
and dies with "mat1 and mat2 must have the same dtype", failing the solver / GARCH-generate /
bit-exact tests that each pass in isolation.

Tests that deliberately need float64 (test_symlog_unit's exact FD gates) set it in their own body,
AFTER this fixture.

A recorded gate's home is a COPY of one home minted for the session: minting is a run of fsyncs,
and a copy is the same record standing in another directory, its own history from the first fact
it files. Whoever serves a record keeps its folds (`spine.advancing`), so copies never share them.

A test marked `card` reads the device, or holds a number read on it (a banked float, a bound sized
on its draw), and is skipped where no CUDA device is visible, so a red on the CPU is a defect.
"""
import shutil

import pytest
import torch

#: The seat a recorded gate appends as.
ACTOR = 'subject-desk-one'


def pytest_configure(config):
    config.addinivalue_line('markers', 'card: reads the card or a number read on it')


def pytest_collection_modifyitems(items):
    """Skips every `card` test where `utils.visible_device_count` counts no device."""
    from derivus import utils

    if not utils.visible_device_count():
        skip = pytest.mark.skip(reason='reads the card or a number read on it, and none is visible')
        for item in items:
            if item.get_closest_marker('card'):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _pin_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield


@pytest.fixture(scope='session')
def minted(tmp_path_factory):
    """One spine home minted for the session, which every `recorded` home is a copy of."""
    from derivus_spine import init_home

    home = tmp_path_factory.mktemp('minted') / 'spine'
    init_home(home, ACTOR)
    return home


@pytest.fixture
def recorded(minted, tmp_path, monkeypatch):
    """A spine home with an actor for the appends, copied from `minted`. Genesis is four events, so
    every head assertion counts from there."""
    home = tmp_path / 'spine'
    shutil.copytree(str(minted), str(home))
    monkeypatch.setenv('DV_SPINE_HOME', str(home))
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield home
