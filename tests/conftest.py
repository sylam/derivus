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
"""
import shutil

import pytest
import torch

#: The seat a recorded gate appends as.
ACTOR = 'subject-desk-one'


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
