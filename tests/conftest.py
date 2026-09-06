"""Suite-wide test hygiene.

Pin the global torch default dtype to float32 at the start of EVERY test. Past ~12 collected
modules the process-wide default flips to float64 at COLLECTION time - a threshold effect on the
imported set, not one culprit - and a value-function net built under it multiplies a float32 input
and dies with "mat1 and mat2 must have the same dtype", failing the solver / GARCH-generate /
bit-exact tests that each pass in isolation.

Tests that deliberately need float64 (test_symlog_unit's exact FD gates) set it in their own body,
AFTER this fixture.
"""
import pytest
import torch


@pytest.fixture(autouse=True)
def _pin_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield
