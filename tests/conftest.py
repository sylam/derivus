"""Suite-wide test hygiene.

Pin the global torch default dtype to float32 at the start of EVERY test. Once enough test modules
are collected in one session the process-wide default flips to float64 at COLLECTION time — a
pre-existing interaction (12+ modules trip it; removing any single module drops it back, so it is a
threshold effect on the imported set, not one culprit, and it is orthogonal to what any test
asserts). A value-function net then built under a float64 default multiplies a float32 input and
dies with "mat1 and mat2 must have the same dtype", failing the solver / GARCH-generate / bit-exact
tests for a reason unrelated to their subject — while each passes in isolation.

Resetting per test makes the full suite green (verified) and is standard torch-test hygiene. Tests
that deliberately need float64 (test_symlog_unit's exact FD gates) set it inside their own body,
AFTER this fixture's setup, so they are unaffected.
"""
import importlib.util
import shutil

import pytest
import torch

#: `utils.hn_log_substep_fused` is `torch.compile`d, and inductor needs a backend for the device
#: it lands on: triton under CUDA, a host C++ compiler under CPU. Without one the deal is skipped
#: CRITICAL, the mark collapses to a scalar zero, and a gate driving the fused sub-step dies three
#: layers downstream on the collapsed frame. The gates that need it state the precondition here
#: instead - a bare Windows box (no MSVC, no triton wheel) cannot run them on either device.
if torch.cuda.is_available():
    HN_FUSED_COMPILES = importlib.util.find_spec('triton') is not None
else:
    HN_FUSED_COMPILES = any(shutil.which(cc) for cc in ('cl', 'g++', 'gcc', 'clang++', 'clang'))
needs_hn_fused = pytest.mark.skipif(
    not HN_FUSED_COMPILES,
    reason='utils.hn_log_substep_fused: torch.compile needs triton (CUDA) or a C++ compiler (CPU), '
           'and this box has neither for its device')


@pytest.fixture(autouse=True)
def _pin_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield
