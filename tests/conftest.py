"""Suite-wide test hygiene.

Pin the global torch default dtype to float32 at the start of EVERY test. Past ~12 collected
modules the process-wide default flips to float64 at COLLECTION time - a threshold effect on the
imported set, not one culprit - and a value-function net built under it multiplies a float32 input
and dies with "mat1 and mat2 must have the same dtype", failing the solver / GARCH-generate /
bit-exact tests that each pass in isolation.

Tests that deliberately need float64 (test_symlog_unit's exact FD gates) set it in their own body,
AFTER this fixture.
"""
import importlib.util
import shutil

import pytest
import torch

#: `utils.hn_log_substep_fused` is `torch.compile`d, and inductor needs a backend for its device:
#: triton under CUDA, a host C++ compiler under CPU. Without one the deal is skipped CRITICAL, the
#: mark collapses to a scalar zero, and the gate dies three layers downstream on the collapsed
#: frame - so the gates that need it state the precondition here instead.
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
