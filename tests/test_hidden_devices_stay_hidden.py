"""A process whose CUDA devices are hidden (`CUDA_VISIBLE_DEVICES` empty) stays on the CPU after
importing the walk-forward harness: nothing at import initialises CUDA, so the engine's device
rule (`utils.calculation_device`, the device count) still answers zero and every job of the
process runs where the environment says. On a box whose runtime ignores an empty list,
`torch.cuda.is_available()` answers True with the devices hidden, and a call made on that answer
initialises CUDA and moves every later job onto card 0 beside whatever runs there.

Killing mutation: the harness's import-time memory cap guarded by `torch.cuda.is_available()`
instead of the device count (`is_initialized()` reads True after the import)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = ("import sys, torch; sys.path.insert(0, sys.argv[1]); import production_walk_forward; "
         "print(int(torch.cuda.is_initialized()), torch.cuda.device_count())")


def test_importing_the_harness_with_the_devices_hidden_initialises_no_cuda():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=os.path.dirname(HERE))
    out = subprocess.run([sys.executable, '-c', PROBE, os.path.join(os.path.dirname(HERE), 'experiments')],
                         env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    initialised, count = out.stdout.split()[-2:]
    assert initialised == '0' and count == '0', out.stdout
