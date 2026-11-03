"""A process whose CUDA devices are hidden (`CUDA_VISIBLE_DEVICES` empty) stays on the CPU: nothing
at import initialises CUDA, and once a training's optimizer step has (torch's graph-capture health
check asks `is_available()`, which answers True with the devices hidden on a box whose runtime
ignores an empty list), the engine's device rule (`utils.calculation_device`) still answers the CPU
for every later job of the process, though the device count then answers with the box's cards."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROBE = ("import sys, torch; sys.path.insert(0, sys.argv[1]); import production_walk_forward; "
         "print(int(torch.cuda.is_initialized()), torch.cuda.device_count())")
JOBS = ("import sys, torch, derivus as rf\n"
        "def device():\n"
        "    cx = rf.Context()\n"
        "    cx.load_json((open(sys.argv[1]).read(), 'job.json'))\n"
        "    return str(cx.run_job()[1].bundle.device)\n"
        "print(device(), device(), int(torch.cuda.is_initialized()), torch.cuda.device_count())")
HIDDEN = ['', ' ', '-1', ' -1', '-1,0', '-1, 0', ',0', ' ,0', 'none', 'NoDevFiles', 'cpu', '-', 'gpu']
SPELLINGS = """
import os, sys, json, torch
asked = []
real = torch.cuda.device_count
torch.cuda.device_count = lambda: asked.append(1) or real()
from derivus import utils, calculation
out = {}
for spelling in json.loads(sys.argv[1]):
    os.environ['CUDA_VISIBLE_DEVICES'] = spelling
    out[spelling] = [utils.visible_device_count(), calculation.worker_count(True), str(utils.calculation_device())]
print(json.dumps([out, int(torch.cuda.is_initialized()), len(asked)]))
"""


def _hidden(*args, **kwargs):
    """`python -c` under `CUDA_VISIBLE_DEVICES` empty: its stdout's last line, split."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=ROOT)
    out = subprocess.run([sys.executable, '-c', *args], env=env, capture_output=True, text=True,
                         timeout=600, **kwargs)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.splitlines()[-1].split()


def test_importing_the_harness_with_the_devices_hidden_initialises_no_cuda():
    """Killing mutation: the harness's import-time memory cap guarded by `torch.cuda.is_available()`
    instead of the device count (`is_initialized()` reads True after the import)."""
    assert _hidden(PROBE, os.path.join(ROOT, 'experiments')) == ['0', '0']


@pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES', '0')[:1] in ('', '-'),
                    reason='opens CUDA on the cards when the runtime ignores an empty list, so only on a '
                           'runner allowed a card')
def test_a_job_after_a_training_step_stays_on_the_cpu(tmp_path):
    """The policy fixture trained by DiffSolverV2 over its last decision step, one fit iteration,
    twice in one process with the devices hidden: both jobs run on the CPU. On Windows with two
    cards the first job's Adam step initialises CUDA and the count then reads 2. 9 s.

    Killing mutation: `utils.visible_device_count` answering the device count alone (the second
    job on card 0). Where the runtime honours an empty list, or the box has no card, the step initialises
    nothing and the count stays 0, so the test holds under either rule; a failure prints whether
    CUDA was initialised and the count."""
    cfg = json.load(open(os.path.join(HERE, 'fixtures', 'policy_test_simulate_only.json')))
    calc = cfg['Calc']['Calculation']
    calc.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': 4, 'Simulation_Batches': 2,
                 'Inner_Sub_Batch': 2, 'Inner_MC_Enabled': 'Yes'})
    calc['Hedging_Problem']['Solver'] = {
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 3,
        'Training_Action_Chunk_Size': 64, 'T_Min': 116, 'DiffV2_Fit_Iters': 1}
    job = tmp_path / 'job.json'
    job.write_text(json.dumps(cfg))
    first, second, initialised, count = _hidden(JOBS, str(job), cwd=ROOT)
    assert first == second == 'cpu', (first, second, initialised, count)


def test_every_hidden_spelling_stays_on_the_cpu_without_asking_the_runtime():
    """Each spelling reads (0, 1, 'cpu'), CUDA is not initialised, and the runtime's count is never
    asked (the probe counts calls to `torch.cuda.device_count`, which it wraps, not replaces). 4 s.

    Killing mutation: `worker_count(True)` returning the count alone (0 workers, red). Not run, since
    it would ask the runtime with the devices named: `visible_device_count` reading
    `torch.cuda.device_count()` before the parse (asked > 0)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONPATH=ROOT)
    run = subprocess.run([sys.executable, '-c', SPELLINGS, json.dumps(HIDDEN)], env=env, capture_output=True,
                         text=True, timeout=300, cwd=ROOT)
    assert run.returncode == 0, run.stderr[-2000:]
    out, initialised, asked = json.loads(run.stdout.splitlines()[-1])
    assert all(out[s] == [0, 1, 'cpu'] for s in HIDDEN), out
    assert (initialised, asked) == (0, 0), (initialised, asked)
