"""The four LogVar2FJ model-validation notebooks, executed headless.

Each notebook is a set of JSON documents run through `derivus.Context` with the prose that reads
them beside it, so the gate IS the execution: a cell that raises is a claim the notebook can no
longer make. Nothing is asserted about a number here - notebook 2's own fits reproduce what
`test_logvar2fj_json.py` section 11 asserts, and notebook 1 states its own bit-identity - what is
asserted is that every cell ran.

NOTEBOOK 3 RUNS BANKED. `DERIVUS_LIVE_BLOOMBERG` is deleted from the environment before the
kernel starts, so the chain notebook loads `notebooks/data/sx5e_chain_20260915.json` and no
request reaches a terminal from here, whatever the box has set.

SLOWEST MEASURED: notebook 2 at 128 s - four fits of a fifteen-quote ladder at `Max_Iterations`
60 - on an RTX 3090 shared with a second session, 93 s with the card to itself. The other three
read 30 to 34 s, and the four together 184 s. A CPU-only box re-times every one of them rather
than reading a defect.
"""
import os
import sys

import pytest

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

nbformat = pytest.importorskip('nbformat')
nbclient = pytest.importorskip('nbclient')

NOTEBOOKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks')
NAMES = ['logvar2fj_01_model_and_fit.ipynb', 'logvar2fj_02_identification.ipynb',
         'logvar2fj_03_listed_chain.ipynb', 'logvar2fj_04_reserve_and_mark.ipynb']


@pytest.mark.parametrize('name', NAMES)
def test_a_notebook_executes_with_no_error_cell(name, monkeypatch):
    """Executed in memory - the shipped outputs are the run they were shipped from and this gate
    does not overwrite them."""
    monkeypatch.delenv('DERIVUS_LIVE_BLOOMBERG', raising=False)
    book = nbformat.read(os.path.join(NOTEBOOKS, name), as_version=4)
    nbclient.NotebookClient(book, timeout=3600, kernel_name='python3',
                            resources={'metadata': {'path': NOTEBOOKS}}).execute()
    failures = [(n, out['ename'], out['evalue'])
                for n, cell in enumerate(book.cells) for out in cell.get('outputs', [])
                if out.get('output_type') == 'error']
    assert not failures, failures
