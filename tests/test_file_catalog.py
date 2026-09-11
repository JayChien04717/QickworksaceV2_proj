"""File/index round trips with numerical fixtures, never hardware measurements."""
import json
import subprocess
import sys

import numpy as np
import pytest

from labtools.catalog import find_experiments, rebuild_catalog
from QickworkspaceV2 import ExperimentData, TraceData


def record():
    return ExperimentData('t1_ge', {'Q1': TraceData(
        np.array([[1+2j], [3+4j]]), ('delay', 'readout'),
        {'delay': [1, 2], 'readout': ['g']}, units={'delay': 'us'})},
        metadata={'acquisition_status': 'completed', 'tags': ['diagnostic']})


def test_native_and_labber_share_search_contract(tmp_path):
    result = record()
    native = result.save(tmp_path/'native.h5', catalog_root=tmp_path)
    labber = result.save_labber(directory=tmp_path, tags=['diagnostic'], save_plot=False)
    rows = find_experiments(data_root=tmp_path, qubit='Q1', tags='diagnostic', experiment='t1_ge')
    assert {r['path'] for r in rows} == {native, labber}
    assert {r['format'] for r in rows} == {'hdf5', 'labber'}
    assert all(r['acquisition_status'] == 'completed' and r['quality'] == 'not_analyzed' for r in rows)
    for path in (native, labber):
        loaded = ExperimentData.load(path)
        np.testing.assert_array_equal(loaded['Q1'].iq, result['Q1'].iq)
    assert find_experiments(data_root=tmp_path, qubit='Q2') == []
    assert len(find_experiments(data_root=tmp_path, run_id=result.run_id, limit=1)) == 1
    assert find_experiments(data_root=tmp_path, start='2099-01-01') == []


def test_rebuild_preserves_files_and_reports_unreadable_entries(tmp_path):
    result = record()
    path = result.save_labber(directory=tmp_path, save_plot=False)
    before = path.read_bytes()
    (tmp_path/'catalog.sqlite').unlink()
    bad = tmp_path/'bad.hdf5'
    bad.write_text('not an HDF5 file')
    rebuilt = rebuild_catalog(tmp_path)
    assert rebuilt['indexed'] == 1 and len(rebuilt['errors']) == 1
    assert rebuilt['errors'][0]['path'] == str(bad)
    assert len(find_experiments(data_root=tmp_path)) == 1
    assert path.read_bytes() == before and bad.read_text() == 'not an HDF5 file'


def test_multiple_targets_and_shots_do_not_replace_each_other(tmp_path):
    result = record()
    result.traces['Q2'] = result['Q1']
    result['Q1'].shots = result['Q1'].iq[..., None]
    result['Q1'].shot_dims = ('delay', 'readout', 'shot')
    for target in result.targets:
        for kind in ('iq', 'shots'):
            result.save_labber(directory=tmp_path, target=target, data=kind, save_plot=False)
    assert len(find_experiments(data_root=tmp_path)) == 4
    assert len(find_experiments(data_root=tmp_path, qubit='Q2')) == 2


def test_index_failure_leaves_successfully_saved_hdf5(tmp_path, monkeypatch):
    import labtools.catalog as catalog
    def unavailable(*args):
        raise OSError('database unavailable')
    monkeypatch.setattr(catalog, 'register_file', unavailable)
    with pytest.warns(RuntimeWarning, match='HDF5 saved'):
        path = record().save_labber(directory=tmp_path, save_plot=False)
    assert ExperimentData.load(path)['Q1'].iq.shape == (2, 1)


def test_tools_import_and_execute_without_sdk(tmp_path):
    code = '''
import sys, importlib.abc
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'QickworkspaceV2', 'qick'}:
            raise AssertionError('Tool imported hardware SDK: '+fullname)
sys.meta_path.insert(0, NoSDK())
from labtools.fitting import fit_exponential, exponential
from labtools.hdf5 import save_hdf5, read_hdf5
from labtools.labber import save_labber
from labtools.catalog import find_experiments
import numpy as np
from types import SimpleNamespace
x=np.linspace(0,10,51)
fit=fit_exponential(x, exponential(x,0.1,0.8,3))
assert fit.success
trace=SimpleNamespace(iq=np.ones((2,1),complex), dims=('x','readout'),
    coords={'x':np.array([1,2]),'readout':np.array(['g'])}, units={},
    shots=None,shot_dims=(),metadata={})
record=dict(experiment='independent',metadata={},run_id='tool-test',
    created_at='2026-09-11T00:00:00+00:00', fits={},analysis_status='not_analyzed',analysis_message='')
class Result(SimpleNamespace):
    def __getitem__(self,key): return self.traces[key]
r=Result(**record,traces={'Q1':trace},targets=('Q1',),path=None)
p=save_labber(r,directory=sys.argv[1],save_plot=False)
assert read_hdf5(p)[1]['Q1']['iq'].shape == (2,1)
assert len(find_experiments(data_root=sys.argv[1]))==1
'''
    completed = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_catalog_bad_fit_keeps_completed_acquisition(tmp_path):
    from labtools.fitting import FitResult
    result = record()
    result.analysis_status = 'failed'
    result.fits = {'Q1': FitResult('exponential', False)}
    result.save_labber(directory=tmp_path, save_plot=False)
    rows = find_experiments(data_root=tmp_path, acquisition_status='completed', quality='bad')
    assert len(rows) == 1
    assert json.loads(json.dumps(rows[0], default=str))['analysis_status'] == 'failed'
