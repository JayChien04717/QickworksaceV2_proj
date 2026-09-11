"""Numerical Labber serialization fixtures; no acquisition or calibration."""
import json

import h5py
import numpy as np
import pytest

from QickworkspaceV2 import ExperimentData, TraceData


def fixture_result():
    iq = (np.arange(12).reshape(2, 3, 2) + 1) * (1 + 2j)
    shots = np.stack([iq + n * (10 - 3j) for n in range(4)], axis=0)
    trace = TraceData(iq, ('gain', 'frequency', 'readout'),
                      {'gain': [0.01, 0.03], 'frequency': [6600., 6600.7, 6602.], 'readout': ['g', 'e']},
                      units={'frequency': 'MHz'}, shots=shots,
                      shot_dims=('shot', 'gain', 'frequency', 'readout'))
    return ExperimentData('resonator_punchout', {'Q7': trace},
                          metadata={'targets': ['Q7'], 'acquisition_status': 'completed',
                                    'resolved_config': {'reps': 3}, 'test_fixture': True})


@pytest.mark.parametrize('kind', ['iq', 'shots'])
def test_export_preserves_complex_data_axes_and_complete_native_record(tmp_path, kind):
    result = fixture_result()
    original = result.path
    path = result.save_labber(tmp_path / (kind + '.hdf5'), data=kind, save_plot=False)
    assert result.path == original
    with h5py.File(path) as h5:
        assert h5['Data'].attrs['Completed']
        raw = h5['Data/Data'][:]
        values = raw[:, -2, :].T + 1j * raw[:, -1, :].T
        expected = result['Q7'].iq.transpose(0, 2, 1).reshape(-1, 3) if kind == 'iq' else result['Q7'].shots.transpose(1, 2, 3, 0).reshape(-1, 4)
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(h5['metagroup/traces/Q7/iq'], result['Q7'].iq)
        np.testing.assert_array_equal(h5['metagroup/traces/Q7/shots'], result['Q7'].shots)
        assert json.loads(h5['metagroup'].attrs['record'])['run_id'] == result.run_id
        freq = h5['Step config/frequency/Step items']['single']
        np.testing.assert_array_equal(freq, [6600., 6600.7, 6602.])
        assert list(h5['Instrument config/Generic - GPIB: , Step channels at localhost'].attrs['___readout___combo_defs']) == ['g', 'e']
    contents = path.read_bytes()
    with pytest.raises(FileExistsError):
        result.save_labber(path, save_plot=False)
    assert path.read_bytes() == contents


def test_export_plot_and_default_path(accepted_record, tmp_path):
    accepted_record.save(tmp_path / 'acquisition.h5')
    original = accepted_record.path
    path = accepted_record.save_labber()
    assert accepted_record.path == original
    with h5py.File(path) as h5:
        assert bytes(h5['metagroup/plots/analysis.png'][:]).startswith(b'\x89PNG')


def test_multitarget_selection_and_missing_shots_are_explicit(tmp_path):
    result = fixture_result()
    result.traces['Q2'] = result['Q7']
    with pytest.raises(ValueError, match='Select target'):
        result.save_labber(tmp_path / 'multi.hdf5', save_plot=False)
    path = result.save_labber(tmp_path / 'multi.hdf5', target='Q2', save_plot=False)
    with h5py.File(path) as h5:
        assert set(h5['metagroup/traces']) == {'Q2', 'Q7'}
    result['Q7'].shots = None
    with pytest.raises(ValueError, match='no captured shots'):
        result.save_labber(tmp_path / 'missing.hdf5', target='Q7', data='shots')


def test_plot_failure_does_not_discard_acquired_data(tmp_path, monkeypatch):
    result = fixture_result()
    def unavailable(**options):
        raise ValueError('No plot for this analysis')
    monkeypatch.setattr(result, 'plot', unavailable)
    path = result.save_labber(tmp_path / 'plot-error.hdf5')
    with h5py.File(path) as h5:
        assert h5['metagroup'].attrs['plot_error'] == 'No plot for this analysis'
        np.testing.assert_array_equal(h5['metagroup/traces/Q7/iq'], result['Q7'].iq)


def test_procedure_exports_children_tables_and_all_targets_once(tmp_path):
    from QickworkspaceV2 import save_labber_results
    parent, child = fixture_result(), fixture_result()
    child.traces['Q2'] = child['Q7']
    parent.metadata['child_runs'] = [child.run_id]
    for result in (parent, child):
        result.save(tmp_path / result.run_id / 'acquisition.h5')
    records = {r.run_id: r for r in (parent, child)}
    saved = save_labber_results({'offset': 0.2, 'pair': [parent], 'rows': [{'run_id': child.run_id}]},
                                load=records.__getitem__, save_plot=False)
    assert len(saved) == 3
    assert all(path.is_file() for path in saved.values())
    assert save_labber_results(None) == {}


def test_notebook_disabled_scan_does_not_export_last_measurement():
    from QickworkspaceV2 import NotebookLab
    lab = NotebookLab.__new__(NotebookLab)
    # No connection or prior result is needed for a disabled procedure.
    assert lab.save_labber(None) == {}
