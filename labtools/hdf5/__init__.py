"""Lossless named-array HDF5 records, independent of an experiment SDK."""
from pathlib import Path
from uuid import uuid4
import json

import h5py
import numpy as np

from .atomic import replace_file
from labtools.serialization import dumps


def result_record(result):
    """Extract the metadata contract from a result-like object."""
    return {key: getattr(result, key) for key in (
        'experiment', 'metadata', 'run_id', 'created_at', 'fits',
        'analysis_status', 'analysis_message')}


def write_record(group, record, traces):
    """Write a record into an open HDF5 group without owning its lifetime."""
    group.attrs['schema_version'] = 2
    group.attrs['record'] = dumps(record)
    output = group.create_group('traces', track_order=True)
    for name, trace in traces.items():
        if not name or '/' in name or '\\' in name:
            raise ValueError('Trace names must be nonempty and contain no path separators')
        trace = trace if isinstance(trace, dict) else vars(trace)
        item = output.create_group(name)
        item.attrs['record'] = dumps({key: trace[key] for key in ('dims', 'units', 'shot_dims', 'metadata')})
        item.create_dataset('iq', data=trace['iq'], compression='gzip')
        if trace.get('shots') is not None:
            item.create_dataset('shots', data=trace['shots'], compression='gzip')
        for dim, values in trace['coords'].items():
            if not dim or '/' in dim or '\\' in dim:
                raise ValueError('Coordinate names must contain no path separators')
            values = np.asarray(values)
            if values.dtype.kind in 'UO':
                item.create_dataset(f'coords/{dim}', data=values.astype(object), dtype=h5py.string_dtype())
            else:
                item.create_dataset(f'coords/{dim}', data=values)


def save_hdf5(path, record, traces, *, catalog_root=None):
    """Atomically save arrays; optionally register the file in a separate catalog.

    Replacement supports acquisition checkpoints. Immutable-run enforcement is
    the caller's responsibility. Staging inherits destination Windows permissions.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f'.run-{uuid4().hex}.h5'
    try:
        with h5py.File(temporary, 'x') as h5:
            write_record(h5, record, traces)
            h5.flush()
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if catalog_root is not None:
        from labtools.catalog import register_file_safely
        register_file_safely(path, catalog_root)
    return path


def read_hdf5(path):
    """Read either a standalone record or a Labber file's embedded record."""
    with h5py.File(path, 'r') as h5:
        group = h5['metagroup'] if 'metagroup' in h5 else h5
        if group.attrs.get('schema_version') != 2:
            raise ValueError('Unsupported HDF5 schema')
        record = json.loads(group.attrs['record'])
        traces = {}
        for name, item in group['traces'].items():
            meta = json.loads(item.attrs['record'])
            coords = {d: v.asstr()[...] if v.dtype.kind in 'OS' else v[...]
                      for d, v in item['coords'].items()}
            traces[name] = dict(iq=item['iq'][...], coords=coords,
                                shots=item['shots'][...] if 'shots' in item else None, **meta)
    return record, traces
