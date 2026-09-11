"""Rebuildable SQLite file index. Scientific arrays remain authoritative in HDF5."""
from contextlib import contextmanager
from datetime import datetime, timezone, time
from pathlib import Path
import json
import sqlite3
import warnings

import h5py


@contextmanager
def _connect(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / 'catalog.sqlite', timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('''CREATE TABLE IF NOT EXISTS files (
            relative_path TEXT PRIMARY KEY, run_id TEXT NOT NULL,
            experiment TEXT NOT NULL, created_at TEXT NOT NULL,
            targets_json TEXT NOT NULL, tags_json TEXT NOT NULL,
            acquisition_status TEXT NOT NULL, analysis_status TEXT NOT NULL,
            quality TEXT NOT NULL, format TEXT NOT NULL, data_kind TEXT NOT NULL,
            comment TEXT NOT NULL)''')
        for column in ('run_id', 'created_at', 'experiment'):
            connection.execute(f'CREATE INDEX IF NOT EXISTS files_{column} ON files({column})')
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def inspect_file(path):
    """Read metadata only; do not load IQ or shots when searching/indexing."""
    with h5py.File(path, 'r') as h5:
        labber = 'metagroup' in h5
        group = h5['metagroup'] if labber else h5
        if group.attrs.get('schema_version') != 2:
            raise ValueError('Unsupported HDF5 schema')
        record = json.loads(group.attrs['record'])
        fits = record.get('fits', {})
        analysis = record.get('analysis_status', 'not_analyzed')
        quality = ('not_analyzed' if analysis == 'not_analyzed' else
                   'good' if analysis == 'completed' and fits and all(f['success'] for f in fits.values()) else 'bad')
        metadata = record.get('metadata', {})
        target = h5.attrs.get('export_target')
        targets = [str(target)] if target is not None else list(group['traces'])
        tags = list(h5['Tags'].attrs.get('Tags', [])) if labber and 'Tags' in h5 else metadata.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]
        tags = [t.decode() if isinstance(t, bytes) else str(t) for t in tags]
        return dict(run_id=record['run_id'], experiment=record['experiment'],
                    created_at=_boundary(record['created_at']), targets_json=json.dumps(targets),
                    tags_json=json.dumps(tags), acquisition_status=metadata.get('acquisition_status', 'unknown'),
                    analysis_status=analysis, quality=quality, format='labber' if labber else 'hdf5',
                    data_kind=str(h5.attrs.get('export_kind', 'record')),
                    comment=str(h5.attrs.get('comment', metadata.get('comment', ''))))


def _row(path, root):
    path, root = Path(path).resolve(), Path(root).expanduser().resolve()
    relative = path.relative_to(root).as_posix()
    return dict(relative_path=relative, **inspect_file(path))


def _insert(connection, row):
    columns = ','.join(row)
    connection.execute(f'INSERT OR REPLACE INTO files ({columns}) VALUES ({",".join("?" for _ in row)})',
                       tuple(row.values()))


def register_file(path, data_root):
    """Index one file; one run can have separate targets, shots and native exports."""
    row = _row(path, data_root)
    with _connect(data_root) as connection:
        _insert(connection, row)


def register_file_safely(path, data_root):
    """An index error must not turn a successful file save into data loss."""
    try:
        register_file(path, data_root)
    except Exception as error:
        warnings.warn(f'HDF5 saved at {path}, but catalog registration failed: {error}. '
                      'Use rebuild_catalog(data_root) to recover the index.', RuntimeWarning, stacklevel=2)


def _boundary(value, end=False):
    text = str(value)
    parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if len(text) == 10:
        parsed = datetime.combine(parsed.date(), time.max if end else time.min)
    return parsed.astimezone(timezone.utc).isoformat()


def find_experiments(*, data_root, experiment=None, qubit=None, tags=(), start=None, end=None,
                     acquisition_status=None, analysis_status=None, quality=None, run_id=None,
                     limit=100):
    """Return metadata and resolved paths; date-only boundaries use local time."""
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError('limit must be a positive integer or None')
    root = Path(data_root).expanduser().resolve()
    if not (root / 'catalog.sqlite').exists():
        rebuild_catalog(root)
    conditions, values = [], []
    for column, value in dict(experiment=experiment, acquisition_status=acquisition_status,
                              analysis_status=analysis_status, quality=quality, run_id=run_id).items():
        if value is not None:
            conditions.append(f'{column}=?')
            values.append(value)
    for value, operator in ((start, '>='), (end, '<=')):
        if value is not None:
            conditions.append(f'created_at {operator} ?')
            values.append(_boundary(value, end=operator == '<='))
    wanted = {tags} if isinstance(tags, str) else set(tags)
    query = 'SELECT * FROM files' + (' WHERE '+' AND '.join(conditions) if conditions else '')
    result = []
    with _connect(root) as connection:
        for row in connection.execute(query+' ORDER BY created_at DESC, relative_path', values):
            row = dict(row)
            row['targets'], row['tags'] = json.loads(row.pop('targets_json')), json.loads(row.pop('tags_json'))
            if qubit is not None and qubit not in row['targets']:
                continue
            if not wanted.issubset(row['tags']):
                continue
            path = (root / row['relative_path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                continue
            row['path'] = path
            result.append(row)
            if limit is not None and len(result) >= limit:
                break
    return result


def rebuild_catalog(data_root):
    """Reindex supported files atomically; report unreadable files without deleting them.

    Symlinks, junctions and hidden staging files are excluded. Partial records are
    retained with their acquisition status. Existing rows remain if their file
    is unreadable during this scan, and scan errors are returned explicitly.
    """
    import os
    root = Path(data_root).expanduser().resolve()
    rows, errors = [], []
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [n for n in folders if not n.startswith('.') and
                      not (Path(directory)/n).is_symlink() and
                      not getattr(os.path, 'isjunction', lambda _: False)(Path(directory)/n)]
        for name in files:
            path = Path(directory)/name
            if name.startswith('.') or path.suffix.lower() not in {'.h5', '.hdf5'} or path.is_symlink():
                continue
            try:
                rows.append(_row(path, root))
            except Exception as error:
                errors.append({'path': str(path), 'error': str(error)})
    with _connect(root) as connection:
        old = list(connection.execute('SELECT relative_path FROM files'))
        for row in old:
            if not (root/row['relative_path']).exists():
                connection.execute('DELETE FROM files WHERE relative_path=?', (row['relative_path'],))
        for row in rows:
            _insert(connection, row)
    return {'indexed': len(rows), 'errors': errors}
