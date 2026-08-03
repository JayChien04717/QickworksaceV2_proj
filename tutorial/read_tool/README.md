# Native HDF5 reader demo

This hardware-independent tutorial creates T1, Rabi, single-shot, RB,
tomography, and SSH optimization files through the public
`ExperimentData.save()` API.

The default archive contains seven measurement dates, repeated T1/Rabi runs,
Q1/Q2 single-shot checks, daily RB, periodic tomography, and selected SSH
optimization runs. Repeated runs share a daily `session_id` and carry date,
qubit, repeat, comment, tag, and quality metadata for query testing.

From the repository root, regenerate the sample archive with:

```bash
python tutorial/read_tool/generate_mock_data.py --clean
```

The history size is configurable:

```bash
python tutorial/read_tool/generate_mock_data.py \
  --clean --days 14 --repeats 5
```

Then open `read_tool.ipynb`. The generated `data/catalog.sqlite` is a cache;
all authoritative data remains in the `.h5` files.

For the current native HDF5 API walkthrough, open
`tutorial/07_data_management.ipynb`. For interactive browsing, run
`python hdf5_viewer_server.py` from the repository root and choose this
`tutorial/read_tool/data` folder in the viewer.
