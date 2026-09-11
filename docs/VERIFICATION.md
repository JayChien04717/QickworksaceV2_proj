# Verification

Run SDK checks from `QickworksaceV2`:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe tests/verify_notebooks.py
.venv/Scripts/ruff.exe check QickworkspaceV2 labtools tests lab measure.py
```

Notebook verification compiles tagged configuration/program cells with the real QICK compiler fixture and skips connection/acquisition cells. Service tests use the single companion HTTP client against in-process transport; catalog tests check complete schemas and project-added experiments without generated files. Numerical fixtures are explicitly test data.

Latest refactor checks and outstanding migration/hardware/UI work are in the [labtools report](../../docs/verification/2026-09-11-labtools.md). Earlier Notebook/package results are in the [Notebook SDK acceptance report](../../docs/V2_NOTEBOOK_SDK_2026-09-11.md). The [catalog cleanup report](../../docs/V2_CATALOG_CLEANUP.md) records the earlier wrapper removal. Earlier [release](verification/release.json), [notebook](verification/notebooks.json) and [Blueprint](verification/blueprint.json) records are historical; the wrapper/discovery implementation has since been removed.

Physical acquisition is separate from software checks. Preserve acquisition when fitting fails; never treat a no-chip run as calibration.
