

## Live display clarification and channel setup

The final-plot-only default was a misunderstanding and is superseded: retain live acquisition updates, clear their display on completion/interruption/failure, then retain the final analysis plot. NotebookLab owns this lifecycle; explicit LivePlot callbacks also clear on terminal events and measurement cleanup. All-in-one setup_device was removed; experiments explicitly declare drive generators and gates, while setup_readout owns resonator/ADC setup. SDK: 215 tests passed; maintained notebooks compile; Ruff passed. Three real Jupyter-kernel display fixtures each retain one final PNG and logs after clearing the live PNG. No hardware acquisition was performed. Evidence: data/verification/2026-09-11-live-cleanup-setup in the repository root.
