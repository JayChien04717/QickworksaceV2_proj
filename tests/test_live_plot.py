from dataclasses import replace

from IPython.core.formatters import DisplayFormatter
import numpy as np
import pytest

from QickworkspaceV2.plotting import LivePlot
from QickworkspaceV2.plotting import plots


@pytest.fixture(autouse=True)
def progress_bars(monkeypatch):
    bars = []
    class Bar:
        def __init__(self, **options):
            self.total, self.n, self.closed = options['total'], 0, False
            bars.append(self)
        def update(self, amount):
            self.n += amount
        def reset(self, total):
            self.total, self.n = total, 0
        def close(self):
            self.closed = True
    monkeypatch.setattr('tqdm.auto.tqdm', Bar)
    return bars


def test_bar_updates_even_when_plot_is_throttled_and_closes(accepted_record, displayed, progress_bars):
    live = LivePlot(min_interval=1000)
    live({'state':'acquiring', 'completed':0, 'total':10})
    for completed in (1, 5):
        live({'state':'acquiring', 'completed':completed, 'total':10, 'result':accepted_record})
    assert len(displayed) == 1
    assert progress_bars[0].n == 5
    live.close()
    assert progress_bars[0].closed
    assert progress_bars[0].n == 5  # Interruption must not claim 100%.


@pytest.fixture
def displayed(monkeypatch):
    outputs = []

    class Handle:
        def update(self, value):
            outputs.append(("update", value))

    def display(value, *, display_id):
        assert display_id is True
        outputs.append(("display", value))
        return Handle()

    monkeypatch.setattr("IPython.display.display", display)
    return outputs


def test_live_plot_emits_png_without_inline_formatter(accepted_record, displayed):
    formatter = DisplayFormatter()
    # A fresh IPython formatter reproduces notebooks without %matplotlib inline.
    assert "image/png" not in formatter.format(accepted_record.plot())[0]
    original_iq = accepted_record["Q1"].iq.copy()
    original_fits = accepted_record.metrics
    live = LivePlot()
    live({"state": "acquiring", "completed": 1, "total": 1000, "result": accepted_record})
    live({"state": "completed", "result": accepted_record})
    assert [kind for kind, _ in displayed] == ["display", "update"]
    for _, value in displayed[:1]:
        assert value.data.startswith(b"\x89PNG\r\n\x1a\n")
        assert "image/png" in formatter.format(value)[0]
    assert displayed[-1][1].data == ""
    assert live._figure is None and live.handle is None
    np.testing.assert_array_equal(accepted_record["Q1"].iq, original_iq)
    assert accepted_record.metrics == original_fits


def test_live_plot_limits_redraws_but_keeps_final_and_new_run(accepted_record, displayed, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(plots, "monotonic", lambda: clock[0], raising=False)
    live = LivePlot(min_interval=0.5)
    event = {"state": "acquiring", "completed": 1, "total": 1000, "result": accepted_record}
    live(event)
    for completed in range(2, 1000):
        live({**event, "completed": completed})
    assert len(displayed) == 1
    clock[0] += 0.5
    live({**event, "completed": 1000})
    live({"state": "completed", "result": accepted_record})
    live({**event, "result": replace(accepted_record, run_id="next-run")})
    assert [kind for kind, _ in displayed] == ["display", "update", "update", "display"]


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_live_plot_rejects_invalid_interval(interval):
    with pytest.raises(ValueError, match="min_interval"):
        LivePlot(min_interval=interval)


def test_live_plot_ignores_events_without_data(displayed):
    LivePlot()({"state": "compiling"})
    assert displayed == []


@pytest.mark.parametrize("kind", ["line", "mesh", "shots", "bars"])
def test_live_plot_reuses_artists_and_updates_values(kind, displayed):
    from QickworkspaceV2 import ExperimentData, TraceData
    from copy import deepcopy
    coords = {"readout": ["g", "e"]}
    if kind == "line":
        trace = TraceData(np.arange(8).reshape(4, 2), ("x", "readout"), {**coords, "x": np.arange(4)})
    elif kind == "mesh":
        trace = TraceData(np.arange(16).reshape(2, 4, 2), ("y", "x", "readout"),
                          {**coords, "y": [1, 2], "x": np.arange(4)})
    else:
        trace = TraceData([1, 2], ("readout",), coords)
        if kind == "shots":
            trace.shots = np.array([[1+1j, 2+2j], [2+1j, 3+2j]])
            trace.shot_dims = ("shot", "readout")
    result = ExperimentData("fixture", {"Q1": trace})
    live = LivePlot(min_interval=0)
    live({"state": "acquiring", "result": result})
    figure = live._figure
    updated = deepcopy(result)
    updated["Q1"].iq *= 3
    if updated["Q1"].shots is not None:
        updated["Q1"].shots *= 3
    live({"state": "acquiring", "result": updated})
    assert live._figure is figure
    ax = figure.axes[0]
    if kind == "line":
        np.testing.assert_allclose(ax.lines[0].get_ydata(), abs(updated["Q1"].iq[:, 0]))
    elif kind == "mesh":
        np.testing.assert_allclose(ax.collections[0].get_array().reshape(2, 4), abs(updated["Q1"].iq[:, :, 0]))
    elif kind == "shots":
        np.testing.assert_allclose(ax.collections[0].get_offsets(), [[3, 3], [6, 3]])
    else:
        assert [bar.get_height() for bar in ax.patches] == [3, 6]
    assert len(displayed) == 2


@pytest.mark.parametrize("state", ["completed", "cancelled", "failed"])
def test_terminal_event_clears_only_live_display_and_close_is_idempotent(accepted_record, displayed, state):
    live = LivePlot()
    live({"state": "acquiring", "result": accepted_record})
    live({"state": state})
    live.close()
    assert [kind for kind, _ in displayed] == ["display", "update"]
    assert displayed[-1][1].data == ""
    assert live._figure is None and live.handle is None
