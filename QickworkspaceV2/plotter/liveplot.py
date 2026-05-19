import threading
import queue
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, update_display
from tqdm.auto import tqdm
import math

from ..tools.system_tool import auto_unit


def _iq_to_complex(iq_list):
    """Convert QICK acquire output into a complex numpy array."""
    arr = np.asarray(iq_list[0][0])
    if arr.ndim > 0 and arr.shape[-1] == 2:
        return arr.dot([1, 1j])
    return arr.astype(complex, copy=False)


def _process_iq(iqdata, iq_process):
    """Select the plotted IQ channel."""
    iq_process = (iq_process or "abs").lower()
    if iq_process in {"real", "i", "avgi"}:
        return np.real(iqdata)
    if iq_process in {"imag", "q", "avgq"}:
        return np.imag(iqdata)
    if iq_process == "phase":
        return np.unwrap(np.angle(iqdata))
    return np.abs(iqdata)


def _process_label(iq_process):
    iq_process = (iq_process or "abs").lower()
    labels = {
        "real": "ADC Units (Real)",
        "i": "ADC Units (Real)",
        "avgi": "ADC Units (Real)",
        "imag": "ADC Units (Imag)",
        "q": "ADC Units (Imag)",
        "avgq": "ADC Units (Imag)",
        "phase": "Phase (rad)",
        "abs": "ADC Units (Abs)",
        "amp": "ADC Units (Abs)",
        "amplitude": "ADC Units (Abs)",
        "all": "IQ Channels",
        "iq": "IQ Channels",
        "channels": "IQ Channels",
        "multi": "IQ Channels",
    }
    return labels.get(iq_process, labels["abs"])


def _is_all_iq(iq_process):
    return str(iq_process or "abs").lower() in {"all", "iq", "channels", "multi"}


def _iq_channel_dict(iqdata):
    """Return all real-valued IQ views used by live plots."""
    return {
        "Abs": np.abs(iqdata),
        "Phase": np.unwrap(np.angle(iqdata)),
        "I": np.real(iqdata),
        "Q": np.imag(iqdata),
    }


def _safe_limits(data, pad_fraction=0.1):
    """Return padded finite axis/color limits for partially filled live data."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    data_min = float(np.min(arr))
    data_max = float(np.max(arr))
    data_range = data_max - data_min
    if data_range <= 0:
        pad = max(abs(data_max) * pad_fraction, 1e-12)
        return data_min - pad, data_max + pad
    pad = pad_fraction * data_range
    return data_min - pad, data_max + pad


def _normalize_rows(data):
    """Normalize each 2D row independently for contrast in live plots."""
    row_mins = data.min(axis=1, keepdims=True)
    row_maxs = data.max(axis=1, keepdims=True)
    ranges = row_maxs - row_mins
    ranges[ranges == 0] = 1
    return (data - row_mins) / ranges


@dataclass
class LivePlotState:
    current_avg: int = 0
    total_avg: int = 1
    interrupted: bool = False


class LivePlotSession:
    """Own notebook display behavior for a live plot."""

    def __init__(
        self,
        fig=None,
        ax=None,
        figsize: Tuple[float, float] = (6, 4),
        display_id: Optional[str] = None,
        clear_output_on_finish: bool = True,
        close_on_finish: bool = True,
    ):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            self.owns_figure = True
        else:
            self.owns_figure = False
        self.fig = fig
        self.ax = ax
        self.display_id = display_id or f"live-plot-v2-{np.random.randint(1e9)}"
        self.clear_output_on_finish = clear_output_on_finish
        self.close_on_finish = close_on_finish

    def show(self):
        display(self.fig, display_id=self.display_id)

    def update(self):
        display(self.fig, display_id=self.display_id, update=True)

    def finish(self):
        if self.clear_output_on_finish:
            clear_output(wait=True)
        if self.close_on_finish and self.owns_figure:
            plt.close(self.fig)


class LineRenderer:
    """Render a 1D software-average trace."""

    def __init__(self, x_axis_vals, x_label, y_label, title_prefix):
        self.x_axis_vals = x_axis_vals
        self.x_label = x_label
        self.y_label = y_label
        self.title_prefix = title_prefix
        self.line = None

    def setup(self, ax):
        (self.line,) = ax.plot(
            self.x_axis_vals,
            np.zeros_like(self.x_axis_vals),
            "o-",
            markersize=5,
            alpha=0.7,
        )
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(f"{self.title_prefix} (Initializing...)")

    def update(self, ax, data, state: LivePlotState):
        self.line.set_ydata(data)
        ax.set_ylim(*_safe_limits(data))
        ax.set_title(
            f"{self.title_prefix} | Average: {state.current_avg + 1} / {state.total_avg}"
        )

    def finalize(self, ax, data, interrupted, last_i):
        title_status = "Interrupted" if interrupted else "Completed"
        ax.set_title(f"{self.title_prefix} ({title_status} at avg {last_i + 1})")
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if data is not None:
            ax.plot(self.x_axis_vals, data, "o-", markersize=5, alpha=0.7)
        else:
            ax.text(
                0.5, 0.5, "No data acquired",
                ha="center", va="center", transform=ax.transAxes,
            )


class MeshRenderer:
    """Render a normalized 2D software-average map."""

    def __init__(self, x_axis_vals, y_axis_vals, x_label, y_label, title_prefix):
        self.x_axis_vals = x_axis_vals
        self.y_axis_vals = y_axis_vals
        self.x_label = x_label
        self.y_label = y_label
        self.title_prefix = title_prefix
        self.mesh = None

    def setup(self, ax):
        self.mesh = ax.pcolormesh(
            self.x_axis_vals,
            self.y_axis_vals,
            np.zeros((len(self.y_axis_vals), len(self.x_axis_vals))),
            cmap="viridis",
        )
        ax.figure.colorbar(self.mesh, ax=ax, label="Normalized Amplitude")
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(f"{self.title_prefix} (Initializing...)")

    def update(self, ax, data, state: LivePlotState):
        self.mesh.set_array(data.ravel())
        vmin, vmax = _safe_limits(data, pad_fraction=0.0)
        self.mesh.set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(
            f"{self.title_prefix} | Average: {state.current_avg + 1} / {state.total_avg}"
        )

    def finalize(self, ax, data, interrupted, last_i):
        title_status = "Interrupted" if interrupted else "Completed"
        ax.set_title(f"{self.title_prefix} ({title_status} at avg {last_i + 1})")
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if data is not None:
            im = ax.pcolormesh(
                self.x_axis_vals,
                self.y_axis_vals,
                data,
                cmap="viridis",
            )
            ax.figure.colorbar(im, ax=ax, label="Normalized Amplitude")
        else:
            ax.text(
                0.5, 0.5, "No data acquired",
                ha="center", va="center", transform=ax.transAxes,
            )


class SoftwareAverageRunner:
    """Acquire repeated QICK averages and emit processed plot data."""

    def __init__(self, prog, soc, py_avg, y_axis_vals=None, iq_process="abs"):
        self.prog = prog
        self.soc = soc
        self.py_avg = py_avg
        self.y_axis_vals = y_axis_vals
        self.iq_process = iq_process

    def run(self, on_update: Callable[[int, Any], None]):
        iq = 0
        iqdata = None
        last_i = 0
        interrupted = False

        try:
            for i in tqdm(range(self.py_avg), desc="Software Average Count", mininterval=0.1):
                last_i = i
                iq_list = self.prog.acquire(self.soc, rounds=1, progress=False)
                iq_data = _iq_to_complex(iq_list)
                iq = iq_data if i == 0 else iq + iq_data
                iqdata = iq / (i + 1)
                plot_data = _process_iq(iqdata, self.iq_process)
                if self.y_axis_vals is not None:
                    plot_data = _normalize_rows(plot_data)
                on_update(i, plot_data)
        except KeyboardInterrupt:
            interrupted = True

        return iqdata, interrupted, last_i + 1


def run_software_average_liveplot(
    prog,
    soc,
    py_avg,
    x_axis_vals,
    y_axis_vals=None,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    show_final_plot=False,
    iq_process="abs",
):
    """Composable implementation for the software-average liveplot path."""
    data_queue = queue.LifoQueue(maxsize=1)
    stop_event = threading.Event()
    state = LivePlotState(total_avg=py_avg)

    session = LivePlotSession()
    is_2d = y_axis_vals is not None
    if is_2d:
        renderer = MeshRenderer(x_axis_vals, y_axis_vals, x_label, y_label, title_prefix)
    else:
        renderer = LineRenderer(
            x_axis_vals,
            x_label,
            _process_label(iq_process),
            title_prefix,
        )

    renderer.setup(session.ax)
    session.show()

    def plotter_thread_func():
        while not stop_event.is_set():
            try:
                current_i, data = data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            state.current_avg = current_i
            renderer.update(session.ax, data, state)
            session.update()
            data_queue.task_done()

    def on_update(i, data):
        try:
            data_queue.put_nowait((i, data))
        except queue.Full:
            try:
                data_queue.get_nowait()
                data_queue.put_nowait((i, data))
            except (queue.Empty, queue.Full):
                pass

    plot_thread = threading.Thread(target=plotter_thread_func, daemon=True)
    plot_thread.start()

    runner = SoftwareAverageRunner(
        prog=prog,
        soc=soc,
        py_avg=py_avg,
        y_axis_vals=y_axis_vals,
        iq_process=iq_process,
    )
    try:
        iqdata, interrupted, avg_count = runner.run(on_update=on_update)
    finally:
        stop_event.set()
        if plot_thread.is_alive():
            plot_thread.join(timeout=1.0)
        session.finish()

    if show_final_plot:
        final_session = LivePlotSession(
            clear_output_on_finish=False,
            close_on_finish=True,
        )
        plot_data = None
        if iqdata is not None:
            plot_data = _process_iq(iqdata, iq_process)
            if is_2d:
                plot_data = _normalize_rows(plot_data)
        renderer.finalize(final_session.ax, plot_data, interrupted, avg_count - 1)
        final_session.show()
        plt.close(final_session.fig)

    return iqdata, interrupted, avg_count


def liveplotfun(
    prog=None,
    soc=None,
    py_avg=1,
    x_axis_vals=None,
    y_axis_vals=None,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    instrument_manager=None,
    yoko_name=None,
    yoko_mode="current",
    yoko_voltage_ramp_step=1e-5,
    yoko_current_ramp_step=1e-8,
    yoko_ramp_interval=0.01,
    scan_x_axis=None,
    scan_y_axis=None,
    get_prog_callback=None,
    show_final_plot=True,
    iq_process="abs",
):
    """
    General-purpose live plotter (Facade pattern).

    Dispatches to one of four specialized internal routines based on the
    combination of arguments supplied:

    - **Yoko sweep** (``instrument_manager`` and ``yoko_name`` are set): outer
      loop steps a Yokogawa source while the inner QICK program sweeps
      ``x_axis_vals``.
    - **2D parameter scan** (``scan_x_axis`` and ``scan_y_axis`` both set):
      a callback generates a fresh program for every (x, y) grid point.
    - **1D parameter scan** (only ``scan_x_axis`` set): a callback generates
      a fresh program for each x point.
    - **Software averaging** (default): repeats a fixed program ``py_avg``
      times and accumulates a running average.

    Returns
    -------
    iqdata : np.ndarray
    interrupted : bool
    n_done : int
    """
    if instrument_manager is not None and yoko_name is not None:
        if y_axis_vals is None:
            raise ValueError("y_axis_vals must be provided for a Yoko sweep.")
        return _liveplot_sweep_yoko(
            prog=prog,
            soc=soc,
            py_avg=py_avg,
            x_axis_vals=x_axis_vals,
            y_axis_vals_yoko=y_axis_vals,
            instrument_manager=instrument_manager,
            yoko_name=yoko_name,
            yoko_mode=yoko_mode,
            yoko_voltage_ramp_step=yoko_voltage_ramp_step,
            yoko_current_ramp_step=yoko_current_ramp_step,
            yoko_ramp_interval=yoko_ramp_interval,
            x_label=x_label,
            y_label=y_label,
            title_prefix=title_prefix,
            iq_process=iq_process,
        )

    elif scan_x_axis is not None:
        if get_prog_callback is None:
            raise ValueError("get_prog_callback must be provided for parameter scan.")

        if scan_y_axis is not None:
            return _liveplot_2d_scan(
                soc=soc,
                py_avg=py_avg,
                scan_x_axis=scan_x_axis,
                scan_y_axis=scan_y_axis,
                get_prog_callback=get_prog_callback,
                x_label=x_label,
                y_label=y_label,
                title_prefix=title_prefix,
                show_final_plot=show_final_plot,
                iq_process=iq_process,
            )
        else:
            return _liveplot_1d_scan(
                soc=soc,
                py_avg=py_avg,
                scan_x_axis=scan_x_axis,
                get_prog_callback=get_prog_callback,
                x_label=x_label,
                title_prefix=title_prefix,
                show_final_plot=show_final_plot,
                iq_process=iq_process,
            )

    else:
        return _liveplot_sw_avg(
            prog=prog,
            soc=soc,
            py_avg=py_avg,
            x_axis_vals=x_axis_vals,
            y_axis_vals=y_axis_vals,
            x_label=x_label,
            y_label=y_label,
            title_prefix=title_prefix,
            show_final_plot=show_final_plot,
            iq_process=iq_process,
        )


def _liveplot_sw_avg(
    prog,
    soc,
    py_avg,
    x_axis_vals,
    y_axis_vals=None,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    show_final_plot=False,
    iq_process="abs",
):
    _y_label_proc = _process_label(iq_process)
    plot_all_iq = _is_all_iq(iq_process)

    iq = 0
    iqdata = None
    last_i = 0
    interrupted = False

    if plot_all_iq and y_axis_vals is None:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        axes_flat = axes.ravel()
        ax = axes_flat[0]
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        axes_flat = None
    plot_display_id = f"live-plot-{np.random.randint(1e9)}"

    is_2d = y_axis_vals is not None
    if plot_all_iq and not is_2d:
        plot_artist = {}
        for channel_ax, channel_name in zip(axes_flat, ("Abs", "Phase", "I", "Q")):
            (line,) = channel_ax.plot(
                x_axis_vals,
                np.zeros_like(x_axis_vals),
                "o-",
                markersize=4,
                alpha=0.75,
            )
            plot_artist[channel_name] = line
            channel_ax.set_title(channel_name)
            channel_ax.set_xlabel(x_label)
            channel_ax.set_ylabel("ADC" if channel_name != "Phase" else "rad")
            channel_ax.grid(True, alpha=0.25)
        fig.suptitle(f"{title_prefix} (Initializing...)")
    elif is_2d:
        plot_artist = ax.pcolormesh(
            x_axis_vals,
            y_axis_vals,
            np.zeros((len(y_axis_vals), len(x_axis_vals))),
            cmap="viridis",
        )
        fig.colorbar(plot_artist, ax=ax, label="Normalized Amplitude")
        ax.set_ylabel(y_label)
    else:
        (plot_artist,) = ax.plot(
            x_axis_vals, np.zeros_like(x_axis_vals), "o-", markersize=5, alpha=0.7
        )
        ax.set_ylabel(_y_label_proc)

    if not (plot_all_iq and not is_2d):
        ax.set_xlabel(x_label)
        ax.set_title(f"{title_prefix} (Initializing...)")
    display(fig, display_id=plot_display_id)

    try:
        for i in tqdm(range(py_avg), desc="Software Average Count", mininterval=0.1):
            last_i = i
            iq_list = prog.acquire(soc, rounds=1, progress=False)
            iq_data = _iq_to_complex(iq_list)
            iq = iq_data if i == 0 else iq + iq_data
            iqdata = iq / (i + 1)

            if plot_all_iq and not is_2d:
                channel_data = _iq_channel_dict(iqdata)
                for channel_ax, (channel_name, data_to_plot) in zip(
                    axes_flat, channel_data.items()
                ):
                    plot_artist[channel_name].set_ydata(data_to_plot)
                    channel_ax.set_ylim(*_safe_limits(data_to_plot))
                fig.suptitle(f"{title_prefix} | Average: {i + 1} / {py_avg}")
            elif is_2d:
                plot_data = _process_iq(iqdata, iq_process)
                data_to_plot = _normalize_rows(plot_data)
                plot_artist.set_array(data_to_plot.ravel())
                plot_artist.set_clim(*_safe_limits(data_to_plot, pad_fraction=0.0))
                ax.set_title(f"{title_prefix} | Average: {i + 1} / {py_avg}")
            else:
                plot_data = _process_iq(iqdata, iq_process)
                data_to_plot = plot_data
                plot_artist.set_ydata(data_to_plot)
                ax.set_ylim(*_safe_limits(data_to_plot))
                ax.set_title(f"{title_prefix} | Average: {i + 1} / {py_avg}")
            update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True

    clear_output(wait=True)
    plt.close(fig)

    if show_final_plot:
        if plot_all_iq and not is_2d:
            final_fig, final_axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
            final_ax = final_axes.ravel()[0]
        else:
            final_fig, final_ax = plt.subplots(figsize=(6, 4))
        title_status = "Interrupted" if interrupted else "Completed"
        if plot_all_iq and not is_2d:
            final_fig.suptitle(f"{title_prefix} ({title_status} at avg {last_i + 1})")
        else:
            final_ax.set_title(f"{title_prefix} ({title_status} at avg {last_i + 1})")
            final_ax.set_xlabel(x_label)

        if iqdata is not None:
            if plot_all_iq and not is_2d:
                for channel_ax, (channel_name, data_to_plot) in zip(
                    final_axes.ravel(), _iq_channel_dict(iqdata).items()
                ):
                    channel_ax.plot(
                        x_axis_vals, data_to_plot, "o-", markersize=4, alpha=0.75
                    )
                    channel_ax.set_title(channel_name)
                    channel_ax.set_xlabel(x_label)
                    channel_ax.set_ylabel("ADC" if channel_name != "Phase" else "rad")
                    channel_ax.grid(True, alpha=0.25)
            elif is_2d:
                plot_data = _process_iq(iqdata, iq_process)
                final_data = _normalize_rows(plot_data)
                im = final_ax.pcolormesh(
                    x_axis_vals, y_axis_vals, final_data, cmap="viridis"
                )
                final_fig.colorbar(im, ax=final_ax, label="Normalized Amplitude")
                final_ax.set_ylabel(y_label)
            else:
                plot_data = _process_iq(iqdata, iq_process)
                final_ax.plot(x_axis_vals, plot_data, "o-", markersize=5, alpha=0.7)
                final_ax.set_ylabel(_y_label_proc)
        else:
            final_ax.text(
                0.5, 0.5, "No data acquired",
                ha="center", va="center", transform=final_ax.transAxes,
            )
        display(final_fig)
        plt.close(final_fig)

    return iqdata, interrupted, last_i + 1


def _liveplot_sweep_yoko(
    prog,
    soc,
    py_avg,
    x_axis_vals,
    y_axis_vals_yoko,
    instrument_manager,
    yoko_name,
    yoko_mode="current",
    yoko_voltage_ramp_step=1e-5,
    yoko_current_ramp_step=1e-8,
    yoko_ramp_interval=0.01,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="Experiment",
    iq_process="abs",
):
    _colorbar_label = _process_label(iq_process)

    if yoko_name is None:
        raise ValueError("yoko_name must be provided when using instrument_manager.")
    yoko_label = yoko_name
    ramp = instrument_manager.ramp(yoko_name)
    voltage_step = ramp.get("voltage_step")
    current_step = ramp.get("current_step")
    interval = ramp.get("interval")
    voltage_step_text = f"{voltage_step:.2e}" if voltage_step is not None else "unknown"
    current_step_text = f"{current_step:.2e}" if current_step is not None else "unknown"
    interval_text = f"{interval * 1e3:.1f}" if interval is not None else "unknown"

    iqdata_full = np.zeros((len(y_axis_vals_yoko), len(x_axis_vals)), dtype=complex)
    data_to_plot = np.zeros((len(y_axis_vals_yoko), len(x_axis_vals)))
    interrupted = False
    last_idx = 0

    fig, ax = plt.subplots(figsize=(6, 4))

    try:
        yoko_unit = "A" if yoko_mode == "current" else "V"
        value_info = auto_unit(y_axis_vals_yoko, yoko_unit)
        plot_x_vals = value_info["value"]
        dynamic_x_label = f"{y_label} ({value_info['unit']})"
        current_yoko_unit = value_info["unit"]
    except NameError:
        plot_x_vals = y_axis_vals_yoko
        dynamic_x_label = y_label
        current_yoko_unit = yoko_unit
    print(
        f"Yoko sweep: {yoko_label} | mode: {yoko_mode} | unit: {current_yoko_unit} | "
        f"V_step: {voltage_step_text} V  "
        f"I_step: {current_step_text} A  "
        f"interval: {interval_text} ms"
    )
    plot_y_vals = x_axis_vals
    dynamic_y_label = x_label

    mesh = ax.pcolormesh(
        plot_x_vals,
        plot_y_vals,
        data_to_plot.T,
        shading="nearest",
        cmap="viridis",
    )

    ax.set_xlabel(dynamic_x_label)
    ax.set_ylabel(dynamic_y_label)

    plot_display_id = f"live-plot-yoko-swapped-{np.random.randint(1e9)}"
    display_handle = display(fig, display_id=plot_display_id)

    try:
        for idx, val in enumerate(
            tqdm(y_axis_vals_yoko, desc=f"Sweeping {yoko_mode} (Plot X-axis)")
        ):
            last_idx = idx
            title = auto_unit(val)
            instrument_manager.set_value(yoko_name, val, mode=yoko_mode)
            suffix = "A" if yoko_mode == "current" else "V"
            ax.set_title(f"{title_prefix} | {title['value']:.2f}{title['unit']}{suffix}")

            iq_list = prog.acquire(soc, rounds=py_avg, progress=False)
            iq_data_row = _iq_to_complex(iq_list)

            iqdata_full[idx, :] = iq_data_row
            data_to_plot = _process_iq(iqdata_full, iq_process)

            mesh.set_array(data_to_plot.T.ravel())

            measured_data = data_to_plot[: idx + 1, :]
            current_max = np.max(measured_data)
            current_min = np.min(measured_data)

            if current_max > current_min:
                mesh.set_clim(vmin=current_min, vmax=current_max)
            elif current_max > 0:
                mesh.set_clim(vmin=0, vmax=current_max)

            update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True
        pass

    clear_output(wait=True)

    if interrupted:
        print(f"KeyboardInterrupt: Interrupted at Yoko step: {last_idx + 1}")

    ax.cla()
    title_status = f"Interrupted at step {last_idx + 1}" if interrupted else "Completed"
    ax.set_title(f"{title_prefix} ({title_status})")

    if len(y_axis_vals_yoko) == 1:
        row = data_to_plot[0, :]
        ax.plot(plot_y_vals, row, lw=1.0)
        ax.set_xlabel(dynamic_y_label)
        ax.set_ylabel("Amplitude")
    else:
        ax.set_xlabel(dynamic_x_label)
        ax.set_ylabel(dynamic_y_label)

        measured_data_final = (
            data_to_plot[: last_idx + 1, :] if interrupted else data_to_plot
        )
        final_min = np.min(measured_data_final) if measured_data_final.size > 0 else 0
        final_max = np.max(measured_data_final) if measured_data_final.size > 0 else 1
        if final_min == final_max:
            final_min = 0

        im = ax.pcolormesh(
            plot_x_vals,
            plot_y_vals,
            data_to_plot.T,
            shading="nearest",
            vmin=final_min,
            vmax=final_max,
        )
        fig.colorbar(im, ax=ax, label=_colorbar_label)

    display(fig)
    plt.close(fig)

    return iqdata_full, interrupted, last_idx + 1


def _liveplot_1d_scan(
    soc,
    py_avg,
    scan_x_axis,
    get_prog_callback,
    x_label="Scan Parameter",
    title_prefix="1D Scan",
    show_final_plot=True,
    iq_process="abs",
):
    _y_label_proc = _process_label(iq_process)
    plot_all_iq = _is_all_iq(iq_process)

    iq_sum = 0
    iqdata = None
    last_avg = 0
    interrupted = False

    if plot_all_iq:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        axes_flat = axes.ravel()
        line = {}
        for channel_ax, channel_name in zip(axes_flat, ("Abs", "Phase", "I", "Q")):
            (channel_line,) = channel_ax.plot(
                scan_x_axis, np.zeros_like(scan_x_axis), "o-", markersize=4, alpha=0.75
            )
            line[channel_name] = channel_line
            channel_ax.set_xlabel(x_label)
            channel_ax.set_ylabel("ADC" if channel_name != "Phase" else "rad")
            channel_ax.set_title(channel_name)
            channel_ax.grid(True, alpha=0.25)
        title = fig.suptitle(f"{title_prefix} (Initializing...)")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        (line,) = ax.plot(
            scan_x_axis, np.zeros_like(scan_x_axis), "o-", markersize=5, alpha=0.7
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(_y_label_proc)
        title = ax.set_title(f"{title_prefix} (Initializing...)")

    plot_display_id = f"live-plot-1d-{np.random.randint(1e9)}"
    display(fig, display_id=plot_display_id)

    try:
        for avg in tqdm(range(py_avg), desc="Average Count"):
            last_avg = avg
            iqlst = []
            for val in scan_x_axis:
                prog = get_prog_callback(val)
                iq_list = prog.acquire(soc, rounds=1, progress=False)
                iqlst.append(_iq_to_complex(iq_list))

            current_iq_data = np.array(iqlst)
            iq_sum = current_iq_data if avg == 0 else iq_sum + current_iq_data
            iqdata = iq_sum / (avg + 1)

            if plot_all_iq:
                for channel_ax, (channel_name, plot_data) in zip(
                    axes_flat, _iq_channel_dict(iqdata).items()
                ):
                    line[channel_name].set_ydata(plot_data)
                    channel_ax.set_ylim(*_safe_limits(plot_data, pad_fraction=0.05))
            else:
                plot_data = _process_iq(iqdata, iq_process)
                line.set_ydata(plot_data)

                ax.set_ylim(*_safe_limits(plot_data, pad_fraction=0.05))

            title.set_text(f"{title_prefix} | Average: {avg + 1} / {py_avg}")
            update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True
    except Exception as e:
        print(f"An error occurred during 1D scan: {e}")
        interrupted = True

    clear_output(wait=True)
    if interrupted:
        print(f"Scan interrupted at average: {last_avg + 1}")

    if show_final_plot:
        title_status = "Interrupted" if interrupted else "Completed"
        if plot_all_iq:
            fig_final, axes_final = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
            fig_final.suptitle(f"{title_prefix} ({title_status} at avg {last_avg + 1})")
        else:
            fig_final, ax_final = plt.subplots(figsize=(6, 4))
            ax_final.set_title(f"{title_prefix} ({title_status} at avg {last_avg + 1})")
            ax_final.set_xlabel(x_label)
            ax_final.set_ylabel(_y_label_proc)

        if iqdata is not None:
            if plot_all_iq:
                for channel_ax, (channel_name, plot_data) in zip(
                    axes_final.ravel(), _iq_channel_dict(iqdata).items()
                ):
                    channel_ax.plot(
                        scan_x_axis, plot_data, "o-", markersize=4, alpha=0.75
                    )
                    channel_ax.set_xlabel(x_label)
                    channel_ax.set_ylabel("ADC" if channel_name != "Phase" else "rad")
                    channel_ax.set_title(channel_name)
                    channel_ax.grid(True, alpha=0.25)
            else:
                ax_final.plot(
                    scan_x_axis,
                    _process_iq(iqdata, iq_process),
                    "o-",
                    markersize=5,
                    alpha=0.7,
                )
        else:
            target_ax = axes_final.ravel()[0] if plot_all_iq else ax_final
            target_ax.text(
                0.5, 0.5, "No data acquired",
                ha="center", va="center", transform=target_ax.transAxes,
            )

        display(fig_final)
        plt.close(fig_final)

    plt.close(fig)
    return iqdata, interrupted, last_avg + 1


def _liveplot_2d_scan(
    soc,
    py_avg,
    scan_x_axis,
    scan_y_axis,
    get_prog_callback,
    x_label="X Axis",
    y_label="Y Axis",
    title_prefix="2D Scan",
    show_final_plot=True,
    iq_process="abs",
):
    _colorbar_label = _process_label(iq_process)

    iqdata_full = np.zeros((len(scan_y_axis), len(scan_x_axis)), dtype=complex)
    data_to_plot = np.zeros((len(scan_y_axis), len(scan_x_axis)))
    interrupted = False
    last_y_idx, last_x_idx = 0, 0

    fig, ax = plt.subplots(figsize=(6, 4))

    mesh = ax.pcolormesh(
        scan_x_axis,
        scan_y_axis,
        data_to_plot,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label=_colorbar_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix} (Initializing...)")

    plot_display_id = f"live-plot-2d-scan-{np.random.randint(1e9)}"
    display(fig, display_id=plot_display_id)

    try:
        for y_idx, y_val in enumerate(
            tqdm(scan_y_axis, desc=f"Outer Sweep: {y_label}")
        ):
            last_y_idx = y_idx
            for x_idx, x_val in enumerate(
                tqdm(scan_x_axis, desc=f"Inner Sweep: {x_label}", leave=False)
            ):
                last_x_idx = x_idx

                prog = get_prog_callback(x_val, y_val)
                iq_list = prog.acquire(soc, rounds=py_avg, progress=False)
                iq_data_pt = _iq_to_complex(iq_list)

                iqdata_full[y_idx, x_idx] = iq_data_pt
                data_to_plot = _process_iq(iqdata_full, iq_process)

                mesh.set_array(data_to_plot.ravel())

                total_measured = y_idx * len(scan_x_axis) + x_idx + 1
                measured_data = data_to_plot.ravel()[:total_measured]

                current_max = np.max(measured_data)
                current_min = np.min(measured_data)

                if current_max > current_min:
                    mesh.set_clim(vmin=current_min, vmax=current_max)
                elif current_max > 0:
                    mesh.set_clim(vmin=0, vmax=current_max)

                ax.set_title(
                    f"{title_prefix} | {y_label}={y_val:.2f}, {x_label}={x_val:.2f}"
                )
                update_display(fig, display_id=plot_display_id)

    except KeyboardInterrupt:
        interrupted = True

    clear_output(wait=True)
    if interrupted:
        print(
            f"Scan interrupted at {y_label}: {scan_y_axis[last_y_idx]}, {x_label}: {scan_x_axis[last_x_idx]}"
        )

    ax.cla()
    title_status = "Interrupted" if interrupted else "Completed"
    ax.set_title(f"{title_prefix} ({title_status})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    total_measured = (
        last_y_idx * len(scan_x_axis) + last_x_idx + 1
        if interrupted
        else data_to_plot.size
    )
    final_measured = data_to_plot.ravel()[:total_measured]
    final_min = np.min(final_measured) if final_measured.size > 0 else 0
    final_max = np.max(final_measured) if final_measured.size > 0 else 1
    if final_min == final_max:
        final_min = 0

    im = ax.pcolormesh(
        scan_x_axis,
        scan_y_axis,
        data_to_plot,
        shading="auto",
        cmap="viridis",
        vmin=final_min,
        vmax=final_max,
    )
    fig.colorbar(im, ax=ax, label=_colorbar_label)

    if show_final_plot:
        display(fig)

    plt.close(fig)

    return iqdata_full, interrupted, py_avg


__all__ = [
    "LivePlotSession",
    "LineRenderer",
    "MeshRenderer",
    "SoftwareAverageRunner",
    "run_software_average_liveplot",
    "liveplotfun",
]
