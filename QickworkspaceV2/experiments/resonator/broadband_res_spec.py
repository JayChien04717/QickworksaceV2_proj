"""Broadband one-tone spectroscopy for locating several resonators."""

from __future__ import annotations

from ...core.base_experiment import BaseExperiment, SweepAxis
from .res_spec import ResonatorSpec, ResonatorSpecProgram


class BroadbandResonatorSpec(ResonatorSpec):
    """Acquire a wide resonator sweep, then locate and plot every resonator.

    The hardware sequence is exactly :class:`ResonatorSpecProgram`.  This
    wrapper only replaces the single-resonator analysis with the existing
    multi-resonator fit and plot.
    """

    EXPT_NAME = "broadband_resonator_spectrum"
    INCLUDE_QUBIT_IN_FILENAME = False
    TAG = "OneTone"
    X_LABEL = "Frequency (MHz)"
    TITLE_PREFIX = "Broadband Resonator Spectrum"
    SWEEP_KEYS_TO_REMOVE = ["res_freq_ge"]
    X_SAVE_NAME = "Frequency"
    X_SAVE_UNIT = "Hz"
    X_SAVE_SCALE = 1e6

    Analysis = None
    PROGRAM = ResonatorSpecProgram
    X_AXIS = SweepAxis.pulse("res_pulse", "freq")

    def run(
        self,
        py_avg,
        count=4,
        *,
        y_mode="abs",
        show=True,
        detection_options=None,
        **kwargs,
    ):
        """Run the sweep and automatically fit/plot ``count`` resonators.

        ``solve_type`` is accepted in ``kwargs`` for compatibility with old
        ``ResonatorSpec`` notebook cells, but is not used by the broadband
        multi-resonator fit.
        """
        kwargs.pop("solve_type", None)
        result = BaseExperiment.run(self, py_avg, **kwargs)

        fig, ax, frequencies = self.plot_n_resonators(
            count=count,
            y_mode=y_mode,
            show=show,
            **dict(detection_options or {}),
        )
        self.resonator_freqs = frequencies
        self.resonator_figure = fig
        self.resonator_axes = ax
        if all(existing is not fig for existing in result.figures):
            result.figures.append(fig)
        return result


__all__ = ["BroadbandResonatorSpec"]
