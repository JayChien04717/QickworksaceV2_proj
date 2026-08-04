"""Step 1: baseline cryoscope using an ordinary constant flux pulse."""

from __future__ import annotations

from qick.asm_v2 import QickSweep1D

from ._common import CryoscopeExperimentBase, CryoscopeProgramBase


class CryoscopeConstProgram(CryoscopeProgramBase):
    """Sweep a constant-pulse length; all lengths must be at least 3 clocks."""

    def _add_flux_pulse(self, cfg):
        """Add flux pulse.

        Parameters
        ----------
        cfg : Any
            Experiment configuration mapping.
        """
        self.add_loop("flux_length_loop", cfg["steps"])
        self.add_pulse(
            ch=cfg["flux_ch"],
            name="flux_pulse",
            style="const",
            length=cfg["flux_length"],
            freq=0,
            phase=0,
            gain=cfg["flux_gain"],
        )


class CryoscopeConst(CryoscopeExperimentBase):
    """Baseline cryoscope for the time range where ``const`` is legal."""

    EXPT_NAME = "cryoscope_const"
    X_LABEL = "Flux pulse length (us)"
    TITLE_PREFIX = "Cryoscope — constant flux pulse"
    SWEEP_KEYS_TO_REMOVE = ["flux_length"]
    X_SAVE_NAME = "Flux pulse length"
    X_SAVE_UNIT = "us"
    X_SAVE_SCALE = 1.0

    def _create_program(self):
        """Create the QICK program for this experiment.

        Returns
        -------
        Any
            Result of the operation.
        """
        return CryoscopeConstProgram(
            self.soccfg,
            reps=self.cfg["reps"],
            final_delay=self.cfg["relax_delay"],
            cfg=self.cfg,
        )

    def _extract_sweep_axis(self, prog):
        """Extract the primary sweep axis from the program.

        Parameters
        ----------
        prog : Any
            Value for ``prog``.

        Returns
        -------
        Any
            Result of the operation.
        """
        return prog.get_pulse_param("flux_pulse", "length", as_array=True)


def configure_const_sweep(cfg, start_us: float, stop_us: float, steps: int):
    """Return a copied config containing the correctly named QICK sweep.

    Parameters
    ----------
    cfg : Any
        Experiment configuration mapping.
    start_us : float
        Value for ``start_us``.
    stop_us : float
        Value for ``stop_us``.
    steps : int
        Value for ``steps``.

    Returns
    -------
    Any
        Result of the operation.
    """

    configured = dict(cfg)
    configured["steps"] = int(steps)
    configured["flux_length"] = QickSweep1D(
        "flux_length_loop", float(start_us), float(stop_us)
    )
    return configured


__all__ = ["CryoscopeConstProgram", "CryoscopeConst", "configure_const_sweep"]
