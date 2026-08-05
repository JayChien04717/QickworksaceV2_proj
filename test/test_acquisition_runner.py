import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from QickworkspaceV2.core.acquisition import decode_acquisition

from QickworkspaceV2.core.experiment_components import (
    AcquisitionResult,
    AcquisitionRunner,
    RunContext,
    SweepAxes,
)


class _Experiment:
    soc = object()
    X_LABEL = "x"
    Y_LABEL = "y"
    TITLE_PREFIX = "test"
    YOKO_VOLTAGE_RAMP_STEP = 1e-5
    YOKO_CURRENT_RAMP_STEP = 1e-8
    YOKO_RAMP_INTERVAL = 0.01

    @staticmethod
    def _get_readout_threshold():
        return None


class _ThresholdExperiment(_Experiment):
    @staticmethod
    def _get_readout_threshold():
        return 0.25


def _context(*, liveplot, py_avg=4, kwargs=None):
    return RunContext(
        py_avg=py_avg,
        iq_process="all",
        show_final_plot=False,
        liveplot=liveplot,
        plot_analysis=False,
        kwargs=kwargs or {},
    )


class AcquisitionRunnerTests(unittest.TestCase):
    def test_liveplot_false_acquires_rounds_directly_without_importing_plotter(self):
        runner = AcquisitionRunner()
        prog = Mock()
        prog.acquire.return_value = [[np.array([[1.0, 2.0], [3.0, 4.0]])]]
        module_name = "QickworkspaceV2.plotter.liveplot"
        with patch.dict(sys.modules):
            sys.modules.pop(module_name, None)
            result = runner.acquire(
                _Experiment(),
                prog,
                SweepAxes(x=np.array([0.0, 1.0]), y=None),
                _context(liveplot=False, py_avg=7),
            )
            self.assertNotIn(module_name, sys.modules)

        prog.acquire.assert_called_once_with(
            _Experiment.soc, rounds=7, progress=True
        )
        np.testing.assert_array_equal(result.raw_iq, np.array([1 + 2j, 3 + 4j]))
        self.assertEqual(result.avg_count, 7)

    def test_liveplot_true_keeps_liveplot_path(self):
        runner = AcquisitionRunner()
        expected = AcquisitionResult(raw_iq=np.array([1 + 2j]), avg_count=3)

        with patch.object(runner, "_liveplot", return_value=expected) as live:
            with patch.object(runner, "_direct") as direct:
                result = runner.acquire(
                    _Experiment(),
                    Mock(),
                    SweepAxes(x=np.array([0.0]), y=None),
                    _context(liveplot=True, py_avg=3),
                )

        self.assertIs(result, expected)
        live.assert_called_once()
        direct.assert_not_called()

    def test_direct_threshold_uses_rounds_and_returns_real_population(self):
        runner = AcquisitionRunner()
        prog = Mock()
        prog.acquire.return_value = [[np.array([0.75])]]

        result = runner.acquire(
            _ThresholdExperiment(),
            prog,
            SweepAxes(x=np.array([0.0]), y=None),
            _context(liveplot=False, py_avg=6),
        )

        prog.acquire.assert_called_once_with(
            _ThresholdExperiment.soc,
            rounds=6,
            progress=True,
            threshold=0.25,
        )
        self.assertEqual(float(result.raw_iq), 0.75)
        self.assertFalse(np.iscomplexobj(result.raw_iq))
        self.assertEqual(result.scalar_result, 0.75)
        self.assertEqual(result.fit_result, {"population": (0.75, None)})
        self.assertTrue(result.metadata["threshold_discrimination"])

    def test_live_threshold_passes_readout_mode_to_liveplot(self):
        runner = AcquisitionRunner()
        population = np.array([0.2, 0.8])

        with patch(
            "QickworkspaceV2.plotter.liveplot.liveplotfun",
            return_value=(population, False, 4),
        ) as liveplot:
            result = runner.acquire(
                _ThresholdExperiment(),
                Mock(),
                SweepAxes(x=np.array([0.0, 1.0]), y=None),
                _context(liveplot=True, py_avg=4),
            )

        self.assertEqual(
            liveplot.call_args.kwargs["threshold"],
            0.25,
        )
        np.testing.assert_array_equal(result.raw_iq, population)
        self.assertFalse(np.iscomplexobj(result.raw_iq))
        self.assertEqual(
            result.fit_result,
            {"population": ([0.2, 0.8], None)},
        )
        self.assertEqual(result.avg_count, 4)

    def test_direct_yoko_sweep_acquires_each_value_with_py_avg_rounds(self):
        runner = AcquisitionRunner()
        prog = Mock()
        prog.acquire.side_effect = [
            [[np.array([[1.0, 2.0]])]],
            [[np.array([[3.0, 4.0]])]],
        ]
        instrument = Mock()

        result = runner.acquire(
            _Experiment(),
            prog,
            SweepAxes(x=np.array([0.0]), y=np.array([0.1, 0.2])),
            _context(
                liveplot=False,
                py_avg=5,
                kwargs={
                    "instrument_manager": instrument,
                    "yoko_name": "q1_flux",
                    "yoko_mode": "voltage",
                },
            ),
        )

        self.assertEqual(prog.acquire.call_count, 2)
        prog.acquire.assert_called_with(
            _Experiment.soc, rounds=5, progress=False
        )
        self.assertEqual(
            instrument.set_value.call_args_list[0].args, ("q1_flux", 0.1)
        )
        self.assertEqual(
            instrument.set_value.call_args_list[0].kwargs, {"mode": "voltage"}
        )
        np.testing.assert_array_equal(
            result.raw_iq, np.array([[1 + 2j], [3 + 4j]])
        )
        self.assertEqual(result.avg_count, 2)

    def test_threshold_decoder_drops_qick_iq_axis(self):
        acquired = [[np.column_stack((np.linspace(0.0, 1.0, 100), np.zeros(100)))]]

        values = decode_acquisition(acquired, threshold=True)

        self.assertEqual(values.shape, (100,))
        np.testing.assert_allclose(values, np.linspace(0.0, 1.0, 100))

if __name__ == "__main__":
    unittest.main()
