"""Hardware-free regression checks for normalized measurement execution."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from QickworkspaceV2_agent.core.experiment_data import ExperimentData, QualityFlag
from QickworkspaceV2_agent.nodes.adapters import execute_experiment, ExperimentContractError


def spec(adapter="standard", identifier="example"):
    return SimpleNamespace(id=identifier, adapter=adapter, class_path="QickworkspaceV2_agent.fake.Experiment")


class NoTruthData(ExperimentData):
    def __bool__(self):
        raise AssertionError("ExperimentData must not be tested for truth")


class AdapterTests(unittest.TestCase):
    def execute(self, cls, *, adapter="standard", run=None, context=None, init=None):
        with patch("QickworkspaceV2_agent.nodes.adapters._resolve_class", return_value=cls):
            return execute_experiment(spec(adapter), {"name": "Q1"}, run or {}, context=context, init_options=init)

    def test_scalarless_result_and_legacy_tuple_preserve_raw_data(self):
        class Legacy:
            def __init__(self, cfg):
                self.cfg = cfg
            def run(self, py_avg, liveplot=True):
                self.liveplot_requested = liveplot
                self.result = NoTruthData(raw_iq=np.array([1 + 2j, 3 + 4j]))
                return np.array([1, 2]), np.array([.1, .2])
        experiment, outputs = self.execute(Legacy, run={"py_avg": 2})
        self.assertFalse(experiment.liveplot_requested)
        self.assertIs(outputs["result"], experiment.result)
        np.testing.assert_equal(outputs["result"].raw_iq, [1 + 2j, 3 + 4j])

    def test_unknown_return_is_not_reported_as_a_measurement(self):
        class Unsupported:
            def __init__(self, cfg):
                pass
            def run(self):
                return {"success": True}
        with self.assertRaises(ExperimentContractError) as raised:
            self.execute(Unsupported)
        self.assertIsInstance(raised.exception.experiment_object, Unsupported)

    def test_rb_processed_mean_is_analysis_data_not_a_second_sweep_axis(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        mean = np.array([.9, .8])
        raw = np.array([[.9 + .1j, .8 + .2j], [.7 + .3j, .6 + .4j]])
        class RB:
            def __init__(self, cfg):
                pass
            def run(self):
                return ExperimentData(
                    data_kind="rb", raw_iq=raw, x_axis=np.array([1, 2]), y_axis=mean,
                    axes={"depth": {"values": [1, 2]}, "sample": {"values": [0, 1]}},
                    dataset_dims={"iq": ["depth", "sample"]},
                    analysis_data={"mean_signal": {"values": mean, "dims": ["depth"]}},
                )
        _, outputs = self.execute(RB)
        self.assertIsNone(outputs["result"].y_axis)
        np.testing.assert_equal(outputs["result"].analysis_data["mean_signal"]["values"], mean)
        with TemporaryDirectory() as folder:
            saved = outputs["result"].save(Path(folder) / "rb.h5", catalog=False)
            loaded = ExperimentData.load(saved)
            self.assertIsNone(loaded.y_axis)
            np.testing.assert_equal(loaded.raw_iq, raw)
            np.testing.assert_equal(loaded.analysis_data["mean_signal"]["values"], mean)

    def test_fit_error_retains_acquired_result_for_runner(self):
        class FailedFit:
            def __init__(self, cfg):
                pass
            def run(self):
                self.result = ExperimentData(raw_iq=np.array([2j]))
                raise ValueError("flat trace cannot fit")
        with self.assertRaisesRegex(ValueError, "flat trace") as raised:
            self.execute(FailedFit)
        np.testing.assert_equal(raised.exception.experiment_object.result.raw_iq, [2j])

    def test_auto_rb_retains_child_data_and_comparison_without_plotting(self):
        class Auto:
            def __init__(self, cfg):
                pass
            def run(self):
                self._rb_objects = {
                    "ref": SimpleNamespace(result=ExperimentData(fit_result={"p": (.99, .001)})),
                    "x180": SimpleNamespace(result=ExperimentData(fit_result={"p": (.98, .002)})),
                }
        _, outputs = self.execute(Auto, adapter="auto_rb")
        self.assertEqual(set(outputs), {"ref", "x180"})
        self.assertAlmostEqual(outputs["x180"].get_param("interleaved_gate_epc"), (1 - .98 / .99) / 2)
        self.assertEqual(outputs["x180"].metadata["reference_experiment_id"], outputs["ref"].experiment_id)

    def test_instrument_requires_registered_context_before_constructor(self):
        class External:
            constructed = False
            def __init__(self, config, pump_source, pump_freqs):
                External.constructed = True
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.execute(External, adapter="external_instrument", init={"pump_freqs": [1]})
        self.assertFalse(External.constructed)

    def test_twpa_keeps_complex_xarray_dimensions_and_turns_pump_off(self):
        class Pump:
            off_count = 0
            def off(self):
                self.off_count += 1
        pump = Pump()
        manager = object()
        reference = object()
        class External:
            def __init__(self, config, pump_source, pump_freqs):
                self.pump = pump_source
            def run(self, instrument_manager, yoko_name, reference=None):
                self.manager = instrument_manager
                self.reference = reference
            def _build_gain_xarray(self):
                return SimpleNamespace(
                    dims=("frequency", "pump_freq"), sizes={"frequency": 2, "pump_freq": 1},
                    coords={"frequency": SimpleNamespace(values=np.array([4, 5]), attrs={"units": "GHz"})},
                    values=np.array([[1j], [2j]]),
                )
        context = SimpleNamespace(resources={"pump": pump, "reference": reference}, instrument_manager=manager)
        experiment, outputs = self.execute(
            External, adapter="external_instrument", context=context,
            init={"pump_source": "pump", "pump_freqs": [1]}, run={"yoko_name": "flux", "reference": "reference"},
        )
        result = outputs["result"]
        self.assertIs(experiment.manager, manager)
        self.assertIs(experiment.reference, reference)
        self.assertEqual(pump.off_count, 1)
        self.assertEqual(result.dataset_dims["iq"], ["frequency", "pump_freq"])
        self.assertEqual(result.axes["frequency"]["unit"], "GHz")
        np.testing.assert_equal(result.raw_iq, [[1j], [2j]])
        self.assertEqual(result.quality, QualityFlag.NO_INFORMATION)

    def test_external_failure_still_turns_pump_off(self):
        pump = SimpleNamespace(off=lambda: cleanup.append("off"))
        cleanup = []
        class External:
            def __init__(self, config, pump_source):
                pass
            def run(self, instrument_manager):
                raise RuntimeError("acquisition failed")
        context = SimpleNamespace(resources={"pump_source": pump, "instrument_manager": object()})
        with self.assertRaisesRegex(RuntimeError, "acquisition failed"):
            self.execute(External, adapter="external_instrument", context=context)
        self.assertEqual(cleanup, ["off"])


class HeadlessAnalysisTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_single_shot_analysis_and_plot_share_cached_metrics(self):
        from QickworkspaceV2_agent.experiments.setup.single_shot import SingleShot_gef
        from QickworkspaceV2_agent.experiments.setup.singleshot_utils import hist
        experiment = SingleShot_gef.__new__(SingleShot_gef)
        rng = np.random.default_rng(5)
        experiment.data = {
            "Ig": rng.normal(-3, .3, 120), "Qg": rng.normal(0, .3, 120),
            "Ie": rng.normal(3, .3, 120), "Qe": rng.normal(0, .3, 120),
        }
        experiment.result = ExperimentData(metadata={"states": ["g", "e"]})
        with patch.dict("os.environ", {"LOKY_MAX_CPU_COUNT": "1"}), patch("QickworkspaceV2_agent.experiments.setup.single_shot.hist", wraps=hist) as histogram:
            with patch.object(plt, "subplots", side_effect=AssertionError("headless analysis made a figure")):
                analyzed = experiment.analyze()
                same = experiment.plot(plot_analysis=False)
            self.assertIs(analyzed, same)
            self.assertEqual(histogram.call_count, 1)
            self.assertEqual(experiment.result.figures, [])
            fidelity = experiment.result.get_param("fidelity")
            with patch.object(plt, "show"):
                experiment.plot()
            self.assertEqual(histogram.call_count, 1)
            self.assertEqual(experiment.result.get_param("fidelity"), fidelity)
            self.assertEqual(len(experiment.result.figures), 1)
        self.assertGreater(fidelity, .95)

    def test_optimizer_preserves_grid_before_analysis_failure(self):
        from QickworkspaceV2_agent.experiments.setup.single_shot import SingleShot_ge_opt
        experiment = SingleShot_ge_opt.__new__(SingleShot_ge_opt)
        experiment.soc = experiment.soccfg = object()
        experiment.cfg = {"ro_length": 1, "res_gain_ge": .01, "res_freq_ge": 7000, "relax_delay": 20}
        iq = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        program = SimpleNamespace(acquire=lambda *args, **kwargs: [iq])
        with patch("QickworkspaceV2_agent.experiments.setup.single_shot.SingleShotOptProgram", return_value=program):
            experiment.run(SHOTS=2, sweep_para={})
        self.assertEqual(experiment.result.dataset_dims["iq"], ["length", "gain", "state", "shot"])
        np.testing.assert_equal(experiment.result.raw_iq[0, 0], [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
        self.assertEqual(experiment.result.quality, QualityFlag.NO_INFORMATION)

    def make_tof(self, cls, trace):
        experiment = cls.__new__(cls)
        experiment.soc = object()
        program = SimpleNamespace(acquire_decimated=lambda *args, **kwargs: [np.column_stack((trace, np.zeros(len(trace))))])
        experiment._create_program = lambda: program
        experiment._extract_sweep_axis = lambda prog: np.arange(len(trace), dtype=float)
        return experiment

    def test_tof_and_mux_tof_find_same_crossing_with_no_figures(self):
        from QickworkspaceV2_agent.experiments.setup.time_of_flight import TOF
        from QickworkspaceV2_agent.experiments_mux.time_of_flight import MuxTOF
        for cls in (TOF, MuxTOF):
            with self.subTest(cls=cls.__name__):
                experiment = self.make_tof(cls, [0, 0, 10, 0])
                experiment._normalized_cfg = lambda: {"ro_chs": [0], "mask": [0], "qubit_names": ["Q1"]}
                with patch.object(plt, "subplots", side_effect=AssertionError("headless TOF made a figure")):
                    result = experiment.liveplot(py_avg=2, plot=False)
                self.assertEqual(result.avg_count, 2)
                self.assertEqual(next(iter(result.fit_result.values()))[0], 2.0)
                self.assertEqual(result.quality, QualityFlag.GOOD)

    def test_flat_tof_does_not_propose_first_sweep_point(self):
        from QickworkspaceV2_agent.experiments.setup.time_of_flight import TOF
        experiment = self.make_tof(TOF, [1, 1, 1, 1])
        result = experiment.run(plot=False)
        self.assertEqual(result.fit_result, {})
        self.assertIsNone(result.scalar_result)
        self.assertEqual(result.quality, QualityFlag.WARNING)


class ProgramTransportTests(unittest.TestCase):
    def test_normal_transport_preserves_inherited_calls(self):
        from QickworkspaceV2_agent.core.program_transport import RoutedAcquisition
        class Native:
            def acquire(self, soc, *args, **kwargs):
                return soc, args, kwargs
            def acquire_decimated(self, soc, *args, **kwargs):
                return "decimated", soc, args, kwargs
        class Program(RoutedAcquisition, Native):
            pass
        soc = object()
        program = Program()
        self.assertEqual(program.acquire(soc, 3, rounds=2), (soc, (3,), {"rounds": 2}))
        self.assertEqual(program.acquire_decimated(soc, rounds=1), ("decimated", soc, (), {"rounds": 1}))

    def test_job_transport_routes_both_modes_with_program_identity(self):
        from QickworkspaceV2_agent.core.program_transport import RoutedAcquisition
        class Native:
            def acquire(self, *args, **kwargs):
                raise AssertionError("local acquisition was used")
            acquire_decimated = acquire
        class Program(RoutedAcquisition, Native):
            pass
        calls = []
        soc = SimpleNamespace(acquire_program=lambda program, **kwargs: calls.append((program, kwargs)))
        program = Program()
        program.acquire(soc, threshold=1.2, rounds=2)
        program.acquire_decimated(soc, rounds=1)
        self.assertIs(calls[0][0], program)
        self.assertEqual(calls[0][1], {"mode": "accumulated", "threshold": 1.2, "rounds": 2})
        self.assertEqual(calls[1][1], {"mode": "decimated", "rounds": 1})
        with self.assertRaises(TypeError):
            program.acquire(soc, 5)

    def test_all_native_program_subclasses_route_without_changing_pulses(self):
        import ast
        from pathlib import Path
        from QickworkspaceV2_agent.core.program_transport import RoutedAcquisition
        from QickworkspaceV2_agent.core.base_program import BaseProgram
        from QickworkspaceV2_agent.experiments_mux import time_of_flight, single_shot, one_tone
        for cls in (BaseProgram, time_of_flight.MuxTOFProgram, single_shot.MuxSingleShotGEProgram, one_tone.MuxOneToneProgram):
            self.assertTrue(issubclass(cls, RoutedAcquisition))
        root = Path(__file__).resolve().parents[1]
        changed = [root / "QickworkspaceV2_agent/core/base_program.py"]
        changed += list((root / "QickworkspaceV2_agent/experiments_mux").glob("*.py"))
        count = 0
        for path in changed:
            old = root / "QickworkspaceV2" / path.relative_to(root / "QickworkspaceV2_agent")
            new_tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            old_tree = ast.parse(old.read_text(encoding="utf-8-sig"))
            for new_class in (node for node in new_tree.body if isinstance(node, ast.ClassDef)):
                if any(isinstance(base, ast.Name) and base.id == "AveragerProgramV2" for base in new_class.bases):
                    count += 1
                    self.assertTrue(any(isinstance(base, ast.Name) and base.id == "RoutedAcquisition" for base in new_class.bases))
                    old_class = next(node for node in old_tree.body if isinstance(node, ast.ClassDef) and node.name == new_class.name)
                    for method in (node for node in old_class.body if isinstance(node, ast.FunctionDef)):
                        new_method = next(node for node in new_class.body if isinstance(node, ast.FunctionDef) and node.name == method.name)
                        self.assertEqual(ast.dump(method), ast.dump(new_method), f"Pulse definition changed: {path.name}:{method.name}")
        self.assertEqual(count, 12)


if __name__ == "__main__":
    unittest.main()
