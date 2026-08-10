import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from QickworkspaceV2.calibration.store import CalibrationStore
from QickworkspaceV2.core.experiment_components import SweepAxes, _infer_iq_dims
from QickworkspaceV2.core.experiment_data import ExperimentData, QualityFlag
from QickworkspaceV2.experiments.characterization.allxy import AllXY
from QickworkspaceV2.experiments.qubit_ef.rabi_ef import PowerRabiEf
from QickworkspaceV2.experiments.qubit_ge.aae import PowerRabiChevron
from QickworkspaceV2.experiments.qubit_ge.drag import DragCalibration
from QickworkspaceV2.experiments.qubit_ge.rabi import (
    PowerRabi,
    PowerRabiReset,
    TimeRabi,
)
from QickworkspaceV2.experiments.qubit_ge.rabi_reset import ActiveResetRabi
from QickworkspaceV2.experiments_mux._common import fit_quality
from QickworkspaceV2.tools.hdf5_store import _normalise_tags, validate_file


class DataContractTests(unittest.TestCase):
    def test_builtin_tags_use_canonical_spelling(self):
        self.assertEqual(
            _normalise_tags([
                "t1",
                "T1",
                " t2 ",
                "one tone",
                "one_tone",
                "TWOTONE",
                "single shot",
                "spin_echo",
                "mux t1",
                "mux-one-tone",
                "allxy",
                "my-custom-tag",
            ]),
            [
                "T1",
                "T2",
                "OneTone",
                "TwoTone",
                "SingleShot",
                "Spin Echo",
                "MuxT1",
                "MuxOneTone",
                "AllXY",
                "my-custom-tag",
            ],
        )
        self.assertEqual(
            _normalise_tags(["power rabi", "time_rabi", "custom"]),
            ["PowerRabi", "TimeRabi", "custom"],
        )
        self.assertEqual(
            _normalise_tags(["Rabi"], "s004_time_rabi_ge"),
            ["TimeRabi"],
        )
        self.assertEqual(
            _normalise_tags(["Rabi"], "s005_power_rabi_ge"),
            ["PowerRabi"],
        )
        self.assertEqual(
            _normalise_tags(["ALLXY", "DRAGCalibration"]),
            ["AllXY", "Drag"],
        )

    def test_experiment_classes_publish_specific_rabi_allxy_drag_tags(self):
        self.assertEqual(TimeRabi.TAG, "TimeRabi")
        for experiment in (
            PowerRabi,
            PowerRabiReset,
            PowerRabiChevron,
            ActiveResetRabi,
            PowerRabiEf,
        ):
            self.assertEqual(experiment.TAG, "PowerRabi")
        self.assertEqual(AllXY.TAG, "AllXY")
        self.assertEqual(DragCalibration.TAG, "Drag")

    def test_experiment_data_is_json_serializable_and_round_trips(self):
        original = ExperimentData(
            experiment_type="demo",
            experiment_id="test-id",
            raw_iq=np.array([1 + 2j, 3 + 4j]),
            x_axis=np.array([0.0, 1.0]),
            metadata={"nested": np.array([1, 2])},
            children=["child-id"],
        )

        payload = original.to_dict()
        json.dumps(payload)
        restored = ExperimentData.from_dict(payload)

        np.testing.assert_array_equal(restored.raw_iq, original.raw_iq)
        np.testing.assert_array_equal(restored.x_axis, original.x_axis)
        self.assertEqual(restored.metadata["nested"], [1, 2])
        self.assertEqual(restored.children, ["child-id"])

    def test_dimension_inference_always_matches_rank(self):
        axes = SweepAxes(x=np.arange(4), y=np.arange(3))
        dims = _infer_iq_dims((2, 3, 4), axes)
        self.assertEqual(dims, ["readout", "y", "x"])

    def test_hdf5_round_trip_preserves_complex_fit_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.h5"
            result = ExperimentData(
                experiment_type="resonator",
                raw_iq=np.array([1 + 1j, 2 + 2j]),
                x_axis=np.array([1.0, 2.0]),
                analysis_data={
                    "fit_curve": {
                        "values": np.array([1.1 + 1.2j, 1.9 + 2.1j]),
                        "dims": ["x"],
                    }
                },
                dataset_dims={"iq": ["x"]},
            )
            result.save(path, catalog=False)

            restored = ExperimentData.load(path)
            np.testing.assert_allclose(
                restored.analysis_data["fit_curve"]["values"],
                result.analysis_data["fit_curve"]["values"],
            )
            self.assertTrue(validate_file(path).valid)

    def test_hdf5_round_trip_preserves_named_multi_readouts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multi-readout.h5"
            readouts = np.array([
                [1 + 1j, 2 + 2j],
                [3 + 3j, 4 + 4j],
            ])
            result = ExperimentData(
                experiment_type="active_reset",
                raw_iq=readouts[0],
                x_axis=np.array([0.0, 1.0]),
                axes={"readout": {"values": ["pre_reset", "post_reset"]}},
                raw_data={"readouts": readouts},
                dataset_dims={
                    "iq": ["x"],
                    "readouts": ["readout", "x"],
                },
            )

            result.save(path, catalog=False)
            restored = ExperimentData.load(path)

            np.testing.assert_array_equal(
                restored.get_readout("pre_reset"), readouts[0]
            )
            np.testing.assert_array_equal(
                restored.get_readout("post_reset"), readouts[1]
            )
            self.assertTrue(validate_file(path).valid)

    def test_calibration_store_saves_atomically_and_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            store = CalibrationStore(path)
            store.set("Q1", "frequency", np.float64(4200.0))

            restored = CalibrationStore(path)
            self.assertEqual(restored.get("Q1", "frequency"), 4200.0)
            self.assertFalse(list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_mux_fit_quality_rejects_acquisition_without_a_fit(self):
        quality, message = fit_quality(True, 0, 2, "Mux T1")
        self.assertIs(quality, QualityFlag.BAD)
        self.assertIn("all fits failed", message)


if __name__ == "__main__":
    unittest.main()
