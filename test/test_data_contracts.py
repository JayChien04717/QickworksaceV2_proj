import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from QickworkspaceV2.calibration.store import CalibrationStore
from QickworkspaceV2.core.experiment_components import SweepAxes, _infer_iq_dims
from QickworkspaceV2.core.experiment_data import ExperimentData, QualityFlag
from QickworkspaceV2.experiments_mux._common import fit_quality
from QickworkspaceV2.tools.hdf5_store import validate_file


class DataContractTests(unittest.TestCase):
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
