import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from QickworkspaceV2.tools.Labber_saver import (
    LogFile,
    createLogFile_ForData,
    getTraceDict,
)


class TestLabberSaverCompatibility(unittest.TestCase):
    def test_public_reader_signatures_match_labber(self):
        self.assertEqual(
            str(inspect.signature(LogFile.getTraceXY)),
            "(self, y_channel=None, x_channel=None, entry=-1)",
        )
        self.assertEqual(
            str(inspect.signature(LogFile.getData)),
            "(self, name=None, entry=None, inner=None, log=-1)",
        )
        self.assertEqual(str(inspect.signature(LogFile.getAxis)), "(self)")
        self.assertEqual(
            str(inspect.signature(LogFile.getNumberOfEntries)),
            "(self, name=None, log=None)",
        )
        self.assertEqual(
            str(inspect.signature(getTraceDict)),
            "(value=[], x0=0.0, dx=1.0, x1=None, logX=False, x=None)",
        )

    def test_vector_trace_defaults_to_first_channel_and_last_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vector.hdf5"
            log = createLogFile_ForData(
                str(path),
                [{"name": "Signal", "unit": "V", "vector": True, "complex": True}],
                [{"name": "Bias", "unit": "V", "values": np.array([10.0, 20.0])}],
                use_database=False,
            )
            log.addEntry(
                {
                    "Signal": getTraceDict(
                        np.array([1 + 2j, 3 + 4j]), x=np.array([5.0, 7.0])
                    )
                }
            )
            log.addEntry(
                {
                    "Signal": getTraceDict(
                        np.array([11 + 12j, 13 + 14j]), x=np.array([15.0, 17.0])
                    )
                }
            )

            self.assertEqual([item["name"] for item in log.getStepChannels()], ["Bias"])
            np.testing.assert_allclose(log.getTraceXY(entry=0)[0], [5.0, 7.0])
            np.testing.assert_allclose(log.getTraceXY(entry=0)[1], [1 + 2j, 3 + 4j])
            np.testing.assert_allclose(log.getTraceXY()[0], [15.0, 17.0])
            np.testing.assert_allclose(log.getTraceXY()[1], [11 + 12j, 13 + 14j])
            np.testing.assert_allclose(
                log.getData(), [[1 + 2j, 3 + 4j], [11 + 12j, 13 + 14j]]
            )
            self.assertEqual(log.getNumberOfEntries(), 2)

    def test_scalar_trace_uses_selected_step_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = createLogFile_ForData(
                str(Path(tmp) / "scalar.hdf5"),
                [{"name": "Signal", "vector": False, "complex": False}],
                [
                    {"name": "X", "values": np.array([1.0, 2.0, 3.0])},
                    {"name": "Y", "values": np.array([10.0, 20.0])},
                ],
                use_database=False,
            )
            log.addEntry({"Signal": np.array([4.0, 5.0, 6.0])})
            log.addEntry({"Signal": np.array([7.0, 8.0, 9.0])})

            np.testing.assert_allclose(log.getTraceXY(entry=0)[0], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(
                log.getTraceXY(x_channel="Y", entry=0)[0], [10.0, 10.0, 10.0]
            )
            np.testing.assert_allclose(log.getData(inner=1), [5.0, 8.0])

            axes = log.getAxis()
            self.assertEqual(
                [axis["axis_name"] for axis in axes],
                ["X", "Y"],
            )
            np.testing.assert_allclose(axes[0]["axis_values"], [1.0, 2.0, 3.0])
            np.testing.assert_allclose(axes[1]["axis_values"], [10.0, 20.0])

    def test_vector_entries_keep_individual_length_and_linear_x_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = createLogFile_ForData(
                str(Path(tmp) / "varying-vector.hdf5"),
                [{"name": "Signal", "vector": True, "complex": False}],
                [{"name": "Bias", "values": np.array([10.0, 20.0])}],
                use_database=False,
            )
            log.addEntry(
                {"Signal": getTraceDict(np.array([1.0, 2.0]), x0=2.0, dx=0.5)}
            )
            log.addEntry(
                {
                    "Signal": getTraceDict(
                        np.array([3.0, 4.0, 5.0]), x0=3.0, dx=0.25
                    )
                }
            )

            data = log.getData()
            self.assertEqual(data.shape, (2, 3))
            np.testing.assert_allclose(data[0, :2], [1.0, 2.0])
            self.assertTrue(np.isnan(data[0, 2]))
            np.testing.assert_allclose(data[1], [3.0, 4.0, 5.0])

            np.testing.assert_allclose(log.getTraceXY(entry=0)[0], [2.0, 2.5])
            np.testing.assert_allclose(log.getTraceXY(entry=0)[1], [1.0, 2.0])
            np.testing.assert_allclose(
                log.getTraceXY(entry=1)[0], [3.0, 3.25, 3.5]
            )
            np.testing.assert_allclose(log.getTraceXY(entry=1)[1], [3.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
