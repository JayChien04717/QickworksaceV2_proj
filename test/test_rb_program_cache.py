import unittest
from unittest.mock import patch

import numpy as np

from QickworkspaceV2.core.base_experiment import BaseExperiment
from QickworkspaceV2.experiments.characterization.rb import RandomizedBenchmarking


class _FakeRBProgram:
    configs = []
    acquisition_order = []

    def __init__(self, soccfg, *, reps, final_delay, cfg):
        self.index = len(type(self).configs)
        type(self).configs.append(dict(cfg))

    def acquire(self, soc, *, rounds, progress):
        type(self).acquisition_order.append(self.index)
        return [[np.array([[float(self.index), 1.0]])]]


class RBProgramCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_runtime = (
            BaseExperiment._soc,
            BaseExperiment._soccfg,
            BaseExperiment._data_path,
        )
        BaseExperiment.setup(object(), object(), "/tmp")

    @classmethod
    def tearDownClass(cls):
        (
            BaseExperiment._soc,
            BaseExperiment._soccfg,
            BaseExperiment._data_path,
        ) = cls.previous_runtime

    def test_programs_compile_once_and_keep_interleaved_average_order(self):
        _FakeRBProgram.configs = []
        _FakeRBProgram.acquisition_order = []
        config = {"name": "Q1", "reps": 10, "relax_delay": 5.0}
        experiment = RandomizedBenchmarking(config)
        experiment.Analysis = None

        sequence_calls = []

        def fake_sequence(*, n_clifford, n_sample, interleave, seed):
            sequence_calls.append((n_clifford, seed))
            return [["I"]]

        with patch(
            "QickworkspaceV2.experiments.characterization.rb.RBProgram",
            _FakeRBProgram,
        ), patch(
            "QickworkspaceV2.tools.rb_generator.single_qb_rb",
            side_effect=fake_sequence,
        ):
            experiment.run(
                py_avg=3,
                max_circuit_depth=5,
                delta_clifford=2,
                number_sample=2,
                seed=7,
            )

        self.assertEqual(len(sequence_calls), 4)
        self.assertEqual(len(_FakeRBProgram.configs), 4)
        self.assertEqual(
            _FakeRBProgram.acquisition_order,
            [0, 1, 2, 3] * 3,
        )
        self.assertNotIn("gate_seq", config)
        self.assertNotIn("prefix", config)


if __name__ == "__main__":
    unittest.main()
