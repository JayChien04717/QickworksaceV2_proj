import unittest
from unittest.mock import patch

import numpy as np

from QickworkspaceV2.experiments.setup import singleshot_utils


class SingleShotGMMReuseTests(unittest.TestCase):
    def test_optimizer_metrics_reuse_histogram_gmm_fit(self):
        rng = np.random.default_rng(3)
        data = {
            "Ig": rng.normal(-1.0, 0.15, 300),
            "Qg": rng.normal(0.0, 0.10, 300),
            "Ie": rng.normal(1.0, 0.15, 300),
            "Qe": rng.normal(0.0, 0.10, 300),
        }
        original_fit = singleshot_utils._fit_gmm

        with patch.object(
            singleshot_utils, "_fit_gmm", wraps=original_fit
        ) as fit_gmm:
            details = singleshot_utils.hist(
                data, plot=False, verbose=False, return_details=True
            )
            metrics = singleshot_utils.histogram_metrics(details)

        self.assertEqual(fit_gmm.call_count, 1)
        self.assertAlmostEqual(metrics["fid"], details.legacy_result[0][0])
        self.assertGreater(metrics["snr"], 0.0)

    def test_legacy_histogram_return_shape_is_unchanged(self):
        rng = np.random.default_rng(4)
        data = {
            "Ig": rng.normal(-1.0, 0.2, 100),
            "Qg": np.zeros(100),
            "Ie": rng.normal(1.0, 0.2, 100),
            "Qe": np.zeros(100),
        }

        result = singleshot_utils.hist(data, plot=False, verbose=False)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
