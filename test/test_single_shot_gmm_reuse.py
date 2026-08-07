import unittest
from unittest.mock import patch

import numpy as np

from QickworkspaceV2.experiments.setup import singleshot_utils
from QickworkspaceV2.experiments.setup.single_shot import (
    SingleShotOptProgram,
    SingleShotProgram_gef,
)


class SingleShotGMMReuseTests(unittest.TestCase):
    def test_fast_metrics_avoid_gmm_and_find_empirical_threshold(self):
        rng = np.random.default_rng(21)
        data = {
            "Ig": rng.normal(-1.0, 0.2, 1000),
            "Qg": rng.normal(0.0, 0.2, 1000),
            "Ie": rng.normal(1.0, 0.2, 1000),
            "Qe": rng.normal(0.2, 0.2, 1000),
        }

        with patch.object(
            singleshot_utils, "_fit_gmm", side_effect=AssertionError("GMM used")
        ):
            metrics = singleshot_utils.fast_histogram_metrics(data)

        self.assertGreater(metrics["fid"], 0.99)
        self.assertLess(metrics["g_to_e_error"], 0.01)
        self.assertLess(metrics["e_to_g_error"], 0.01)
        self.assertTrue(np.isfinite(metrics["threshold"]))
        self.assertGreater(metrics["e_core_fraction"], 0.95)
        self.assertLess(metrics["e_tail_fraction"], 0.05)
        self.assertAlmostEqual(
            metrics["readout_score"],
            metrics["fid"] * metrics["e_core_fraction"]
            * (1.0 - metrics["e_to_g_error"]),
        )
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

    def test_t1_weighted_score_prefers_lower_e_to_g_error(self):
        ordinary = singleshot_utils.weighted_assignment_score(
            np.array([0.02, 0.12]), np.array([0.20, 0.12]), e_to_g_weight=1.0
        )
        weighted = singleshot_utils.weighted_assignment_score(
            np.array([0.02, 0.12]), np.array([0.20, 0.12]), e_to_g_weight=3.0
        )

        self.assertGreater(ordinary[0], ordinary[1])
        self.assertGreater(weighted[1], weighted[0])
        self.assertAlmostEqual(
            singleshot_utils.weighted_assignment_score(0.10, 0.20, 1.0),
            0.85,
        )

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

    def test_skewed_density_uses_all_shots_for_empirical_confusion(self):
        rng = np.random.default_rng(12)
        g = np.concatenate((rng.normal(-1.0, 0.15, 2400), rng.normal(-2.2, 0.30, 600)))
        e = np.concatenate((rng.normal(1.0, 0.20, 2100), rng.normal(2.4, 0.40, 900)))
        details = singleshot_utils.hist(
            {"Ig": g, "Qg": np.zeros(g.size), "Ie": e, "Qe": np.zeros(e.size)},
            plot=False, verbose=False, return_details=True,
        )
        metrics = singleshot_utils.histogram_metrics(details)

        threshold = details.thresholds[0]
        state_order = np.argsort([np.mean(values) for values in details.projections])
        expected = np.zeros((2, 2))
        for prepared_state, values in enumerate(details.projections):
            predicted = np.where(values <= threshold, state_order[0], state_order[1])
            for declared_state in range(2):
                expected[prepared_state, declared_state] = np.mean(
                    predicted == declared_state
                )
        np.testing.assert_allclose(details.confusion_matrix_pct / 100.0, expected)
        self.assertAlmostEqual(metrics["leakage"], expected[1, 0])
        self.assertAlmostEqual(metrics["thermal"], expected[0, 1])

        g_core, g_retained = singleshot_utils._dominant_core(g)
        e_core, e_retained = singleshot_utils._dominant_core(e)
        self.assertLess(g_retained, 1.0)
        self.assertLess(e_retained, 1.0)
        self.assertLess(np.std(g_core), np.std(g))
        self.assertLess(np.std(e_core), np.std(e))
    def test_three_state_rotation_optimizes_all_prepared_states(self):
        rng = np.random.default_rng(7)
        centers = [(1.558, 1.264), (0.681, -1.010), (-0.246, -0.350)]
        clouds = [
            rng.normal(i, 0.45, 5000) + 1j * rng.normal(q, 0.45, 5000)
            for i, q in centers
        ]

        gf_angle = -np.arctan2(
            np.mean(clouds[2].imag) - np.mean(clouds[0].imag),
            np.mean(clouds[2].real) - np.mean(clouds[0].real),
        )
        optimized_angle = singleshot_utils._optimize_multistate_rotation(
            [(cloud.real, cloud.imag) for cloud in clouds]
        )

        def fidelity(angle):
            projections = [
                cloud.real * np.cos(angle) - cloud.imag * np.sin(angle)
                for cloud in clouds
            ]
            _, thresholds, confusion = singleshot_utils._ordered_threshold_classifier(
                projections
            )
            self.assertLess(thresholds[0], thresholds[1])
            return float(np.mean(np.diag(confusion)))

        self.assertGreater(fidelity(optimized_angle), fidelity(gf_angle) + 0.05)
    def test_scatter_alpha_tracks_local_density(self):
        dense_x = np.zeros(100)
        dense_y = np.zeros(100)
        x = np.concatenate((dense_x, [10.0]))
        y = np.concatenate((dense_y, [10.0]))

        alpha = singleshot_utils._local_density_alpha(x, y, bins=20)

        self.assertAlmostEqual(alpha[-1], 0.025)
        self.assertAlmostEqual(alpha[0], 0.95)
        self.assertTrue(np.all((alpha >= 0.025) & (alpha <= 0.95)))

    def test_state_shots_are_drawn_in_one_interleaved_collection(self):
        fig, ax = singleshot_utils.plt.subplots()
        try:
            collection = singleshot_utils._density_scatter_interleaved(
                ax,
                [
                    (np.arange(5), np.zeros(5), "tab:blue", "g"),
                    (np.arange(7), np.ones(7), "tab:orange", "e"),
                ],
            )
            self.assertEqual(len(collection.get_offsets()), 12)
            self.assertEqual(ax.get_legend_handles_labels()[1], ["g", "e"])
            self.assertTrue(collection.get_rasterized())
        finally:
            singleshot_utils.plt.close(fig)
    def test_optimizer_and_single_shot_share_the_same_qick_program(self):
        self.assertIs(SingleShotOptProgram._initialize, SingleShotProgram_gef._initialize)
        self.assertIs(SingleShotOptProgram._body, SingleShotProgram_gef._body)

if __name__ == "__main__":
    unittest.main()
