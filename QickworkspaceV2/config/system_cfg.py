"""
system_cfg — experiment configuration template.

This file keeps a sample ``config_list`` that users can copy and edit for a
specific measurement setup. It is intentionally not imported from
``QickworkspaceV2.config`` as package API.

Typical notebook usage:
    from QickworkspaceV2.tools.system_tool import ExperimentConfig
    from QickworkspaceV2.config.system_cfg import config_list

    qubit = "Q1"
    config_all = ExperimentConfig(config_list)
    run_cfg = config_all.get_qubit(qubit)

    # dot-notation nested update
    config_all.update("res.res_gain_ge", 0.4, q_index=qubit)

    # flat key update
    config_all.update("qb_freq_ge", 4998.7, q_index=qubit)

    # flat cfg with list-of-tuples (plain dict .update() syntax)
    run_cfg = config_all.get_qubit(qubit)
    run_cfg.update([
        ("steps", 101),
        ("res_freq_ge", QickSweep1D("freqloop", START, STOP)),
        ("relax_delay", 0),
    ])
"""

# One dict per qubit.  Nested sub-dicts (ch, res, qb, cooling) are flattened
# by ExperimentConfig.get_qubit(), so experiments always see a flat dict.
# Dot-notation updates work on the nested keys:
#   config_all.update("res.res_gain_ge", 0.4, q_index="Q1")
# Plain-key updates (auto-search) also work:
#   config_all.update("res_gain_ge", 0.4, q_index="Q1")

config_list = [
    {
        "name": "Q1",
        "ch": {
            "res_ch": 0,
            "qb_ch": 0,
            "qb_ch_ef": 2,
            "ro_ch": 0,
        },
        "res": {
            "res_freq_ge": 6717,
            "res_length": 10,
            "res_gain_ge": 0.2,
            "res_phase": 0,
            "res_sigma": 0.005,
            "ro_length": 5,
            "nqz_res": 2,
            "chi": False,
        },
        "qb": {
            "pulse_type": "arb",
            "qb_freq_ge": 2872.649825,
            "qb_mixer": 2872.649825,
            "qb_gain_ge": 0.1,
            "qb_phase": 0,
            "sigma_ge": 0.05,
            "pi_gain_ge": 0.1,
            "pi2_gain_ge": 0.1,
            "qb_flat_top_length_ge": 0.1,
            "qb_freq_ef": 4000,
            "qb_mixer_ef": 4000,
            "qb_gain_ef": 0.1,
            "qb_phase_ef": 0,
            "sigma_ef": 0.05,
            "pi_gain_ef": 0.1,
            "pi2_gain_ef": 0.1,
            "qb_flat_top_length_ef": 0.1,
            "virtual_detune": 2,
            "nqz_qb": 1,
            "nqz_qb_ef": 1,
        },
        "cooling": {
            "cooling": False,
            "cool_ch1": 2,
            "cool_freq_1": 5400,
            "cool_gain_1": 0.1,
            "nqz_cool_ch1": 2,
            "cool_length": 100,
            "cool_ch2": 3,
            "cool_freq_2": 5400,
            "cool_gain_2": 0.1,
            "nqz_cool_ch2": 2,
        },
        "reps": 100,
        "trig_time": 0.5,
        "relax_delay": 10,
    },
    {
        "name": "Q2",
        "ch": {
            "res_ch": 0,
            "qb_ch": 0,
            "qb_ch_ef": 2,
            "ro_ch": 0,
        },
        "res": {
            "res_freq_ge": 6740,
            "res_length": 10,
            "res_gain_ge": 0.1,
            "res_phase": 0,
            "res_sigma": 0.005,
            "ro_length": 5,
            "nqz_res": 2,
            "chi": False,
        },
        "qb": {
            "pulse_type": "arb",
            "qb_freq_ge": 3620.80596,
            "qb_mixer": 3620.80596,
            "qb_gain_ge": 0.1,
            "qb_phase": 0,
            "sigma_ge": 0.05,
            "pi_gain_ge": 0.1,
            "pi2_gain_ge": 0.1,
            "qb_flat_top_length_ge": 0.1,
            "qb_freq_ef": 4000,
            "qb_mixer_ef": 4000,
            "qb_gain_ef": 0.1,
            "qb_phase_ef": 0,
            "sigma_ef": 0.05,
            "pi_gain_ef": 0.1,
            "pi2_gain_ef": 0.1,
            "qb_flat_top_length_ef": 0.1,
            "virtual_detune": 2,
            "nqz_qb": 1,
            "nqz_qb_ef": 1,
        },
        "cooling": {
            "cooling": False,
            "cool_ch1": 2,
            "cool_freq_1": 5400,
            "cool_gain_1": 0.1,
            "nqz_cool_ch1": 2,
            "cool_length": 100,
            "cool_ch2": 3,
            "cool_freq_2": 5400,
            "cool_gain_2": 0.1,
            "nqz_cool_ch2": 2,
        },
        "reps": 100,
        "trig_time": 0.5,
        "relax_delay": 10,
    },
    {
        "name": "Q3",
        "ch": {
            "res_ch": 0,
            "qb_ch": 0,
            "qb_ch_ef": 2,
            "ro_ch": 0,
        },
        "res": {
            "res_freq_ge": 6813,
            "res_length": 10,
            "res_gain_ge": 0.1,
            "res_phase": 0,
            "res_sigma": 0.005,
            "ro_length": 5,
            "nqz_res": 2,
            "chi": False,
        },
        "qb": {
            "pulse_type": "arb",
            "qb_freq_ge": 2777.488323,
            "qb_mixer": 2777.488323,
            "qb_gain_ge": 0.1,
            "qb_phase": 0,
            "sigma_ge": 0.05,
            "pi_gain_ge": 0.1,
            "pi2_gain_ge": 0.1,
            "qb_flat_top_length_ge": 0.1,
            "qb_freq_ef": 4000,
            "qb_mixer_ef": 4000,
            "qb_gain_ef": 0.1,
            "qb_phase_ef": 0,
            "sigma_ef": 0.05,
            "pi_gain_ef": 0.1,
            "pi2_gain_ef": 0.1,
            "qb_flat_top_length_ef": 0.1,
            "virtual_detune": 2,
            "nqz_qb": 1,
            "nqz_qb_ef": 1,
        },
        "cooling": {
            "cooling": False,
            "cool_ch1": 2,
            "cool_freq_1": 5400,
            "cool_gain_1": 0.1,
            "nqz_cool_ch1": 2,
            "cool_length": 100,
            "cool_ch2": 3,
            "cool_freq_2": 5400,
            "cool_gain_2": 0.1,
            "nqz_cool_ch2": 2,
        },
        "reps": 100,
        "trig_time": 0.5,
        "relax_delay": 10,
    },
    {
        "name": "Q4",
        "ch": {
            "res_ch": 0,
            "qb_ch": 0,
            "qb_ch_ef": 2,
            "ro_ch": 0,
        },
        "res": {
            "res_freq_ge": 6890,
            "res_length": 10,
            "res_gain_ge": 0.1,
            "res_phase": 0,
            "res_sigma": 0.005,
            "ro_length": 5,
            "nqz_res": 2,
            "chi": False,
        },
        "qb": {
            "pulse_type": "arb",
            "qb_freq_ge": 3312.01532,
            "qb_mixer": 3312.01532,
            "qb_gain_ge": 0.1,
            "qb_phase": 0,
            "sigma_ge": 0.05,
            "pi_gain_ge": 0.1,
            "pi2_gain_ge": 0.1,
            "qb_flat_top_length_ge": 0.1,
            "qb_freq_ef": 4000,
            "qb_mixer_ef": 4000,
            "qb_gain_ef": 0.1,
            "qb_phase_ef": 0,
            "sigma_ef": 0.05,
            "pi_gain_ef": 0.1,
            "pi2_gain_ef": 0.1,
            "qb_flat_top_length_ef": 0.1,
            "virtual_detune": 2,
            "nqz_qb": 1,
            "nqz_qb_ef": 1,
        },
        "cooling": {
            "cooling": False,
            "cool_ch1": 2,
            "cool_freq_1": 5400,
            "cool_gain_1": 0.1,
            "nqz_cool_ch1": 2,
            "cool_length": 100,
            "cool_ch2": 3,
            "cool_freq_2": 5400,
            "cool_gain_2": 0.1,
            "nqz_cool_ch2": 2,
        },
        "reps": 100,
        "trig_time": 0.5,
        "relax_delay": 10,
    },
]
