from importlib import import_module
from dataclasses import replace
import inspect
import pytest
from QickworkspaceV2 import Device, Session, QICKBackend
from QickworkspaceV2.experiments import EXPERIMENT_MODULES


def test_each_experiment_owns_native_protocol_and_schema(session):
    for name in EXPERIMENT_MODULES:
        module = import_module("QickworkspaceV2.experiments." + name)
        spec = session.registry.get(name)
        assert spec.build.__module__ == module.__name__ + ".program"
        assert spec.parameters.__module__ == module.__name__ + ".parameters"
        programs = [
            v
            for v in vars(module).values()
            if inspect.isclass(v) and v.__module__ == module.__name__ + ".program" and v.__name__.endswith("Program")
        ]
        assert len(programs) == 1
        assert hasattr(programs[0], "_initialize") and hasattr(programs[0], "_body")
        assert programs[0].EXPERIMENT is spec
        assert callable(spec.plot)
        if name.endswith(("_ge", "_ef")):
            assert "transition" not in spec.parameters.model_fields


def test_replacing_ge_experiment_does_not_replace_ef(session):
    ef = session.registry.get("t1_ef")
    ge = session.registry.get("t1_ge")
    session.register(replace(ge, version="2.0.0"), replace=True)
    assert session.registry.get("t1_ef") is ef
    assert session.registry.get("t1_ge").parameters is not ef.parameters


@pytest.mark.parametrize(
    "name,params",
    [
        ("resonator_spec_e", {}),
        ("resonator_spec_f", {}),
        ("power_rabi_chevron_ge", {}),
        ("power_rabi_chevron_ef", {}),
        ("drag_ge", {"delta_mhz": -200}),
        ("drag_ef", {"delta_mhz": -200}),
        ("resonator_flux", {"bias_port": "coupler_c12", "bias_length_us": 100}),
        ("twpa_probe", {"points": 11}),
    ],
)
def test_specialist_native_compilation(name, params, device, native_config, tmp_path):
    from qick import QickConfig

    session = Session(device, QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path)
    _, _, plan, _, _, _ = session._prepare(name, params)
    program = session.backend.compile(plan)
    session._check_readouts(program, plan)
    assert len(program.binprog["pmem"]) > 0


@pytest.mark.parametrize("name", ["qubit_spec_ef", "power_rabi_ef"])
def test_ef_calibration_proposal_updates_only_ef(name, session):
    result = session.run(name)
    from QickworkspaceV2 import FitResult

    parameters = {"center": 2601.0} if name == "qubit_spec_ef" else {"pi_gain": 0.2, "pi2_gain": 0.1}
    result.fits = {"Q1": FitResult("fixture", True, parameters)}
    proposal = session.propose(result)
    assert all(".transitions.ef." in key for key in proposal.updates)
    ge_before = session.device.config.qubits["Q1"].transitions["ge"].model_dump()
    session.commit(proposal)
    assert session.device.config.qubits["Q1"].transitions["ge"].model_dump() == ge_before


def test_separate_ef_drive_declares_both_channels(device, native_config, tmp_path):
    from qick import QickConfig

    config = device.config.model_dump()
    hardware = device.hardware.model_dump()
    hardware["generators"]["ef_q1"] = {**hardware["generators"]["drive_q1"], "channel": 5}
    config["qubits"]["Q1"]["transitions"]["ef"]["drive"] = "ef_q1"
    session = Session(
        Device(hardware, config), QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path
    )
    p = session.compile("t1_ef")
    assert 0 in p.gen_chs and 5 in p.gen_chs
    assert p.pulses["Q1__x_ge"].gen_chs == [0]
    assert p.pulses["Q1__x_ef"].gen_chs == [5]


def test_native_joint_tomography_compiles_all_bases(device, native_config, tmp_path):
    from qick import QickConfig

    configured = device.with_updates(
        {f"readout_groups.{device.readout(q)[0]}.members.{q}.threshold": 0.5 for q in ("Q1", "Q2")}
    )
    session = Session(configured, QICKBackend(None, QickConfig(native_config)), data_dir=tmp_path)
    _, _, plan, _, _, _ = session._prepare("state_tomography", {"target": "Q1,Q2"})
    program = session.backend.compile(plan)
    session._check_readouts(program, plan)
    assert len(plan.readout_events) == 9
    assert program.ro_chs[0]["trigs"] == 9
