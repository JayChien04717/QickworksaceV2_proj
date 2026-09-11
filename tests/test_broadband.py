import numpy as np
import pytest
from dataclasses import asdict
from copy import deepcopy

from QickworkspaceV2 import ExperimentData, TraceData
from labtools.fitting import lorentzian, fit_lorentzian
from QickworkspaceV2.experiments.broadband_resonator_spectrum import analyze, plot, updates
from QickworkspaceV2.experiments.broadband_resonator_spectrum.parameters import BroadbandParameters


def spectrum(y=None, **settings):
    x = np.linspace(6600, 7000, 1601)
    if y is None:
        y = 1 + lorentzian(x, 0, -0.4, 6700, 2) + lorentzian(x, 0, -0.3, 6900, 3)
    return ExperimentData(
        "broadband_resonator_spectrum",
        {
            "Q1": TraceData(
                np.asarray(y)[:, None],
                ("frequency", "readout"),
                {"frequency": x, "readout": ["measurement"]},
                {"frequency": "MHz"},
            )
        },
        metadata={
            "acquisition_status": "completed",
            "iq_process": "abs",
            "parameters": {"count": 2, **settings},
        },
    )


def test_broadband_fits_multiple_resonances_and_does_not_choose_calibration():
    result = spectrum()
    before = deepcopy(result.metadata)
    result.fits = analyze(result)
    fit = result.fits["Q1"]
    assert fit.success and fit.model == "broadband"
    assert sorted(fit.parameters.values()) == pytest.approx([6700, 6900], abs=0.02)
    assert len(result["Q1"].metadata["multi_resonator_fit"]["details"]["indices"]) == 2
    assert result.metadata == before
    assert not updates(result, "Q1").working
    figure = plot(result)
    assert len(figure.axes[0].texts) == 2


def test_candidate_limit_and_prominence_are_analysis_settings():
    result = spectrum(count=1)
    result.fits = analyze(result)
    assert list(result.fits["Q1"].parameters.values()) == pytest.approx([6700], abs=0.02)
    result = spectrum(detection_options={"prominence_sigma": 100})
    result.fits = analyze(result)
    assert not result.fits["Q1"].success and result.fits["Q1"].parameters == {}
    assert plot(result).axes[0].lines


def test_flat_data_preserved_when_no_resonance_is_found():
    result = spectrum(np.ones(1601))
    before = result["Q1"].iq.copy()
    fit = analyze(result)["Q1"]
    assert not fit.success and "No magnitude contrast" in fit.message
    np.testing.assert_array_equal(result["Q1"].iq, before)


def test_narrow_analysis_retains_the_existing_lorentzian_fit():
    from QickworkspaceV2.experiments.resonator_spec import analyze as analyze_narrow

    result = spectrum()
    result.experiment = "resonator_spec"
    result.metadata["parameters"] = {"fit": {"model": "lorentzian"}}
    actual = analyze_narrow(result)["Q1"]
    expected = fit_lorentzian(result["Q1"].coords["frequency"], abs(result["Q1"].iq[:, 0]), unit="MHz")
    assert asdict(actual) == asdict(expected)


@pytest.mark.parametrize(
    "settings",
    [{"count": 0}, {"count": 1.5}, {"detection_options": {"prominence_sigma": -1}}, {"y_mode": "typo"}],
)
def test_invalid_broadband_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        BroadbandParameters(**settings)


def test_catalog_host_sweep_uses_the_same_broadband_analyzer(session):
    result = session.run("broadband_resonator_spectrum", start=-200, stop=200, points=8, count=3)
    assert result.analysis_status == "completed"
    assert result.fits["Q1"].model == "broadband"
    assert result.metadata["resolved_config"]["count"] == 3
    assert not result.fits["Q1"].success  # Constant transport fixture has no resonances.


def test_notebook_native_broadband_run_shows_only_final_plot_and_saves_result(
    session, device, native_config, tmp_path, monkeypatch
):
    from qick import QickConfig
    from QickworkspaceV2 import NotebookLab, Measurement, ExperimentConfig, QickSweep1D
    from QickworkspaceV2.experiments.broadband_resonator_spectrum import BroadbandResonatorSpecProgram

    lab = NotebookLab(working_config=tmp_path / "working.yaml")
    lab._measurement = Measurement(object(), QickConfig(native_config), data_path=tmp_path / "runs")
    lab.config = ExperimentConfig(device)
    figures = []

    class Handle:
        def update(self, value):
            figures[0] = value

    def display(value, **kwargs):
        figures.append(value)
        return Handle() if kwargs.get("display_id") else None

    monkeypatch.setattr("IPython.display.display", display)
    cfg = lab.config["Q1"].for_run(
        steps=401,
        res_freq_ge=QickSweep1D("freqloop", 6600, 7000),
        count=4,
        detection_options={"prominence_sigma": 0.08},
    )
    result = lab.run(BroadbandResonatorSpecProgram, cfg, py_avg=2)
    assert result.experiment == "broadband_resonator_spectrum"
    assert result.analysis_status == "completed" and result.fits["Q1"].model == "broadband"
    assert result["Q1"].iq.shape == (401, 1) and len(figures) == 2
    assert figures[0].data == ""
    assert result.path.name == "acquisition.h5"
    assert (result.path.parent / "broadband_resonator_spectrum.png").exists()
    assert lab.load(result.run_id).fits["Q1"].model == "broadband"


def test_broadband_has_its_own_catalog_identity_and_same_native_sequence(session):
    narrow = session.compile("resonator_spec", points=8)
    broad = session.compile("broadband_resonator_spectrum", points=8)
    np.testing.assert_array_equal(narrow.binprog["pmem"], broad.binprog["pmem"])
    assert type(narrow).EXPERIMENT.id == "resonator_spec"
    assert type(broad).EXPERIMENT.id == "broadband_resonator_spectrum"
    narrow_schema = session.registry.get("resonator_spec").schema()["parameters"]
    assert "count" not in narrow_schema["properties"]
    broad_schema = session.registry.get("broadband_resonator_spectrum").schema()["parameters"]
    assert {"count", "detection_options", "y_mode"} <= broad_schema["properties"].keys()
