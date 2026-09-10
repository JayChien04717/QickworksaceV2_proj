"""Daily editable settings. Importing this file does not connect to hardware."""

from pathlib import Path
from QickworkspaceV2 import ExperimentConfig
from QickworkspaceV2.device.models import ProjectConfig, read_yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "lab/project.yaml"
project = ProjectConfig.model_validate(read_yaml(PROJECT))

# Change these here, or overwrite them in the notebook connection cell.
CONNECTION = project.connection.model_dump()
DATA_PATH = (PROJECT.parent / project.data_dir).resolve()


def make_config():
    """Load wiring/initial values, then apply your daily Python edits below."""
    config_all = ExperimentConfig.from_project(PROJECT)
    config_all["Q1"].update(res_gain_ge=0.15)
    config_all.update_all(res_sigma=0.01)
    # Constant readout has no envelope. Use "flat_top" to apply res_sigma.
    config_all.update_all(res_pulse_type="const")
    # config_all["Q1"].update(qb_freq_ge=2872.65, pi_gain_ge=0.2)
    return config_all
