"""Experiment-owned update values; shared validation, without experiment dispatch."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CalibrationUpdates:
    """Working dictionary values and their optional strict device paths.

    Experiments own both mappings in analysis.py. Values without a device path
    are Notebook working settings/metrics, outside the strict device schema.
    Constructing this value never writes configuration.
    """

    working: dict = field(default_factory=dict)
    device_paths: dict[str, str] = field(default_factory=dict)
    reason: str = "This experiment has no automatic calibration updates"

    def for_device(self):
        return {path: self.working[key] for key, path in self.device_paths.items()}


def accepted_fit(result, target=None):
    """Return accepted fit parameters only for a completed acquisition."""
    if target is None:
        if len(result.targets) != 1:
            raise ValueError("Select a fit target for a multi-qubit acquisition")
        target = result.targets[0]
    if result.metadata.get("acquisition_status") != "completed":
        raise ValueError("Partial acquisitions cannot update calibration")
    fit = result.fits.get(target)
    if (
        target not in result.targets
        or result.analysis_status != "completed"
        or fit is None
        or not fit.success
    ):
        message = fit.message if fit else result.analysis_message or "No fit for selected target"
        raise ValueError("Fit rejected; inspect data and tune the experiment before applying it: " + message)
    return fit.parameters
