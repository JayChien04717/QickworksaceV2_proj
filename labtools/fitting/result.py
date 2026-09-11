from dataclasses import dataclass, field

@dataclass
class FitResult:
    model: str
    success: bool
    parameters: dict[str, float] = field(default_factory=dict)
    errors: dict[str, float | None] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    r_squared: float | None = None
    message: str = ""
    x_fit: list[float] = field(default_factory=list)
    y_fit: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
