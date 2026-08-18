from __future__ import annotations
from dataclasses import dataclass, field

from zpk.parametrization import ZPKParams
from bmi.solution import BMISolution


@dataclass
class ZPKRun:
    """Artifact from a single ZPK alternating minimization trajectory."""
    zpk_params: ZPKParams | None      # best ZPKParams found in this run
    bmi_solution: BMISolution | None  # corresponding BMI solution
    residual_history: list[float]
    converged: bool
    n_iterations: int
    init_label: str
    seed: int


@dataclass
class ZPKResult:
    """Aggregate artifact from the full multi-start ZPK solve."""
    best_zpk: ZPKParams | None
    best_bmi: BMISolution | None
    runs: list[ZPKRun] = field(default_factory=list)
    n_feasible: int = 0

    def feasible_runs(self) -> list[ZPKRun]:
        return [r for r in self.runs if r.zpk_params is not None]