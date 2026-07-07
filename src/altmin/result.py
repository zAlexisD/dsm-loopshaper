from dataclasses import dataclass, field
from typing import Any

@dataclass
class AltMinRun:
    """Artifact from a single trajectory. `solution` is domain-specific."""
    solution: Any
    gamma_history: list[float]
    converged: bool
    n_iterations: int
    init_label: str
    seed: int


@dataclass
class AltMinResult:
    """Aggregate artifact from a full multi-start solve."""
    best: Any
    runs: list[AltMinRun] = field(default_factory=list)
    n_feasible: int = 0

    def feasible_runs(self) -> list[AltMinRun]:
        return [r for r in self.runs if r.solution is not None]