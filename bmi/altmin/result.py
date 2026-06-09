"""
Alternating Minimization Module - 
"""
from dataclasses import dataclass

@dataclass
class AltMinRun:
    """Artifact from a single trajectory."""
    solution: BMISolution          # best solution found in this run
    gamma_history: list[float]     # objective per iteration
    converged: bool
    n_iterations: int
    init_label: str                # which strategy produced K⁽⁰⁾
    seed: int

@dataclass  
class AltMinResult:
    """Aggregate artifact from the full multi-start solve."""
    best: BMISolution              # globally best feasible solution
    runs: list[AltMinRun]          # all runs, for analysis
    n_feasible: int
    
    def feasible_runs(self) -> list[AltMinRun]: ...
    def cluster_summary(self, n_clusters: int) -> list[BMISolution]: ...
    # returns best-per-cluster, exposing the distinct local optima landscape