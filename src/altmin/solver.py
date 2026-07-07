from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from altmin.step import AlternatingStep
from altmin.result import AltMinRun


@dataclass(frozen=True)
class AltMinSpec:
    """
    Picklable bundle describing one trajectory's problem.
    step_a / step_b must themselves be picklable (no live CVXPY state).
    """
    step_a: AlternatingStep
    step_b: AlternatingStep


def _infeasible_run(seed: int, init_label: str) -> AltMinRun:
    return AltMinRun(
        solution=None,
        gamma_history=[],
        converged=False,
        n_iterations=0,
        init_label=init_label,
        seed=seed)


class AltMinSolver:
    """
    Generic alternating minimization: fix block_a, solve for block_b;
    fix block_b, solve for block_a; repeat until the objective converges.

    Domain-specific meaning of "block" and "objective" lives entirely
    inside the AlternatingStep implementations (step_a, step_b).
    """

    def __init__(
        self,
        spec: AltMinSpec,
        max_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False):

        self.spec = spec
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def run(self, block_b_init: Any, init_label: str, seed: int) -> AltMinRun:
        block_b = block_b_init
        history: list[float] = []

        for i in range(self.max_iter):
            block_a, obj = self.spec.step_a.solve(block_b)
            if block_a is None:
                return _infeasible_run(seed, init_label)

            block_b, obj = self.spec.step_b.solve(block_a)
            if block_b is None:
                return _infeasible_run(seed, init_label)

            history.append(float(obj))
            if self.verbose:
                print(f"  iter {i+1:3d} | objective = {obj:.6f}")
                print(49*"-"+"-\n")

            if self._converged(history):
                break

        return AltMinRun(
            solution=(block_a, block_b, history[-1] if history else float("inf")),
            gamma_history=history,
            converged=self._converged(history),
            n_iterations=len(history),
            init_label=init_label,
            seed=seed)

    def _converged(self, history: list[float]) -> bool:
        if len(history) < 2:
            return False
        return abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12) < self.tol