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
        residual_history=[],
        converged=False,
        n_iterations=0,
        init_label=init_label,
        seed=seed,
    )


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
        window: int = 3,
        verbose: bool = False):

        self.spec = spec
        self.max_iter = max_iter
        self.tol = tol
        self.window = window
        self.verbose = verbose

    def run(self, block_b_init: Any, init_label: str, seed: int) -> AltMinRun:
        block_b = block_b_init
        residual_history: list[float] = []
        best_solution = None
        best_residual = float("inf")

        for i in range(self.max_iter):
            # P-step: returns ((P, Pf), mu)
            block_a, res_a = self.spec.step_a.solve(block_b)
            if block_a is None:
                return _infeasible_run(seed, init_label)

            # K-step: returns (filter_matrices, mu)
            block_b, res_b = self.spec.step_b.solve(block_a)
            if block_b is None:
                return _infeasible_run(seed, init_label)

            res = max(res_a, res_b)
            residual_history.append(res)

            # Track best solution seen so far
            if res < best_residual:
                best_residual = res
                best_solution = (block_a, block_b, res)

            if self.verbose:
                print(f"  iter {i+1:3d} | residual = {res:.6e}")

            if self._converged(residual_history):
                break

            # Early stopping if residual has been increasing for `window` steps
            if self._diverging(residual_history):
                break

        if best_solution is None:
            return _infeasible_run(seed, init_label)
        
        return AltMinRun(
            solution=best_solution,
            residual_history=residual_history,
            converged=self._converged(residual_history),
            n_iterations=len(residual_history),
            init_label=init_label,
            seed=seed)

    def _converged(self, history: list[float], window: int = 3) -> bool:
        if len(history) < window:
            return False
        recent = history[-window:]
        max_change = max(
            abs(recent[i] - recent[i-1]) / (abs(recent[i-1]) + 1e-12)
            for i in range(1, len(recent))
        )
        return max_change < self.tol

    def _diverging(self, history: list[float]) -> bool:
        if len(history) < self.window:
            return False
        recent = history[-self.window:]
        return all(recent[i] > recent[i-1] for i in range(1, len(recent)))