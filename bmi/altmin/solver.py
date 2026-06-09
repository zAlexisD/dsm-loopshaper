"""
Alternating Minimizatin Module - Solver
"""
import numpy as np

class AltMinSolver:
    """
    Single-trajectory alternating minimization.
    
    Alternates between:
      - P-step: fix K, solve convex SDP for (P, gamma)
      - K-step: fix P, solve convex SDP for (K, gamma)
    """
    def __init__(
        self,
        problem_data: AltMinProblemData,  # frozen snapshot, safe to pickle
        max_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
    ): ...

    def run(self, K_init: np.ndarray, seed: int) -> AltMinRun:
        gamma_history = []
        K = K_init
        for i in range(self.max_iter):
            P, gamma_p = self._solve_p_step(K)       # convex SDP
            if P is None:                              # infeasible — bail
                return self._infeasible_run(seed)
            K, gamma_k = self._solve_k_step(P)       # convex SDP
            if K is None:
                return self._infeasible_run(seed)
            gamma_history.append(gamma_k)
            if self._converged(gamma_history):
                break
        solution = BMISolution(P=P, A_f=..., gamma=gamma_k, feasible=True)
        return AltMinRun(solution, gamma_history, converged=..., ...)

    def _solve_p_step(self, K) -> tuple[np.ndarray | None, float]: ...
    def _solve_k_step(self, P) -> tuple[np.ndarray | None, float]: ...
    def _converged(self, history) -> bool:
        if len(history) < 2: return False
        return abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12) < self.tol