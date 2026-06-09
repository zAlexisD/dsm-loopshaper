"""
Alternating Minimization Module - Problem definition
"""
from scipy.signal import StateSpace
from joblib import Parallel
import numpy as np

class AltMinProblem:
    def __init__(
        self,
        plant: StateSpace,           # from scipy.signal
        osr: int,
        order: int | None = None,    # None → OrderStatus.UNSET
        init_strategy: InitStrategy | None = None,  # None → MixedInit default
        n_starts: int = 20,
        max_iter: int = 50,
        tol: float = 1e-4,
        n_jobs: int = -1,
    ):
        self._order_status = OrderStatus.UNSET
        if order is not None:
            self.set_order(order)
        self._init_strategy = init_strategy or self._default_init()
        ...

    # --- order management (your existing pattern) ---
    def set_order(self, order: int) -> None:
        self._order = order
        self._order_status = OrderStatus.FIXED
        self._init_variables()

    def _set_derived_order(self, order: int) -> None:
        self._order = order
        self._order_status = OrderStatus.DERIVED
        self._init_variables()

    def _init_variables(self) -> None:
        # set up CVXPY variable shapes now that order is known
        ...

    @property
    def is_ready(self) -> bool:
        return self._order_status != OrderStatus.UNSET

    # --- main entry point ---
    def solve(self, seed: int = 0) -> AltMinResult:
        if not self.is_ready:
            raise RuntimeError("Filter order not set. Call set_order() first.")
        
        rng = np.random.default_rng(seed)
        problem_data = self._freeze()           # → AltMinProblemData
        
        # 1. generate init pool
        inits = self._init_strategy.generate(self.n_starts, rng)  # [(K, label)]
        
        # 2. run in parallel
        seeds = rng.integers(0, 2**31, size=len(inits))
        solver = AltMinSolver(problem_data, self.max_iter, self.tol)
        
        runs: list[AltMinRun] = Parallel(n_jobs=self.n_jobs)(
            delayed(solver.run)(K, int(s))
            for (K, _label), s in zip(inits, seeds)
        )
        
        # 3. aggregate
        return self._aggregate(runs)

    def _freeze(self) -> AltMinProblemData: ...

    def _aggregate(self, runs: list[AltMinRun]) -> AltMinResult:
        feasible = [r for r in runs if r.solution.feasible]
        if not feasible:
            # return an infeasible AltMinResult, don't raise
            return AltMinResult(best=_infeasible_solution(), runs=runs, n_feasible=0)
        best = min(feasible, key=lambda r: r.solution.gamma)
        return AltMinResult(best=best.solution, runs=runs, n_feasible=len(feasible))