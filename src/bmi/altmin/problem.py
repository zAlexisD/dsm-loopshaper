from __future__ import annotations
import numpy as np
from joblib import Parallel, delayed

from altmin.solver import AltMinSolver, AltMinSpec
from altmin.result import AltMinRun, AltMinResult
from altmin.init import InitStrategy, MixedInit, LHSInit, ZeroInit

from bmi.problem import BMIProblem
from bmi.solution import BMISolution
from bmi.altmin.steps import BMIPStep, BMIKStep
from bmi.altmin.init import SynthesizeNTFInit, WarmStartInit
from bmi.altmin.parametrization import FreeFilterParametrization


def _worker(spec: AltMinSpec, K_init, label: str, seed: int, max_iter: int, 
            tol: float, max_consecutive_failures: int,verbose: bool) -> AltMinRun:
    solver = AltMinSolver(spec, max_iter=max_iter, tol=tol, 
                          max_consecutive_failures=max_consecutive_failures, verbose=verbose)
    return solver.run(K_init, label, seed)


class BMIAltMinProblem:
    """User-facing entry point for BMI alternating minimization."""

    def __init__(
        self,
        bmi_problem: BMIProblem,
        osr: int,
        filter_order: int,
        n_inputs: int = 1,
        n_outputs: int = 1,
        init_strategy: InitStrategy | None = None,
        n_starts: int = 20,
        max_iter: int = 50,
        tol: float = 1e-4,
        n_jobs: int = -1,
        max_consecutive_failures: int = 3,
        verbose: bool = False):

        self.bmi_problem = bmi_problem
        self.osr = osr
        self.filter_order = filter_order
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.init_strategy = init_strategy or self._default_init()
        self.max_consecutive_failures = max_consecutive_failures
        self.verbose = verbose

    def _default_init(self) -> MixedInit:
        return MixedInit([
            (SynthesizeNTFInit(self.osr, self.filter_order), 0.5),
            (WarmStartInit(self.bmi_problem), 0.1),
            (LHSInit(self.filter_order, self.n_inputs, self.n_outputs), 0.3),
            (ZeroInit(self.filter_order, self.n_inputs, self.n_outputs), 0.1)
            ])

    def solve(self, seed: int = 0) -> AltMinResult:
        rng = np.random.default_rng(seed)

        spec = AltMinSpec(
            step_a=BMIPStep(self.bmi_problem),
            step_b=BMIKStep(self.bmi_problem, FreeFilterParametrization(self.filter_order, self.n_inputs, self.n_outputs)),
        )

        inits = self.init_strategy.generate(self.n_starts, rng, order=self.filter_order, gamma=self.bmi_problem.gamma)
        seeds = rng.integers(0, 2**31, size=len(inits))

        if self.n_jobs == 1:
            # Serial path — exceptions surface cleanly with full tracebacks
            runs = [
                _worker(spec, K, label, int(s), self.max_iter, self.tol, self.max_consecutive_failures, self.verbose)
                for (K, label), s in zip(inits, seeds)
            ]
        else:
            runs = Parallel(n_jobs=self.n_jobs)(
                delayed(_worker)(spec, K, label, int(s), self.max_iter, self.tol, self.max_consecutive_failures,self.verbose)
                for (K, label), s in zip(inits, seeds)
            )

        return self._aggregate(runs)

    def _aggregate(self, runs: list[AltMinRun]) -> AltMinResult:
        feasible = [r for r in runs if r.solution is not None]
        if not feasible:
            return AltMinResult(best=
                                BMISolution(feasible=False), 
                                runs=runs, n_feasible=0)

        # Best = lowest final mu across feasible runs
        best_run = min(
            feasible,
            key=lambda r: r.residual_history[-1] if r.residual_history else float("inf"))

        (P, P_f), (Af, Bf, Cf, Df), mu = best_run.solution

        best = BMISolution(
            P=P, Pf=P_f,
            Af=Af, Bf=Bf, Cf=Cf, Df=Df,
            mu=mu,
            feasible=True)
        
        return AltMinResult(best=best, runs=runs, n_feasible=len(feasible))