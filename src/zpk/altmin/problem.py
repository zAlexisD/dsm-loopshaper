from __future__ import annotations
import numpy as np
from joblib import Parallel, delayed

from bmi.problem import BMIProblem
from bmi.solution import BMISolution
from zpk.parametrization import ZPKParams
from zpk.altmin.steps import ZPKPoleStep, ZPKZeroStep
from zpk.altmin.result import ZPKRun, ZPKResult
from zpk.altmin.init import InitStrategy, MixedZPKInit, UniformPoleInit, WarmStartZPKInit, SynthesizeNTFZPKInit


def _infeasible_run(seed: int, init_label: str) -> ZPKRun:
    return ZPKRun(
        zpk_params=None,
        bmi_solution=BMISolution(),
        residual_history=[],
        converged=False,
        n_iterations=0,
        init_label=init_label,
        seed=seed)


def _worker(
    bmi_problem: BMIProblem,
    zpk_init: ZPKParams,
    n_zeros: int,
    init_label: str,
    seed: int,
    max_iter: int,
    tol: float,
    tol_nontrivial: float = 0.1,
    force_dc_zero: bool = True,
    verbose: bool = False
    ) -> ZPKRun:

    pole_step = ZPKPoleStep(bmi_problem,n_zeros,tol_nontrivial,force_dc_zero)
    zero_step = ZPKZeroStep(bmi_problem,tol_nontrivial,force_dc_zero)

    pole_params = zpk_init.poles
    zero_params = zpk_init.zeros
    residual_history: list[float] = []
    bmi_solution = None

    for i in range(max_iter):
        # ── Pole step: LMI in (Cf, P, Pf, mu) ───────────────────────────
        new_zeros, P, Pf, mu_p = pole_step.solve(pole_params)
        if new_zeros is None:
            return _infeasible_run(seed, init_label)
        # # P regularization
        # min_eig = np.min(np.linalg.eigvalsh(P))
        # if min_eig < 1e-7:
        #     P = P + (1e-7 - min_eig + 1e-8) * np.eye(P.shape[0])
        
        zero_params = new_zeros

        # ── Zero step: LMI in (a_var, mu) with P, Pf fixed ───────────────
        new_poles, bmi_solution, mu_z = zero_step.solve(
            (zero_params, pole_params, P, Pf)
        )
        if new_poles is None:
            return _infeasible_run(seed, init_label)
        pole_params = new_poles

        # Track worst mu across both steps this iteration
        residual_history.append(max(mu_p, mu_z))

        if verbose:
                print(f"  iter {i+1:3d} | mu_p = {mu_p:.6e} | mu_z = {mu_z:.6e}")

        if _converged(residual_history, tol):
            break

    best_zpk = ZPKParams(poles=pole_params, zeros=zero_params)
    return ZPKRun(
        zpk_params=best_zpk,
        bmi_solution=bmi_solution,
        residual_history=residual_history,
        converged=_converged(residual_history, tol),
        n_iterations=len(residual_history),
        init_label=init_label,
        seed=seed,
    )


def _converged(history: list[float], tol: float) -> bool:
    if len(history) < 2:
        return False
    return abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12) < tol


class ZPKAltMinProblem:
    """
    User-facing entry point for ZPK alternating minimization.

    Outer loop alternates between two LMI steps:
      - Pole step: fix poles → LMI in (Cf, P, Pf, mu)
      - Zero step: fix zeros + (P, Pf) → LMI in (a_var, mu)

    Both steps are true LMIs — no nested alternation needed.

    Parameters
    ----------
    bmi_problem : BMIProblem
        Holds output filter and LMI builders.
    n_poles : int
        Number of poles (must be even for conjugate pairs only).
    n_zeros : int
        Number of zeros (real valued).
    init_strategy : InitStrategy, optional
        ZPK initialization strategy. Defaults to UniformPoleInit.
    n_starts : int
        Number of random restarts.
    max_iter : int
        Maximum outer pole/zero alternation iterations.
    tol : float
        Convergence tolerance on relative mu improvement.
    n_jobs : int
        Joblib parallelism. 1 = serial.
    """

    def __init__(
        self,
        bmi_problem: BMIProblem,
        n_poles: int,
        n_zeros: int,
        init_strategy: InitStrategy | None = None,
        n_starts: int = 10,
        max_iter: int = 20,
        tol: float = 1e-4,
        n_jobs: int = 1,
        tol_nontrivial: float = 0.1,
        force_dc_zero=True,
        verbose: bool = False):

        self.bmi_problem = bmi_problem
        self.n_poles = n_poles
        self.n_zeros = n_zeros
        self.init_strategy = init_strategy or self._default_init()
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.tol_nontrivial = tol_nontrivial
        self.force_dc_zero  = force_dc_zero
        self.verbose = verbose

    def _default_init(self) -> MixedZPKInit:
        return MixedZPKInit([
            (UniformPoleInit(self.n_poles, self.n_zeros),0.4),
            (WarmStartZPKInit(self.bmi_problem),0.1),
            (SynthesizeNTFZPKInit(self.bmi_problem.n_f),0.5)
            ])

    def solve(self, seed: int = 0) -> ZPKResult:
        rng = np.random.default_rng(seed)
        inits = self.init_strategy.generate(self.n_starts, rng, order=self.bmi_problem.n_f, gamma=self.bmi_problem.gamma)
        seeds = rng.integers(0, 2**31, size=len(inits))

        if self.n_jobs == 1:
            runs = [
                _worker(
                    self.bmi_problem,
                    zpk, self.n_zeros, label, int(s), self.max_iter, self.tol,
                    self.tol_nontrivial,self.verbose,self.force_dc_zero)

                for (zpk, label), s in zip(inits, seeds)
            ]
        else:
            runs = Parallel(n_jobs=self.n_jobs)(
                delayed(_worker)(
                    self.bmi_problem,
                    zpk, self.n_zeros, label, int(s), self.max_iter, self.tol, 
                    self.tol_nontrivial,self.verbose,self.force_dc_zero)

                for (zpk, label), s in zip(inits, seeds)
            )

        return self._aggregate(runs)

    def _aggregate(self, runs: list[ZPKRun]) -> ZPKResult:
        feasible = [r for r in runs if r.zpk_params is not None]
        if not feasible:
            return ZPKResult(
                best_zpk=None,
                best_bmi=None,
                runs=runs,
                n_feasible=0)
        
        best = min(
            feasible,
            key=lambda r: r.residual_history[-1] if r.residual_history else float("inf"))
        
        return ZPKResult(
            best_zpk=best.zpk_params,
            best_bmi=best.bmi_solution,
            runs=runs,
            n_feasible=len(feasible))